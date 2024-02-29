# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from utils.text_control_example import collate_all
import pickle
from manotorch.manolayer import ManoLayer
from pytorch3d.transforms import rotation_6d_to_matrix,axis_angle_to_matrix,matrix_to_axis_angle
from os.path import join
from utils.data_util import *
import torch.nn as nn

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    # n_frames = 196
    # n_frames = 60
    if args.dataset == 'gazehoi_stage1':
        n_frames = 60
    elif args.dataset == 'gazehoi_stage2':
        n_frames = 180
    elif args.dataset == 'gazehoi_stage0_flag2_lowfps' or args.dataset == 'gazehoi_stage0_flag2_lowfps_global' or args.dataset == 'gazehoi_stage0_1obj':
        n_frames = 69
    elif args.dataset.startswith('gazehoi_stage0'):
        n_frames = 345

    is_using_data = True
    # mean = np.load('dataset/gazehoi_mean.npy') 
    # std = np.load('dataset/gazehoi_std.npy')
    # is_using_data = not any([args.text_prompt])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')

    hints = None

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)

    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []
    all_hint = []
    all_hint_for_vis = []
    all_seqs = []
    all_obj_index = []
    all_obj_activeflag = []
    all_gt = []

    all_active_flag = []
    all_move_flag = []
    all_flag = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        length = model_kwargs['y']['lengths'].cpu()
        sample_fn = diffusion.p_sample_loop
        # print(args.batch_size, model.njoints, model.nfeats, n_frames)
        if not args.dataset.startswith('gazehoi_stage0'):
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample.permute(0, 3, 2, 1).squeeze(2).contiguous().cpu()
          
            global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_motion_6d_mean.npy'))
            global_std = torch.from_numpy(np.load('dataset/gazehoi_global_motion_6d_std.npy'))
            local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_motion_6d_mean.npy'))
            local_std = torch.from_numpy(np.load('dataset/gazehoi_local_motion_6d_std.npy'))

            
            # sample = sample * global_std + local_mean 
            sample = sample * local_std + local_mean 
            sample = local2global_rot6d_by_matrix_repair(sample,length)
            print('repair')
            sample = rot6d2axis(sample)
            # 转为绝对表示
            # sample = torch.cumsum(sample_r,dim=1)
            # sample[:,:,9:] = sample_r[:,:,9:]
            # # rot6d 转为 轴角
            # sample_axis = matrix_to_axis_angle(rotation_6d_to_matrix(sample[:,:,3:].reshape(81,-1,16,6))).reshape(81,-1,48)
            # sample = torch.cat([sample[:,:,:3],sample_axis],dim=-1)
            if args.dataset == 'gazehoi_stage1':
                hint = model_kwargs['y']['hint'].cpu()
                hint = hint * global_std + global_mean
                hint = rot6d2axis(hint)
                all_hint.append(hint.data.cpu().numpy())  
        elif args.dataset == 'gazehoi_stage0_1':
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample.permute(0, 3, 2, 1).squeeze(2).contiguous().cpu()
            obj_global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_global_std = torch.from_numpy(np.load('dataset/gazehoi_global_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_local_std = torch.from_numpy(np.load('dataset/gazehoi_local_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            sample = sample.unsqueeze(2).repeat(1,1,4,1).reshape(81,345,-1)
            sample = sample * obj_local_std + obj_local_mean  # bs, nf, 36
            bs,nf,_ = sample.shape
            # sample = sample.reshape(bs,nf,4,9)
            sample = obj_local2global_matrix(sample)
        elif args.dataset=='gazehoi_stage0_flag2_lowfps_global':
            obj_global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_global_std = torch.from_numpy(np.load('dataset/gazehoi_global_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            gt = model_kwargs['y']['gt']
            gt_flag = model_kwargs['y']['flag']
            bs = gt.shape[0]
            nf = gt.shape[2]
            # print(gt.shape)
            gt = gt.permute(0,2,1,3).contiguous().reshape(bs,nf,-1).cpu()
            gt = gt * obj_global_std + obj_global_mean  
            gt = obj_rot6d2matrix(gt)
            all_gt.append(gt.numpy())
            sample,flag = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample.permute(0, 3, 2, 1).squeeze(2).contiguous().cpu()
            
            sample = sample * obj_global_std + obj_global_mean  # bs, nf, 36
            # sample = obj_local2global_matrix(sample)
            sample = obj_rot6d2matrix(sample)

            active_flag = flag[0]
            move_flag = flag[1]
            print(active_flag.shape, move_flag.shape)
            logsoftmax = nn.LogSoftmax()
            active_flag = logsoftmax(active_flag)
            move_flag = logsoftmax(move_flag)
            # print(active_flag,move_flag)
            print(torch.argmax(active_flag,dim=-1),torch.argmax(move_flag,dim=-1))

            all_active_flag.append(torch.argmax(active_flag,dim=-1).cpu().numpy())
            all_move_flag.append(torch.argmax(move_flag,dim=-1).cpu().numpy())
            all_flag.append(gt_flag.cpu().numpy())
            

            # 处理flag ([81, 345, 4])
            # pred_move = torch.sum(flag,dim=1) # #b,4
            # obj_index = torch.argmax(pred_move,dim=-1)
            # all_obj_index.append(obj_index)
            # flag = flag.permute(0,2,1)
            # active_flag = flag[torch.arange(bs),obj_index] # b,nf
            # all_obj_activeflag.append(active_flag)
            # all_obj_activeflag.append(flag.cpu().numpy())
            # tgt_obj = obj_mesh[torch.arange(bs),obj_index]
        elif args.dataset.startswith('gazehoi_stage0_flag'):
            sample,flag = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample.permute(0, 3, 2, 1).squeeze(2).contiguous().cpu()
            obj_global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_global_std = torch.from_numpy(np.load('dataset/gazehoi_global_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_local_std = torch.from_numpy(np.load('dataset/gazehoi_local_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            sample = sample * obj_local_std + obj_local_mean  # bs, nf, 36
            sample = obj_local2global_matrix(sample)


            # 处理flag ([81, 345, 4])
            pred_move = torch.sum(flag,dim=1) # #b,4
            obj_index = torch.argmax(pred_move,dim=-1)
            all_obj_index.append(obj_index)
            flag = flag.permute(0,2,1)
            # active_flag = flag[torch.arange(bs),obj_index] # b,nf
            # all_obj_activeflag.append(active_flag)
            all_obj_activeflag.append(flag.cpu().numpy())
            # tgt_obj = obj_mesh[torch.arange(bs),obj_index]


        else:

            obj_global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_global_std = torch.from_numpy(np.load('dataset/gazehoi_global_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            obj_local_std = torch.from_numpy(np.load('dataset/gazehoi_local_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            # sample = sample * obj_global_std + obj_local_mean  # bs, nf, 36
            sample = sample * obj_local_std + obj_local_mean  # bs, nf, 36
            sample = obj_local2global_matrix(sample)


        all_seqs.append(model_kwargs['y']['seq_name'])

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    print(total_num_samples)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_seqs =  [element for sublist in all_seqs for element in sublist]
    # all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    if args.dataset == 'gazehoi_stage1':
        all_hint = np.concatenate(all_hint, axis=0)[:total_num_samples]
        # all_hint_for_vis = np.concatenate(all_hint_for_vis, axis=0)[:total_num_samples]
    # if args.dataset.startswith("gazehoi_stage0_flag"):
    #     all_obj_index = all_obj_index[0]
    #     all_obj_activeflag = np.concatenate(all_obj_activeflag, axis=0)[:total_num_samples]
    if args.dataset.startswith("gazehoi_stage0_flag2_lowfps_global"):
        all_gt = np.concatenate(all_gt,axis=0)
        all_flag = np.concatenate(all_flag,axis=0)
        all_active_flag = np.concatenate(all_active_flag,axis=0)
        all_move_flag = np.concatenate(all_move_flag,axis=0)
        
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    if args.dataset == 'gazehoi_stage1' or args.dataset == 'gazehoi_stage1_new':
        np.save(npy_path,
                {'motion': all_motions, 'lengths': all_lengths, "hint": all_hint, 'seqs':all_seqs,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    elif args.dataset == 'gazehoi_stage2' or args.dataset=='gazehoi_stage0':
        np.save(npy_path,
                {'motion': all_motions, 'lengths': all_lengths, 'seqs':all_seqs,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    elif args.dataset=="gazehoi_stage0_flag2_lowfps_global":
        np.save(npy_path,
                {'motion': all_motions, 'lengths': all_lengths, 'seqs':all_seqs,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,
                "obj_index":all_obj_index,'flag':all_obj_activeflag,'all_gt':all_gt,
                'all_flag':all_flag, 'all_active_flag':all_active_flag, 'all_move_flag':all_move_flag})

    elif args.dataset.startswith("gazehoi_stage0_flag"):
        np.save(npy_path,
                {'motion': all_motions, 'lengths': all_lengths, 'seqs':all_seqs,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,
                "obj_index":all_obj_index,'flag':all_obj_activeflag})
        
    # if len(all_hint) != 0:
    #     from utils.simple_eval import simple_eval
    #     results = simple_eval(all_motions, all_hint, n_joints)
    #     print(results)
    res_save_path = join(out_path, 'eval_results.txt')
    all_motions = torch.tensor(all_motions)
    all_hint = torch.tensor(all_hint)
    stage1_eval(all_motions, all_hint, all_seqs, res_save_path)
    # vis_gen(all_motions,all_seqs,res_save_path,vis_num)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

def cal_goal_err(x, hint):
    """
    每个joints的平均偏差?
    """
    mask_hint = hint.view(hint.shape[0], hint.shape[1], -1).sum(dim=-1, keepdim=True) != 0
    x_ = x
    bs,nf,_ = x_.shape
    loss = torch.norm((x_.reshape(bs,nf,-1) - hint.reshape(bs,nf,-1)) * mask_hint, dim=-1)
    # print(loss[:,0])
    # loss = loss[:,0] # bs,17
    # loss = torch.sum(loss) / bs
    loss = torch.mean(loss)
    return loss

def stage1_eval(pred_motion, all_hint, all_seqs, res_save_path):
    pred_motion = pred_motion
    hint = all_hint
    goal_err = cal_goal_err(pred_motion,hint)

    num_samples = pred_motion.shape[0]


    datapath = '/root/code/seqs/1205_data/'
    manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
    obj_path = '/root/code/seqs/object/'

    total_hand_T_error = 0
    total_hand_R_error = 0
    total_mpjpe = 0 # root-relative
    total_traj_invalid_10 = 0
    total_traj_invalid_20 = 0
    total_traj_num = 0
    total_hand_R_error = 0

    for i in range(pred_motion.shape[0]):
        seq = all_seqs[i]
        seq_path = join(datapath,seq)

        meta_path = join(seq_path,'meta.pkl')
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
            
        active_obj = meta['active_obj']
        obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))
        
        goal_index = meta['goal_index']
        obj_pose = torch.tensor(obj_pose[:goal_index+1]).float()

        if goal_index < 59:
            hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))[:goal_index+1]
        else:
            hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))[goal_index-59:goal_index+1]

        hand_trans = hand_params[:,:3]
        hand_rot = hand_params[:,3:6]
        hand_theta = hand_params[:,3:51]
        mano_beta = hand_params[:,51:]

        ## 倒序
        # pred_trans = torch.flip(pred_motion[i,:goal_index+1,:3],dims=[0])
        # pred_theta = torch.flip(pred_motion[i,:goal_index+1,3:],dims=[0])
        # pred_rot = torch.flip(pred_motion[i,:goal_index+1,3:6],dims=[0])

        ## 正序
        if goal_index < 59:
            pred_trans = pred_motion[i,59-goal_index:,:3]
            pred_theta = pred_motion[i,59-goal_index:,3:]
            pred_rot = pred_motion[i,59-goal_index:,3:6]
        else:
            pred_trans = pred_motion[i,:,:3]
            pred_theta = pred_motion[i,:,3:]
            pred_rot = pred_motion[i,:,3:6]

        # print(goal_index,pred_theta.shape, mano_beta.shape)
        pred_output = manolayer(pred_theta, mano_beta)
        # 相对表示
        pred_joints = pred_output.joints - pred_output.joints[:, 0].unsqueeze(1)
        gt_output = manolayer(hand_theta, mano_beta)
        # 相对表示
        gt_joints = gt_output.joints - gt_output.joints[:, 0].unsqueeze(1)
        mpjpe = torch.sum(torch.norm(pred_joints - gt_joints, dim=-1)) / (hand_trans.shape[0] * 21)
        total_mpjpe += mpjpe

        hand_T_error = torch.sum(torch.norm(hand_trans-pred_trans,p=2,dim=-1)) / hand_trans.shape[0]
        total_hand_T_error += hand_T_error

        hand_rot = axis_angle_to_matrix(hand_rot)
        pred_rot = axis_angle_to_matrix(pred_rot)
        hand_rot = torch.einsum('...ij->...ji', [hand_rot])
        hand_R_error = (torch.einsum('fpn,fnk->fpk',hand_rot,pred_rot) - torch.eye(3).unsqueeze(0).repeat(hand_rot.shape[0],1,1)).reshape(-1,9) # nf,3,3
        hand_R_error =  torch.sum(torch.norm(hand_R_error,dim=-1)) /hand_trans.shape[0]
        total_hand_R_error += hand_R_error

        traj_error = torch.norm(hand_trans-pred_trans,p=2,dim=1) 
        traj_invalid_10 = torch.sum(traj_error > 0.1)
        traj_invalid_20 = torch.sum(traj_error > 0.2)
        total_traj_invalid_10 += traj_invalid_10
        total_traj_invalid_20 += traj_invalid_20
        total_traj_num += traj_error.shape[0]
    
    goal_err = goal_err.numpy()
    traj_err_10 = (total_traj_invalid_10/total_traj_num).numpy()
    traj_err_20 = (total_traj_invalid_20/total_traj_num).numpy()
    hand_T_err = (total_hand_T_error/num_samples).numpy()
    hand_R_err = (total_hand_R_error/num_samples).numpy()
    mpjpe = (total_mpjpe/num_samples).numpy()
    with open(res_save_path, 'w') as f:
        f.write(f'Goal Error:{goal_err:.6f}\n')
        f.write(f'Traj Error (<10cm):{traj_err_10:.6f}\n')
        f.write(f'Traj Error (<20cm):{traj_err_20:.6f}\n')
        f.write(f'Hand Trans Error:{hand_T_err:.6f}\n')
        f.write(f'Hand Rot Error:{hand_R_err:.6f}\n')
        f.write(f'MPJPE :{mpjpe:.6f}\n')




def load_dataset(args, max_frames, n_frames):
    # print(args.dataset)
    # data = get_dataset_loader(name=args.dataset,
    #                           batch_size=args.batch_size,
    #                           num_frames=max_frames,
    #                           split='train',
    #                           hml_mode='train',hint_type=args.hint_type)
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train',hint_type=args.hint_type)
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
