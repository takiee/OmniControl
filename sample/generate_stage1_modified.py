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
    all_seqs = []
    all_gt = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        length = model_kwargs['y']['lengths'].cpu()
        sample_fn = diffusion.p_sample_loop
        if not args.dataset.startswith('gazehoi_stage0'):
            global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_motion_6d_mean.npy'))
            global_std = torch.from_numpy(np.load('dataset/gazehoi_global_motion_6d_std.npy'))
            local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_motion_6d_mean.npy'))
            local_std = torch.from_numpy(np.load('dataset/gazehoi_local_motion_6d_std.npy'))
            print('!!!!!')
            if args.dataset.startswith('gazehoi_stage1'):
                hint = model_kwargs['y']['hint'].cpu()
                # print(torch.sum(hint[0]))
                mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0
                hint = (hint * global_std + global_mean)*mask_hint
                hint = rot6d2axis(hint)
                all_hint.append(hint.data.cpu().numpy()) 
                # print(hint[0],hint[0][length[0]-1])
            dump_steps = 750
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=dump_steps,
                noise=None,
                const_noise=False,
            )
            sample = sample.permute(0, 3, 2, 1).squeeze(2).contiguous().cpu()
          
            
            
            sample = sample * local_std + local_mean 
            sample = local2global_rot6d_by_matrix(sample)
            sample = rot6d2axis(sample)
             

        all_seqs.append(model_kwargs['y']['seq_name'])

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    print(total_num_samples)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_seqs =  [element for sublist in all_seqs for element in sublist]
    if args.dataset.startswith('gazehoi_stage1'):
        all_hint = np.concatenate(all_hint, axis=0)[:total_num_samples]
        
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    if args.dataset.startswith('gazehoi_stage1') :
        np.save(npy_path,
                {'motion': all_motions, 'lengths': all_lengths, "hint": all_hint, 'seqs':all_seqs,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    res_save_path = join(out_path, 'eval_results.txt')
    stage1_eval(all_motions, all_seqs, res_save_path)
    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

def stage1_eval(pred_motion, seqs,res_save_path):
    pred_motion = torch.tensor(pred_motion)
    num_samples = pred_motion.shape[0]

    datapath = '/root/code/seqs/0303_data/'
    manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
    obj_path = '/root/code/seqs/object/'

    total_hand_T_error = 0
    total_hand_R_error = 0
    total_mpjpe = 0 # root-relative
    total_traj_invalid_10 = 0
    total_traj_invalid_20 = 0
    total_traj_num = 0
    total_hand_R_error = 0
    total_goal_mpjpe = 0

    for i in range(num_samples):
        seq = seqs[i]
        seq_path = join(datapath,seq)

        meta_path = join(seq_path,'meta.pkl')
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
            
        active_obj = meta['active_obj']
        obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))
        
        goal_index = meta['goal_index']
        obj_pose = torch.tensor(obj_pose[:goal_index]).float()

        if goal_index < 59:
            hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))[:goal_index]
            pred_motion_i = pred_motion[i,:goal_index]

        else:
            hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))[goal_index-60:goal_index]
            pred_motion_i = pred_motion[i]
        


        hand_trans = hand_params[:,:3]
        hand_rot = hand_params[:,3:6]
        hand_theta = hand_params[:,3:51]
        mano_beta = hand_params[:,51:]

        pred_trans = pred_motion_i[:,:3]
        pred_rot = pred_motion_i[:,3:6]
        pred_theta = pred_motion_i[:,3:51]


        pred_output = manolayer(pred_theta, mano_beta)
        pred_joints = pred_output.joints - pred_output.joints[:, 0].unsqueeze(1)

        gt_output = manolayer(hand_theta, mano_beta)
        gt_joints = gt_output.joints - gt_output.joints[:, 0].unsqueeze(1)

        goal_mpjpe = torch.mean(torch.norm((gt_joints[-1] - pred_joints[-1]),dim=-1))
        total_goal_mpjpe += goal_mpjpe

        mpjpe = torch.sum(torch.norm(pred_joints - gt_joints, dim=-1)) / (hand_trans.shape[0] * 21)
        total_mpjpe += mpjpe

        hand_T_error = torch.sum(torch.norm(hand_trans-pred_trans,dim=-1)) / hand_trans.shape[0]
        total_hand_T_error += hand_T_error

        hand_rot = axis_angle_to_matrix(hand_rot)
        pred_rot = axis_angle_to_matrix(pred_rot)
        hand_rot = torch.einsum('...ij->...ji', [hand_rot])
        hand_R_error = (torch.einsum('fpn,fnk->fpk',hand_rot,pred_rot) - torch.eye(3).unsqueeze(0).repeat(hand_rot.shape[0],1,1)).reshape(-1,9) # nf,3,3 --nf,9
        hand_R_error =  torch.mean(torch.norm(hand_R_error,dim=-1))
        total_hand_R_error += hand_R_error

        traj_error = torch.norm(hand_trans-pred_trans,p=2,dim=1) 
        traj_invalid_10 = torch.sum(traj_error > 0.1)
        traj_invalid_20 = torch.sum(traj_error > 0.2)
        total_traj_invalid_10 += traj_invalid_10
        total_traj_invalid_20 += traj_invalid_20
        total_traj_num += traj_error.shape[0]
    
    traj_err_10 = (total_traj_invalid_10/total_traj_num).numpy()
    traj_err_20 = (total_traj_invalid_20/total_traj_num).numpy()
    hand_T_err = (total_hand_T_error/num_samples).numpy()
    hand_R_err = (total_hand_R_error/num_samples).numpy()
    mpjpe = (total_mpjpe/num_samples).numpy()
    goal_mpjpe = (total_goal_mpjpe/num_samples).numpy()
    with open(res_save_path, 'w') as f:
        f.write(f'Goal MPJPE:{goal_mpjpe:.6f}\n')
        f.write(f'Traj Error (>10cm):{traj_err_10:.6f}\n')
        f.write(f'Traj Error (>20cm):{traj_err_20:.6f}\n')
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
                              split='val_random',
                            #   split='val_gaze',
                              # split='test',
                              hml_mode='train',hint_type=args.hint_type)
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
