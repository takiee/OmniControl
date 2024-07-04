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
from pytorch3d.transforms import rotation_conversions as rc
from tqdm import *

def slerp(q0, q1, t):
    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / result.norm()
    theta_0 = dot.acos()
    sin_theta_0 = theta_0.sin()
    theta = theta_0 * t
    sin_theta = theta.sin()
    s0 = ((1.0 - t) * theta).cos()
    s1 = sin_theta / sin_theta_0
    return (s0 * q0) + (s1 * q1)

def interpolate_rotations_and_translations(rot_matrices, translations, timestamps_original, timestamps_target):
    # 将旋转矩阵转换为四元数
    quaternions = rc.matrix_to_quaternion(rot_matrices)
    
    # 插值结果容器
    quaternions_interpolated = []
    translations_interpolated = []

    for i in range(len(timestamps_target)):
        # 计算当前目标时间戳在原始时间戳的位置
        t_norm = timestamps_target[i] * (len(timestamps_original) - 1)
        idx = int(t_norm)
        t = t_norm - idx

        if idx < len(timestamps_original) - 1:
            # 对四元数进行SLERP插值
            q_interp = slerp(quaternions[idx], quaternions[idx + 1], t)
            quaternions_interpolated.append(q_interp)

            # 对平移向量进行线性插值
            trans_interp = (1 - t) * translations[idx] + t * translations[idx + 1]
            translations_interpolated.append(trans_interp)
        else:
            # 直接使用最后一个四元数和平移向量
            quaternions_interpolated.append(quaternions[-1])
            translations_interpolated.append(translations[-1])

    # 将插值后的四元数转换回旋转矩阵
    quaternions_interpolated = torch.stack(quaternions_interpolated)
    rot_matrices_interpolated = rc.quaternion_to_matrix(quaternions_interpolated)
    
    # 将插值后的平移向量转换为合适的形式
    translations_interpolated = torch.stack(translations_interpolated)

    return rot_matrices_interpolated, translations_interpolated
def get_new_obj_verts(obj_pose,obj_verts):
    # nf,3,4 , nf,N,3
    obj_R = obj_pose[:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    obj_T = obj_pose[:,:3,3].unsqueeze(1)
    # print(obj_verts.shape,obj_R.shape,obj_T.shape)
    new_obj_verts = torch.einsum('fpn,fnk->fpk',obj_verts,obj_R) + obj_T
    return new_obj_verts

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
    elif args.dataset == args.dataset == 'gazehoi_stage0_1obj' or args.dataset == 'gazehoi_stage0_norm' or args.dataset == 'gazehoi_stage0_point' or args.dataset == 'gazehoi_stage0_noatt':
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

        sample_fn = diffusion.p_sample_loop
        # print(args.batch_size, model.njoints, model.nfeats, n_frames)
        if args.dataset == 'gazehoi_stage0_1obj'or args.dataset == 'gazehoi_stage0_noatt' or args.dataset == 'gazehoi_stage0_norm' or args.dataset == 'gazehoi_stage0_point':
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
            obj_global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_obj_mean.npy'))
            obj_global_std = torch.from_numpy(np.load('dataset/gazehoi_global_obj_std.npy'))
            obj_local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_obj_mean.npy'))
            obj_local_std = torch.from_numpy(np.load('dataset/gazehoi_local_obj_std.npy'))
            sample = sample * obj_global_std + obj_global_mean  # bs, nf, 36
            bs,nf,_ = sample.shape # bs,nf,9
            R = rotation_6d_to_matrix(sample[:,:,3:])
            T = sample[:,:,:3].unsqueeze(-1)
            sample = torch.cat((R,T),dim=-1)
            print(sample.shape)
            gt = model_kwargs['y']['gt']
            R = rotation_6d_to_matrix(gt[:,:,3:])
            T = gt[:,:,:3].unsqueeze(-1)
            gt = torch.cat((R,T),dim=-1)
        

        all_gt.append(gt.cpu().numpy())
        all_seqs.append(model_kwargs['y']['seq_name'])
        
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    print(total_num_samples)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_seqs =  [element for sublist in all_seqs for element in sublist]
    all_gt = np.concatenate(all_gt,axis=0)
        
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")

    np.save(npy_path,
            {'motion': all_motions, 'lengths': all_lengths, 'seqs':all_seqs,
            'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,
           'all_gt':all_gt})

        
    res_save_path = join(out_path, 'stage0_eval_results.txt')
    pred_motion = torch.tensor(all_motions)
    seqs = all_seqs
    bs = pred_motion.shape[0]
    timestamps_original = np.linspace(0, 1, 69)
    timestamps_target = np.linspace(0, 1, 345)
    all_motion = []
    for i in range(bs):
        motion = pred_motion[i]
        rot_matrices = motion[:,:3,:3]
        translations = motion[:,:3,3]
        rot_matrices_interpolated, translations_interpolated = interpolate_rotations_and_translations(
                                    rot_matrices, translations, timestamps_original, timestamps_target)
        motion_inter = torch.cat([rot_matrices_interpolated, translations_interpolated.unsqueeze(-1)],dim=-1)
        motion_inter = motion_inter.unsqueeze(0).numpy()
        all_motion.append(motion_inter)

    all_motion_inter = np.concatenate(all_motion, axis=0)
    print(all_motion_inter.shape)

    gt_path = '/root/code/seqs/0303_data/'
    all_smooth_motion = []
    all_goal_index = []
    all_mpvpe_global = 0
    all_final_loc = 0

    all_mpjpe_local = 0
    all_goal_mpjpe = 0

    all_traj = 0
    total_traj_num = 0
    contact_num = 0
    pene_num = 0
    T_error = 0
    R_error = 0
    num_samples =0
    for i in tqdm(range(len(seqs))):
        seq = seqs[i]
        # if seq != '0535':
        #     continue
        # print(seq)
        gt_seq_path = join(gt_path,seq)
        gaze = np.load(join(gt_seq_path,'gaze_point.npy'))
        seq_len = gaze.shape[0]
        meta_path = join(gt_seq_path,'meta.pkl')
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        gaze_obj = meta['gaze_obj']
        active_obj = meta['active_obj']
        gaze = np.load(join(gt_seq_path,'fake_goal.npy'))
        num_frames = gaze.shape[0]

        gt_init_obj_pose = torch.tensor(np.load(join(gt_seq_path,gaze_obj+'_pose_trans.npy')).reshape(-1,3,4)[0])
        
        motion_inter_i = torch.tensor(all_motion_inter[i,:seq_len])
        motion_inter_T = motion_inter_i[:,:3,3]
        vel = torch.norm(motion_inter_T[11:] - motion_inter_T[10:-1],dim=-1)
        goal_index = torch.nonzero(vel > 5e-4)[0].item() + 10
        pred_init10_obj_pose = motion_inter_i[goal_index+10]
        interval = 10
        timestamps_original = np.linspace(0, 1, 2)
        timestamps_target = np.linspace(0, 1, 10)

        new_motion = torch.cat((gt_init_obj_pose.unsqueeze(0),pred_init10_obj_pose.unsqueeze(0)),dim=0)
        rot_matrices = new_motion[:,:3,:3]
        translations = new_motion[:,:3,3]
        rot_matrices_interpolated, translations_interpolated = interpolate_rotations_and_translations(
                                    rot_matrices, translations, timestamps_original, timestamps_target)
        motion_10 = torch.cat([rot_matrices_interpolated, translations_interpolated.unsqueeze(-1)],dim=-1)
        motion_inter_i[goal_index:goal_index+10] = motion_10
        motion_inter_i[:goal_index] = gt_init_obj_pose
        pred_path = join(out_path,'pred_obj')
        os.makedirs(pred_path,exist_ok=True)
        output_path = join(pred_path,f'{seq}_pred_obj_and_goal.npy')
        # output_path = join(gt_seq_path,'pred_obj_and_goal.npy')
        np.save(output_path,{'pred_obj_pose':motion_inter_i.numpy(),'goal_index':goal_index,'seqs':seqs})


        if gaze_obj == active_obj:
            gt_pose = torch.tensor(np.load(join(gt_seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4))
            pred_pose = torch.tensor(all_motion_inter[i,:num_frames])
            gt_T = gt_pose[:,:3,3]
            pred_T = pred_pose[:,:3,3]
            T_error += torch.mean(torch.norm(gt_T - pred_T,p=2,dim=-1))

            gt_R = gt_pose[:,:3,:3]
            pred_R = pred_pose[:,:3,:3]
            gt_R = torch.einsum('...ij->...ji', [gt_R])
            hand_R_error = (torch.einsum('fpn,fnk->fpk',gt_R,pred_R) - torch.eye(3).unsqueeze(0).repeat(gt_R.shape[0],1,1)).reshape(-1,9) # nf,3,3
            hand_R_error =  torch.mean(torch.norm(hand_R_error,dim=-1))
            R_error += hand_R_error 

            
        else:
            obj_name_list = meta['obj_name_list']
            for obj in obj_name_list:
                gt_pose = torch.tensor(np.load(join(gt_seq_path,obj+'_pose_trans.npy')).reshape(-1,3,4))
                if obj == gaze_obj:
                    pred_pose = torch.tensor(all_motion_inter[i,:num_frames])
                else:
                    pred_pose = gt_pose[0].unsqueeze(0).repeat(num_frames,1,1)
                    # print(pred_pose.shape)

                gt_T = gt_pose[:,:3,3]
                pred_T = pred_pose[:,:3,3]
                T_error += torch.mean(torch.norm(gt_T - pred_T,p=2,dim=-1))

                gt_R = gt_pose[:,:3,:3]
                pred_R = pred_pose[:,:3,:3]
                gt_R = torch.einsum('...ij->...ji', [gt_R])
                hand_R_error = (torch.einsum('fpn,fnk->fpk',gt_R,pred_R) - torch.eye(3).unsqueeze(0).repeat(gt_R.shape[0],1,1)).reshape(-1,9) # nf,3,3
                hand_R_error =  torch.mean(torch.norm(hand_R_error,dim=-1))
                R_error += hand_R_error 
        
        gt_obj_verts = torch.tensor(np.load(join('/root/code/seqs/object/',active_obj,'resampled_500_trans.npy'))).unsqueeze(0).repeat(seq_len,1,1).float()
        pred_obj_verts = torch.tensor(np.load(join('/root/code/seqs/object/',gaze_obj,'resampled_500_trans.npy'))).unsqueeze(0).repeat(seq_len,1,1).float()
        gt_obj_pose = torch.tensor(np.load(join(gt_seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4)).float()
        pred_obj_pose = torch.tensor(all_motion_inter[i][:seq_len]).float()
        gt_obj_verts = get_new_obj_verts(gt_obj_pose,gt_obj_verts)
        pred_obj_verts = get_new_obj_verts(pred_obj_pose,pred_obj_verts) # nf,N,3

        mpvpe = torch.mean(torch.norm(gt_obj_verts-pred_obj_verts,dim=-1))
        all_mpvpe_global += mpvpe

        gt_final_T = gt_obj_pose[-1][:3,3]
        pred_final_T = pred_obj_pose[-1][:3,3]
        final_loc = torch.norm(gt_final_T - pred_final_T)
        all_final_loc += final_loc

        obj_traj_error = torch.norm(gt_obj_pose[:,:3,3]-pred_obj_pose[:,:3,3],p=2,dim=1) 
        traj_invalid_10 = torch.sum(obj_traj_error > 0.1)
        all_traj += traj_invalid_10
        total_traj_num += obj_traj_error.shape[0]

    traj_err_10 = (all_traj/total_traj_num).numpy()
    contact = (contact_num/total_traj_num)
    pene = (pene_num/total_traj_num)
    num_samples = 130
    mpjpe = (all_mpjpe_local/num_samples)
    mpvpe = (all_mpvpe_global/num_samples)
    final_loc = (all_final_loc/num_samples)
    goal_mpjpe = (all_goal_mpjpe/num_samples)
    # mpjpe = (all_mpjpe_local/num_samples).numpy()
    # mpvpe = (all_mpvpe_global/num_samples).numpy()
    # final_loc = (all_final_loc/num_samples).numpy()
    # goal_mpjpe = (all_goal_mpjpe/num_samples).numpy()



    print(T_error / len(seqs))
    print(R_error / len(seqs))

    with open(res_save_path, 'w') as f:
        # f.write(f'Goal goal_mpjpe Error:{goal_mpjpe:.6f}\n')
        f.write(f'Traj Error (<10cm):{traj_err_10:.6f}\n')
        # f.write(f'Traj Error (<20cm):{traj_err_20:.6f}\n')
        f.write(f'final_loc:{final_loc:.6f}\n')
        f.write(f'mpvpe:{mpvpe:.6f}\n')




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
