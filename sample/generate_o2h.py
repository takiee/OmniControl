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
from tqdm import *
import numpy as np
import torch
import trimesh
from pysdf import SDF
from manotorch.manolayer import ManoLayer
from tqdm import *
def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
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

    all_hand_motions = []
    all_obj_motions = []
    all_lengths = []
    all_seqs = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        length = model_kwargs['y']['lengths'].cpu()
        sample_fn = diffusion.p_sample_loop
        # print(args.batch_size, model.njoints, model.nfeats, n_frames)
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
        sample = sample.permute(0, 3, 2, 1).squeeze(2).contiguous().cpu() # global 6d 表示 99 hand 9 obj
        
        global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_motion_6d_mean.npy'))
        global_std = torch.from_numpy(np.load('dataset/gazehoi_global_motion_6d_std.npy'))
        obj_global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_obj_mean.npy'))
        obj_global_std = torch.from_numpy(np.load('dataset/gazehoi_global_obj_std.npy'))

        sample = sample * global_std + global_mean

        hand_motion = sample
        hand_motion = rot6d2axis(hand_motion)

        obj_motion = model_kwargs['y']['obj_pose'].cpu()
        obj_motion = obj_motion * obj_global_std + obj_global_mean
        print(obj_motion.shape)
        obj_motion = torch.cat((rotation_6d_to_matrix(obj_motion[:,:,3:]),obj_motion[:,:,:3].unsqueeze(-1)),dim=-1)
        print(obj_motion.shape)

        all_seqs.append(model_kwargs['y']['seq_name'])

        all_hand_motions.append(hand_motion.cpu().numpy())
        all_obj_motions.append(obj_motion.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())



    all_hand_motions = np.concatenate(all_hand_motions, axis=0)
    all_obj_motions = np.concatenate(all_obj_motions, axis=0)
    all_seqs =  [element for sublist in all_seqs for element in sublist]
        
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'all_hand_motions': all_hand_motions,'all_obj_motions':all_obj_motions, 'lengths': all_lengths, 'seqs':all_seqs})
        
    res_save_path = join(out_path, 'eval_results.txt')
    # # all_motions = torch.tensor(all_motions)
    # # all_hint = torch.tensor(all_hint)
    # stage1_eval(all_motions, all_hint, all_seqs, res_save_path)
    # vis_gen(all_motions,all_seqs,res_save_path,vis_num)
    all_hand_motions = torch.tensor(all_hand_motions)
    all_obj_motions = torch.tensor(all_obj_motions)
    eval_all(res_save_path,all_hand_motions,all_obj_motions,all_seqs)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

def get_new_obj_verts(obj_pose,obj_verts):
    # nf,3,4 , nf,N,3
    obj_R = obj_pose[:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    obj_T = obj_pose[:,:3,3].unsqueeze(1)
    # print(obj_verts.shape,obj_R.shape,obj_T.shape)
    new_obj_verts = torch.einsum('fpn,fnk->fpk',obj_verts,obj_R) + obj_T
    return new_obj_verts

def eval_all(res_path,all_hand_motions,all_obj_motions,seqs):
    
    num_samples = len(seqs)
    manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
    all_mpvpe_global = 0
    all_final_loc = 0

    all_mpjpe_local = 0
    all_goal_mpjpe = 0

    all_traj = 0
    total_traj_num = 0
    contact_num = 0
    pene_num = 0
    for i in tqdm(range(num_samples)):
        seq = seqs[i]
        gt_seq_path = join('/root/code/seqs/0303_data/',seq)
        gaze = np.load(join(gt_seq_path,'gaze_point.npy')) 
        seq_len = gaze.shape[0]
        meta_path = join(gt_seq_path,'meta.pkl')
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        ######## object #############
        gaze_obj = meta['gaze_obj']
        active_obj = meta['active_obj']
        gt_obj_verts = torch.tensor(np.load(join('/root/code/seqs/object/',active_obj,'resampled_500_trans.npy'))).unsqueeze(0).repeat(seq_len,1,1).float()
        pred_obj_verts = torch.tensor(np.load(join('/root/code/seqs/object/',gaze_obj,'resampled_500_trans.npy'))).unsqueeze(0).repeat(seq_len,1,1).float()
        gt_obj_pose = torch.tensor(np.load(join(gt_seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4)).float()
        pred_obj_pose = torch.tensor(all_obj_motions[i][:seq_len]).float()

        gt_obj_verts = get_new_obj_verts(gt_obj_pose,gt_obj_verts)
        pred_obj_verts = get_new_obj_verts(pred_obj_pose,pred_obj_verts) # nf,N,3

        mpvpe = torch.mean(torch.norm(gt_obj_verts-pred_obj_verts,dim=-1))
        all_mpvpe_global += mpvpe

        gt_final_T = gt_obj_pose[-1][:3,3]
        pred_final_T = pred_obj_pose[-1][:3,3]
        final_loc = torch.norm(gt_final_T - pred_final_T)
        all_final_loc += final_loc

        ######## hand ################
        hand_params = torch.tensor(np.load(join(gt_seq_path,'mano/poses_right.npy')))
        hand_trans = hand_params[:,:3]
        hand_rot = hand_params[:,3:6]
        hand_theta = hand_params[:,3:51]
        mano_beta = hand_params[:,51:]

        pred_hand_motion = torch.tensor(all_hand_motions[i])[:seq_len]
        # print(pred_hand_motion.shape)
        pred_trans = pred_hand_motion[:,:3]
        pred_theta = pred_hand_motion[:,3:]

        pred_output = manolayer(pred_theta, mano_beta)
        pred_joints = pred_output.joints - pred_output.joints[:, 0].unsqueeze(1)

        gt_output = manolayer(hand_theta, mano_beta)
        gt_joints = gt_output.joints - gt_output.joints[:, 0].unsqueeze(1)

        mpjpe = torch.mean(torch.norm(pred_joints - gt_joints, dim=-1))
        all_mpjpe_local += mpjpe

        pred_joints = pred_joints + pred_trans.unsqueeze(1)
        gt_joints = gt_joints + hand_trans.unsqueeze(1)

        goal_mpjpe = torch.mean(torch.norm((gt_joints[-1] - pred_joints[-1]),dim=-1))
        all_goal_mpjpe += goal_mpjpe

        ##### all ######
        # traj
        hand_traj_error = torch.norm(hand_trans-pred_trans,p=2,dim=1) 
        obj_traj_error = torch.norm(gt_obj_pose[:,:3,3]-pred_obj_pose[:,:3,3],p=2,dim=1) 
        traj_error = hand_traj_error + obj_traj_error
        traj_invalid_20 = torch.sum(traj_error > 0.1)
        all_traj += traj_invalid_20
        total_traj_num += traj_error.shape[0]

        # contact and penne
        def convert_to_obj_frame(pc, obj_rot, obj_trans):
            pc = (obj_rot.T @ (pc - obj_trans).T).T
            return pc
        
        obj_mesh_path = join('/root/code/seqs/object/',gaze_obj,'simplified_scan_processed.obj')
        obj_mesh = trimesh.load(obj_mesh_path)
        obj_rot = pred_obj_pose[:,:,:3]
        obj_trans = pred_obj_pose[:,:,3]
        obj_sdf = SDF(obj_mesh.vertices,obj_mesh.faces)
        # print(pred_output.verts.shape, pred_output.joints[:, 0].shape, pred_trans.unsqueeze(1).shape)
        mano_verts = pred_output.verts - pred_output.joints[:, 0].unsqueeze(1) + pred_trans.unsqueeze(1)
        for i in range(mano_verts.shape[0]):
            mano_verts_i = convert_to_obj_frame(mano_verts[i],
                                                obj_rot[i],
                                                obj_trans[i])
            contact = obj_sdf(mano_verts_i)
            close_num = np.sum(np.abs(contact) < 0.01)
            if close_num > 10:
                contact_num += 1
            pene = np.sum(contact[contact>0])
            if pene > 0.3:
                pene_num += 1

    
    traj_err_10 = (all_traj/total_traj_num).numpy()
    contact = (contact_num/total_traj_num)
    pene = (pene_num/total_traj_num)

    mpjpe = (all_mpjpe_local/num_samples).numpy()
    mpvpe = (all_mpvpe_global/num_samples).numpy()
    final_loc = (all_final_loc/num_samples).numpy()
    goal_mpjpe = (all_goal_mpjpe/num_samples).numpy()

    with open(res_path, 'w') as f:
        f.write(f'Goal MPJPE:{goal_mpjpe:.6f}\n')
        f.write(f'Traj Error (>10cm):{traj_err_10:.6f}\n')
        f.write(f'MPJPE local :{mpjpe:.6f}\n')
        f.write(f'MPVPE global:{mpvpe:.6f}\n')
        f.write(f'Final Loc:{final_loc:.6f}\n')
        f.write(f'contact rate:{contact:.6f}\n')
        f.write(f'pene rate:{pene:.6f}\n')
    return


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
