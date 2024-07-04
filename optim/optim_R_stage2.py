import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import torch
import manotorch
from manotorch.manolayer import ManoLayer
from os.path import join 
from tqdm import *
import pickle

manolayer = ManoLayer(
    mano_assets_root='/root/code/CAMS/data/mano_assets/mano',
    side='right'
).cuda()
def convert_to_obj_frame(points, obj_pose):
    # points(frames,5,3)
    # obj_pose (frames,3,4)

    # pc = (obj_rot.T @ (pc - obj_trans).T).T
    obj_T = obj_pose[:,:3,3].unsqueeze(1)
    # print(obj_T.shape)
    points = points - obj_T
    # hand_rot = torch.einsum('...ij->...ji', [hand_rot])
    # hand_R_error = (torch.einsum('fpn,fnk->fpk',hand_rot,pred_rot) - torch.eye(3).unsqueeze(0).repeat(hand_rot.shape[0],1,1)).reshape(-1,9) # nf,3,3
       
    points = torch.einsum('...ij->...ji', [points])
    obj_R = obj_pose[:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    new_points = torch.einsum('fpn,fnk->fpk',obj_R,points)
    new_points = torch.einsum('...ij->...ji', [new_points])

    return new_points

def get_trans_obj_verts(obj_verts,obj_pose):
    """
    obj_verts: (N,3) -- T,N
    obj_pose: (T,3,4)
    """
    nf = obj_pose.shape[0]
    N = obj_verts.shape[0]
    # obj_pose = obj_pose.unsqueeze(1).repeat(1,N,1,1)
    obj_R = obj_pose[:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    obj_T = obj_pose[:,:3,3].unsqueeze(1)
    
    obj_verts = obj_verts.unsqueeze(0).repeat(nf,1,1)
    trans_obj_verts = torch.einsum('fpn,fnk->fpk',obj_verts,obj_R) + obj_T
    return trans_obj_verts


class Fitter:
    def __init__(self):
        # datapath = '/root/code/seqs/gazehoi_list_train_new.txt'
        # datapath = '/root/code/seqs/gazehoi_list_test_0303.txt'
        self.root = '/root/code/seqs/0303_data/'
        self.obj_path = '/root/code/seqs/object/'
        self.output_path = '/root/code/OmniControl/optim/val_gaze'
        # with open(datapath,'r') as f:
        #     info_list = f.readlines()
        # self.seqs = []
        # for info in info_list:
        #     seq = info.strip()
        #     self.seqs.append(seq)

        stage1_hand_res_path = 'save/8000_DSG/samples_final_000008000_seed10_new_dsg750_random/results.npy'
        stage1_hand_res = np.load(stage1_hand_res_path,allow_pickle=True).item()
        self.hint = torch.tensor(stage1_hand_res['hint']).float().cuda()
        self.stage1_hand_hand_params = torch.tensor(stage1_hand_res['motion']).float().cuda()
        self.seqs = stage1_hand_res['seqs']
        
        # hand_params_list = []
        # hint_list = []
        # for seq in self.seqs:
        #     seq_path = join(self.root,seq)
        #     meta_path = join(seq_path,'meta.pkl')
        #     mano_right_path = join(seq_path, 'mano/poses_right.npy')
        #     hand_params = torch.tensor(np.load(mano_right_path)).unsqueeze(0)
        #     hand_params_list.append(hand_params)
        #     hint = torch.tensor(np.load(join(seq_path,'tips_closest_point.npy'))).unsqueeze(0)
        #     hint_list.append(hint)
        #     # print(hint.shape, hint)
        #     # break
        # self.hand_params = torch.cat(hand_params_list)
        # self.hints = torch.cat(hint_list)
        # print(self.hand_params.shape)
        # print(self.hint.shape)
    
    def fit_all(self,num_iters):
        hand_list = []
        seqs = []
        index = 0
        for i in tqdm(range(len(self.seqs))):
            hand_params_after = self.fit_seq_pred(i, num_iters)
        #     hand_list.append(hand_params_after)
        #     seqs.append(seq)
        #     index = index + 1
            if index > 2:
                break
        # all_motions = np.concatenate(hand_list, axis=0)
        # all_seqs =  [element for sublist in all_seqs for element in sublist]
        # print(all_motions.shape)
        # npy_path = os.path.join(self.output_path, 'results.npy')
        # print(f"saving results file to [{npy_path}]")
        # np.save(npy_path,{'motion': all_motions,'seqs':all_seqs})
            # break
    
    def fit_seq(self,seq, num_iters):
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        goal_index = meta['goal_index']
        active_obj = meta['active_obj']
        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        hand_params = torch.tensor(np.load(mano_right_path)).cuda()
        # obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
        # obj_mesh = trimesh.load(obj_mesh_path)
        obj_verts = torch.tensor(np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))).float().cuda()
        obj_pose = torch.tensor(np.load(join(seq_path,active_obj+'_pose_trans.npy'))).float().cuda()

        
        obj_pose_s2 = obj_pose[goal_index:]
        seq_len = obj_pose_s2.shape[0]
        
        hand_params_stage2 = hand_params[goal_index].unsqueeze(0).repeat(seq_len,1)
        
        hand_T = torch.tensor(np.load(join(seq_path,'contact_hand_T_from_obj_T.npy'))).float().cuda()
        # print(hand_params_stage2.shape,hand_T.shape)
        hand_params_stage2[:,:3] = hand_T
        # hand_params_stage2.requires_grad = True
        # hand_params_stage2[:,3:6].requires_grad = True
        
        hand_rot = hand_params_stage2[:,3:6].clone().cuda()
        hand_rot.requires_grad = True

        hand_params_stage2[:,3:6] = hand_rot
        hand_trans = hand_params_stage2[:,:3]
        hand_theta = hand_params_stage2[:,3:51]
        mano_beta = hand_params_stage2[:,51:]

        # obj_verts = get_trans_obj_verts(obj_verts,obj_pose)

        hint = torch.tensor(np.load(join(seq_path,'tips_closest_id_and_dis.npy'))).float().cuda()
        # print(hint)
        obj_ids = hint[:,0].long()
        obj_dis = hint[:,1]
        # print(obj_ids)
        # print(obj_dis)
        tgt_points = obj_verts[obj_ids].unsqueeze(0)
        # print(tgt_points)

        optimizer = torch.optim.SGD([
            dict(params = hand_rot, lr = 1e-3, weight_decay = 0.1),
        ])
        tbar = trange(num_iters)
        for i in tbar:
            optimizer.zero_grad()
            # print(hand_theta.device,mano_beta.device)
            hand_theta[:,:3] = hand_rot
            gt_output = manolayer(hand_theta, mano_beta)
            # print(hand_theta.device)
            # print(hand_theta.shape, mano_beta.shape)
            gt_joints = gt_output.joints - gt_output.joints[:, 0].unsqueeze(1) + hand_trans.unsqueeze(1)
            # print(gt_joints.device)
            # print(gt_joints.shape)
            tips = [15,3,6,12,9]
            tips_T = gt_joints[:,tips]
            tips_to_obj = convert_to_obj_frame(tips_T, obj_pose_s2) # (frames,5,3) (frames,3,4)
            now_dis = torch.norm(tips_to_obj - tgt_points,dim=-1)
            # print(now_dis.shape,now_dis)
            """
            # folder: opt_res
            loss = torch.sum(abs(now_dis - obj_dis.unsqueeze(0))) # sum(frames,1)
            """
            # folder: opt_weight_res
            all_dis = torch.sum(abs(now_dis - obj_dis.unsqueeze(0)),dim=-1)
            # print(all_dis.shape)
            max_dis = max(all_dis)
            weight = all_dis / max_dis
            loss = torch.sum(weight * all_dis)

            loss.backward(retain_graph=True)
            optimizer.step()
            tbar.set_description(f'all loss = {loss.detach().item(): 04f}')
        
        hand_params_after = hand_params_stage2.unsqueeze(0)
        hand_params_after[:,:,3:6] = hand_rot.detach()
        hand_params_after = hand_params_after.detach().cpu().numpy()
        # return hand_params_after
        npy_path = os.path.join(self.output_path, f'{seq}.npy')
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path,{'motion': hand_params_after,'seq':seq})


    def fit_seq_pred(self,i, num_iters):
        seq = self.seqs[i]
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        active_obj = meta['gaze_obj']
        gaze_path = join(seq_path, 'fake_goal.npy')
        gaze = torch.tensor(np.load(gaze_path)).cuda()
        seq_len = gaze.shape[0]
        print(seq,gaze.shape)
        # goal_index = meta['goal_index']
        # goal index && obj pose
        obj_res_path = join('save/0303_stage0_1obj/samples_0303_stage0_1obj_000030000_seed10_predefined/pred_obj',f'{seq}_pred_obj_and_goal.npy')
        obj_res = np.load(obj_res_path,allow_pickle=True).item()
        goal_index = obj_res['goal_index']
        obj_pose = torch.tensor(obj_res['pred_obj_pose']).float().cuda()

        
        

        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        gt_hand_params = torch.tensor(np.load(mano_right_path)).cuda()
        obj_verts = torch.tensor(np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))).float().cuda()
        

        
        obj_pose_s2 = obj_pose[goal_index:]
        obj_seq_len = obj_pose_s2.shape[0]
        print(gaze[goal_index:].shape[0],obj_seq_len)
        assert gaze[goal_index:].shape[0]== obj_seq_len

        if goal_index >59:
            hand_params_stage2 = self.hint[i,-1].unsqueeze(0).repeat(obj_seq_len,1)
            hand_params_stage1 = self.stage1_hand_hand_params[i,:goal_index]
        else:
            hand_params_stage2 = self.hint[i,goal_index].unsqueeze(0).repeat(obj_seq_len,1)
            # hand_params_stage1 = self.stage1_hand_hand_params[i]
            hand_params_stage1_pad = torch.zeros((goal_index+1,51)).cuda() + self.stage1_hand_hand_params[i,0]
            hand_params_stage1_pad[i,-60:] = self.stage1_hand_hand_params[i]
            hand_params_stage1 = hand_params_stage1_pad
            # pad = hand_params_stage1
        # print(hint)
        
        # hand_params_stage2 = hint[goal_index].unsqueeze(0).repeat(obj_seq_len,1)

        hand_T = torch.tensor(np.load(join(seq_path,'gaze_contact_hand_T_from_obj_T.npy'))).float().cuda()
        print(hand_params_stage2.shape,hand_T.shape)
        hand_params_stage2[:,:3] = hand_T

        npy_path = os.path.join(self.output_path, f'{seq}_no_opt_R.npy') # 只有stage2的数据
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path,{'hand_motion': hand_params_stage2,'obj_motion':obj_pose_s2,'seq':seq})

        # hand_params_stage2.requires_grad = True
        # hand_params_stage2[:,3:6].requires_grad = True
        
        hand_rot = hand_params_stage2[:,3:6].clone().cuda()
        hand_rot.requires_grad = True

        # hand_params_stage2[:,3:6] = hand_rot
        hand_trans = hand_params_stage2[:,:3]
        hand_theta = hand_params_stage2[:,3:51]
        mano_beta = gt_hand_params[0,51:].unsqueeze(0).repeat(obj_seq_len,1)

        # obj_verts = get_trans_obj_verts(obj_verts,obj_pose)

        hint = torch.tensor(np.load(join(seq_path,'tips_closest_id_and_dis.npy'))).float().cuda()
        # print(hint)
        obj_ids = hint[:,0].long()
        obj_dis = hint[:,1]
        # print(obj_ids)
        # print(obj_dis)
        tgt_points = obj_verts[obj_ids].unsqueeze(0)
        # print(tgt_points)

        optimizer = torch.optim.SGD([
            dict(params = hand_rot, lr = 1e-3, weight_decay = 0.1),
        ])
        tbar = trange(num_iters)
        for i in tbar:
            optimizer.zero_grad()
            # print(hand_theta.device,mano_beta.device)
            hand_theta[:,:3] = hand_rot
            gt_output = manolayer(hand_theta, mano_beta)
            # print(hand_theta.device)
            # print(hand_theta.shape, mano_beta.shape)
            gt_joints = gt_output.joints - gt_output.joints[:, 0].unsqueeze(1) + hand_trans.unsqueeze(1)
            # print(gt_joints.device)
            # print(gt_joints.shape)
            tips = [15,3,6,12,9]
            tips_T = gt_joints[:,tips]
            tips_to_obj = convert_to_obj_frame(tips_T, obj_pose_s2) # (frames,5,3) (frames,3,4)
            now_dis = torch.norm(tips_to_obj - tgt_points,dim=-1)
            # print(now_dis.shape,now_dis)
            """
            # folder: opt_res
            loss = torch.sum(abs(now_dis - obj_dis.unsqueeze(0))) # sum(frames,1)
            """
            # folder: opt_weight_res
            all_dis = torch.sum(abs(now_dis - obj_dis.unsqueeze(0)),dim=-1)
            weight = 1
            # print(all_dis.shape)
            # max_dis = max(all_dis)
            # weight = all_dis / max_dis
            loss = torch.sum(weight * all_dis)

            loss.backward(retain_graph=True)
            optimizer.step()
            tbar.set_description(f'all loss = {loss.detach().item(): 04f}')
        
        hand_params_after = hand_params_stage2.unsqueeze(0)
        hand_params_after[:,:,3:6] = hand_rot.detach()
        hand_params_after = hand_params_after.detach().cpu().numpy()
        # return hand_params_after
        npy_path = os.path.join(self.output_path, f'{seq}.npy')
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path,{'motion': hand_params_after,'seq':seq})



    

fit = Fitter()
fit.fit_all(2000)