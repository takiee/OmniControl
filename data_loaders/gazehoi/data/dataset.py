import torch
from torch.utils import data
import numpy as np
import os
from os.path import join 
import random
from tqdm import *
import spacy
import pickle
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
import trimesh
from utils.data_util import axis2rot6d,global2local_axis,global2local_axis_by_matrix,obj_global2local_matrix, obj_matrix2rot6d
from tqdm import *
from manotorch.manolayer import ManoLayer
manolayer = ManoLayer(
    mano_assets_root='/root/code/CAMS/data/mano_assets/mano',
    side='right'
)
def read_xyz(path):
    data = []
    with open(path,'r') as f:
        line = f.readline()
        ls = line.strip().split(' ')
        data.append([float(ls[0]),float(ls[1]),float(ls[2])])
        while line:
            ls = f.readline().strip().split(' ')
            # print(ls)
            if ls != ['']:
                data.append([float(ls[0]),float(ls[1]),float(ls[2])])
            else:
                line = None
    data = np.array(data)
    return data

class GazeHOIDataset_stage1(data.Dataset):
    def __init__(self, mode='stage1', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.global_mean = np.load('dataset/gazehoi_global_motion_6d_mean.npy')
        self.global_std = np.load('dataset/gazehoi_global_motion_6d_std.npy')
        self.local_mean = np.load('dataset/gazehoi_local_motion_6d_mean.npy')
        self.local_std = np.load('dataset/gazehoi_local_motion_6d_std.npy')
        self.hint_type = hint_type
        # for seq in seqs:
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        gaze_path = join(seq_path,'fake_goal.npy')

        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
        active_obj = meta['active_obj']
        obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
        obj_mesh = trimesh.load(obj_mesh_path)
        obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
        obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))

        hand_params = np.load(mano_right_path)
        hand_pose_axis = torch.tensor(hand_params[:,:51])
        hand_shape = hand_params[:,51:]
        gaze = np.load(gaze_path)

        ##
        goal_index = meta['goal_index']
        goal_obj_pose = obj_pose[goal_index].reshape(3,-1) 
        obj_verts = obj_verts @ goal_obj_pose[:3,:3].T + goal_obj_pose[:3,3].reshape(1,3)

        seq_length = goal_index + 1
        length = 60
        if goal_index == 0:
            goal_index = 1
        if goal_index < 59:
            ## 不足的用第一帧补充
            pad_hand_pose_axis = torch.zeros(60,51)
            pad_hand_pose_axis[-goal_index:] = hand_pose_axis[:goal_index]
        else:
            pad_hand_pose_axis = hand_pose_axis[-60:]
        
        local_hand_pose_axis = global2local_axis_by_matrix(pad_hand_pose_axis.unsqueeze(0))
        local_hand_pose_6d = axis2rot6d(local_hand_pose_axis).squeeze(0).numpy()
        local_hand_pose_6d = (local_hand_pose_6d - self.local_mean) / self.local_std


        global_hand_pose_6d = axis2rot6d(hand_pose_axis.unsqueeze(0)).squeeze(0).numpy()
        goal_hand_pose = global_hand_pose_6d[goal_index]
        init_hand_pose = global_hand_pose_6d[0]
        goal_hand_pose = (goal_hand_pose - self.global_mean) / self.global_std
        init_hand_pose = (init_hand_pose - self.global_mean) / self.global_std
        hint = np.zeros((60,99))
        # hint[0] = init_hand_pose
        hint[-1] = goal_hand_pose
        
        # local_hand_pose_6d = local_hand_pose_6d.cpu().numpy()
        """
        goal_hand_pose = hand_pose[goal_index]
        # hint = np.zeros((60,51)) # 绝对表示
        hint = np.zeros((60,99)) # 绝对表示 rot6d

        stage1_global_hand_pose =  hand_pose[:goal_index+1] # 包含goal_pose

        ## Motion相对表示 正序### 
        # hint[-1] = goal_hand_pose

        ## 把角度转为6d表示
        stage1_global_hand_pose = torch.tensor(stage1_global_hand_pose)
        hand_rot = (stage1_global_hand_pose[:,3:].reshape(-1,16,3)).contiguous()
        # print(hand_rot.shape)
        hand_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(hand_rot)).reshape(-1,96)
        # print(hand_rot.shape)
        stage1_global_hand_pose = torch.cat((stage1_global_hand_pose[:,:3],hand_rot6d),dim=-1)
        hint[-1] = stage1_global_hand_pose[goal_index] # 绝对表示


        # 转为相对表示
        stage1_local_hand_pose = stage1_global_hand_pose[1:] - stage1_global_hand_pose[:-1]
        stage1_local_hand_pose = np.concatenate([stage1_global_hand_pose[0].reshape(1,-1),stage1_local_hand_pose],axis=0)

        # 只对global RT相对表示
        stage1_local_hand_pose[:,9:] = stage1_global_hand_pose[:,9:]
        


        length = 60
        if goal_index < 59:
            ## 不足的用第一帧补充
            # print(stage1_hand_pose.shape)
            # past1 = stage1_hand_pose[0]
            pad_width = ((59-goal_index,0), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            stage1_local_hand_pose = np.pad(stage1_local_hand_pose, pad_width, mode='constant',constant_values=0) 
            
        else:
            stage1_local_hand_pose = stage1_local_hand_pose[-60:]
        """
        # print(stage1_global_hand_pose.shape[0] == 60)
        
        # print(stage1_local_hand_pose.shape)
        # break
        """
        ## Motion绝对表示 倒序### 
        hint[0] = goal_hand_pose
        # 倒序
        stage1_hand_pose = stage1_hand_pose[::-1, :].copy()
        
        ## 补全 和 倒序
        length = 60
        if goal_index < 59:
            pad_width = ((0, 59-goal_index), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            stage1_hand_pose = np.pad(stage1_hand_pose, pad_width, mode='edge') 
        else:
            stage1_hand_pose = stage1_hand_pose[:60]
        """

        return local_hand_pose_6d, hint, goal_obj_pose, obj_verts,seq_length,seq
        # return local_hand_pose_6d, hint, goal_obj_pose, obj_verts,length,seq

class GazeHOIDataset_stage1_new(data.Dataset):
    def __init__(self, mode='stage1', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.global_mean = np.load('dataset/gazehoi_global_motion_6d_mean.npy')
        self.global_std = np.load('dataset/gazehoi_global_motion_6d_std.npy')
        self.local_mean = np.load('dataset/gazehoi_local_motion_6d_mean.npy')
        self.local_std = np.load('dataset/gazehoi_local_motion_6d_std.npy')
        self.hint_type = hint_type
        # for seq in seqs:
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        gaze_path = join(seq_path,'fake_goal.npy')

        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
        active_obj = meta['active_obj']
        obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
        obj_mesh = trimesh.load(obj_mesh_path)
        obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
        obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))

        hand_params = np.load(mano_right_path)
        hand_pose_axis = torch.tensor(hand_params[:,:51])
        hand_trans = hand_pose_axis[:,:3]
        hand_theta = hand_pose_axis[:,3:]
        mano_beta = torch.tensor(hand_params[:,51:])
        gaze = np.load(gaze_path)

        ##
        goal_index = meta['goal_index']
        goal_obj_pose = obj_pose[goal_index].reshape(3,-1) 
        obj_verts = obj_verts @ goal_obj_pose[:3,:3].T + goal_obj_pose[:3,3].reshape(1,3)
        goal_obj_pose_6d = matrix_to_rotation_6d(torch.tensor(goal_obj_pose[:3,:3]).unsqueeze(0)).squeeze(0).numpy()
        # print(goal_obj_pose[:3,3].shape,goal_obj_pose_6d.shape)
        goal_obj_pose_rot6d = np.concatenate((goal_obj_pose[:3,3],goal_obj_pose_6d))
        # print(goal_obj_pose_rot6d.shape)

        seq_length = goal_index + 1
        # length = 60
        

        """0226晚pad版本 直接全补成60帧 不用mask了"""
        # if goal_index == 0:
        #     goal_index = 1
        # if goal_index < 59:
        #     pad_hand_pose_axis = hand_pose_axis[:goal_index].numpy()
        #     pad_width = ((length-seq_length,0), (0, 0))
        #     pad_hand_pose_axis = np.pad(pad_hand_pose_axis, pad_width, mode='edge')
        #     pad_hand_pose_axis = torch.tensor(pad_hand_pose_axis).unsqueeze(0)
        #     seq_length = 60
        # else:
        #     pad_hand_pose_axis = hand_pose_axis[-60:].unsqueeze(0)
        #     seq_length = 60
            
        # local_hand_pose_axis = global2local_axis_by_matrix(pad_hand_pose_axis)
        # local_hand_pose_6d = axis2rot6d(local_hand_pose_axis).squeeze(0).numpy()
        # local_hand_pose_6d = (local_hand_pose_6d - self.local_mean) / self.local_std

        """0226 凌晨pad版本 先变成local表示 然后前边补零 结果存在飘忽的情况 即反复张开 合上手掌"""
        local_hand_pose_axis = global2local_axis_by_matrix(hand_pose_axis.unsqueeze(0))
        if goal_index == 0:
            goal_index = 1
        if goal_index < 59:
            pad_hand_pose_axis = torch.zeros(60,51)
            pad_hand_pose_axis[-goal_index:] = local_hand_pose_axis[0,:goal_index]
        else:
            pad_hand_pose_axis = local_hand_pose_axis[0,-60:]
            seq_length = 60
        # print(pad_hand_pose_axis.shape)
        local_hand_pose_6d = axis2rot6d(pad_hand_pose_axis.unsqueeze(0)).squeeze(0).numpy()
        local_hand_pose_6d = (local_hand_pose_6d - self.local_mean) / self.local_std

        """
        用keypoint做hint
        """
        # mano_output = manolayer(hand_theta, mano_beta)
        # # print(mano_output.joints.shape)
        # mano_joints = mano_output.joints - mano_output.joints[:,0].unsqueeze(1) + hand_trans.unsqueeze(1) 
        # # print(mano_joints.shape)
        # hint = np.zeros((60,21,3))
        # hint[0] = mano_joints[0]
        # hint[-1] = mano_joints[goal_index]
        # hint = hint.reshape(-1,63)
        
        """
        用mano参数做hint
        """
        global_hand_pose_6d = axis2rot6d(hand_pose_axis.unsqueeze(0)).squeeze(0).numpy()
        goal_hand_pose = global_hand_pose_6d[goal_index]
        init_hand_pose = global_hand_pose_6d[0]
        goal_hand_pose = (goal_hand_pose - self.global_mean) / self.global_std
        init_hand_pose = (init_hand_pose - self.global_mean) / self.global_std
        hint = np.zeros((60,99))
        # hint[0] = init_hand_pose
        hint[-1] = goal_hand_pose
        # print(mano_beta)
        

        return local_hand_pose_6d, hint, init_hand_pose, goal_hand_pose, goal_obj_pose_rot6d,mano_beta.numpy()[:60], obj_verts,seq_length,seq
        # return local_hand_pose_6d, hint, goal_obj_pose, obj_verts,length,seq

class GazeHOIDataset_stage1_repair(data.Dataset):
    def __init__(self, mode='stage1', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.global_mean = np.load('dataset/gazehoi_global_motion_6d_mean.npy')
        self.global_std = np.load('dataset/gazehoi_global_motion_6d_std.npy')
        self.local_mean = np.load('dataset/gazehoi_local_motion_6d_mean.npy')
        self.local_std = np.load('dataset/gazehoi_local_motion_6d_std.npy')
        self.hint_type = hint_type
        # for seq in seqs:
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        gaze_path = join(seq_path,'fake_goal.npy')

        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
        active_obj = meta['active_obj']
        obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
        obj_mesh = trimesh.load(obj_mesh_path)
        obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
        obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))

        hand_params = np.load(mano_right_path)
        hand_pose_axis = torch.tensor(hand_params[:,:51])
        hand_trans = hand_pose_axis[:,:3]
        hand_theta = hand_pose_axis[:,3:]
        mano_beta = torch.tensor(hand_params[:,51:])
        gaze = np.load(gaze_path)

        ##
        # goal_index = meta['goal_index'] + 1 #这样能包含goal_index这一帧
        goal_index = meta['goal_index'] 
        goal_obj_pose = obj_pose[goal_index].reshape(3,-1) 
        obj_verts = obj_verts @ goal_obj_pose[:3,:3].T + goal_obj_pose[:3,3].reshape(1,3)
        goal_obj_pose_6d = matrix_to_rotation_6d(torch.tensor(goal_obj_pose[:3,:3]).unsqueeze(0)).squeeze(0).numpy()
        # print(goal_obj_pose[:3,3].shape,goal_obj_pose_6d.shape)
        goal_obj_pose_rot6d = np.concatenate((goal_obj_pose[:3,3],goal_obj_pose_6d))
        # print(goal_obj_pose_rot6d.shape)

        seq_length = goal_index + 1
        length = 60
        

        """0229晚pad版本 补在后边"""
        if goal_index <= 59:
            pad_hand_pose_axis = hand_pose_axis[:goal_index].numpy()
            pad_width = ((0,length-seq_length+1), (0, 0))
            pad_hand_pose_axis = np.pad(pad_hand_pose_axis, pad_width, mode='edge')
            pad_hand_pose_axis = torch.tensor(pad_hand_pose_axis).unsqueeze(0)
        else:
            pad_hand_pose_axis = hand_pose_axis[goal_index-60:goal_index].unsqueeze(0)
            seq_length = 60
        # print(pad_hand_pose_axis.shape)
        # if pad_hand_pose_axis.shape[1] == 61:
        #     print(seq,seq_length,length)
        local_hand_pose_axis = global2local_axis_by_matrix(pad_hand_pose_axis)
        local_hand_pose_6d = axis2rot6d(local_hand_pose_axis).squeeze(0).numpy()
        local_hand_pose_6d = (local_hand_pose_6d - self.local_mean) / self.local_std

        """
        用mano参数做hint
        """
        global_hand_pose_6d = axis2rot6d(pad_hand_pose_axis).squeeze(0).numpy()
        goal_hand_pose = global_hand_pose_6d[-1]
        init_hand_pose = global_hand_pose_6d[0]
        goal_hand_pose = (goal_hand_pose - self.global_mean) / self.global_std
        init_hand_pose = (init_hand_pose - self.global_mean) / self.global_std
        hint = np.zeros((60,99))
        # hint[0] = init_hand_pose
        if goal_index <59:
            hint[goal_index] = goal_hand_pose
        else:
            hint[-1] = goal_hand_pose
        # print(mano_beta)
        

        return local_hand_pose_6d, hint, init_hand_pose, goal_hand_pose, goal_obj_pose_rot6d,mano_beta.numpy()[:60], obj_verts,seq_length,seq
        # return local_hand_pose_6d, hint, goal_obj_pose, obj_verts,length,seq

class GazeHOIDataset_stage2(data.Dataset):
    def __init__(self, mode='stage1', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.global_mean = np.load('dataset/gazehoi_global_motion_6d_mean.npy')
        self.global_std = np.load('dataset/gazehoi_global_motion_6d_std.npy')
        self.local_mean = np.load('dataset/gazehoi_local_motion_6d_mean.npy')
        self.local_std = np.load('dataset/gazehoi_local_motion_6d_std.npy')

        self.obj_global_mean = np.load('dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('dataset/gazehoi_local_obj_std.npy')

        self.hint_type = hint_type

    def __len__(self):
            return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        gaze_path = join(seq_path,'fake_goal.npy')

        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
        active_obj = meta['active_obj']
        obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
        obj_mesh = trimesh.load(obj_mesh_path)
        # obj_verts = np.load(join(self.obj_path,active_obj,'resampled_200_trans.npy'))
        obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
        obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))

        hand_params = np.load(mano_right_path)
        hand_pose = hand_params[:,:51]
        hand_shape = hand_params[:,51:]
        gaze = np.load(gaze_path)

        ## 
        goal_index = meta['goal_index']
        goal_obj_pose = obj_pose[goal_index].reshape(3,-1) 
        # obj_verts = obj_verts @ goal_obj_pose[:3,:3].T + goal_obj_pose[:3,3].reshape(1,3)
        obj_pose = obj_pose[goal_index:]
        hand_shape = hand_shape[goal_index:]

        # hand_pose = (hand_pose - self.mean) / self.std

        # goal_hand_pose = hand_pose[goal_index]
        length = 180
        seq_len = hand_pose.shape[0] - goal_index # 包含goal的seq长度
        hand_pose_axis = hand_pose[goal_index:]
        if self.hint_type == 'root_dis':
            hint = np.load(join(seq_path,'global_distance.npy'))[0].reshape(1,-1)
            hint = np.repeat(hint,length,axis=0)
        elif self.hint_type == 'tip_dis':
            hint = np.load(join(seq_path,'tip_distance.npy'))[0].reshape(1,5)
            hint = np.repeat(hint,length,axis=0)
        elif self.hint_type == 'tips_closest_point':
            hint = np.load(join(seq_path,'tips_closest_point.npy')).reshape(1,-1)
            hint = np.repeat(hint,length,axis=0)
        elif self.hint_type == 'hand_T':
            hint_hand_T = np.load(join(seq_path,'contact_hand_T_from_obj_T.npy')) #(seq_len, 3) 全局绝对表示
            hint = np.zeros((length,3))
            true_len = min(seq_len, length)
            hint[:true_len,:3] = hint_hand_T[:true_len]
            hint[true_len:,:3] = hint_hand_T[-1] 

        # print(hint.shape)
        # hint = np.repeat(hint,length,axis=0)
        # print(hint.shape)
        # print(hint.shape)
        if seq_len <= length:
            pad_width = ((0,length-seq_len), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            hand_pose_axis = np.pad(hand_pose_axis, pad_width, mode='edge') 
            hand_shape = np.pad(hand_shape, pad_width, mode='edge') 
            obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)
        else:
            hand_pose_axis = hand_pose_axis[:length]
            hand_shape = hand_shape[:length]
            obj_pose = obj_pose[:length]

        pad_hand_pose_axis = torch.tensor(hand_pose_axis)
        local_hand_pose_axis = global2local_axis_by_matrix(pad_hand_pose_axis.unsqueeze(0))
        local_hand_pose_6d = axis2rot6d(local_hand_pose_axis).squeeze(0).numpy()
        local_hand_pose_6d = (local_hand_pose_6d - self.local_mean) / self.local_std
        # print(local_hand_pose_6d.shape)

        # print(hint.shape, hint)
        return local_hand_pose_6d, hint,obj_pose, obj_verts, seq_len, seq,hand_shape

def read_xyz(path):
    data = []
    with open(path,'r') as f:
        line = f.readline()
        ls = line.strip().split(' ')
        data.append([float(ls[0]),float(ls[1]),float(ls[2])])
        while line:
            ls = f.readline().strip().split(' ')
            # print(ls)
            if ls != ['']:
                data.append([float(ls[0]),float(ls[1]),float(ls[2])])
            else:
                line = None
    data = np.array(data)
    return data

class GazeHOIDataset_stage0(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.obj_global_mean = np.load('dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('dataset/gazehoi_local_obj_std.npy')
        self.hint_type = hint_type
        self.table_plane = read_xyz("dataset/table_plane_750.xyz")

        # for seq in seqs:
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        gaze_path = join(seq_path,'fake_goal.npy')
        gaze = np.load(gaze_path) # (num_frames, 3)
        num_frames = gaze.shape[0]
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
        active_obj = meta['active_obj']
        gaze_obj = meta['gaze_obj']

        obj_name_list = meta['obj_name_list']
        obj_verts_list = []
        obj_pose_list = []
        for obj in obj_name_list:
            # obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
            # obj_mesh = trimesh.load(obj_mesh_path)
            obj_verts = np.load(join(self.obj_path,obj,'resampled_500_trans.npy'))
            obj_pose = np.load(join(seq_path,obj+'_pose_trans.npy')).reshape(-1,3,4)

            new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)
            obj_verts_list.append(new_verts)
            obj_pose_list.append(obj_pose)
        obj_num = len(obj_name_list)

        length = 345
        if num_frames < length:
            # 在后边补充
            pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            for i in range(obj_num):
                obj_pose = obj_pose_list[i]
                pad_obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)
                obj_pose_list[i] = pad_obj_pose

            gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')

        local_obj_pose_6d_list = []
        global_obj_pose_6d_list = []

        for i in range(obj_num):
            obj_pose = obj_pose_list[i]
            obj_pose = torch.tensor(obj_pose)
            obj_pose_local = obj_global2local_matrix(obj_pose.unsqueeze(0))
            obj_pose_local_6d = obj_matrix2rot6d(obj_pose_local).numpy()
            obj_pose_global_6d = obj_matrix2rot6d(obj_pose.unsqueeze(0)).numpy()
            local_obj_pose_6d_list.append(obj_pose_local_6d.squeeze(0))
            global_obj_pose_6d_list.append(obj_pose_global_6d.squeeze(0))


        if obj_num < 4:
            obj_pose_paddding = [np.zeros((length,9))] * (4-obj_num) # (4, num_frames, 12)
            local_obj_pose_6d_list = local_obj_pose_6d_list + obj_pose_paddding
            global_obj_pose_6d_list = global_obj_pose_6d_list + obj_pose_paddding
            
            obj_verts_paddding = [np.zeros((500,3))] * (4-obj_num)
            obj_verts_list = obj_verts_list + obj_verts_paddding  # 4 500 3

        # local & global normalization
        for i in range(4):
            local_obj_pose_6d_list[i] = (local_obj_pose_6d_list[i] - self.obj_local_mean) / self.obj_local_std
            global_obj_pose_6d_list[i] = (global_obj_pose_6d_list[i] - self.obj_global_mean) / self.obj_global_std


        local_obj_pose_6d = np.array(local_obj_pose_6d_list) # (4,num_frames,9)
        global_obj_pose_6d = np.array(global_obj_pose_6d_list) # (4, num_frames, 9)
        global_initial_obj_pose_6d = global_obj_pose_6d[:,0].flatten() # (36)
        hint = np.zeros((length,36))
        hint[0] = global_initial_obj_pose_6d
        # obj_verts = np.array(obj_verts_list)
        obj_verts = np.vstack((np.array(obj_verts_list).reshape(-1,3),self.table_plane))
        # (345, 3) (4, 345, 9) (4, 9) (4, 500, 3) 150 0677
        # print(gaze.shape, local_obj_pose_6d.shape, global_initial_obj_pose_6d.shape, obj_verts.shape,num_frames,seq)
        return  local_obj_pose_6d, hint,gaze, obj_verts,num_frames,seq
        # return local_hand_pose_6d, hint, goal_obj_pose, obj_verts,length,seq

class GazeHOIDataset_stage0_flag(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.obj_global_mean = np.load('dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('dataset/gazehoi_local_obj_std.npy')
        self.hint_type = hint_type
        self.table_plane = read_xyz("dataset/table_plane_750.xyz")

        # for seq in seqs:
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        gaze_path = join(seq_path,'fake_goal.npy')
        gaze = np.load(gaze_path) # (num_frames, 3)
        num_frames = gaze.shape[0]
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
        active_obj = meta['active_obj']
        gaze_obj = meta['gaze_obj']
        goal_index = meta['goal_index']

        obj_name_list = meta['obj_name_list']
        obj_verts_list = []
        obj_pose_list = []
        for obj in obj_name_list:
            # obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
            # obj_mesh = trimesh.load(obj_mesh_path)
            obj_verts = np.load(join(self.obj_path,obj,'resampled_500_trans.npy'))
            obj_pose = np.load(join(seq_path,obj+'_pose_trans.npy')).reshape(-1,3,4)

            new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)
            obj_verts_list.append(new_verts)
            obj_pose_list.append(obj_pose)
        obj_num = len(obj_name_list)

        length = 345
        flag = np.zeros((4,length))
        # active_flag
        obj_index = obj_name_list.index(active_obj)
        flag[obj_index][goal_index:num_frames] = 1
        if num_frames < length:
            # 在后边补充
            pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            for i in range(obj_num):
                obj_pose = obj_pose_list[i]
                pad_obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)
                obj_pose_list[i] = pad_obj_pose

            gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')

        local_obj_pose_6d_list = []
        global_obj_pose_6d_list = []

        for i in range(obj_num):
            obj_pose = obj_pose_list[i]
            obj_pose = torch.tensor(obj_pose)
            obj_pose_local = obj_global2local_matrix(obj_pose.unsqueeze(0))
            obj_pose_local_6d = obj_matrix2rot6d(obj_pose_local).numpy()
            obj_pose_global_6d = obj_matrix2rot6d(obj_pose.unsqueeze(0)).numpy()
            local_obj_pose_6d_list.append(obj_pose_local_6d.squeeze(0))
            global_obj_pose_6d_list.append(obj_pose_global_6d.squeeze(0))


        if obj_num < 4:
            obj_pose_paddding = [np.zeros((length,9))] * (4-obj_num) # (4, num_frames, 12)
            local_obj_pose_6d_list = local_obj_pose_6d_list + obj_pose_paddding
            global_obj_pose_6d_list = global_obj_pose_6d_list + obj_pose_paddding
            
            obj_verts_paddding = [np.zeros((500,3))] * (4-obj_num)
            obj_verts_list = obj_verts_list + obj_verts_paddding  # 4 500 3

        # local & global normalization
        for i in range(4):
            # local_obj_pose_6d_list[i] = (local_obj_pose_6d_list[i] - self.obj_local_mean) / self.obj_global_std
            local_obj_pose_6d_list[i] = (local_obj_pose_6d_list[i] - self.obj_local_mean) / self.obj_local_std
            global_obj_pose_6d_list[i] = (global_obj_pose_6d_list[i] - self.obj_global_mean) / self.obj_global_std


        local_obj_pose_6d = np.array(local_obj_pose_6d_list) # (4,num_frames,9)
        global_obj_pose_6d = np.array(global_obj_pose_6d_list) # (4, num_frames, 9)
        global_initial_obj_pose_6d = global_obj_pose_6d[:,0].flatten() # (36)
        hint = np.zeros((length,36))
        hint[0] = global_initial_obj_pose_6d
        # obj_verts = np.array(obj_verts_list)
        obj_verts = np.vstack((np.array(obj_verts_list).reshape(-1,3),self.table_plane))
        # (345, 3) (4, 345, 9) (4, 9) (4, 500, 3) 150 0677
        # print(gaze.shape, local_obj_pose_6d.shape, global_initial_obj_pose_6d.shape, obj_verts.shape,num_frames,seq)
        return  local_obj_pose_6d, hint,gaze, obj_verts,flag, num_frames,seq
        # return local_hand_pose_6d, hint, goal_obj_pose, obj_verts,length,seq

class GazeHOIDataset_stage0_flag_lowfps(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.obj_global_mean = np.load('dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('dataset/gazehoi_local_obj_std.npy')
        self.hint_type = hint_type
        self.table_plane = read_xyz("dataset/table_plane_750.xyz")
        self.datalist = []
        self.fps = 6
        for seq in self.seqs:
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            mano_right_path = join(seq_path, 'mano/poses_right.npy')
            gaze_path = join(seq_path,'fake_goal.npy')
            gaze = np.load(gaze_path) # (num_frames, 3)
            num_frames = gaze.shape[0]
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            
            active_obj = meta['active_obj']
            gaze_obj = meta['gaze_obj']
            goal_index = meta['goal_index']

            obj_name_list = meta['obj_name_list']
            obj_verts_list = []
            obj_pose_list = []
            for obj in obj_name_list:
                # obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
                # obj_mesh = trimesh.load(obj_mesh_path)
                obj_verts = np.load(join(self.obj_path,obj,'resampled_500_trans.npy'))
                obj_pose = np.load(join(seq_path,obj+'_pose_trans.npy')).reshape(-1,3,4)

                new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)
                obj_verts_list.append(new_verts)
                obj_pose_list.append(obj_pose)
            obj_num = len(obj_name_list)
            step = int(30/self.fps)
            for i in range(step):
                gaze_ = gaze[i::step]
                num_frames_ = gaze_.shape[0]
                goal_index_ = int(goal_index/step)
                # goal_index_ = int((goal_index/30) * self.fps + i)
                obj_pose_list_ = []
                for obj_pose in obj_pose_list:
                    obj_pose = obj_pose[i::step]
                    obj_pose_list_.append(obj_pose)
                data = {"gaze":gaze_, #x
                        "obj_pose_list":obj_pose_list_, #x
                        "obj_verts_list": obj_verts_list,
                        "obj_name_list":obj_name_list,
                        "seq":seq,
                        "active_obj":active_obj,
                        "num_frames":num_frames_,#x
                        "goal_index":goal_index_}#x
                self.datalist.append(data)

        # for seq in seqs:
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]

        length = 69
        flag = np.zeros((4,length))
        # active_flag
        obj_index = data['obj_name_list'].index(data['active_obj'])
        # print(obj_index)
        num_frames = data['num_frames']
        goal_index = data['goal_index']
        seq = data['seq']
        assert goal_index < num_frames,  f"spilt wrong {seq},{goal_index},{num_frames}"
        flag[obj_index][goal_index:num_frames] = 1
        obj_pose_list = data['obj_pose_list']
        obj_verts_list = data['obj_verts_list']
        gaze = data['gaze']
        obj_num = len(obj_pose_list)
        if num_frames < length:
            # 在后边补充
            pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            for i in range(obj_num):
                obj_pose = obj_pose_list[i]
                pad_obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)
                obj_pose_list[i] = pad_obj_pose
            gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')

        local_obj_pose_6d_list = []
        global_obj_pose_6d_list = []

        for i in range(obj_num):
            obj_pose = obj_pose_list[i]
            obj_pose = torch.tensor(obj_pose)
            obj_pose_local = obj_global2local_matrix(obj_pose.unsqueeze(0))
            obj_pose_local_6d = obj_matrix2rot6d(obj_pose_local).numpy()

            obj_pose_global_6d = obj_matrix2rot6d(obj_pose.unsqueeze(0)).numpy()

            local_obj_pose_6d_list.append(obj_pose_local_6d.squeeze(0))
            global_obj_pose_6d_list.append(obj_pose_global_6d.squeeze(0))

        if obj_num < 4:
            obj_pose_paddding = [np.zeros((length,9))] * (4-obj_num) # (4, num_frames, 12)
            local_obj_pose_6d_list = local_obj_pose_6d_list + obj_pose_paddding
            global_obj_pose_6d_list = global_obj_pose_6d_list + obj_pose_paddding
            
            obj_verts_paddding = [np.zeros((500,3))] * (4-obj_num)
            obj_verts_list = obj_verts_list + obj_verts_paddding  # 4 500 3

        # local & global normalization
        for i in range(4):
            local_obj_pose_6d_list[i] = (local_obj_pose_6d_list[i] - self.obj_local_mean) / self.obj_local_std
            global_obj_pose_6d_list[i] = (global_obj_pose_6d_list[i] - self.obj_global_mean) / self.obj_global_std


        local_obj_pose_6d = np.array(local_obj_pose_6d_list) # (4,num_frames,9)
        global_obj_pose_6d = np.array(global_obj_pose_6d_list) # (4, num_frames, 9)
        global_initial_obj_pose_6d = global_obj_pose_6d[:,0].flatten() # (36)
        hint = np.zeros((length,36))
        hint[0] = global_initial_obj_pose_6d
        # obj_verts = np.array(obj_verts_list)
        obj_verts = np.vstack((np.array(obj_verts_list).reshape(-1,3),self.table_plane))
        # (345, 3) (4, 345, 9) (4, 9) (4, 500, 3) 150 0677
        # print(gaze.shape, local_obj_pose_6d.shape, global_initial_obj_pose_6d.shape, obj_verts.shape,num_frames,seq)
        return  local_obj_pose_6d, hint,gaze, obj_verts,flag, num_frames,seq
        # return local_hand_pose_6d, hint, goal_obj_pose, obj_verts,length,seq

class GazeHOIDataset_stage0_flag_lowfps_global(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.obj_global_mean = np.load('dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('dataset/gazehoi_local_obj_std.npy')
        self.hint_type = hint_type
        self.table_plane = read_xyz("dataset/table_plane_750.xyz")
        self.datalist = []
        self.fps = 6
        for seq in self.seqs:
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            mano_right_path = join(seq_path, 'mano/poses_right.npy')
            gaze_path = join(seq_path,'fake_goal.npy')
            gaze = np.load(gaze_path) # (num_frames, 3)
            num_frames = gaze.shape[0]
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            
            active_obj = meta['active_obj']
            gaze_obj = meta['gaze_obj']
            goal_index = meta['goal_index']

            obj_name_list = meta['obj_name_list']
            obj_verts_list = []
            obj_pose_list = []
            for obj in obj_name_list:
                # obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
                # obj_mesh = trimesh.load(obj_mesh_path)
                obj_verts = np.load(join(self.obj_path,obj,'resampled_500_trans.npy'))
                obj_pose = np.load(join(seq_path,obj+'_pose_trans.npy')).reshape(-1,3,4)

                new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)
                obj_verts_list.append(new_verts)
                obj_pose_list.append(obj_pose)
            obj_num = len(obj_name_list)
            step = int(30/self.fps)
            for i in range(step):
                gaze_ = gaze[i::step]
                num_frames_ = gaze_.shape[0]
                goal_index_ = int(goal_index/step)
                # goal_index_ = int((goal_index/30) * self.fps + i)
                obj_pose_list_ = []
                for obj_pose in obj_pose_list:
                    obj_pose = obj_pose[i::step]
                    obj_pose_list_.append(obj_pose)
                data = {"gaze":gaze_, #x
                        "obj_pose_list":obj_pose_list_, #x
                        "obj_verts_list": obj_verts_list,
                        "obj_name_list":obj_name_list,
                        "seq":seq,
                        "active_obj":active_obj,
                        "num_frames":num_frames_,#x
                        "goal_index":goal_index_}#x
                self.datalist.append(data)

        # for seq in seqs:
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]

        length = 69
        flag = np.zeros((4,length))
        # active_flag
        obj_index = data['obj_name_list'].index(data['active_obj'])
        # print(obj_index)
        num_frames = data['num_frames']
        goal_index = data['goal_index']
        seq = data['seq']
        assert goal_index < num_frames,  f"spilt wrong {seq},{goal_index},{num_frames}"
        flag[obj_index][goal_index:num_frames] = 1
        obj_pose_list = data['obj_pose_list']
        obj_verts_list = data['obj_verts_list']
        gaze = data['gaze']
        obj_num = len(obj_pose_list)
        if num_frames < length:
            # 在后边补充
            pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            for i in range(obj_num):
                obj_pose = obj_pose_list[i]
                pad_obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)
                obj_pose_list[i] = pad_obj_pose
            gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')

        global_obj_pose_6d_list = []

        for i in range(obj_num):
            obj_pose = obj_pose_list[i]
            obj_pose = torch.tensor(obj_pose)
            obj_pose_global_6d = obj_matrix2rot6d(obj_pose.unsqueeze(0)).numpy()
            global_obj_pose_6d_list.append(obj_pose_global_6d.squeeze(0))

        if obj_num < 4:
            obj_pose_paddding = [np.zeros((length,9))] * (4-obj_num) # (4, num_frames, 12)
            global_obj_pose_6d_list = global_obj_pose_6d_list + obj_pose_paddding
            
            obj_verts_paddding = [np.zeros((500,3))] * (4-obj_num)
            obj_verts_list = obj_verts_list + obj_verts_paddding  # 4 500 3

        # local & global normalization
        for i in range(4):
            global_obj_pose_6d_list[i] = (global_obj_pose_6d_list[i] - self.obj_global_mean) / self.obj_global_std


        global_obj_pose_6d = np.array(global_obj_pose_6d_list) # (4, num_frames, 9)
        global_initial_obj_pose_6d = global_obj_pose_6d[:,0].flatten() # (36)
        hint = np.zeros((length,36))
        hint[0] = global_initial_obj_pose_6d
        obj_verts = np.vstack((np.array(obj_verts_list).reshape(-1,3),self.table_plane))
        # (345, 3) (4, 345, 9) (4, 9) (4, 500, 3) 150 0677
        # print(gaze.shape, local_obj_pose_6d.shape, global_initial_obj_pose_6d.shape, obj_verts.shape,num_frames,seq)
        return  global_obj_pose_6d, hint,gaze, obj_verts,flag, num_frames,seq


class GazeHOIDataset_stage0_1obj(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.obj_global_mean = np.load('dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('dataset/gazehoi_local_obj_std.npy')
        self.hint_type = hint_type
        self.table_plane = read_xyz("dataset/table_plane_750.xyz")
        self.datalist = []
        self.fps = 6
        for seq in self.seqs:
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            mano_right_path = join(seq_path, 'mano/poses_right.npy')
            gaze_path = join(seq_path,'fake_goal.npy')
            gaze = np.load(gaze_path) # (num_frames, 3)
            num_frames = gaze.shape[0]
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            if split == 'test':
                active_obj = meta['gaze_obj']
            else:
                active_obj = meta['active_obj']
            gaze_obj = meta['gaze_obj']
            goal_index = meta['goal_index']

            obj_name_list = meta['obj_name_list']
            obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
            obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4)

            new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)

            step = int(30/self.fps)
            for i in range(step):
                gaze_ = gaze[i::step]
                num_frames_ = gaze_.shape[0]
                goal_index_ = int(goal_index/step)
                obj_pose_ = obj_pose[i::step]
                data = {"gaze":gaze_, #x
                        "obj_pose":obj_pose_, #x
                        "obj_verts": new_verts,
                        "seq":seq,
                        "active_obj":active_obj,
                        "num_frames":num_frames_,#x
                        "goal_index":goal_index_}#x
                
                self.datalist.append(data)
                if split == 'test':
                    break
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]
        flag = np.zeros(4)
        length = 69
        num_frames = data['num_frames']
        goal_index = data['goal_index']
        seq = data['seq']
        assert goal_index < num_frames,  f"spilt wrong {seq},{goal_index},{num_frames}"
        obj_pose = data['obj_pose']
        obj_verts = data['obj_verts']
        gaze = data['gaze']
        if num_frames < length:
            # 在后边补充
            pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)
            gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')

        obj_pose = torch.tensor(obj_pose)
        obj_pose_global_6d = obj_matrix2rot6d(obj_pose.unsqueeze(0)).squeeze(0).numpy()

        obj_pose_global_6d = (obj_pose_global_6d - self.obj_global_mean) / self.obj_global_std

        # print(obj_pose_global_6d.shape)
        hint = np.zeros((length,9))
        hint[0] = obj_pose_global_6d[0]
        obj_verts = np.vstack(((obj_verts,self.table_plane)))
        return  obj_pose_global_6d, hint,gaze, obj_verts,flag, num_frames,seq


class GazeHOIDataset_stage0_1(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.obj_global_mean = np.load('dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('dataset/gazehoi_local_obj_std.npy')
        self.hint_type = hint_type
        self.table_plane = read_xyz("dataset/table_plane_750.xyz")

        # for seq in seqs:
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_path = join(self.root,seq)
        meta_path = join(seq_path,'meta.pkl')
        mano_right_path = join(seq_path, 'mano/poses_right.npy')
        gaze_path = join(seq_path,'fake_goal.npy')
        gaze = np.load(gaze_path) # (num_frames, 3)
        num_frames = gaze.shape[0]
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
        active_obj = meta['active_obj']
        gaze_obj = meta['gaze_obj']

        obj_name_list = meta['obj_name_list']
        obj_verts = np.load(join(self.obj_path,gaze_obj,'resampled_500_trans.npy'))
        obj_pose = np.load(join(seq_path,gaze_obj+'_pose_trans.npy')).reshape(-1,3,4)
        new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)

        length = 345
        if num_frames < length:
            # 在后边补充
            pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)

            gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')


        obj_pose = torch.tensor(obj_pose)
        obj_pose_local = obj_global2local_matrix(obj_pose.unsqueeze(0))
        obj_pose_local_6d = obj_matrix2rot6d(obj_pose_local).squeeze(0).numpy()
        obj_pose_global_6d = obj_matrix2rot6d(obj_pose.unsqueeze(0)).squeeze(0).numpy()


        obj_pose_local_6d = (obj_pose_local_6d - self.obj_local_mean) / self.obj_local_std
        obj_pose_global_6d = (obj_pose_global_6d - self.obj_global_mean) / self.obj_global_std
        # print()
        global_initial_obj_pose_6d = obj_pose_global_6d[0]
        # print(global_initial_obj_pose_6d.shape) # (36)
        hint = np.zeros((length,9))
        hint[0] = global_initial_obj_pose_6d
        # obj_verts = np.array(obj_verts_list)
        obj_verts = np.vstack((new_verts,self.table_plane))
        # (345, 3) (4, 345, 9) (4, 9) (4, 500, 3) 150 0677
        # print(gaze.shape, local_obj_pose_6d.shape, global_initial_obj_pose_6d.shape, obj_verts.shape,num_frames,seq)
        return  obj_pose_local_6d, hint, gaze, obj_verts,num_frames,seq


class GazeHOIDataset_pretrain(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        self.split = split
        if split == 'test':
            print("Use Test DATA")
            datapath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
        self.root = '/root/code/seqs/1205_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.obj_global_mean = np.load('dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('dataset/gazehoi_local_obj_std.npy')
        self.hint_type = hint_type
        self.datapool = []
        if self.split == 'train':
            self.extend_data()
        # for seq in seqs:
    
    def __len__(self):
        if self.split == 'train':
            return len(self.datapool)
        else:
            return len(self.seqs)

    def extend_data(self):
        for seq in tqdm(self.seqs):
        # seq = self.seqs[index]
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            mano_right_path = join(seq_path, 'mano/poses_right.npy')
            gaze_path = join(seq_path,'fake_goal.npy')
            raw_gaze = np.load(gaze_path) # (num_frames, 3)
            num_frames = raw_gaze.shape[0]
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            
            active_obj = meta['active_obj']
            obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
            raw_obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4)
                

            length = 345
            if num_frames < length:
                # 在后边补充
                for i in range(10):
                    for k in range(10):
                        i = min(length-num_frames,i)
                        k = min(length-num_frames,k)
                        obj_pad_width = ((i,length-num_frames-i), (0, 0))  # 在第一维度上填充0行，使总行数变为10
                        obj_pose = np.pad(raw_obj_pose.reshape(-1,12), obj_pad_width, mode='edge').reshape(-1,3,4)
                        
                        gaze_pad_width = ((k,length-num_frames-k), (0, 0))
                        gaze = np.pad(raw_gaze.reshape(-1,3), gaze_pad_width, mode='edge')

                        pad_obj_pose = torch.tensor(obj_pose)
                        obj_pose_global_6d = obj_matrix2rot6d(pad_obj_pose.unsqueeze(0)).squeeze(0).numpy()

                        obj_pose_global_6d = (obj_pose_global_6d - self.obj_global_mean) / self.obj_global_std
                        data = {'obj_T':obj_pose_global_6d[:,:3],'gaze': gaze, 'obj_verts':obj_verts,'num_frames':num_frames,'seq':seq}
                        self.datapool.append(data)
                
            else:
                pad_obj_pose = torch.tensor(raw_obj_pose)
                obj_pose_global_6d = obj_matrix2rot6d(pad_obj_pose.unsqueeze(0)).squeeze(0).numpy()

                obj_pose_global_6d = (obj_pose_global_6d - self.obj_global_mean) / self.obj_global_std
                data = {'obj_T':obj_pose_global_6d[:,:3],'gaze': gaze, 'obj_verts':obj_verts,'num_frames':num_frames,'seq':seq}
                # print(obj_pose_global_6d[:,:3].shape)
                self.datapool.append(data)



    def __getitem__(self, index):
        if self.split == 'test':
            seq = self.seqs[index]
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            mano_right_path = join(seq_path, 'mano/poses_right.npy')
            gaze_path = join(seq_path,'fake_goal.npy')
            gaze = np.load(gaze_path) # (num_frames, 3)
            num_frames = gaze.shape[0]
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            
            active_obj = meta['active_obj']
            obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
            obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4)
                

            length = 345
            if num_frames < length:
                # 在后边补充
                pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
                obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)

                gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')

            pad_obj_pose = torch.tensor(obj_pose)
            obj_pose_global_6d = obj_matrix2rot6d(pad_obj_pose.unsqueeze(0)).squeeze(0).numpy()

            obj_pose_global_6d = (obj_pose_global_6d - self.obj_global_mean) / self.obj_global_std

            return  obj_pose_global_6d[:,:3], gaze, obj_verts,num_frames,seq
        else:
            data = self.datapool[index]
            obj_T = data['obj_T']
            gaze = data['gaze']
            obj_verts = data['obj_verts']
            num_frames = data['num_frames']
            seq = data['seq']

            return  obj_T, gaze, obj_verts,num_frames,seq


            








                
