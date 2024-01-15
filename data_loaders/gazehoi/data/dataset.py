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

class GazeHOIDataset(data.Dataset):
    def __init__(self, mode='stage1', datapath='/root/code/seqs/gazehoi_list_train_new.txt', split='train'):
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

        hand_pose = np.load(mano_right_path)
        gaze = np.load(gaze_path)

        ##
        goal_index = meta['goal_index']

        goal_hand_pose = hand_pose[goal_index]
        goal_obj_pose = obj_pose[goal_index].reshape(3,-1)
        obj_verts = obj_verts @ goal_obj_pose[:3,:3].T + goal_obj_pose[:3,3].reshape(1,3)

        stage1_hand_pose =  hand_pose[:goal_index+1] # 包含goal_pose
        # 倒序
        stage1_hand_pose = stage1_hand_pose[::-1, :].copy()
        
        ## 补全 和 倒序
        length = 60
        if goal_index < 59:
            pad_width = ((0, 59-goal_index), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            stage1_hand_pose = np.pad(stage1_hand_pose, pad_width, mode='edge') 
        else:
            stage1_hand_pose = stage1_hand_pose[:60]

        return stage1_hand_pose, goal_hand_pose, goal_obj_pose, obj_verts,length
        

            








                
