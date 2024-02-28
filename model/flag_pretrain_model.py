import os
import sys
sys.path.append('/root/code/OmniControl/')
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.base_cross_model import PerceiveEncoder,PerceiveDecoder
from .transformer import *
from model.base_cross_model import *
from model.pointnet_plus2 import *



class flag_model(torch.nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None,hint_dim=99,length=None,*args, **kargs):
        super().__init__()

        self.hint_dim = hint_dim
        self.length = length

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.emb_trans_dec = emb_trans_dec

        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        # --- MDM ---

        
        self.encode_obj_pose = nn.Linear(9,self.latent_dim)
            
        self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
        self.fp_layer = MyFPModule()
        self.gaze_linear = nn.Linear(self.latent_dim, self.latent_dim)  
        # print(self.latent_dim,self.length)
        # self.length = 12
        # self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
        #                                 n_latent=self.length,
        #                                 n_latent_channels=self.latent_dim,
        #                                 n_self_att_heads=1,
        #                                 n_self_att_layers=1,
        #                                 dropout=0.5)
        # self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
        #                                 n_latent=self.length,
        #                                 n_latent_channels=self.latent_dim,
        #                                 n_self_att_heads=1,
        #                                 n_self_att_layers=1,
        #                                 dropout=0.5)
        self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
                                        n_latent=1,
                                        n_latent_channels=self.latent_dim,
                                        n_self_att_heads=2,
                                        n_self_att_layers=1,
                                        dropout=0.1)
        # self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
        #                                 n_latent=1,
        #                                 n_latent_channels=self.latent_dim,
        #                                 n_self_att_heads=4,
        #                                 n_self_att_layers=3,
        #                                 dropout=0.1)
        # self.move_flag_pool = nn.MaxPool1d(kernel_size=self.length)
        # self.move_flag_linear = nn.Linear(self.latent_dim, 4)
        self.move_flag_linear = nn.Linear(self.latent_dim*3, 1)
        # self.move_flag_linear = nn.Sequential( nn.Linear(self.latent_dim*12, 32), nn.ELU(),
        #                                 nn.Linear(32,4) )
        # self.move_flag_linear = nn.Sequential( nn.Linear(self.latent_dim, 32), nn.ELU(),
        #                                 nn.Linear(32,4) )

        # self.move_index_linear = nn.Sequential( nn.Linear(self.latent_dim, 32), nn.ELU(),
        #                                 nn.Linear(32,8), nn.ELU(),
        #                                 nn.Linear(8,1) )

    def gaze_encoder(self,gaze,points,points_feat):
        gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
        gaze_feat = self.gaze_linear(gaze_emb)
        gaze_feat = self.encode_gaze(gaze_feat)
        return gaze_feat

    def forward(self, motion,y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
     
        bs, nf, _ = y['gaze'].shape
        init_obj_pose = y['hint'][:,0]
        gt_flag = y['flag']

        obj_pose_emb = self.encode_obj_pose(init_obj_pose.reshape(bs,4,9))
        # print(obj_pose_emb.shape)

        points = y['obj_points']
        obj_mesh = points[:,:2000].reshape(bs,4,500,3)
        obj0 = obj_mesh[:,0].contiguous()
        obj1 = obj_mesh[:,1].contiguous()
        obj2 = obj_mesh[:,2].contiguous()
        obj3 = obj_mesh[:,3].contiguous()
        points_feat0, global_obj_feat0= self.encode_obj_mesh(obj0.repeat(1, 1, 2))
        points_feat1, global_obj_feat1= self.encode_obj_mesh(obj1.repeat(1, 1, 2))
        points_feat2, global_obj_feat2= self.encode_obj_mesh(obj2.repeat(1, 1, 2))
        points_feat3, global_obj_feat3= self.encode_obj_mesh(obj3.repeat(1, 1, 2))

        gaze = y['gaze'][:,:self.length].contiguous()
        # print(gaze.shape)
        gaze_feat0 = self.gaze_encoder(gaze,obj0,points_feat0).squeeze(1)
        gaze_feat1 = self.gaze_encoder(gaze,obj1,points_feat1).squeeze(1)
        gaze_feat2 = self.gaze_encoder(gaze,obj2,points_feat2).squeeze(1)
        gaze_feat3 = self.gaze_encoder(gaze,obj3,points_feat3).squeeze(1)
        # gaze_feat0 = self.gaze_encoder(gaze,obj0,points_feat0).squeeze(1)
        # gaze_feat1 = self.gaze_encoder(gaze,obj1,points_feat1).squeeze(1)
        # gaze_feat2 = self.gaze_encoder(gaze,obj2,points_feat2).squeeze(1)
        # gaze_feat3 = self.gaze_encoder(gaze,obj3,points_feat3).squeeze(1)

        # gaze_obj0 = torch.cat((gaze_feat0,global_obj_feat0,obj_pose_emb[:,0]),dim=-1)
        # gaze_obj1 = torch.cat((gaze_feat1,global_obj_feat1,obj_pose_emb[:,1]),dim=-1)
        # gaze_obj2 = torch.cat((gaze_feat2,global_obj_feat2,obj_pose_emb[:,2]),dim=-1)
        # gaze_obj3 = torch.cat((gaze_feat3,global_obj_feat3,obj_pose_emb[:,3]),dim=-1)
        gaze_obj0 = torch.cat((gaze_feat0,global_obj_feat0,obj_pose_emb[:,0]),dim=-1).unsqueeze(1)
        gaze_obj1 = torch.cat((gaze_feat1,global_obj_feat1,obj_pose_emb[:,1]),dim=-1).unsqueeze(1)
        gaze_obj2 = torch.cat((gaze_feat2,global_obj_feat2,obj_pose_emb[:,2]),dim=-1).unsqueeze(1)
        gaze_obj3 = torch.cat((gaze_feat3,global_obj_feat3,obj_pose_emb[:,3]),dim=-1).unsqueeze(1)
        # print(gaze_feat0.shape)
        # print(global_obj_feat0.shape)

        feat = torch.cat((gaze_obj0,gaze_obj1,gaze_obj2,gaze_obj3),dim=1)
        # print(feat.shape)

        
        # feat = obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()
        # feat = feat.permute(1,2,0).contiguous()



        # active_flag = self.move_flag_linear(feat) # bs,4
        active_flag = self.move_flag_linear(feat).squeeze(-1) # bs,4
        # print(active_flag.shape)

        return active_flag
