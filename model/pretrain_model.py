import os
import sys
sys.path.append('/root/code/OmniControl/')
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.base_cross_model import PerceiveEncoder,PerceiveDecoder


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x   
    
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024) # [B,1024]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            # print(x.shape)
            global_feat = x
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # N  [B, 1024, N]
            # print(x.shape,pointfeat.shape)
            return global_feat, torch.cat([x, pointfeat], 1), trans, trans_feat



class gaze_obj_model(torch.nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None,hint_dim=99,*args, **kargs):
        super().__init__()

        self.hint_dim = hint_dim

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

        
        self.encode_obj_mesh = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        self.pointnet_emb = nn.Linear(1024,self.latent_dim)

        self.gaze_linear = nn.Linear(3, 16)  
        self.encode_gaze = PerceiveEncoder(n_input_channels=16,
                                        n_latent=345,
                                        n_latent_channels=self.latent_dim,
                                        n_self_att_heads=4,
                                        n_self_att_layers=3,
                                        dropout=0.1)
        self.obj_linear = nn.Linear(9, 16)  
        self.encode_obj = PerceiveEncoder(n_input_channels=16,
                                        n_latent=345,
                                        n_latent_channels=self.latent_dim,
                                        n_self_att_heads=4,
                                        n_self_att_layers=3,
                                        dropout=0.1)

        self.gaze_decoder = PerceiveDecoder(n_query_channels=self.latent_dim,
                n_query=345,
                n_latent_channels=self.latent_dim,
                dropout=0.1)
        self.gaze_linear2 = nn.Sequential(
                                nn.Linear(self.latent_dim, 32), nn.ELU(),
                                nn.Linear(32,3), nn.ELU()
                            )

        self.obj_decoder = PerceiveDecoder(n_query_channels=self.latent_dim,
                n_query=345,
                n_latent_channels=self.latent_dim,
                dropout=0.1)
        self.obj_linear2 = nn.Sequential(
                                nn.Linear(self.latent_dim, 32), nn.ELU(),
                                nn.Linear(32,9), nn.ELU()
                            )
    
    def forward(self, motion,y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
     
        
        bs, nf, _ = y['gaze'].shape
        obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
        obj_mesh = y['obj_points']
        # reshape(bs,-1,3).permute(0,2,1).contiguous()

        global_obj_feat,points_feat, _ ,_ = self.encode_obj_mesh(obj_mesh.permute(0,2,1).contiguous())
        # print(global_obj_feat.shape)
        obj_shape_emb += self.pointnet_emb(global_obj_feat).unsqueeze(0)
            
        gaze_emb = self.encode_gaze(self.gaze_linear(y['gaze'])).permute(1,0,2).contiguous()
        obj_pose_emb = self.encode_obj(self.gaze_linear(y['obj_pose'])).permute(1,0,2).contiguous()
        # print(x.shape, obj_pose_emb.shape, obj_shape_emb.shape, gaze_emb.shape)

        obj_emb =  obj_pose_emb + obj_shape_emb 

        pre_gaze = self.gaze_linear2(self.gaze_decoder(obj_emb))
        pre_obj = self.obj_linear2(self.obj_decoder(gaze_emb))
        print(gaze_emb.shape,obj_emb.shape,pre_gaze.shape,pre_obj.shape)
        return gaze_emb,obj_emb,pre_gaze,pre_obj