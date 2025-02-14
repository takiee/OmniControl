# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
import torch
import torch.nn as nn
import clip
from model.rotation2xyz import Rotation2xyz
from .transformer import *
from torch.autograd import Variable
from model.base_cross_model import PerceiveEncoder
from model.base_cross_model import *
from model.pointnet_plus2 import *


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


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class CMDM(torch.nn.Module):
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
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # if self.cond_mode != 'no_cond':
        #     if 'text' in self.cond_mode:
        #         self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        #         print('EMBED TEXT')
        #         print('Loading CLIP...')
        #         self.clip_version = clip_version
        #         self.clip_model = self.load_and_freeze_clip(clip_version)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        # ------
        # --- CMDM ---
        # input 263 or 6 * 3 or 3
        # n_joints = 22 if njoints == 263 else 21
        # n_joints = 17
        # if self.dataset == 'gazehoi_stage1':
        #     self.input_hint_block = HintBlock(self.data_rep, 99, self.latent_dim)
        # elif self.dataset == 'gazehoi_stage2':
        
        # if self.hint_dim != 99 and not (self.dataset.startswith('gazehoi_stage0')):
        #     # print("wrong")
        #     self.encode_obj_pose = nn.Linear(12,self.latent_dim)
        #     self.encode_obj_mesh = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        #     self.pointnet_emb = nn.Linear(1024,self.latent_dim)
        if self.dataset == 'gazehoi_stage0':
            # print("correct")
            """ 原版
            self.encode_obj_pose = nn.Linear(36,self.latent_dim)
            self.encode_obj_mesh = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
            self.pointnet_emb = nn.Linear(1024,self.latent_dim)
            self.obj_linear = nn.Linear(self.latent_dim*4,self.latent_dim)
            self.gaze_linear = nn.Linear(3, 16)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=16,
                                            n_latent=345,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            """
            # 参照GIMO修改后
            self.encode_obj_pose = nn.Linear(36,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.fp_layer = MyFPModule()
            self.gaze_linear = nn.Linear(self.latent_dim, self.latent_dim)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
                                            
        elif self.dataset == 'gazehoi_stage0_flag' or self.dataset == 'gazehoi_stage0_flag2' or self.dataset == 'gazehoi_stage0_flag2_lowfps' :
        # elif self.dataset == 'gazehoi_stage0_flag' or self.dataset == 'gazehoi_stage0_flag2' or self.dataset == 'gazehoi_stage0_flag2_lowfps' or self.dataset == 'gazehoi_stage0_flag2_lowfps_global':
            # print("correct")
            """ 原版
            self.encode_obj_pose = nn.Linear(36,self.latent_dim)
            self.encode_obj_mesh = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
            self.pointnet_emb = nn.Linear(1024,self.latent_dim)
            self.obj_linear = nn.Linear(self.latent_dim*4,self.latent_dim)
            self.gaze_linear = nn.Linear(3, 16)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=16,
                                            n_latent=345,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            """
            # 参照GIMO修改后
            self.encode_obj_pose = nn.Linear(36,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.fp_layer = MyFPModule()
            self.gaze_linear = nn.Linear(self.latent_dim, self.latent_dim)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            self.move_flag = nn.Sequential( nn.Linear(self.latent_dim, 32), nn.ELU(),
                                            nn.Linear(32,4) )

        elif  self.dataset == 'gazehoi_stage0_flag2_lowfps_global':
            print("correct")
            self.encode_obj_pose = nn.Linear(36,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.fp_layer = MyFPModule()
            self.gaze_linear = nn.Linear(self.latent_dim, self.latent_dim)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            self.move_flag_pool = nn.MaxPool1d(kernel_size=self.length)
            self.move_flag_linear = nn.Sequential( nn.Linear(self.latent_dim, 32), nn.ELU(),
                                            nn.Linear(32,4) )

            self.move_index_linear = nn.Sequential( nn.Linear(self.latent_dim, 32), nn.ELU(),
                                            nn.Linear(32,8), nn.ELU(),
                                            nn.Linear(8,1) )

        elif  self.dataset == 'gazehoi_stage0_1obj':
            self.encode_obj_pose = nn.Linear(9,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.fp_layer = MyFPModule()
            self.gaze_linear = nn.Linear(self.latent_dim, self.latent_dim)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)

        elif  self.dataset == 'gazehoi_stage0_noatt':
            self.encode_obj_pose = nn.Linear(9,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.fp_layer = MyFPModule()
            self.gaze_linear = nn.Linear(3, self.latent_dim)  
            # self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
            #                                 n_latent=self.length,
            #                                 n_latent_channels=self.latent_dim,
            #                                 n_self_att_heads=4,
            #                                 n_self_att_layers=3,
            #                                 dropout=0.1)
        elif self.dataset == 'gazehoi_g2ho':
            self.encode_obj_pose = nn.Linear(9,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.fp_layer = MyFPModule()
            self.gaze_linear = nn.Linear(self.latent_dim, self.latent_dim)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            self.encode_hand_pose = nn.Sequential(nn.Linear(99,128), nn.ELU(),
                                                nn.Linear(128,self.latent_dim) )
        elif self.dataset == 'gazehoi_o2h':
            self.encode_obj_pose = nn.Linear(9,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.encode_obj = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            self.encode_hand_pose = nn.Sequential(nn.Linear(99,128), nn.ELU(),
                                                nn.Linear(128,self.latent_dim) )
        elif self.dataset == 'gazehoi_o2h_mid':
            self.encode_obj_pose = nn.Linear(12,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.encode_obj = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            self.encode_hand_pose = nn.Sequential(nn.Linear(291,128), nn.ELU(),
                                                nn.Linear(128,self.latent_dim) )


        elif  self.dataset == 'gazehoi_stage0_norm':
            self.encode_obj_pose = nn.Linear(9,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.gaze_linear = nn.Linear(6, 32)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=32,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)

        elif  self.dataset == 'gazehoi_stage0_point':
            self.encode_obj_pose = nn.Linear(9,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.gaze_linear = nn.Linear(3, 32)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=32,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            
            

        elif self.dataset == 'gazehoi_stage1_new' or self.dataset == 'gazehoi_stage1_repair':
            self.encode_obj_pose = nn.Sequential(nn.Linear(9,64), nn.ELU(),
                                                nn.Linear(64,self.latent_dim) )
            self.encode_hand_pose = nn.Sequential(nn.Linear(99,128), nn.ELU(),
                                                nn.Linear(128,self.latent_dim) )
            self.encode_obj_mesh = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
            self.pointnet_emb = nn.Linear(1024,self.latent_dim)

        elif self.dataset == 'gazehoi_stage1_simple':
            self.encode_hand_pose = nn.Sequential(nn.Linear(99,128), nn.ELU(),
                                                nn.Linear(128,self.latent_dim) )
            

        # if self.dataset != 'gazehoi_stage':
        if not (self.dataset.startswith('gazehoi_stage0')) and not (self.dataset == 'gazehoi_stage1_simple') and not(self.dataset == 'gazehoi_g2ho') and not(self.dataset == 'gazehoi_o2h') and not(self.dataset == 'gazehoi_o2h_mid'):
        # if not (self.dataset.startswith('gazehoi_stage0')) and self.dataset != 'gazehoi_stage1_repair':
        # if not (self.dataset.startswith('gazehoi_stage0')) and not (self.dataset == 'gazehoi_stage1_new'):
            print(self.data_rep, self.hint_dim, self.latent_dim)
            self.input_hint_block = HintBlock(self.data_rep, self.hint_dim, self.latent_dim)

            self.c_input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

            self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

            print("ccTRANS_ENC init")
            seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            self.c_seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers,
                                                        return_intermediate=True)

            self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
            
            self.c_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def cmdm_forward(self, x, timesteps, y=None, weight=1.0):
        """
        Realism Guidance
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        emb = self.c_embed_timestep(timesteps)  # [1, bs, d]

        seq_mask = y['hint'].sum(-1) != 0
        # print(y['hint'].shape)
        guided_hint = self.input_hint_block(y['hint'].float())  # [bs, d]

        force_mask = y.get('uncond', False)
        # if 'text' in self.cond_mode:
        #     enc_text = self.encode_text(y['text'])
        #     emb += self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        x = self.c_input_process(x)
        # print(x.shape,guided_hint.shape,seq_mask.shape)

        x += guided_hint * seq_mask.permute(1, 0).unsqueeze(-1)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.c_seqTransEncoder(xseq)  # [seqlen+1, bs, d]

        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight
        return control
    
    def mdm_forward(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        # print("emb shape:", emb.shape)

        force_mask = y.get('uncond', False)
        # if 'text' in self.cond_mode:
        #     enc_text = self.encode_text(y['text'])
        #     emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        
        x = self.input_process(x)
        if self.dataset == 'gazehoi_stage2':
            # print(x.shape)
            """
            提取物体pose和shape特征
            """
            bs, nf, _,_ = y['obj_pose'].shape
            obj_pose_emb = self.encode_obj_pose(y['obj_pose'].reshape(bs,nf,-1)).permute(1,0,2).contiguous()
            # print(obj_pose_emb.shape)
            obj_mesh = y['obj_points'].permute(0,2,1)
            global_obj_feat,points_feat, _ ,_ = self.encode_obj_mesh(obj_mesh)
            # print(global_obj_feat.shape)
            obj_shape_emb = self.pointnet_emb(global_obj_feat).unsqueeze(0)
            # print(obj_pose_emb.shape, obj_shape_emb.shape)
            
            x = x + obj_pose_emb + obj_shape_emb
        elif self.dataset == 'gazehoi_stage1_new' or self.dataset == 'gazehoi_stage1_repair':
            # print(x.shape)
            """
            提取物体pose和shape特征
            """
            # print(y['goal_obj_pose'].shape)
            bs, _ = y['goal_obj_pose'].shape
            obj_pose_emb = self.encode_obj_pose(y['goal_obj_pose'])
            # print(obj_pose_emb.shape)
            obj_mesh = y['obj_points'].permute(0,2,1)
            global_obj_feat,points_feat, _ ,_ = self.encode_obj_mesh(obj_mesh)
            # print(global_obj_feat.shape)
            obj_shape_emb = self.pointnet_emb(global_obj_feat).unsqueeze(0)
            # print(obj_pose_emb.shape, obj_shape_emb.shape)
            init_hand_emb = self.encode_hand_pose(y['init_hand_pose']) #b,D
            goal_hand_emb = self.encode_hand_pose(y['goal_hand_pose'])
            # print(x.shape)
            x = x + obj_pose_emb + obj_shape_emb + init_hand_emb + goal_hand_emb
        elif self.dataset == 'gazehoi_stage1_simple':
            # print(x.shape)
            """
            提取物体pose和shape特征
            """
            # print(y['goal_obj_pose'].shape)
            init_hand_emb = self.encode_hand_pose(y['init_hand_pose']) #b,D
            goal_hand_emb = self.encode_hand_pose(y['goal_hand_pose'])
            x = x + init_hand_emb + goal_hand_emb
        elif self.dataset == 'gazehoi_stage0':
            # print(x.shape)
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape
            init_obj_pose = y['hint'][:,0]
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            # print(obj_pose_emb.shape)
            points = y['obj_points']
            # reshape(bs,-1,3).permute(0,2,1).contiguous()
            table = points[:,2000:]
            obj_mesh = points[:,:2000].reshape(bs,4,500,3)
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze_emb)
            gaze_feat = self.encode_gaze(gaze_feat)

            # # global_obj_feat,points_feat, _ ,_ = self.encode_obj_mesh(obj_mesh[:,0].permute(0,2,1).contiguous())
            # points_feat, global_obj_feat= self.encode_obj_mesh(table.permute(0,2,1).contiguous())
            # obj_shape_emb = global_obj_feat
            # scene_feat = points_feat
            # # obj_shape_emb = self.pointnet_emb(global_obj_feat).unsqueeze(0)
            # for i in range(0, 4):
            #     points_feat, global_obj_feat = self.encode_obj_mesh(obj_mesh[:,i].permute(0,2,1).contiguous())
            #     # print(global_obj_feat.shape)
            #     obj_shape_emb += self.pointnet_emb(global_obj_feat).unsqueeze(0)
            #     # obj_shape_emb += global_obj_feat
            #     # scene_feat
                
            # gaze_emb = self.encode_gaze(self.gaze_linear(y['gaze'])).permute(1,0,2).contiguous()
            # print(x.shape, obj_pose_emb.shape, global_obj_feat.shape, gaze_feat.shape)

            x = x + obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous() 
        elif self.dataset == 'gazehoi_stage0_flag':
            # print(x.shape)
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape
            init_obj_pose = y['hint'][:,0]
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            # print(obj_pose_emb.shape)
            points = y['obj_points']
            # reshape(bs,-1,3).permute(0,2,1).contiguous()
            table = points[:,2000:]
            obj_mesh = points[:,:2000].reshape(bs,4,500,3)
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze_emb)
            gaze_feat = self.encode_gaze(gaze_feat)

            x = x + obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()
            
            feat = obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()
            feat = feat.permute(1,0,2).contiguous()
            flag = self.move_flag(feat)
        elif self.dataset == 'gazehoi_stage0_flag2' or self.dataset == 'gazehoi_stage0_flag2_lowfps' :
        # elif self.dataset == 'gazehoi_stage0_flag2' or self.dataset == 'gazehoi_stage0_flag2_lowfps' or self.dataset == 'gazehoi_stage0_flag2_lowfps_global':
            # print(x.shape)
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape
            init_obj_pose = y['hint'][:,0]
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            # print(obj_pose_emb.shape)
            points = y['obj_points']
            # reshape(bs,-1,3).permute(0,2,1).contiguous()
            table = points[:,2000:]
            obj_mesh = points[:,:2000].reshape(bs,4,500,3)
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze_emb)
            gaze_feat = self.encode_gaze(gaze_feat)

            
            feat = obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()
            feat = feat.permute(1,0,2).contiguous()
            flag = self.move_flag(feat) # b,nf,4
            # pred_flag = flag.permute(0,2,1).contiguous() # bs frame 4 -- b 4 frames
            pred_move = torch.sum(flag,dim=1) # #b,4
            obj_index = torch.argmax(pred_move,dim=-1)
            tgt_obj = obj_mesh[torch.arange(bs),obj_index]
            # print(tgt_obj.shape)
            _ , tgt_obj_feat= self.encode_obj_mesh(tgt_obj.repeat(1, 1, 2))

            x = x + obj_pose_emb + global_obj_feat + tgt_obj_feat + gaze_feat.permute(1,0,2).contiguous()

        elif self.dataset == 'gazehoi_stage0_flag2_lowfps_global':
            # print(x.shape)
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape

            init_obj_pose = y['hint'][:,0]
            gt_flag = y['flag']
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            # print(obj_pose_emb.shape)
            points = y['obj_points']
            # reshape(bs,-1,3).permute(0,2,1).contiguous()
            table = points[:,2000:]
            obj_mesh = points[:,:2000].reshape(bs,4,500,3)
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze_emb)
            gaze_feat = self.encode_gaze(gaze_feat)

            
            feat = obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()
            feat = feat.permute(1,2,0).contiguous()
            active_flag = self.move_flag_linear(self.move_flag_pool(feat).squeeze(-1)) # bs,4

            gt_move = torch.sum(gt_flag,dim=-1) # #b,4
            obj_index = torch.argmax(gt_move,dim=-1)
            tgt_obj = obj_mesh[torch.arange(bs),obj_index]
            # print(tgt_obj.shape)
            _ , tgt_obj_feat= self.encode_obj_mesh(tgt_obj.repeat(1, 1, 2))
            # print(gaze_feat.shape,tgt_obj_feat.shape)
            move_flag = self.move_index_linear(gaze_feat + tgt_obj_feat.unsqueeze(1)).squeeze(-1)
            # print(move_feat.shape)

            # pred_move_index = 

            x = x + obj_pose_emb + global_obj_feat + tgt_obj_feat + gaze_feat.permute(1,0,2).contiguous()

        elif self.dataset == 'gazehoi_stage0_1obj':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape
            init_obj_pose = y['hint'][:,0]
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            points = y['obj_points']
            table = points[:,500:]
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze_emb)
            gaze_feat = self.encode_gaze(gaze_feat)

            x = x + obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()
        elif self.dataset == 'gazehoi_stage0_noatt':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape
            init_obj_pose = y['hint'][:,0]
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            points = y['obj_points']
            table = points[:,500:]
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            # gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze)
            # gaze_feat = self.encode_gaze(gaze_feat)

            x = x + obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()

        elif self.dataset == 'gazehoi_g2ho':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape
            init_obj_pose = y['init_obj_pose']
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            points = y['obj_points']
            table = points[:,500:]
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze_emb)
            gaze_feat = self.encode_gaze(gaze_feat)

            init_hand_emb = self.encode_hand_pose(y['init_hand_pose']) #b,D

            x = x + obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous() + init_hand_emb 

        elif self.dataset == 'gazehoi_o2h':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['obj_pose'].shape
            obj_pose = y['obj_pose']
            # print(obj_pose.shape)
            obj_pose_emb = self.encode_obj_pose(obj_pose)

            points = y['obj_points']
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))

            obj_feat = self.encode_obj(obj_pose_emb)

            init_hand_emb = self.encode_hand_pose(y['init_hand_pose']) #b,D

            x = x  + global_obj_feat + obj_feat.permute(1,0,2).contiguous() + init_hand_emb 
        elif self.dataset == 'gazehoi_o2h_mid':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['obj_pose'].shape
            obj_pose = y['obj_pose']
            # print(obj_pose.shape)
            obj_pose_emb = self.encode_obj_pose(obj_pose)

            points = y['obj_points']
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))

            obj_feat = self.encode_obj(obj_pose_emb)

            init_hand_emb = self.encode_hand_pose(y['init_hand_pose']) #b,D

            x = x  + global_obj_feat + obj_feat.permute(1,0,2).contiguous() + init_hand_emb 

        elif self.dataset == 'gazehoi_stage0_norm' or self.dataset == 'gazehoi_stage0_point':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape
            init_obj_pose = y['hint'][:,0]
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            points = y['obj_points']
            table = points[:,500:]
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_feat = self.gaze_linear(gaze)
            gaze_feat = self.encode_gaze(gaze_feat)

            x = x + obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()



        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, control=control)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        # print(self.dataset)
        if self.dataset == 'gazehoi_stage0_flag' or self.dataset == 'gazehoi_stage0_flag2' or self.dataset == 'gazehoi_stage0_flag2_lowfps':
        # if self.dataset == 'gazehoi_stage0_flag' or self.dataset == 'gazehoi_stage0_flag2' or self.dataset == 'gazehoi_stage0_flag2_lowfps' or self.dataset == 'gazehoi_stage0_flag2_lowfps_global':
            # print("return here")
            return output, flag
        elif self.dataset == 'gazehoi_stage0_flag2_lowfps_global':
            # print("return here")
            return output, (active_flag,move_flag)
        else:
            # print("wrong return")
            return output

    def mdm_forward_test(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        # print("emb shape:", emb.shape)

        force_mask = y.get('uncond', False)
        # if 'text' in self.cond_mode:
        #     enc_text = self.encode_text(y['text'])
        #     emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        
        x = self.input_process(x)
        
        if self.dataset == 'gazehoi_stage0_flag2_lowfps_global':
            # print(x.shape)
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape

            init_obj_pose = y['hint'][:,0]
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            # print(obj_pose_emb.shape)
            points = y['obj_points']
            # reshape(bs,-1,3).permute(0,2,1).contiguous()
            table = points[:,2000:]
            obj_mesh = points[:,:2000].reshape(bs,4,500,3)
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze_emb)
            gaze_feat = self.encode_gaze(gaze_feat)

            
            feat = obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()
            feat = feat.permute(1,2,0).contiguous()
            active_flag = self.move_flag_linear(self.move_flag_pool(feat).squeeze(-1)) # bs,4

            # flag给ground truth
            gt_flag = y['flag']
            gt_move = torch.sum(gt_flag,dim=-1) # #b,4
            obj_index = torch.argmax(gt_move,dim=-1)
            # print('gt_flag!')

            # test 时候该用的
            # obj_index = torch.argmax(active_flag,dim=-1)

            tgt_obj = obj_mesh[torch.arange(bs),obj_index]
            # print(tgt_obj.shape)
            _ , tgt_obj_feat= self.encode_obj_mesh(tgt_obj.repeat(1, 1, 2))
            # print(gaze_feat.shape,tgt_obj_feat.shape)
            move_flag = self.move_index_linear(gaze_feat + tgt_obj_feat.unsqueeze(1)).squeeze(-1)
            # print(move_feat.shape)

            # pred_move_index = 

            x = x + obj_pose_emb + global_obj_feat + tgt_obj_feat + gaze_feat.permute(1,0,2).contiguous()

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, control=control)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output, (active_flag,move_flag)

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # if 'hint' in y.keys() :
        # if 'hint' in y.keys() and not (self.dataset.startswith('gazehoi_stage0')) and self.dataset != 'gazehoi_stage1_repair' :
        if 'hint' in y.keys() and not (self.dataset.startswith('gazehoi_stage0')) and not (self.dataset == 'gazehoi_stage1_new') and not (self.dataset == 'gazehoi_stage1_simple') and not(self.dataset == 'gazehoi_g2ho') and not(self.dataset == 'gazehoi_o2h') and not(self.dataset == 'gazehoi_o2h_mid'):
            control = self.cmdm_forward(x, timesteps, y)
        else:
            control = None

        output = self.mdm_forward(x, timesteps, y, control)
        return output

    def forward_test(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # if 'hint' in y.keys() :
        if 'hint' in y.keys() and not (self.dataset.startswith('gazehoi_stage0')):
            control = self.cmdm_forward(x, timesteps, y)
        else:
            control = None

        output = self.mdm_forward_test(x, timesteps, y, control)
        return output

    def _apply(self, fn):
        super()._apply(fn)
        # self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        # self.rot2xyz.smpl_model.train(*args, **kwargs)


class HintBlock(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.ModuleList([
            nn.Linear(self.input_feats, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        ])

    def forward(self, x):
        # print(x.shape)
        x = x.permute((1, 0, 2))

        for module in self.poseEmbedding:
            # print(x.shape,self.input_feats,self.latent_dim)
            # print(x.shape)
            x = module(x)  # [seqlen, bs, d]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        # print(x.shape)
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec','mid']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec','mid']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output