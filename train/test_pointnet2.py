import os
import sys
sys.path.append('/root/code/OmniControl/')
import torch
from model.base_cross_model import *
from model.pointnet_plus2 import *

scene_encoder = PointNet2SemSegSSGShape({'feat_dim': 128}).cuda()
fp_layer = MyFPModule().cuda()

bs =64
nf = 354
scenes = torch.randn((bs,3000,3)).cuda()
gazes = torch.randn((bs,nf,3)).cuda()

scene_feats, scene_global_feats = scene_encoder(scenes.repeat(1, 1, 2))
print(scene_feats.shape, scene_global_feats.shape)
gaze_embedding = fp_layer(gazes, scenes, scene_feats).permute((0, 2, 1))
print(gaze_embedding.shape)
