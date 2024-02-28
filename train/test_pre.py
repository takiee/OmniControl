import os
import sys
sys.path.append('/root/code/OmniControl/')
sys.path.append(os.getcwd())
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import torch.nn as nn
from model.pretrain_model import gaze_obj_model
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from diffusion.nn import mean_flat, sum_flat
import torch.nn.functional as F
import time
import numpy as np


def masked_l2(a, b, mask):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    l2_loss = lambda a, b: (a - b) ** 2
    loss = l2_loss(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1] * a.shape[2]
    non_zero_elements = sum_flat(mask) * n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / non_zero_elements
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val
def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263 # + 66
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    elif args.dataset == 'gazehoi_stage1' or args.dataset == 'gazehoi_stage2':
        # print('gazehoi!!')
        data_rep = 'rot6d'
        # njoints = 51
        njoints = 99
        nfeats = 1
        if args.hint_type == 'root_dis':
            hint_dim = 1
        elif args.hint_type == 'tip_dis':
            hint_dim = 5
        elif args.hint_type == 'goal_pose':
            hint_dim = 99
        elif args.hint_type == 'tips_closest_point':
            hint_dim = 20
        elif args.hint_type == 'hand_T':
            hint_dim = 3
    elif args.dataset == 'gazehoi_pretrain':
        # print('gazehoi!!')
        data_rep = 'rot6d'
        # njoints = 51
        njoints = 9
        nfeats = 1
        args.latent_dim = 128
        if args.hint_type == 'init_pose':
            hint_dim = 36 # 4*9

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}


def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    dist_util.setup_dist(args.device)
    out_path = os.path.join(args.save_dir,'result')
    os.makedirs(out_path,exist_ok=True)
    print("creating data loader...")
    test_data = get_dataset_loader(name=args.dataset, batch_size=1, num_frames=args.num_frames,hint_type=args.hint_type, split='test')

    model = gaze_obj_model(**get_model_args(args, test_data))
    checkpoint = torch.load('save/0220_gaze_obj_pretrain/2024022016020102S/000033_model_57.056800842285156.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model.to(dist_util.dev())
    device = dist_util.dev()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Testing...")
    for motion,cond in test_data:
        motion = motion.to(device)
        cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
        mask = cond['y']['mask']
        gaze = cond['y']['gaze']
        obj_pose = cond['y']['obj_pose']
        seq = cond['y']['seq_name']
        gaze_emb,obj_emb,pre_gaze,pre_obj = model(motion,cond['y'])
        gaze_emb = gaze_emb.detach().cpu().numpy()
        np.save(os.path.join(out_path,f'{seq}_gaze_emb.npy'),gaze_emb)
        break


if __name__ == "__main__":
    main()