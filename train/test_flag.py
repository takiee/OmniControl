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
from model.flag_pretrain_model import flag_model
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from diffusion.nn import mean_flat, sum_flat
import torch.nn.functional as F
import time

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
    if args.dataset == 'gazehoi_pretrain':
        # print('gazehoi!!')
        data_rep = 'rot6d'
        # njoints = 51
        njoints = 9
        nfeats = 1
        args.latent_dim = 128
        if args.hint_type == 'init_pose':
            hint_dim = 36 # 4*9
    elif  args.dataset == 'gazehoi_stage0_flag2_lowfps_global':
        # print('gazehoi!!')
        data_rep = 'rot6d'
        # njoints = 51
        njoints = 36
        nfeats = 1
        args.latent_dim = 128
        length = 69
        if args.hint_type == 'init_pose':
            hint_dim = 36 # 4*9

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,'length':length}

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')


    dist_util.setup_dist(args.device)

    print("creating data loader...")
    test_data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,hint_type=args.hint_type, split='test')
    # train_data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,hint_type=args.hint_type)
    
    model = flag_model(**get_model_args(args, test_data))
    model.to(dist_util.dev())
    device = dist_util.dev()

    state_dict = torch.load(args.save_dir, map_location='cpu')['model_state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("unexpected_keys: ", unexpected_keys)


    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Test...")
    total_loss = 0
    right = 0
    model.eval()
    with torch.no_grad():
        for motion,cond in test_data:
            cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
            gt_flag = cond['y']['flag']
            pred_flag = model(motion,cond['y'])

            gt_move_flag = gt_flag.sum(dim=-1)>0
            gt_move_index = torch.where(gt_move_flag==1)
            gt_active_flag = gt_flag[gt_move_index]
            gt_class = gt_move_index[1].long()

            C = nn.CrossEntropyLoss()
            loss = C(pred_flag,gt_class)
            # print(loss)
            obj_index = torch.argmax(pred_flag,dim=-1)
            right += torch.sum(obj_index == gt_class)
            total_loss += loss
        total_loss = total_loss / 12
        print(f"Test Loss",total_loss.item(),f"Acc:{right/405}")
    
    # save_dir = args.save_dir
    # os.makedirs(save_dir,exist_ok=True)
    # save_path = os.path.join(save_dir, str(epoch).zfill(6)+f'_model_{best_loss}.pt')
    # torch.save({
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict()
    #             }, save_path)
    # print("Saved model to:\n{}".format(save_path))



if __name__ == "__main__":
    main()