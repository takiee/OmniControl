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

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,hint_type=args.hint_type)
    model = gaze_obj_model(**get_model_args(args, data))
    model.to(dist_util.dev())
    device = dist_util.dev()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    scheduler = StepLR(optimizer, step_size=300, gamma=0.1)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    num_epoch = 10000
    best_loss = 100
    writer = SummaryWriter(log_dir=args.save_dir)
    iter = 0
    for epoch in range(num_epoch):
        print("EPOCH:",epoch)
        for motion,cond in data:
            motion = motion.to(device)
            cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
            mask = cond['y']['mask']
            gaze = cond['y']['gaze']
            obj_pose = cond['y']['obj_pose']

            gaze_emb,obj_emb,pre_gaze,pre_obj = model(motion,cond['y'])
            
            KLD = nn.KLDivLoss(reduction='batchmean')
            kl_loss1 = KLD(gaze_emb, obj_emb)
            kl_loss2 = KLD(obj_emb, gaze_emb)
            kl_loss = (kl_loss1 + kl_loss2) / 2

            gaze_rec_loss = masked_l2(gaze,pre_gaze,mask)
            obj_rec_loss = masked_l2(obj_pose,pre_obj,mask)

            loss = (kl_loss + gaze_rec_loss + obj_rec_loss).mean()
            print(loss)
            # print(loss)
            writer.add_scalar("toal loss",loss,iter)
            writer.add_scalar("kl_loss",kl_loss.mean(),iter)
            writer.add_scalar("gaze_rec loss",gaze_rec_loss.mean(),iter)
            writer.add_scalar("obj rec loss",obj_rec_loss.mean(),iter)
            # writer.add_scalar("time_smooth",terms['time_smooth'].mean(),iter)

            if loss < best_loss:
                print("Epoch: ", epoch)
                print('################## BEST PERFORMANCE {:0.2f} ########'.format(loss))
                best_loss = loss
                if best_loss < 0.5:
                    save_path = os.path.join(args.save_dir, str(epoch).zfill(6)+'_model.pt')
                    torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                                }, save_path)
                    print("Saved model to:\n{}".format(save_path))
            loss.backward()
            optimizer.step()
            iter = iter + 1



if __name__ == "__main__":
    main()