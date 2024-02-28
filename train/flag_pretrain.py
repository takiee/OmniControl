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
        args.latent_dim = 8
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
    test_data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,hint_type=args.hint_type, split='test')
    train_data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,hint_type=args.hint_type)
    print(len(train_data))
    print(len(test_data))
    
    model = flag_model(**get_model_args(args, train_data))
    model.to(dist_util.dev())
    device = dist_util.dev()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=0)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1) #zuobian
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1) # 右边
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    num_epoch = 100
    best_loss = 100
    writer = SummaryWriter(log_dir=args.save_dir)
    iter = 0
    name = time.strftime("%Y%m%d%H%M%S%MS",time.localtime(time.time()))
    for epoch in range(num_epoch):
        # print("EPOCH:",epoch)
        total_loss = 0
        right = 0
        for motion,cond in train_data:
            cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
            gt_flag = cond['y']['flag']
            pred_flag = model(motion,cond['y'])

            gt_move_flag = gt_flag.sum(dim=-1)>0
            gt_move_index = torch.where(gt_move_flag==1)
            gt_active_flag = gt_flag[gt_move_index]
            gt_class = gt_move_index[1].long()

            C = nn.CrossEntropyLoss()
            # loss = C(pred_flag,gt_class) * 10
            loss = C(pred_flag,gt_class) 
            # print(loss)
            obj_index = torch.argmax(pred_flag,dim=-1)
            right += torch.sum(obj_index == gt_class)
            # writer.add_scalar("toal train loss",loss,iter)
            total_loss += loss
            loss.backward()
            optimizer.step()
            iter = iter + 1
        total_loss = total_loss / 106
        print(f"Epoch{epoch} -- TrainLoss",total_loss.item(),f"Acc:{right/3420}")
        writer.add_scalar("toal train loss",total_loss,epoch)
        writer.add_scalar("toal train Acc",right/3420,epoch)
        

        # save_dir = args.save_dir
        # os.makedirs(save_dir,exist_ok=True)
        # save_path = os.path.join(save_dir, str(epoch).zfill(6)+f'_model_{best_loss}.pt')
        # torch.save({
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict()
        #             }, save_path)
        # print("Saved model to:\n{}".format(save_path))
        
        # if total_loss < 400:
        with torch.no_grad():
            data_num = 0
            right = 0
            total_loss = 0
            for motion,cond in test_data:
                # print("in")
                cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                gt_flag = cond['y']['flag']
                pred_flag = model(motion,cond['y'])

                gt_move_flag = gt_flag.sum(dim=-1)>0
                gt_move_index = torch.where(gt_move_flag==1)
                gt_active_flag = gt_flag[gt_move_index]
                gt_class = gt_move_index[1].long()
                
                C = nn.CrossEntropyLoss()
                loss = C(pred_flag,gt_class)

                obj_index = torch.argmax(pred_flag,dim=-1)
                # print(obj_index)
                # print(gt_class)
                # print(gt_class==obj_index)
                data_num += gt_flag.shape[0]
                right += torch.sum(obj_index == gt_class)
                total_loss += loss
            
            total_loss = total_loss / 12
                
            writer.add_scalar("toal test loss",total_loss,epoch)
            writer.add_scalar("toal test Acc",right/405,epoch)
            print(f"Epoch{epoch} -- TestLoss",loss.item(),f'Acc:{right/405}')
        print('#######################################################################################')
            # if loss < best_loss:
            # #     print("Epoch: ", epoch)
            # #     print('################## BEST PERFORMANCE {:0.2f} ########'.format(loss))
            #     best_loss = loss
            #     if best_loss < 100:
                    
            #         save_dir = os.path.join(args.save_dir,name)
            #         os.makedirs(save_dir,exist_ok=True)
            #         save_path = os.path.join(save_dir, str(epoch).zfill(6)+f'_model_{best_loss}.pt')
            #         torch.save({
            #                     'model_state_dict': model.state_dict(),
            #                     'optimizer_state_dict': optimizer.state_dict()
            #                     }, save_path)
            #         print("Saved model to:\n{}".format(save_path))
            



if __name__ == "__main__":
    main()