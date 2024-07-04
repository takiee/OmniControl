def cal_goal_err(x, hint):
    """
    x (bs,nf,99)
    只计算goal pose 的 error
    不算 init pose
    """
    hint[:,0] = 0
    mask_hint = hint.reshape(hint.shape[0], hint.shape[1], -1).sum(dim=-1, keepdim=True) != 0
    loss = torch.sum(torch.norm((x - hint) * mask_hint, dim=-1))
    loss = torch.mean(loss)
    return loss

def stage1_eval(pred_motion, seqs, hint, res_save_path):
    pred_motion = torch.tensor(pred_motion)
    hint = torch.tensor(hint)
    goal_err = cal_goal_err(pred_motion,hint)

    num_samples = pred_motion.shape[0]


    datapath = '/root/code/seqs/1205_data/'
    manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
    obj_path = '/root/code/seqs/object/'

    total_hand_T_error = 0
    total_hand_R_error = 0
    total_mpjpe = 0 # root-relative
    total_traj_invalid_10 = 0
    total_traj_invalid_20 = 0
    total_traj_num = 0
    total_hand_R_error = 0

    for i in range(num_samples):
        seq = seqs[i]
        seq_path = join(datapath,seq)

        meta_path = join(seq_path,'meta.pkl')
        with open(meta_path,'rb')as f:
            meta = pickle.load(f)
            
        active_obj = meta['active_obj']
        obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))
        
        goal_index = meta['goal_index']
        obj_pose = torch.tensor(obj_pose[:goal_index]).float()

        if goal_index < 59:
            hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))[:goal_index]
            pred_motion_i = pred_motion[i,:goal_index]
        else:
            hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))[goal_index-60:goal_index]
            pred_motion_i = pred_motion[i]

        hand_trans = hand_params[:,:3]
        hand_rot = hand_params[:,3:6]
        hand_theta = hand_params[:,3:51]
        mano_beta = hand_params[:,51:]

        pred_trans = pred_motion_i[:,:3]
        pred_rot = pred_motion_i[:,3:6]
        pred_theta = pred_motion_i[:,3:51]

        # print(goal_index,pred_theta.shape, mano_beta.shape)
        pred_output = manolayer(pred_theta, mano_beta)
        pred_joints = pred_output.joints - pred_output.joints[:, 0].unsqueeze(1)

        gt_output = manolayer(hand_theta, mano_beta)
        gt_joints = gt_output.joints - gt_output.joints[:, 0].unsqueeze(1)

        mpjpe = torch.sum(torch.norm(pred_joints - gt_joints, dim=-1)) / (hand_trans.shape[0] * 21)
        total_mpjpe += mpjpe

        hand_T_error = torch.sum(torch.norm(hand_trans-pred_trans,dim=-1)) / hand_trans.shape[0]
        total_hand_T_error += hand_T_error

        hand_rot = axis_angle_to_matrix(hand_rot)
        pred_rot = axis_angle_to_matrix(pred_rot)
        hand_rot = torch.einsum('...ij->...ji', [hand_rot])
        hand_R_error = (torch.einsum('fpn,fnk->fpk',hand_rot,pred_rot) - torch.eye(3).unsqueeze(0).repeat(hand_rot.shape[0],1,1)).reshape(-1,9) # nf,3,3 --nf,9
        hand_R_error =  torch.mean(torch.norm(hand_R_error,dim=-1))
        total_hand_R_error += hand_R_error

        traj_error = torch.norm(hand_trans-pred_trans,p=2,dim=1) 
        traj_invalid_10 = torch.sum(traj_error > 0.1)
        traj_invalid_20 = torch.sum(traj_error > 0.2)
        total_traj_invalid_10 += traj_invalid_10
        total_traj_invalid_20 += traj_invalid_20
        total_traj_num += traj_error.shape[0]
    
    goal_err = goal_err.numpy()
    traj_err_10 = (total_traj_invalid_10/total_traj_num).numpy()
    traj_err_20 = (total_traj_invalid_20/total_traj_num).numpy()
    hand_T_err = (total_hand_T_error/num_samples).numpy()
    hand_R_err = (total_hand_R_error/num_samples).numpy()
    mpjpe = (total_mpjpe/num_samples).numpy()
    with open(res_save_path, 'w') as f:
        f.write(f'Goal Error:{goal_err:.6f}\n')
        f.write(f'Traj Error (>10cm):{traj_err_10:.6f}\n')
        f.write(f'Traj Error (>20cm):{traj_err_20:.6f}\n')
        f.write(f'Hand Trans Error:{hand_T_err:.6f}\n')
        f.write(f'Hand Rot Error:{hand_R_err:.6f}\n')
        f.write(f'MPJPE :{mpjpe:.6f}\n')


# res = np.load(path,allow_pickle=True).item()
# pred_motion = torch.tensor(res['motion'])
# seqs = res['seqs']