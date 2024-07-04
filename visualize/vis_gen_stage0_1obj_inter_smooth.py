import cv2
import os
import manotorch
import torch
import numpy as np
import json
from tqdm import *
import copy
import sys
sys.path.append(os.getcwd())
# from model.rotation2xyz import Rotation2xyz
from pytorch3d.transforms import rotation_6d_to_matrix,axis_angle_to_matrix
import torch
import trimesh
import pyrender
import pickle
from manotorch.manolayer import ManoLayer
from os.path import join
import argparse


os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
def vis_smpl(out_path, image, nf, vertices, faces, camera_R, camera_T, K = np.array([1031.8450927734375, 0.0, 932.1596069335938, 0.0, 1022.4588623046875, 541.9437255859375, 0.0, 0.0, 1.0]).reshape((3,3))):
    outname = os.path.join(out_path, '{:06d}.jpg'.format(nf))
    # if os.path.exists(outname): return
    render_data = {}
    assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
    render_data = {"vertices": vertices, "faces": faces, "vid":0, "name": "human_{}_0".format(nf)}

    camera = {"K": K,
        "R": camera_R,
        "T":camera_T}
    from renderer import Renderer
    render = Renderer(height=3840, width=2160, faces=None)
    image_vis, depth = render.render(render_data, camera, image, add_back=True)
    # print(depth)
    # outname = os.path.join(out_path, '{:06d}.jpg'.format(nf))
    # cv2.imwrite(outname, image_vis)
    return image_vis, depth  

def img2video(image_path, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    height = 2160
    weight = 3840 * 2
    fps = 30
    videowriter = cv2.VideoWriter(video_path, fourcc, fps, (weight, height))  # 创建一个写入视频对象

    imgs = sorted(os.listdir(image_path))
    for img in tqdm(imgs):
        frame = cv2.imread(join(image_path, img))
        videowriter.write(frame)
    videowriter.release()

parser = argparse.ArgumentParser(description='Visualization')
parser.add_argument('--view',default=1,type=int,help='choose a camera view to render') # 1 10比较好
parser.add_argument('--render_path',type=str,help='choose a camera view to render') # 1 10比较好

# path = '/root/code/OmniControl/save/guide_delay1/samples_guide_delay1_000050000_seed10_predefined/results.npy'
# path = 'save/my_omnicontrol2/samples_my_omnicontrol2_000050000_seed10_predefined/results.npy'
# view = 3
args = parser.parse_args()
view = args.view
render_path = args.render_path
os.makedirs(render_path,exist_ok=True)

testpath = '/root/code/seqs/gazehoi_list_test_new.txt'
        
with open(testpath,'r') as f:
    test_list = f.readlines()

seqs = []
for info in test_list:
    seq = info.strip()
    seqs.append(seq)

# res = np.load(path,allow_pickle=True).item()
# pred_motion = torch.tensor(res['motions']) # obj_motion (bs,nf,4,3,4)
# print(pred_motion.shape)
# seqs = res['seqs']
datapath = '/root/code/seqs/1205_data/'
manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
hand_faces = manolayer.th_faces

calib_path = "/root/code/seqs/calibration_all.json"
obj_path = '/root/code/seqs/object/'
with open(calib_path) as f:
    calib_dome = json.load(f)
    f.close()
camera_pose = np.vstack((np.asarray(calib_dome[str(view)]['RT']).reshape((3,4)), np.ones(4) ))
K = np.asarray(calib_dome[str(view)]['K']).reshape((3,3))

seq_index = 0
length = 345 # 345
# fail_seq = ['1270', '1004', '0650', '0827', '1188', '0535', '0776', '1299', '0620', '0331', '0796', '1294']
for i in range(len(seqs)): # 每个seq遍历
# for i in range(1): # 每个seq遍历
    # seq = seqs[i]
    seq = seqs[i]
    print(seq)
    # if seq in fail_seq:
    #     continue
    seq_path = join(datapath,seq)

    meta_path = join(seq_path,'meta.pkl')
    with open(meta_path,'rb')as f:
            meta = pickle.load(f)
    
    active_obj = meta['gaze_obj']
    num_frames = meta['num_frames']

    obj_mesh_path = join(obj_path,active_obj,'simplified_scan_processed.obj')
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_verts = obj_mesh.vertices
    obj_faces = obj_mesh.faces
    obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))
    
    seq_len = obj_pose.shape[0]

    hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))
    hand_trans = hand_params[:,:3]
    hand_rot = hand_params[:,3:6]
    hand_theta = hand_params[:,3:51]
    mano_beta = hand_params[:,51:]


    gt_output = manolayer(hand_theta, mano_beta)
    gt_hand_verts = gt_output.verts - gt_output.joints[:, 0].unsqueeze(1) + hand_trans.unsqueeze(1)

    obj_verts_gt = torch.tensor(obj_verts).unsqueeze(0).repeat(seq_len,1,1).float()
    obj_verts_pred = torch.tensor(obj_verts).unsqueeze(0).repeat(seq_len,1,1).float()
    # obj_verts_pred = torch.tensor(obj_verts).unsqueeze(0).repeat(length,1,1).float()

    obj_pose = torch.tensor(obj_pose[:seq_len]).float()
    obj_R = obj_pose[:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    obj_T = obj_pose[:,:3,3].unsqueeze(1)  
    gt_obj_verts = torch.einsum('fpn,fnk->fpk',obj_verts_gt,obj_R) + obj_T

    # pred_obj_pose = pred_motion[i,:].float()
    pred_obj_pose = torch.tensor(np.load(join(seq_path,'pred_obj_and_goal.npy'),allow_pickle=True).item()['pred_obj_pose']).float()
    print(pred_obj_pose.shape)
    obj_R = pred_obj_pose[:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    obj_T = pred_obj_pose[:,:3,3].unsqueeze(1)
    # print(type(obj_verts_pred),obj_verts_pred.dtype)
    # print(obj_T.dtype, obj_R.dtype)
    pred_obj_verts = torch.einsum('fpn,fnk->fpk',obj_verts_pred,obj_R) + obj_T

    # print(obj_verts.shape,mano_verts.shape)
    render_out = join(render_path,seq,str(view))
    os.makedirs(render_out,exist_ok=True)
    for k in range(seq_len): ## 每帧遍历
        img = np.zeros((2160, 3840,3), np.uint8) + 255
        # if k >= seq_len:
        #     k_gt = seq_len - 1
        # else:
        #     k_gt = k
        # print(k)
        gt_obj_verts_k = gt_obj_verts[k].numpy()
        pred_obj_verts_k = pred_obj_verts[k].numpy()

        pred_all_verts = np.vstack((gt_hand_verts[k].numpy(),pred_obj_verts_k))
        gt_all_verts = np.vstack((gt_hand_verts[k].numpy(),gt_obj_verts_k))
        faces = np.vstack((hand_faces,obj_faces+778))
        pred_img, _ = vis_smpl(render_out, img, k, pred_all_verts, faces,camera_pose[:3,:3],camera_pose[:3,3],K)
        gt_img, _ = vis_smpl(render_out, img, k, gt_all_verts, faces,camera_pose[:3,:3],camera_pose[:3,3],K)
        cv2.putText(gt_img,'Ground Truth',(100,300),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,0),20)
        cv2.putText(pred_img,'Generation',(100,300),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,0),20)
        image_vis = cv2.hconcat([gt_img, pred_img])
        outname = os.path.join(render_out, '{:06d}.jpg'.format(k))
        cv2.imwrite(outname, image_vis)
        # break
    video_name = 'seq' + seq + '_view' + str(view) + '.mp4' 
    video_path = join(render_path,'render_videos')
    os.makedirs(video_path,exist_ok=True)
    video_path = join(video_path,video_name)
    img2video(render_out,video_path)
    # seq_index += 1
    # if render_num == seq_index:
    #     break
    # break



