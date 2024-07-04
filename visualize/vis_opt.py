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
    from renderer_raw import Renderer
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
    fps = 5
    videowriter = cv2.VideoWriter(video_path, fourcc, fps, (weight, height))  # 创建一个写入视频对象

    imgs = sorted(os.listdir(image_path))
    for img in tqdm(imgs):
        frame = cv2.imread(join(image_path, img))
        videowriter.write(frame)
    videowriter.release()

parser = argparse.ArgumentParser(description='Visualization')
parser.add_argument('--res_path',type=str,help='result.npy root path')
parser.add_argument('--view',default=1,type=int,help='choose a camera view to render') # 1 10比较好
parser.add_argument('--render_num',default=1,type=int,help='the number of seqs that you want to render') # 1 10比较好
# parser.add_argument('--stage',default='stage1',type=str,help='stage?') # 1 10比较好

# path = '/root/code/OmniControl/save/guide_delay1/samples_guide_delay1_000050000_seed10_predefined/results.npy'
# path = 'save/my_omnicontrol2/samples_my_omnicontrol2_000050000_seed10_predefined/results.npy'
# view = 3
args = parser.parse_args()
path = args.res_path
view = args.view
render_num = args.render_num


datapath = '/root/code/seqs/0303_data/'
manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
hand_faces = manolayer.th_faces

calib_path = "/root/code/seqs/calibration_all.json"
obj_path = '/root/code/seqs/object/'
with open(calib_path) as f:
    calib_dome = json.load(f)
    f.close()
camera_pose = np.vstack((np.asarray(calib_dome[str(view)]['RT']).reshape((3,4)), np.ones(4) ))
K = np.asarray(calib_dome[str(view)]['K']).reshape((3,3))

# testpath = '/root/code/seqs/gazehoi_list_test_0303.txt'
# with open(testpath,'r') as f:
#     info_list = f.readlines()
# seqs = []
# for info in info_list:
#     seq = info.strip()
#     seqs.append(seq)
seqs = ['0009']
for seq in seqs:
    # seq = seqs[i]
    res = np.load((path),allow_pickle=True).item()
    # res = np.load(join(path,f'{seq}.npy'),allow_pickle=True).item()
    pred_motion = torch.tensor(res['hand_motion']).cpu()
    obj_pose = torch.tensor(res['obj_motion']).cpu().float()
    seq = res['seq']
    print(seq)
    seq_path = join(datapath,seq)

    meta_path = join(seq_path,'meta.pkl')
    with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
    active_obj = meta['active_obj']
    obj_mesh_path = join(obj_path,active_obj,'simplified_scan_processed.obj')
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_verts = obj_mesh.vertices
    obj_faces = obj_mesh.faces
    # obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy'))
    
    
    seq_len = obj_pose.shape[0]
    
   
    obj_verts = torch.tensor(obj_verts).unsqueeze(0).repeat(seq_len,1,1).float()
    
    
    pred_trans = pred_motion[:,:3]
    pred_theta = pred_motion[:,3:51]
    pred_rot = pred_motion[:,3:6]
    length = pred_motion.shape[0]


    # hand_params[-1,:51] = torch.tensor(res['hint'][i][0]).reshape(1,-1)
    hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))
    mano_beta = hand_params[0,51:].unsqueeze(0).repeat(seq_len,1)

    pred_output = manolayer(pred_theta, mano_beta)
    pred_verts = pred_output.verts - pred_output.joints[:, 0].unsqueeze(1) + pred_trans.unsqueeze(1)


    obj_R = obj_pose[:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    obj_T = obj_pose[:,:3,3].unsqueeze(1)
    
    # print(obj_verts.shape,obj_R.shape,obj_T.shape)
    obj_verts = torch.einsum('fpn,fnk->fpk',obj_verts,obj_R) + obj_T

    # print(obj_verts.shape,mano_verts.shape)
    render_path = path.replace('.npy','render')
    print(render_path)
    render_out = join(render_path,seq,str(view))
    os.makedirs(render_out,exist_ok=True)
    for k in range(len(pred_verts)):
        img = np.zeros((2160, 3840,3), np.uint8) + 255
        # print(k)
        pred_all_verts = np.vstack((pred_verts[k].numpy(),obj_verts[k].numpy()))
        faces = np.vstack((hand_faces,obj_faces+778))
        pred_img, _ = vis_smpl(render_out, img, k, pred_all_verts, faces,camera_pose[:3,:3],camera_pose[:3,3],K)
        outname = os.path.join(render_out, '{:06d}.jpg'.format(k))
        cv2.imwrite(outname, pred_img)
        # break
    # video_name = 'seq' + seq + '_view' + str(view) + '.mp4' 
    # video_path = join(render_path,'render_videos')
    # os.makedirs(video_path,exist_ok=True)
    # video_path = join(video_path,video_name)
    # img2video(render_out,video_path)
    # # seq_index += 1
    # # if render_num == seq_index:
    # #     break
    # # break



