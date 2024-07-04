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
from manotorch.manolayer import ManoLayer as manotorch
from manopth.manolayer import ManoLayer
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

import os
import pickle
path = '/root/code/CAMS/data/mano_assets/mano/MANO_RIGHT.pkl'
# os.listdir(path)
with open(path,'rb') as f:
    mano=pickle.load(f,encoding='latin1')
print(mano.keys())
mano_mean = mano['hands_mean']


path = 'final_results/contactgen_500_2000'
hand_path = join(path,'0009/results/67/new_hand_params.npy')
hand_params = torch.tensor(np.load(hand_path).reshape(1,-1))
offset = torch.tensor(np.load(join(path,'0009/offset.npy')))

manotorch = manotorch(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
hand_faces = manotorch.th_faces

print(mano_mean,mano_mean.shape)
calib_path = "/root/code/seqs/calibration_all.json"
with open(calib_path) as f:
    calib_dome = json.load(f)
    f.close()
view = '3'
camera_pose = np.vstack((np.asarray(calib_dome[str(3)]['RT']).reshape((3,4)), np.ones(4) ))
K = np.asarray(calib_dome[str(3)]['K']).reshape((3,3))

# hand_trans = hand_params[:,:3] 
hand_trans = hand_params[:,:3] + offset
hand_rot = hand_params[:,3:6]
hand_params[:,6:51] = hand_params[:,6:51]
# hand_params[:,6:51] = hand_params[:,6:51] 
hand_theta = hand_params[:,3:51] 
mano_beta = hand_params[:,51:]
"""加上mean pose"""

gt_output = manotorch(hand_theta, mano_beta)
# gt_verts = gt_output.verts - gt_output.joints[:, 0].unsqueeze(1) 
gt_verts = gt_output.verts + hand_trans.unsqueeze(1)
# gt_verts = gt_output.verts - gt_output.joints[:, 0].unsqueeze(1) + hand_trans.unsqueeze(1)
render_path = hand_path.replace('.npy',f'_render{str(view)}')
print(render_path)
render_out = render_path
os.makedirs(render_out,exist_ok=True)
img = np.zeros((2160, 3840,3), np.uint8) + 255

gt_img, _ = vis_smpl(render_out, img, 0, gt_verts[0].numpy(), hand_faces,camera_pose[:3,:3],camera_pose[:3,3],K)


outname = os.path.join(render_out, '{:06d}.jpg'.format(0))
cv2.imwrite(outname, gt_img)

