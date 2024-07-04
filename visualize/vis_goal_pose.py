import pyrender
import json
import os
from os.path import join
import pickle
from pysdf import SDF
import trimesh
import numpy as np
from tqdm import *
import cv2
import torch
from manotorch.manolayer import ManoLayer
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
def vis_smpl(out_path, image, nf, vertices, faces, camera_R, camera_T, K = np.array([1031.8450927734375, 0.0, 932.1596069335938, 0.0, 1022.4588623046875, 541.9437255859375, 0.0, 0.0, 1.0]).reshape((3,3))):
    # outname = os.path.join(out_path, '{:06d}.jpg'.format(nf))
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
    if nf=='gt':
        outname = os.path.join(out_path, nf+'.jpg')
    else: 
        outname = os.path.join(out_path, nf.zfill(4)+'.jpg')
    cv2.imwrite(outname, image_vis)
    return image_vis, depth 

calib_path = "/root/code/seqs/calibration_all.json"
obj_path = '/root/code/seqs/object/'
with open(calib_path) as f:
    calib_dome = json.load(f)
    f.close()
view = 3
camera_pose = np.vstack((np.asarray(calib_dome[str(view)]['RT']).reshape((3,4)), np.ones(4) ))
K = np.asarray(calib_dome[str(view)]['K']).reshape((3,3))
path = 'final_results/check_vis/1354'
seqs = sorted(os.listdir(path))
output_path = join(path,'goal_pose_render')
os.makedirs(output_path,exist_ok=True)
for seq in tqdm(seqs):
    seq_path = join(path,seq)
    res = np.load(join(seq_path),allow_pickle=True).item()
    hand_verts = res['hand_verts']
    hand_faces = res['hand_faces']
    obj_verts = res['obj_verts']
    obj_faces = res['obj_faces']
    print(hand_verts.shape, obj_verts.shape)
    verts = np.vstack((hand_verts,obj_verts))
    faces = np.vstack((hand_faces,obj_faces+778))
    img = np.zeros((2160, 3840,3), np.uint8) + 255
    # render_out = join(output_path,str(index).zfill(4)+'.jpg')
    index = seq.split('_')[0]
    print(output_path)
    pred_img, _ = vis_smpl(output_path, img, index , verts, faces,camera_pose[:3,:3],camera_pose[:3,3],K)