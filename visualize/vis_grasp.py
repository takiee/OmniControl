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
    from renderer import Renderer
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
path = '/root/code/OmniControl/final_results/contactgen_500_2000_0306/'
gt_path = '/root/code/seqs/0303_data/'
seqs = sorted(os.listdir(path))
manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
gt_faces = manolayer.th_faces

for seq in tqdm(seqs):
    # seq = '0067'
    # print(seq)
    seq_path = join(path,seq)
    offset = np.load(join(seq_path,'offset.npy'))
    # valid_hand = list(np.load(join(seq_path,'valid_seq.npy')))
    # meta_path = join(seq_path,'meta.pkl')
    output_path = join(seq_path,'render_view3')
    os.makedirs(output_path,exist_ok=True)
    # with open(meta_path,'rb')as f:
    #     meta = pickle.load(f)
    # obj_name = meta['obj_name']
    mesh_path = join(seq_path,'mesh_res')
    # obj_path = join(mesh_path, obj_name+'.obj')
    # obj_mesh = trimesh.load(obj_path)
    # obj_verts = obj_mesh.vertices + offset
    # obj_faces = obj_mesh.faces

    valid_hand = ['6']
    for index in valid_hand:
        # index = '0'
        # print("#################")
        hand_path = join(mesh_path,f'filter_grasp_{index}.obj')
        hand_mesh = trimesh.load(hand_path)
        hand_verts = hand_mesh.vertices + offset
        hand_faces = hand_mesh.faces
        # verts = np.vstack((hand_verts,obj_verts))
        # faces = np.vstack((hand_faces,obj_faces+778))
        img = np.zeros((2160, 3840,3), np.uint8) + 255
        # render_out = join(output_path,str(index).zfill(4)+'.jpg')
        pred_img, _ = vis_smpl(output_path, img, index, hand_verts, hand_faces,camera_pose[:3,:3],camera_pose[:3,3],K)
     #gt
    # with open(join(gt_path,seq,'meta.pkl'),'rb')as f:
    #     gt_meta = pickle.load(f)
    # goal_index = gt_meta['goal_index']
    # gt_hand_path = join(gt_path,seq,'mano/poses_right.npy')
    # hand_params = torch.tensor(np.load(gt_hand_path))[goal_index].unsqueeze(0)
    # hand_trans = hand_params[:,:3]
    # hand_rot = hand_params[:,3:6]
    # hand_theta = hand_params[:,3:51]
    # mano_beta = hand_params[:,51:]
    # gt_output = manolayer(hand_theta, mano_beta)
    # gt_verts = gt_output.verts - gt_output.joints[:, 0].unsqueeze(1) + hand_trans.unsqueeze(1)
    # gt_verts = gt_verts[0]
    # verts = np.vstack((gt_verts,obj_verts))
    # faces = np.vstack((gt_faces,obj_faces+778))
    # img = np.zeros((2160, 3840,3), np.uint8) + 255
    # # render_out = join(output_path,str(index).zfill(4)+'.jpg')
    # gt_img, _ = vis_smpl(output_path, img, 'gt', verts, faces,camera_pose[:3,:3],camera_pose[:3,3],K)
    # # break