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
def vis_smpl_color(out_path, image, nf, vertices, faces, camera_R, camera_T, K,index=778,face_index=1538):
    outname = os.path.join(out_path, '{:06d}.jpg'.format(nf))
    # if os.path.exists(outname): return
    render_data = {}
    assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
    # render_data = {"vertices": vertices, "faces": faces, "vid":0, "name": "human_{}_0".format(nf)}
    render_data_hand = {"vertices": vertices[:index], "faces": faces[:face_index], "vid":0, "name": "human_{}_0_h".format(nf),"col_code":(60,100,160)}
    render_data_obj = {"vertices": vertices[index:], "faces": faces[face_index:]-778, "vid":0, "name": "human_{}_0_o".format(nf),"col_code":(180, 140, 40)}
    #(112,128,144)
    camera = {"K": K,
        "R": camera_R,
        "T":camera_T}
    from renderer import Renderer
    render = Renderer(height=3840, width=2160, faces=None)
    image_vis, depth = render.render([render_data_hand,render_data_obj], camera, image, add_back=True)
    # print(depth)
    # outname = os.path.join(out_path, '{:06d}.jpg'.format(nf))
    # cv2.imwrite(outname, image_vis)
    return image_vis, depth  

def img2video(image_path, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    height = 2160
    weight = 3840
    fps = 10
    videowriter = cv2.VideoWriter(video_path, fourcc, fps, (weight, height))  # 创建一个写入视频对象

    imgs = sorted(os.listdir(image_path))
    for img in tqdm(imgs):
        frame = cv2.imread(join(image_path, img))
        videowriter.write(frame)
    videowriter.release()

datapath = '/root/code/seqs/0303_data/'
res_path = 'final_results/check_vis/gt'
manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right')
hand_faces = manolayer.th_faces
print(hand_faces.shape)

calib_path = "/root/code/seqs/calibration_all.json"
obj_path = '/root/code/seqs/object/'
with open(calib_path) as f:
    calib_dome = json.load(f)
    f.close()
view = 1
camera_pose = np.vstack((np.asarray(calib_dome[str(view)]['RT']).reshape((3,4)), np.ones(4) ))
K = np.asarray(calib_dome[str(view)]['K']).reshape((3,3))

# select_seqs = ['0155','0327','0641','1016','0009']
select_seqs = ['1327']
# path = 'save/0304_g2ho/samples_0304_g2ho_000150000_seed10_0307_test/results.npy'
# path = 'save/0303_o2h/samples_0303_o2h_000150000_seed10_0307_test/results.npy'
# seqs = ['0009','0071','0155','0327','0553','0641','0664','1016','1327','1354']
# res = np.load(path,allow_pickle=True).item()
# all_hand_motions = torch.tensor(res['all_hand_motions']).cpu().float() # obj_motion (bs,nf,4,3,4)
# all_obj_motions = torch.tensor(res['all_obj_motions']).cpu().float() # obj_motion (bs,nf,4,3,4)
# seqs = res['seqs']
for seq in select_seqs:
    # break
    # i = seqs.index(seq)
    # print(i)
    # print(seqs[i])
    # continue
    # pred_path = join(res_path,seq,f'{seq}.npy')
    # res = np.load((pred_path),allow_pickle=True).item()
    if seq == '0009':
        seq_len = 345
    if seq == '0155':
        seq_len = 150
    if seq == '0327':
        seq_len = 155
    if seq == '0641':
        seq_len = 160
    if seq == '1016':
        seq_len = 163
    if seq == '1313':
        seq_len = 80
    if seq == '1327':
        seq_len = 193

    seq_path = join(datapath,seq)
    pred_motion =  torch.tensor(np.load(join(seq_path, 'mano/poses_right.npy'))).float()[:seq_len]
    # pred_motion = all_hand_motions[i][:seq_len]
    # obj_pose = all_obj_motions[i][:seq_len]
    # print(pred_motion.shape,obj_pose.shape)

    

    meta_path = join(seq_path,'meta.pkl')
    with open(meta_path,'rb')as f:
            meta = pickle.load(f)
        
    active_obj = meta['active_obj']
    obj_pose = torch.tensor(np.load(join(seq_path,active_obj+'_pose_trans.npy'))).reshape(-1,3,4)[:seq_len].float()
    obj_name_list = meta['obj_name_list']
    print(obj_name_list)

    obj_mesh_path = join(obj_path,active_obj,'simplified_scan_processed.obj')
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_verts = obj_mesh.vertices
    obj_faces = obj_mesh.faces
    
    
    # seq_len = obj_pose.shape[0]
    obj_verts = torch.tensor(obj_verts).unsqueeze(0).repeat(seq_len,1,1).float()
    obj_R = obj_pose[:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    obj_T = obj_pose[:,:3,3].unsqueeze(1)
    obj_verts = (torch.einsum('fpn,fnk->fpk',obj_verts,obj_R) + obj_T).numpy()

    for obj in obj_name_list:
        if obj == active_obj:
            continue
        obj_mesh_path = join(obj_path,obj,'simplified_scan_processed.obj')
        obj_mesh = trimesh.load(obj_mesh_path)
        obj_verts_ = obj_mesh.vertices
        obj_faces_ = obj_mesh.faces
        
        obj_verts_ = torch.tensor(obj_verts_).unsqueeze(0).repeat(seq_len,1,1).float()
        obj_pose_ = torch.tensor(np.load(join(seq_path,obj+'_pose_trans.npy'))).reshape(-1,3,4)[0].repeat(seq_len,1,1).float()
        obj_R = obj_pose_[:,:3,:3]
        obj_R = torch.einsum('...ij->...ji', [obj_R])
        obj_T = obj_pose_[:,:3,3].unsqueeze(1)
        print(obj_verts_.shape,obj_pose_.shape)
        obj_verts_ = (torch.einsum('fpn,fnk->fpk',obj_verts_,obj_R) + obj_T).numpy()
        obj_faces = np.vstack((obj_faces,obj_faces_+obj_verts.shape[1]))
        obj_verts = np.concatenate((obj_verts,obj_verts_),axis=1)
    print(obj_verts.shape)
    
    pred_trans = pred_motion[:,:3]
    pred_theta = pred_motion[:,3:51]
    pred_rot = pred_motion[:,3:6]
    length = pred_motion.shape[0]

    hand_params = torch.tensor(np.load(join(seq_path,'mano/poses_right.npy')))
    mano_beta = hand_params[0,51:].unsqueeze(0).repeat(seq_len,1)

    pred_output = manolayer(pred_theta, mano_beta)
    pred_verts = pred_output.verts - pred_output.joints[:, 0].unsqueeze(1) + pred_trans.unsqueeze(1)


    render_out = join(res_path,seq,str(view))
    os.makedirs(render_out,exist_ok=True)
    for k in tqdm(range(len(pred_verts))):
        img = np.zeros((2160, 3840,3), np.uint8) + 255
        # print(k)
        pred_all_verts = np.vstack((pred_verts[k].numpy(),obj_verts[k]))
        faces = np.vstack((hand_faces,obj_faces+778))
        pred_img, _ = vis_smpl_color(render_out, img, k, pred_all_verts, faces,camera_pose[:3,:3],camera_pose[:3,3],K)
        outname = os.path.join(render_out, '{:06d}.jpg'.format(k))
        # print(outname)
        cv2.imwrite(outname, pred_img)
        # break
    video_name = 'seq' + seq + '_view' + str(view) + '.mp4' 
    video_path = join(render_out,video_name)
    img2video(render_out,video_path)