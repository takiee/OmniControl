import torch
from pytorch3d.transforms import  quaternion_to_axis_angle,axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_rotation_6d,rotation_6d_to_matrix,matrix_to_axis_angle
from data_loaders.humanml.common.quaternion import qinv, qmul

def axis2rot6d(motion):
    # motion (bs, frames, 51)
    bs, nf, _ = motion.shape
    motion_rot_axis = motion[:,:,3:].reshape(bs,nf,-1,3)
    motion_rot_6d = matrix_to_rotation_6d(axis_angle_to_matrix(motion_rot_axis))
    motion_rot_6d = motion_rot_6d.reshape(bs, nf, -1)
    motion = torch.cat((motion[:,:,:3], motion_rot_6d),dim=-1)
    return motion

def rot6d2axis(motion):
    # motion (bs, frames, 99)
    bs, nf, _ = motion.shape
    motion_rot_6d = motion[:,:,3:].reshape(bs,nf,-1,6)
    motion_axis = matrix_to_axis_angle(rotation_6d_to_matrix(motion_rot_6d))
    motion_axis = motion_axis.reshape(bs, nf, -1)
    motion = torch.cat((motion[:,:,:3], motion_axis),dim=-1)
    return motion

def get_vel_axis(motion):
    # motion (bs, frames, 51)
    trans = motion[:,:,:3]
    rot_axis = motion[:,:,3:6]
    l_velocity = trans[:,1:] - trans[:,:-1] #bs, frames-1 ,3

    rot_qua = axis_angle_to_quaternion(rot_axis)
    # print(rot_qua[:,:-1].shape)
    r_velocity = qmul(rot_qua[:,1:], qinv(rot_qua[:,:-1]))
    r_velocity = quaternion_to_axis_angle(r_velocity)

    return l_velocity, r_velocity


def global2local_axis(motion):
    # 只对global RT做相对变换
    l_velocity, r_velocity = get_vel_axis(motion)
    # motion[:,1:,:3] = l_velocity
    # motion[:,1:,3:6] = r_velocity
    new_motion = motion.clone()
    new_motion[:, 1:, :3] = l_velocity
    new_motion[:, 1:, 3:6] = r_velocity
    return new_motion


def local2global_axis(motion):
    # 只对global RT做相对变换
    l_velocity = motion[:,:,:3]
    r_velocity = motion[:,:,3:6]

    trans = torch.cumsum(l_velocity,dim=1)

    r_velocity = axis_angle_to_quaternion(r_velocity)
    nf = l_velocity.shape[1]
    rot = r_velocity[:,0].unsqueeze(1)
    cur_r = rot
    for i in range(1,nf):
        rot_i = qmul(r_velocity[:,i].unsqueeze(1), cur_r)
        cur_r = rot_i
        # print(rot_i.shape)
        rot = torch.cat((rot,rot_i),dim=1)
    rot = quaternion_to_axis_angle(rot)
    # motion[:,:,:3] = trans
    # motion[:,:,3:6] = rot
    new_motion = motion.clone()
    new_motion[:,:,:3] = trans
    new_motion[:,:,3:6] = rot
    return new_motion

def get_vel_axis_by_matrix(motion):
    trans = motion[:,:,:3]
    rot_axis = motion[:,:,3:6]
    l_velocity = trans[:,1:] - trans[:,:-1] #bs, frames-1 ,3

    rot_matrix = axis_angle_to_matrix(rot_axis) # bs. frames, 3, 3
    matrix_0 = rot_matrix[:,:-1] # 需要求逆
    matrix_1 = rot_matrix[:,1:]
    matrix_0_inv = torch.einsum('...ij->...ji', [matrix_0])
    r_velocity = torch.einsum('fipn,fink->fipk',matrix_1,matrix_0_inv)
    r_velocity = matrix_to_axis_angle(r_velocity)
    return l_velocity, r_velocity

def obj_get_vel_matrix(motion):
    # bs, num_frames,3,4
    # 相对最后一帧的表示
    num_frames = motion.shape[1]
    # motion = motion.reshape(-1,num_frames,3,4)
    trans = motion[:,:,:3,3]
    rot_matrix = motion[:,:,:3,:3]
    l_velocity =  trans[:,:-1] - trans[:,1:] #bs, frames-1 ,3

    matrix_1 = rot_matrix[:,:-1] # 需要求逆
    matrix_0 = rot_matrix[:,1:]
    matrix_0_inv = torch.einsum('...ij->...ji', [matrix_0])
    r_velocity = torch.einsum('fipn,fink->fipk',matrix_1,matrix_0_inv)
    # r_velocity = matrix_to_rotation_6d(r_velocity)
    return l_velocity, r_velocity

def global2local_axis_by_matrix(motion):
    l_velocity, r_velocity = get_vel_axis_by_matrix(motion)
    # motion[:,1:,:3] = l_velocity
    # motion[:,1:,3:6] = r_velocity
    new_motion = motion.clone()
    new_motion[:, 1:, :3] = l_velocity
    new_motion[:, 1:, 3:6] = r_velocity
    return new_motion

def obj_global2local_matrix(motion):
    l_velocity, r_velocity = obj_get_vel_matrix(motion)
    # motion[:,1:,:3] = l_velocity
    # motion[:,1:,3:6] = r_velocity
    new_motion = motion.clone()
    new_motion[:, :-1, :3,3] = l_velocity
    new_motion[:, :-1, :3,:3] = r_velocity
    return new_motion

def obj_matrix2rot6d(motion):
    # bs, 
    R_matrix = motion[:,:,:3,:3]
    R_rot6d = matrix_to_rotation_6d(R_matrix)
    motion = torch.cat((motion[:,:,:3,3], R_rot6d),dim=-1)
    return motion

def obj_rot6d2matrix(motion):
    # bs, nf,36 -- bs,nf,4,3,4
    bs,nf ,_= motion.shape
    device = motion.device
    motion = motion.reshape(bs,nf,4,9)
    R_rot6d = motion[:,:,:,3:]
    R_matrix = rotation_6d_to_matrix(R_rot6d)

    new_motion = torch.zeros((bs,nf,4,3,4)).to(device)
    new_motion[:,:,:,:3,3] = motion[:,:,:,:3]
    new_motion[:,:,:,:3,:3] = R_matrix
    return new_motion


def local2global_axis_by_matrix(motion):
     # 只对global RT做相对变换
    l_velocity = motion[:,:,:3]
    r_velocity = motion[:,:,3:6]

    trans = torch.cumsum(l_velocity,dim=1)

    r_velocity = axis_angle_to_matrix(r_velocity)
    nf = l_velocity.shape[1]
    rot = r_velocity[:,0].unsqueeze(1) #bs,1,3,3
    cur_r = rot
    for i in range(1,nf):
        rot_i = torch.einsum('fipn,fink->fipk',r_velocity[:,i].unsqueeze(1),cur_r)
        cur_r = rot_i
        rot = torch.cat((rot,rot_i),dim=1)

    rot = matrix_to_axis_angle(rot)
    new_motion = motion.clone()
    new_motion[:,:,:3] = trans
    new_motion[:,:,3:6] = rot
    return new_motion


def local2global_rot6d_by_matrix(motion):
     # 只对global RT做相对变换
    l_velocity = motion[:,:,:3]
    r_velocity = motion[:,:,3:9]

    trans = torch.cumsum(l_velocity,dim=1)

    r_velocity = rotation_6d_to_matrix(r_velocity)
    nf = l_velocity.shape[1]
    rot = r_velocity[:,0].unsqueeze(1) #bs,1,3,3
    cur_r = rot
    for i in range(1,nf):
        rot_i = torch.einsum('fipn,fink->fipk',r_velocity[:,i].unsqueeze(1),cur_r)
        cur_r = rot_i
        rot = torch.cat((rot,rot_i),dim=1)

    rot = matrix_to_rotation_6d(rot)
    new_motion = motion.clone()
    new_motion[:,:,:3] = trans
    new_motion[:,:,3:9] = rot
    return new_motion

def local2global_rot6d_by_matrix_repair(motion,length):
     # 只对global RT做相对变换
    bs,nf,_ = motion.shape
    device = motion.device
    mask_length = nf-length
    mask = torch.arange(nf).to(device).unsqueeze(0) < mask_length.unsqueeze(-1)
    mask = mask.to(device)
    # print(mask.shape)
    motion[mask] = 0
    # print(motion)
    l_velocity = motion[:,:,:3]
    r_velocity = motion[:,:,3:9]

    trans = torch.cumsum(l_velocity,dim=1)

    r_velocity = rotation_6d_to_matrix(r_velocity)
    nf = l_velocity.shape[1]
    rot = r_velocity[:,0].unsqueeze(1) #bs,1,3,3
    cur_r = rot
    for i in range(1,nf):
        rot_i = torch.einsum('fipn,fink->fipk',r_velocity[:,i].unsqueeze(1),cur_r)
        cur_r = rot_i
        rot = torch.cat((rot,rot_i),dim=1)

    rot = matrix_to_rotation_6d(rot)
    new_motion = motion.clone()
    new_motion[:,:,:3] = trans
    new_motion[:,:,3:9] = rot
    return new_motion

def local2global_axis_by_matrix(motion):
    # 只对global RT做相对变换
    bs,nf ,_= motion.shape # bs,nf,D
    device = motion.device
    l_velocity = motion[:,:,:3]
    r_velocity = motion[:,:,3:9]

    trans = torch.cumsum(l_velocity,dim=1)

    r_velocity = rotation_6d_to_matrix(r_velocity)
    rot = r_velocity[:,0].unsqueeze(1) #bs,1,3,3
    cur_r = rot
    for i in range(1,nf):
        rot_i = torch.einsum('fipn,fink->fipk',r_velocity[:,i].unsqueeze(1),cur_r)
        cur_r = rot_i
        rot = torch.cat((rot,rot_i),dim=1)
    # print(rot.shape)
    rot = matrix_to_axis_angle(rot)
    rot_local = matrix_to_axis_angle(rotation_6d_to_matrix(motion[:,:,9:].reshape(bs,nf,15,6)))

    # print(rot.shape)
    
    new_motion = torch.zeros((bs,nf,51)).to(device)
    new_motion[:,:,:3] = trans
    new_motion[:,:,3:6] = rot.reshape(bs,nf,-1)
    new_motion[:,:,6:] = rot_local.reshape(bs,nf,-1)
    return new_motion

def local2global_axis_by_matrix_repair(motion):
    # 只对global RT做相对变换
    bs,nf ,_= motion.shape # bs,nf,D
    device = motion.device
    # motion[]
    
    l_velocity = motion[:,:,:3]
    r_velocity = motion[:,:,3:9]

    trans = torch.cumsum(l_velocity,dim=1)

    r_velocity = rotation_6d_to_matrix(r_velocity)
    rot = r_velocity[:,0].unsqueeze(1) #bs,1,3,3
    cur_r = rot
    for i in range(1,nf):
        rot_i = torch.einsum('fipn,fink->fipk',r_velocity[:,i].unsqueeze(1),cur_r)
        cur_r = rot_i
        rot = torch.cat((rot,rot_i),dim=1)
    # print(rot.shape)
    rot = matrix_to_axis_angle(rot)
    rot_local = matrix_to_axis_angle(rotation_6d_to_matrix(motion[:,:,9:].reshape(bs,nf,15,6)))

    # print(rot.shape)
    
    new_motion = torch.zeros((bs,nf,51)).to(device)
    new_motion[:,:,:3] = trans
    new_motion[:,:,3:6] = rot.reshape(bs,nf,-1)
    new_motion[:,:,6:] = rot_local.reshape(bs,nf,-1)
    return new_motion

def obj_local2global_rot6d_by_matrix(motion):
     # 只对global RT做相对变换
    #  反过来 最后一帧是参考
    new_motion = motion.clone()
    bs,nf ,_= motion.shape
    motion = motion.reshape(bs,nf,4,9)
    new_motion = motion.clone()
    for k in range(4):
        l_velocity = torch.flip(motion[:,:,k,:3],dims=[1])
        r_velocity = motion[:,:,k,3:9]

        trans = torch.flip(torch.cumsum(l_velocity,dim=1),dims=[1])

        r_velocity = rotation_6d_to_matrix(r_velocity)
        nf = l_velocity.shape[1]
        rot = r_velocity[:,-1].unsqueeze(1) #bs,1,3,3
        cur_r = rot
        for i in range(nf-2,-1,-1):
            # print(i)
            rot_i = torch.einsum('fipn,fink->fipk',r_velocity[:,i].unsqueeze(1),cur_r)
            cur_r = rot_i
            rot = torch.cat((rot_i,rot),dim=1)

        rot = matrix_to_rotation_6d(rot)
        
        new_motion[:,:,k,:3] = trans
        new_motion[:,:,k,3:9] = rot
    new_motion = new_motion.reshape(bs,nf,-1)
    return new_motion

def obj_local2global_matrix(motion):
     # 只对global RT做相对变换
    #  反过来 最后一帧是参考
    bs,nf ,_= motion.shape
    device = motion.device
    new_motion = torch.zeros((bs,nf,4,3,4)).to(device)
    motion = motion.reshape(bs,nf,4,9)
    # new_motion = motion.clone()
    for k in range(4):
        l_velocity = torch.flip(motion[:,:,k,:3],dims=[1])
        r_velocity = motion[:,:,k,3:9]

        trans = torch.flip(torch.cumsum(l_velocity,dim=1),dims=[1])

        r_velocity = rotation_6d_to_matrix(r_velocity)
        nf = l_velocity.shape[1]
        rot = r_velocity[:,-1].unsqueeze(1) #bs,1,3,3
        cur_r = rot
        for i in range(nf-2,-1,-1):
            # print(i)
            rot_i = torch.einsum('fipn,fink->fipk',r_velocity[:,i].unsqueeze(1),cur_r)
            cur_r = rot_i
            rot = torch.cat((rot_i,rot),dim=1)

        # rot = matrix_to_rotation_6d(rot)
        # print(trans.shape, rot.shape)
        # print( new_motion[:,:,k,:3,3].shape)
        new_motion[:,:,k,:3,3] = trans
        new_motion[:,:,k,:3,:3] = rot
    # new_motion = new_motion.reshape(bs,nf,-1)
    return new_motion