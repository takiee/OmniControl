# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import torch

# mask后部
def lengths_to_mask_after(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

# mask前部
def lengths_to_mask_before(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask



def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_stage1(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask_before(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = [b['obj_points']for b in notnone_batches]
        cond['y'].update({'obj_points': torch.as_tensor(obj_points)})
    
    if 'hint' in notnone_batches[0] and notnone_batches[0]['hint'] is not None:
        hint = [b['hint']for b in notnone_batches]
        # cond['y'].update({'hint': hint})
        cond['y'].update({'hint': torch.as_tensor(hint)})
    
    if 'goal_obj_pose' in notnone_batches[0] and notnone_batches[0]['goal_obj_pose'] is not None:
        goal_obj_pose = [b['goal_obj_pose']for b in notnone_batches]
        cond['y'].update({'goal_obj_pose': torch.as_tensor(goal_obj_pose)})
    
    seqbatch = [b['seq_name'] for b in notnone_batches] 
    cond['y']['seq_name']= seqbatch
    
    return motion, cond

def collate_stage2(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask_after(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = [b['obj_points']for b in notnone_batches]
        cond['y'].update({'obj_points': torch.as_tensor(obj_points).float()})
    
    if 'hint' in notnone_batches[0] and notnone_batches[0]['hint'] is not None:
        hint = [b['hint']for b in notnone_batches]
        # cond['y'].update({'hint': hint})
        cond['y'].update({'hint': torch.as_tensor(hint)})
    
    if 'obj_pose' in notnone_batches[0] and notnone_batches[0]['obj_pose'] is not None:
        goal_obj_pose = [b['obj_pose']for b in notnone_batches]
        cond['y'].update({'obj_pose': torch.as_tensor(goal_obj_pose).float()})

    if 'hand_shape' in notnone_batches[0] and notnone_batches[0]['hand_shape'] is not None:
        hand_shape = [b['hand_shape']for b in notnone_batches]
        cond['y'].update({'hand_shape': torch.as_tensor(hand_shape).float()})
    
    seqbatch = [b['seq_name'] for b in notnone_batches] 
    cond['y']['seq_name']= seqbatch
    
    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'hint': b[-1],
    } for b in batch]
    return collate(adapted_batch)

def g2m_stage1_collate(batch):
    # print(b[0])
    adapted_batch = [{
        'inp': torch.tensor(b[0].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'hint': b[1],
        'goal_obj_pose':b[2],
        'lengths': b[-2],
        'obj_points':b[3],
        'seq_name':b[-1]
    } for b in batch]
    return collate_stage1(adapted_batch)

def g2m_stage2_collate(batch):
    # print(b[0])
    adapted_batch = [{
        'inp': torch.tensor(b[0].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'hint': b[1],
        'obj_pose':b[2],
        'lengths': b[-3],
        'obj_points':b[3],
        'seq_name':b[-2],
        'hand_shape':b[-1]
    } for b in batch]
    return collate_stage2(adapted_batch)

def g2m_stage0_collate(batch):
    # print(b[0])
    adapted_batch = [{
        'inp': torch.tensor(b[0]).permute(1,0,2).contiguous().reshape(-1,36).T.float().unsqueeze(1), # [4,nf,9] -> [nf,4,9]
        'hint': b[1],
        'gaze':b[2],
        'lengths': b[-2],
        'obj_points':b[3],
        'seq_name':b[-1]
    } for b in batch]
    return collate_stage0(adapted_batch)

def g2m_pretrain_collate(batch):
    # print(b[0])
    adapted_batch = [{
        'inp': torch.tensor(b[0].T).float().unsqueeze(1), # [4,nf,9] -> [nf,4,9]
        'obj_pose':b[0],
        'gaze':b[1],
        'lengths': b[-2],
        'obj_points':b[2],
        'seq_name':b[-1]
    } for b in batch]
    return collate_stage0(adapted_batch)

def collate_stage0(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask_after(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}


    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = [b['obj_points']for b in notnone_batches]
        cond['y'].update({'obj_points': torch.as_tensor(obj_points).float()})
    
    if 'hint' in notnone_batches[0] and notnone_batches[0]['hint'] is not None:
        hint = [b['hint']for b in notnone_batches]
        # cond['y'].update({'hint': hint})
        cond['y'].update({'hint': torch.as_tensor(hint).float()})
    
    if 'obj_pose' in notnone_batches[0] and notnone_batches[0]['obj_pose'] is not None:
        goal_obj_pose = [b['obj_pose']for b in notnone_batches]
        cond['y'].update({'obj_pose': torch.as_tensor(goal_obj_pose).float()})

    if 'gaze' in notnone_batches[0] and notnone_batches[0]['gaze'] is not None:
        gaze = [b['gaze']for b in notnone_batches]
        cond['y'].update({'gaze': torch.as_tensor(gaze).float()})

    
    seqbatch = [b['seq_name'] for b in notnone_batches] 
    cond['y']['seq_name']= seqbatch
    
    return motion, cond