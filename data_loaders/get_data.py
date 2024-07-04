# This code is based on https://github.com/GuyTevet/motion-diffusion-model
from torch.utils.data import DataLoader
# from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import *

def get_dataset_class(name):
    print(name)
    if name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == 'gazehoi_stage0':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage0
        return GazeHOIDataset_stage0
    elif name == 'gazehoi_stage0_flag' or name == 'gazehoi_stage0_flag2':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage0_flag
        return GazeHOIDataset_stage0_flag
    elif name == 'gazehoi_stage0_flag2_lowfps':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage0_flag_lowfps
        return GazeHOIDataset_stage0_flag_lowfps
    elif name == 'gazehoi_stage0_flag2_lowfps_global':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage0_flag_lowfps_global
        return GazeHOIDataset_stage0_flag_lowfps_global
    elif name == 'gazehoi_stage0_1obj' or name == 'gazehoi_stage0_point' or name == 'gazehoi_stage0_noatt':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage0_1obj
        return GazeHOIDataset_stage0_1obj
    elif name == 'gazehoi_stage0_norm':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage0_norm
        return GazeHOIDataset_stage0_norm
    elif name == 'gazehoi_stage1':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage1
        return GazeHOIDataset_stage1
    elif name == 'gazehoi_stage1_new':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage1_new
        return GazeHOIDataset_stage1_new
    elif name == 'gazehoi_stage1_repair':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage1_repair
        return GazeHOIDataset_stage1_repair
    elif name == 'gazehoi_stage1_simple':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage1_simple
        return GazeHOIDataset_stage1_simple
    elif name == 'gazehoi_stage2':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage2
        return GazeHOIDataset_stage2
    elif name == 'gazehoi_g2ho':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_g2ho
        return GazeHOIDataset_g2ho
    elif name == 'gazehoi_o2h':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_o2h
        return GazeHOIDataset_o2h
    elif name == 'gazehoi_o2h_mid':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_o2h_mid
        print("GazeHOIDataset_o2h_mid")
        return GazeHOIDataset_o2h_mid
    elif name == 'gazehoi_pretrain':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_pretrain
        return GazeHOIDataset_pretrain
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    if name == 'gazehoi_stage1':
        return g2m_stage1_collate
    if name == 'gazehoi_stage1_new':
        return g2m_stage1_new_collate
    if name == 'gazehoi_stage1_repair':
        return g2m_stage1_repair_collate
    if name == 'gazehoi_stage1_simple':
        return g2m_stage1_simple_collate
    if name == 'gazehoi_stage2':
        return g2m_stage2_collate
    if name == 'gazehoi_stage0':
        return g2m_stage0_collate
    if name == 'gazehoi_stage0_flag' or name == 'gazehoi_stage0_flag2' or name == 'gazehoi_stage0_flag2_lowfps' or name == 'gazehoi_stage0_flag2_lowfps_global' :
        return g2m_stage0_flag_collate
    if name == 'gazehoi_stage0_1obj' or name == 'gazehoi_stage0_norm' or name == 'gazehoi_stage0_point' or name == 'gazehoi_stage0_noatt':
        return g2m_stage0_1obj_collate
    if name == 'gazehoi_pretrain':
        return g2m_pretrain_collate
    if name == 'gazehoi_g2ho':
        return g2ho_collate
    if name == 'gazehoi_o2h':
        return o2h_collate
    if name == 'gazehoi_o2h_mid':
        return o2h_mid_collate
    else:
        return all_collate


def get_dataset(name, num_frames,hint_type, split='train', hml_mode='train', control_joint=0, density=100):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, control_joint=control_joint, density=density)
    elif name.startswith("gazehoi"):
        print(split)
        dataset = DATA(split=split,hint_type=hint_type)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, hint_type, split='train', hml_mode='train', control_joint=0, density=100):
    dataset = get_dataset(name, num_frames,hint_type, split, hml_mode, control_joint, density)
    # print(len(dataset))
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, drop_last=True, collate_fn=collate,
    )

    return loader