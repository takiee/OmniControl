# This code is based on https://github.com/GuyTevet/motion-diffusion-model
from torch.utils.data import DataLoader
# from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate,g2m_stage1_collate, g2m_stage2_collate,g2m_stage0_collate,g2m_pretrain_collate

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
    elif name == 'gazehoi_stage1':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage1
        return GazeHOIDataset_stage1
    elif name == 'gazehoi_stage2':
        from data_loaders.gazehoi.data.dataset import GazeHOIDataset_stage2
        return GazeHOIDataset_stage2
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
    if name == 'gazehoi_stage2':
        return g2m_stage2_collate
    if name == 'gazehoi_stage0':
        return g2m_stage0_collate
    if name == 'gazehoi_pretrain':
        return g2m_pretrain_collate
    else:
        return all_collate


def get_dataset(name, num_frames,hint_type, split='train', hml_mode='train', control_joint=0, density=100):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, control_joint=control_joint, density=density)
    elif name.startswith("gazehoi"):
        dataset = DATA(split=split,hint_type=hint_type)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, hint_type, split='train', hml_mode='train', control_joint=0, density=100):
    dataset = get_dataset(name, num_frames,hint_type, split, hml_mode, control_joint, density)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate,
    )

    return loader