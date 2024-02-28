# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        # self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        self.dataset = self.model.dataset

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['only_text', 'only_spatial', 'both_text_spatial']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        # out = self.model(x, timesteps, y)
        if self.dataset=='gazehoi_stage0_flag2_lowfps_global':
            out,flag = self.model.forward_test(x, timesteps, y)
            out_uncond,_ = self.model.forward_test(x, timesteps, y_uncond)
            output = out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
            return output,flag
        elif self.dataset.startswith('gazehoi_stage0_flag'):
            out,flag = self.model(x, timesteps, y)
            out_uncond,_ = self.model(x, timesteps, y_uncond)
            output = out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
            return output,flag
        else:
            out = self.model(x, timesteps, y)
            out_uncond = self.model(x, timesteps, y_uncond)
            return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

