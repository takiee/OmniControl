# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import pickle
import numpy as np
import torch
import torch as th
import torch.nn as nn
from copy import deepcopy
from diffusion.nn import mean_flat, sum_flat
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from os.path import join as pjoin
from utils.data_util import *
from manotorch.manolayer import ManoLayer
manolayer = ManoLayer(
    mano_assets_root='/root/code/CAMS/data/mano_assets/mano',
    side='right'
).cuda()

def convert_to_obj_frame(points, obj_pose):
    # points(bs, frames,5,3)
    # obj_pose (bs, frames,3,4)

    obj_T = obj_pose[:,:,:3,3].unsqueeze(-2)
    points = points - obj_T
    points = torch.einsum('...ij->...ji', [points])
    obj_R = obj_pose[:,:,:3,:3]
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    new_points = torch.einsum('bfpn,bfnk->bfpk',obj_R,points)
    new_points = torch.einsum('...ij->...ji', [new_points])
    return new_points

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_rcxyz=0.,
        lambda_vel=0.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_fc=0.,
        dataset='humanml',
    ):
        self.dataset = dataset
        self.log_guide_dict = {}
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

        if dataset == 'humanml':
            spatial_norm_path = './dataset/humanml_spatial_norm'
            data_root = './dataset/HumanML3D'
        elif dataset == 'kit':
            spatial_norm_path = './dataset/kit_spatial_norm'
            data_root = './dataset/KIT-ML'
        elif dataset.startswith('gazehoi_stage1') or dataset == 'gazehoi_stage2':
            # self.raw_mean = torch.from_numpy(np.load('/root/code/OmniControl/dataset/gazehoi_mean.npy')) # 原始关节点的mean,std
            # self.raw_std = torch.from_numpy(np.load('/root/code/OmniControl/dataset/gazehoi_std.npy'))
            # self.mean = torch.from_numpy(np.load('/root/code/OmniControl/dataset/gazehoi_mean.npy')).float() # humanl3d转换后的mean,std
            # self.std = torch.from_numpy(np.load('/root/code/OmniControl/dataset/gazehoi_std.npy')).float()
            self.global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_motion_6d_mean.npy'))
            self.global_std = torch.from_numpy(np.load('dataset/gazehoi_global_motion_6d_std.npy'))
            self.local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_motion_6d_mean.npy'))
            self.local_std = torch.from_numpy(np.load('dataset/gazehoi_local_motion_6d_std.npy'))
        elif dataset == 'gazehoi_stage0' or dataset == 'gazehoi_stage0_flag' or dataset == 'gazehoi_stage0_flag2'or dataset == 'gazehoi_stage0_flag2_lowfps' or dataset == 'gazehoi_stage0_flag2_lowfps_global':
            self.obj_global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            self.obj_global_std = torch.from_numpy(np.load('dataset/gazehoi_global_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            self.obj_local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_obj_mean.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
            self.obj_local_std = torch.from_numpy(np.load('dataset/gazehoi_local_obj_std.npy')).reshape(1,-1).repeat(4,1).reshape(1,-1)
        elif  dataset == 'gazehoi_stage0_1obj' or dataset == 'gazehoi_stage0_norm' or dataset == 'gazehoi_stage0_point':
            self.obj_global_mean = torch.from_numpy(np.load('dataset/gazehoi_global_obj_mean.npy'))
            self.obj_global_std = torch.from_numpy(np.load('dataset/gazehoi_global_obj_std.npy'))
            self.obj_local_mean = torch.from_numpy(np.load('dataset/gazehoi_local_obj_mean.npy'))
            self.obj_local_std = torch.from_numpy(np.load('dataset/gazehoi_local_obj_std.npy'))
        else:
            raise NotImplementedError('Dataset not recognized!!')
        
        self.manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right').cuda()
        # self.manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side='right',cuda=True)
        
        # self.raw_mean = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))) # 原始关节点的mean,std
        # self.raw_std = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Std_raw.npy')))
        # self.mean = torch.from_numpy(np.load(pjoin(data_root, 'Mean.npy'))).float() # humanl3d转换后的mean,std
        # self.std = torch.from_numpy(np.load(pjoin(data_root, 'Std.npy'))).float()

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # print(x.shape)
        if self.dataset.startswith('gazehoi_stage0_flag'):
            model_output,flag = model(x, self._scale_timesteps(t), **model_kwargs)
            # print(model_output.shape,flag.shape)
        else:
            model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
            inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
            model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            # print('model_output', model_output.shape, model_output)
            # print('inpainting_mask', inpainting_mask.shape, inpainting_mask[0,0,0,:])
            # print('inpainted_motion', inpainted_motion.shape, inpainted_motion)

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        #################################################################################
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        #################################################################################
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        if self.dataset.startswith('gazehoi_stage0_flag'):
            return {
                "mean": model_mean,
                "variance": model_variance,
                "log_variance": model_log_variance,
                "pred_xstart": pred_xstart,
                "flag":flag,
            }
        else:
            return {
                "mean": model_mean,
                "variance": model_variance,
                "log_variance": model_log_variance,
                "pred_xstart": pred_xstart,
            }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def gradients(self, out, x_prev, hint, mask_hint, length):
        mask_hint_new = mask_hint.clone()
        mask_hint_new = torch.where(mask_hint_new, 1.0, 0.0).to(torch.float32)
        # mask_hint_new[:,55:,:] = 1.0
        # print(f'mask_hint_new:\n {mask_hint_new[0]}')
        mask_hint_new = mask_hint_new.repeat(1, 1, 99)

        # with torch.enable_grad():
        x_0_pred = out['pred_xstart']
        # x_t.requires_grad_(True)

        x_ = x_0_pred.permute(0, 3, 2, 1).contiguous()
        x_ = x_.squeeze(2)
        x_ = x_ * self.local_std + self.local_mean
        x_ = local2global_rot6d_by_matrix(x_)
        # x_ = local2global_rot6d_by_matrix(x_)
        bs,nf,_ = x_.shape

        # print(f'mask_hint:{mask_hint.shape}')
        # print(f'hint:{hint.shape}')
        # print(x_.reshape(bs,nf,-1).shape)
        # loss = torch.linalg.norm((x_.reshape(bs,nf,-1) - hint) * mask_hint, dim=-1)
        difference = (x_.reshape(bs, nf, -1) - hint) * mask_hint_new
        # print('difference: ', difference.shape)
        # print('x_prev: ', x_prev.shape)
        print('difference 0-2: ', difference[:,:,3].pow(2).mean())
        print('difference 3-9: ', difference[:,:,3:9].pow(2).mean())
        print('difference 9-: ', difference[:,:,9:].pow(2).mean())

        loss = torch.sum(torch.norm((x_.reshape(bs, nf, -1) - hint) * mask_hint, dim=-1), dim=-1)
        grad = torch.autograd.grad([loss.sum()], [x_prev])[0]

        print(f"Loss: ", torch.mean(loss))

        # loss = torch.linalg.norm(difference.reshape(bs,-1), dim=[1]).mean()
        # grad = torch.autograd.grad([loss], [x_prev])[0]

        # norm_grad = grad * (1. - mask_hint.reshape(1, -1, 1, 60))

        # print('norm_grad: ', norm_grad)

        # print(grad[:])
        # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
        # grad[..., 0] = 0
        # x_t.detach()
        return loss, grad

    def gradients_stage0(self, x, hint, mask_hint, joint_ids=None):
        # print('gradients_stage0')
        with torch.enable_grad():
            x.requires_grad_(True)

            x_ = x.permute(0, 3, 2, 1).contiguous()
            x_ = x_.squeeze(2)
            x_ = x_ * self.obj_local_std + self.obj_local_mean
            # x_ = x_ * self.obj_global_std + self.obj_local_mean
            x_ = obj_local2global_rot6d_by_matrix(x_)
            bs,nf,_ = x_.shape

            loss = torch.norm((x_.reshape(bs,nf,-1) - hint) * mask_hint, dim=-1)
            # torch.autograd.set_detect_anomaly(True)
            grad = torch.autograd.grad([loss.sum()], [x])[0]
            # print(grad[:])
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            # grad[..., 0] = 0
            x.detach()
        return loss, grad

    def gradients_stage0_global(self, out, x, hint, mask_hint, joint_ids=None):
        # print('gradients_stage0')
        # with torch.enable_grad():
        #     x.requires_grad_(True)
        x_0_pred = out['pred_xstart']

        x_ = x_0_pred.permute(0, 3, 2, 1).contiguous()
        x_ = x_.squeeze(2)
        x_ = x_ * self.obj_global_std + self.obj_global_mean
        # x_ = x_ * self.obj_global_std + self.obj_local_mean
        # x_ = obj_local2global_rot6d_by_matrix(x_)
        bs,nf,_ = x_.shape

        loss = torch.norm((x_.reshape(bs,nf,-1) - hint) * mask_hint, dim=-1)
        # torch.autograd.set_detect_anomaly(True)
        grad = torch.autograd.grad([loss.sum()], [x])[0]
        # print(grad[:])
        # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
        # grad[..., 0] = 0
        # x.detach()
        return loss, grad
    def gradients_stage0_1(self, x, hint, mask_hint, joint_ids=None):
        # print('gradients_stage0')
        with torch.enable_grad():
            x.requires_grad_(True)

            x_ = x.permute(0, 3, 2, 1).contiguous()
            x_ = x_.squeeze(2)
            x_ = x_ * self.obj_local_std + self.obj_local_mean
            bs,nf,_ = x_.shape
            x_ = x_.unsqueeze(2).repeat(1,1,4,1).reshape(bs,nf,-1)
            x_ = obj_local2global_rot6d_by_matrix(x_)
            x_ = x_.reshape(bs,nf,4,9)[:,:,0]


            loss = torch.norm((x_.reshape(bs,nf,-1) - hint) * mask_hint, dim=-1)
            # torch.autograd.set_detect_anomaly(True)
            grad = torch.autograd.grad([loss.sum()], [x])[0]
            # print(grad[:])
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            # grad[..., 0] = 0
            x.detach()
        return loss, grad

    def gradients_stage2(self, x, obj_pose, hint, hand_shape=None, obj_verts=None):
        with torch.enable_grad():
            x.requires_grad_(True)

            x_ = x.permute(0, 3, 2, 1).contiguous()
            x_ = x_.squeeze(2)
            bs,nf,_ = x_.shape
            x_ = x_ * self.local_std + self.local_mean
            x_ = local2global_rot6d_by_matrix(x_)
            hand_T = x_[:,:,:3]
            # print(obj_pose.shape)
            obj_T = obj_pose[:,:,:3,3]
            # print(hint.shape)
            if hint.shape[-1] == 1:                
                pred_dis = torch.norm(hand_T-obj_T,dim=-1).unsqueeze(-1) # bs,nf,1
            elif hint.shape[-1] == 5:
                x_axis = rot6d2axis(x_).reshape(-1,51)
                hand_shape = hand_shape.reshape(-1,10)
                hand_T = hand_T.reshape(-1,3)
                mano_output = self.manolayer(x_axis[:,3:], hand_shape)
                joints = mano_output.joints - mano_output.joints[:, 0].unsqueeze(1) + hand_T.unsqueeze(1)
                joints = joints.reshape(bs,nf,-1,3)
                tips = [15,3,6,12,9]
                tips_T = joints[:,:,tips]
                # print(tips_T.shape,obj_T.shape)
                pred_dis = torch.norm(tips_T - obj_T.unsqueeze(2),dim=-1)
                # print(joints.shape)
                # print(pred_dis.shape)
            elif hint.shape[-1] == 20:
                x_axis = rot6d2axis(x_).reshape(-1,51)
                hand_shape = hand_shape.reshape(-1,10)
                hand_T = hand_T.reshape(-1,3)
                mano_output = self.manolayer(x_axis[:,3:], hand_shape)
                joints = mano_output.joints - mano_output.joints[:, 0].unsqueeze(1) + hand_T.unsqueeze(1)
                joints = joints.reshape(bs,nf,-1,3)
                tips_index = [15,3,6,12,9]
                tips_T = joints[:,:,tips_index]
                tips_to_obj = convert_to_obj_frame(tips_T,obj_pose) # bs, nf, 5, 3
                # print(tips_to_obj.shape)
                hint = hint.reshape(-1,nf,5,4)
                tgt_point = hint[:,:,:,:3]
                hint = hint[:,:,:,3]
                pred_dis = torch.norm(tips_to_obj-tgt_point,dim=-1)
            elif hint.shape[-1] == 3:
                # hint: (bs, f,3)
                # mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0
                # hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
                pred_dis = x_[:,:,:3]
                 

            # loss = torch.norm((x_.reshape(bs,nf,-1) - hint) * mask_hint, dim=-1)
            # print(pred_dis.shape, hint.shape)
            loss = torch.norm(pred_dis-hint, dim=-1)
            # torch.autograd.set_detect_anomaly(True)
            grad = torch.autograd.grad([loss.sum()], [x])[0]
            # print(grad[:])
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            # grad[..., 0] = 0
            x.detach()
        return loss, grad

    def calc_grad_scale(self, mask_hint):
        # assert mask_hint.shape[1] == 196
        # print(mask_hint.shape)
        num_keyframes = mask_hint.sum(dim=-1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        # print(num_keyframes.shape)
        # print(max_keyframes)
        scale = 20 / max_keyframes
        # print('scale:',scale)
        return scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def calc_grad_scale_kp(self, mask_hint):
        num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        scale = 20 / max_keyframes
        return scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def guide(self, out, x, t, use_guidance, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        n_joint = 17
        # model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        # model_variance = torch.exp(model_log_variance)
        model_variance = out['variance']
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        # if train:
        #     if t[0] < 20:
        #         n_guide_steps = 100
        #     else:
        #         n_guide_steps = 20
        # else:
        #     if t[0] < 10:
        #         n_guide_steps = 500
        #     else:
        #         n_guide_steps = 10
        n_guide_steps = 1

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()
        length = model_kwargs['y']['lengths'].clone().detach().long().to(x.device)
        # print(hint.shape)
        # mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0
        # print(mask_hint.shape)
        if self.global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.local_mean = self.local_mean.to(hint.device)
            self.local_std = self.local_std.to(hint.device)
            self.global_mean = self.global_mean.to(hint.device)
            self.global_std = self.global_std.to(hint.device)
        hint = hint * self.global_std + self.global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        # joint id
        # joint_ids = []
        # for m in mask_hint:
        #     joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
        #     joint_ids.append(joint_id)
        
        # if not train:
        #     scale = self.calc_grad_scale(mask_hint)
        # if t > 500:
        #     scale = 100
        # else:
        #     scale = 0.1
        # print(f'scale:{scale}')

        for _ in range(n_guide_steps):
            loss, grad = self.gradients(out, x, hint, mask_hint, length)
            # print(grad.shape)
            # print(grad[0,:,0,1])
            # print(grad[0,:,0,0])
            # grad = model_variance * grad
            grad = grad
            # print(grad)
            # print('loss: ', loss)

            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

            if not use_guidance:
                return sample

            mask_hint = torch.where(mask_hint, 1.0, 0.0).to(torch.float32)
            mask_hint = mask_hint.repeat(1, 1, 99)

            # print('mask_hint:', mask_hint.shape)
            # print('hint:', hint.shape)
            # print('sample:', sample.shape)
            # print('self.q_sample(hint, t):', self.q_sample(hint, t).shape)
            # print('out["log_variance"].dtype', out["log_variance"].dtype)

            # x = sample * (1. - mask_hint.reshape(1, -1, 1, 60)) - scale * grad + \
            #     self.q_sample(hint, t).reshape(1, -1, 1, 60) * mask_hint.reshape(1, -1, 1, 60)
            # print('x.dtype', x.dtype)
            # x = x.to(torch.float32)

            r = torch.linalg.norm(sample - out["mean"], dim=[1, 2, 3]).mean()
            # print(f'r:{r}')
            # print(f't:{t}')

            if t[0] % 1 == 0:
                eps = 1e-20
                guidance_rate = 0.1
                grad_norm = torch.linalg.norm(grad, dim=[1, 2, 3])
                grad_norm = grad_norm.reshape(grad_norm.shape[0], 1, 1, 1)
                # print(f'grad_norm:{grad_norm.shape}')
                grad2 = grad / (grad_norm + eps)
                direction1 = -r * grad2
                direction2 = sample - out["mean"]
                mix_direction = direction2 + guidance_rate * (direction1 - direction2)
                mix_direction_norm = torch.linalg.norm(mix_direction, dim=[1, 2, 3])
                mix_direction_norm = mix_direction_norm.reshape(mix_direction_norm.shape[0], 1, 1, 1)
                print(f'mix_direction_norm:{mix_direction_norm.shape}')
                mix_step = mix_direction / (mix_direction_norm + eps) * r
                x = out["mean"] + mix_step
            else:
                x = sample


            # x = sample - scale * grad

            # print('hint: ', hint.shape)
            # print('mask_hint: ', mask_hint.shape)


            # if t[0] >= t_stopgrad:
            #     x = x - scale * grad
        # print("Guide: ",torch.mean(loss[:,0]))
        # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_stage2(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()
        if self.global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.local_mean = self.local_mean.to(hint.device)
            self.local_std = self.local_std.to(hint.device)
            self.global_mean = self.global_mean.to(hint.device)
            self.global_std = self.global_std.to(hint.device)
        
        # if not train:
        #     scale = self.calc_grad_scale(mask_hint)

        obj_pose = model_kwargs['y']['obj_pose']
        hand_shape = model_kwargs['y']['hand_shape']
        obj_verts = model_kwargs['y']['obj_points']
        for _ in range(n_guide_steps):
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            loss, grad = self.gradients_stage2(x, obj_pose, hint, hand_shape,obj_verts)
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        print("Guide: ",torch.mean(loss))
        # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_stage0(self, out, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        # if train:
        #     if t[0] < 20:
        #         n_guide_steps = 100
        #     else:
        #         n_guide_steps = 20
        # else:
        #     # if t[0] < 10:
        #     #     n_guide_steps = 10
        #     # else:
        #     #     n_guide_steps = 2
        #     if t[0] < 10:
        #         n_guide_steps = 500
        #     else:
        #         n_guide_steps = 10

        n_guide_steps = 1

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach() # hint (bs, nf, 36)
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0 
        if self.obj_global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.obj_local_mean = self.obj_local_mean.to(hint.device)
            self.obj_local_std = self.obj_local_std.to(hint.device)
            self.obj_global_mean = self.obj_global_mean.to(hint.device)
            self.obj_global_std = self.obj_global_std.to(hint.device)
        hint = hint * self.obj_global_std + self.obj_global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        
        if not train:
            scale = self.calc_grad_scale(mask_hint)

        
        for i in range(n_guide_steps):
            # print(t[0].item(),i)
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            if self.dataset == 'gazehoi_stage0_flag2_lowfps_global' or self.dataset == 'gazehoi_stage0_1obj':
                loss, grad = self.gradients_stage0_global(out, x,  hint,mask_hint)
            else:
                loss, grad = self.gradients_stage0(x,  hint,mask_hint)
            # grad = model_variance * grad
            # if t[0] >= t_stopgrad:
            #     x = x - scale * grad
            grad = grad
            # print(grad)
            # print('loss: ', loss)
            print("Loss: ", torch.mean(torch.sum(loss, dim=-1)))

            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

            # if not use_guidance:
            #     return sample

            # mask_hint = torch.where(mask_hint, 1.0, 0.0).to(torch.float32)
            # mask_hint = mask_hint.repeat(1, 1, 99)

            # print('mask_hint:', mask_hint.shape)
            # print('hint:', hint.shape)
            # print('sample:', sample.shape)
            # print('self.q_sample(hint, t):', self.q_sample(hint, t).shape)
            # print('out["log_variance"].dtype', out["log_variance"].dtype)

            # x = sample * (1. - mask_hint.reshape(1, -1, 1, 60)) - scale * grad + \
            #     self.q_sample(hint, t).reshape(1, -1, 1, 60) * mask_hint.reshape(1, -1, 1, 60)
            # print('x.dtype', x.dtype)
            # x = x.to(torch.float32)

            r = torch.linalg.norm(sample - out["mean"], dim=[1, 2, 3]).mean()
            # print(f'r:{r}')
            # print(f't:{t}')

            if t[0] % 1 == 0:
                eps = 1e-20
                guidance_rate = 1.
                grad_norm = torch.linalg.norm(grad, dim=[1, 2, 3])
                grad_norm = grad_norm.reshape(grad_norm.shape[0], 1, 1, 1)
                # print(f'grad_norm:{grad_norm.shape}')
                grad2 = grad / (grad_norm + eps)
                direction1 = -r * grad2
                direction2 = sample - out["mean"]
                mix_direction = direction2 + guidance_rate * (direction1 - direction2)
                mix_direction_norm = torch.linalg.norm(mix_direction, dim=[1, 2, 3])
                mix_direction_norm = mix_direction_norm.reshape(mix_direction_norm.shape[0], 1, 1, 1)
                print(f'mix_direction_norm:{mix_direction_norm.shape}')
                mix_step = mix_direction / (mix_direction_norm + eps) * r
                x = out["mean"] + mix_step
            else:
                x = sample


        # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_stage0_1(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 10
            else:
                n_guide_steps = 2
            # if t[0] < 10:
            #     n_guide_steps = 500
            # else:
            #     n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach() # hint (bs, nf, 36)
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0 
        if self.obj_global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.obj_local_mean = self.obj_local_mean.to(hint.device)
            self.obj_local_std = self.obj_local_std.to(hint.device)
            self.obj_global_mean = self.obj_global_mean.to(hint.device)
            self.obj_global_std = self.obj_global_std.to(hint.device)
        hint = hint * self.obj_global_std + self.obj_global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        
        if not train:
            scale = self.calc_grad_scale(mask_hint)

        
        for i in range(n_guide_steps):
            # print(t[0].item(),i)
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            loss, grad = self.gradients_stage0_1(x,  hint,mask_hint)
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        # print("Guide: ",torch.mean(loss))
        # print("Guide: ",loss.shape,grad.shape)
        return x
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        x = x.requires_grad_()
        sample = None
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # if 'hint' in model_kwargs['y'].keys():
        #     # spatial guidance/classifier guidance
        #     hint = model_kwargs['y']['hint']
        #
        #     if hint.shape[-1] == 99:
        #         # out['mean'] = self.guide(out, x, t, model_kwargs=model_kwargs)
        #         use_guidance = True
        #         sample = self.guide(out, x, t,use_guidance, model_kwargs=model_kwargs)
            # elif hint.shape[-1] == 36:
            #     out['mean'] = self.guide_stage0(out['mean'], t, model_kwargs=model_kwargs)
            # elif hint.shape[-1] == 9:
            #     out['mean'] = self.guide_stage0(out['mean'], t, model_kwargs=model_kwargs)
            # else:
            #     out['mean'] = self.guide_stage2(out['mean'], t, model_kwargs=model_kwargs)
        # print(f'self.dataset:{self.dataset}')
        # print("self.dataset == 'gazehoi_stage0_1obj':",self.dataset == 'gazehoi_stage0_1_obj')
        if 'hint' in model_kwargs['y'].keys():
            # spatial guidance/classifier guidance
            hint = model_kwargs['y']['hint']
            # if hint.shape[-1] == 99:
            use_guidance = True
            if self.dataset.startswith('gazehoi_stage1') :
            # if self.dataset == 'gazehoi_stage1' or self.dataset == 'gazehoi_stage1_new':
                sample = self.guide(out, x, t, use_guidance, model_kwargs=model_kwargs)
            # elif self.dataset == 'gazehoi_stage1_new' or self.dataset == 'gazehoi_stage1_repair':
            #     sample = self.guide(out, x, t, use_guidance, model_kwargs=model_kwargs)
                # out['mean'] = self.guide_new(out['mean'], t, model_kwargs=model_kwargs)
            elif self.dataset == 'gazehoi_stage0':
            # elif hint.shape[-1] == 36:
                out['mean'] = self.guide_stage0(out['mean'], t, model_kwargs=model_kwargs)
            elif self.dataset == 'gazehoi_stage0_flag2_lowfps':
                out['mean'] = self.guide_stage0(out['mean'], t, model_kwargs=model_kwargs)
            # elif self.dataset == 'gazehoi_stage0_flag2_lowfps_global':
            #     out['mean'] = self.guide_stage0(out['mean'], t, model_kwargs=model_kwargs)
            elif self.dataset == 'gazehoi_stage0_1':
            # elif hint.shape[-1] == 9:
                out['mean'] = self.guide_stage0_1(out['mean'], t, model_kwargs=model_kwargs)
            elif self.dataset == 'gazehoi_stage0_1obj':
                sample = self.guide_stage0(out, x, t, model_kwargs=model_kwargs)
            elif self.dataset == 'gazehoi_stage2':
                out['mean'] = self.guide_stage2(out['mean'], t, model_kwargs=model_kwargs)

        ## GMD在此处增加了inpainting
        if (self.dataset.startswith('gazehoi_stage1') or self.dataset == 'gazehoi_stage0_1_obj'):
            sample = sample.detach()
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        else:
            if const_noise:
                noise = th.randn_like(x[0])
                noise = noise[None].repeat(x.shape[0], 1, 1, 1)
            else:
                noise = th.randn_like(x)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
            # print('mean', out["mean"].shape, out["mean"])
            # print('log_variance', out["log_variance"].shape, out["log_variance"])
            # print('nonzero_mask', nonzero_mask.shape, nonzero_mask)
            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            sample = sample.detach()
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
        )):
            if dump_steps is not None and i == dump_steps:
                # dump.append(deepcopy(sample["pred_xstart"].detach().clone()))
                return deepcopy(sample["pred_xstart"].detach().clone())
            final = sample
        # if dump_steps is not None:
        #     return dump
        if self.dataset.startswith('gazehoi_stage0_flag'):
            return final["sample"],final['flag']
        else:
            return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            if const_noise:
                img = th.randn(*shape[1:], device=device)
                img = img[None].repeat(shape[0], 1, 1, 1)
            else:
                img = th.randn(*shape, device=device).requires_grad_()

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            # with th.no_grad():
            sample_fn = self.p_sample
            out = sample_fn(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                const_noise=const_noise,
            )
            yield out
            img = out["sample"]

    def training_losses_stage1(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        x_t = self.guide(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
        # print(target.shape)
        
        """
        # 相对表示转为绝对表示
        target_global = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output_global = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()
        target_global = local2global_rot6d_by_matrix(target_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        output_global = local2global_rot6d_by_matrix(output_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        # model_output = torch.cumsum(model_output,dim=-1)
        terms['global_mse'] = self.masked_l2(target_global, output_global, mask)
       """

        # terms['global_mse'] = self.masked_l2(target, model_output, mask)
        # global linear vel
        # gt_vel = target[:,:,:,1:] - target[:,:,:,:-1]
        # pred_vel = model_output[:,:,:,1:] - model_output[:,:,:,:-1]
        # print(mask.shape)
        # terms['vel_mse'] = self.masked_l2(gt_vel, pred_vel,mask[:,:,:,:-1])
        # 只计算global位置的
        # terms['vel_mse'] = self.masked_l2(gt_vel[:,:9], pred_vel[:,:9],mask[:,:9,:,1:])

        # gt_acc = gt_vel[:,:,:,1:] - gt_vel[:,:,:,:-1]
        # pred_acc = pred_vel[:,:,:,1:] - pred_vel[:,:,:,:-1]
        # terms['acc_mse'] = self.masked_l2(gt_acc, pred_acc, mask[:,:,:,:-2])
        # 只计算global位置的
        # terms['acc_mse'] = self.masked_l2(gt_acc[:,:9], pred_acc[:,:9], mask[:,:9,:,2:])

        terms['time_smooth'] = self.masked_l2(model_output[:,:9], torch.zeros_like(model_output[:,:9]),mask)
        # 0123 调整后
        terms["loss"] = terms["rot_mse"] + terms['global_mse'] + terms['time_smooth'] * 0.1
        # terms["loss"] = terms["rot_mse"] + terms['time_smooth'] * 0.1

        # terms["loss"] = terms["rot_mse"]*10 + terms['global_mse']*0.01 + terms['vel_mse'] + terms['acc_mse'] + terms['time_smooth'] * 0.1
        
        # 0121 weight1
        # terms["loss"] = terms["rot_mse"] + terms['global_mse'] + terms['vel_mse'] + terms['acc_mse'] + terms['time_smooth'] * 0.1

        # 0121 weight2
        # terms["loss"] = terms["rot_mse"] + terms['global_mse']*0.1 + terms['vel_mse'] + terms['acc_mse'] + terms['time_smooth'] * 0.1



        # print(torch.mean(terms["rot_mse"]),torch.mean(terms["global_mse"]),torch.mean(terms['time_smooth']))
        # print(torch.mean(terms["rot_mse"]),torch.mean(terms["global_mse"]),torch.mean(terms['vel_mse']), torch.mean(terms['acc_mse']),torch.mean(terms['time_smooth']))
        

        """
        # global linear vel
        gt_vel = target[:,:,:,1:] - target[:,:,:,:-1]
        pred_vel = model_output[:,:,:,1:] - model_output[:,:,:,:-1]
        # print(mask.shape)
        terms['vel_mse'] = self.masked_l2(gt_vel, pred_vel,mask[:,:,:,:-1])

        gt_acc = gt_vel[:,:,:,1:] - gt_vel[:,:,:,:-1]
        pred_acc = pred_vel[:,:,:,1:] - pred_vel[:,:,:,:-1]
        terms['acc_mse'] = self.masked_l2(gt_acc, pred_acc, mask[:,:,:,:-2])

        terms['time_smooth'] = self.masked_l2(pred_vel, torch.zeros_like(pred_vel),mask[:,:,:,:-1])

        terms["loss"] = terms["rot_mse"] + terms['vel_mse'] + terms['acc_mse'] + terms['time_smooth']
        print(torch.mean(terms["rot_mse"]),torch.mean(terms['vel_mse']), torch.mean(terms['acc_mse']),torch.mean(terms['time_smooth']))
        """
        return terms


    def training_losses_stage1_new(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        # x_t = self.guide(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        bs = target.shape[0]
        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
        # print(target.shape)


        # 相对表示转为绝对表示
        target_global = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output_global = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()
        target_global = local2global_rot6d_by_matrix(target_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        output_global = local2global_rot6d_by_matrix(output_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        # model_output = torch.cumsum(model_output,dim=-1)
        terms['global_mse'] = self.masked_l2(target_global, output_global, mask)

        target_goal = torch.cat((target_global[:,:9,:,-1].unsqueeze(-1), target[:,9:,:,-1].unsqueeze(-1)),dim=1)
        output_goal = torch.cat((output_global[:,:9,:,-1].unsqueeze(-1), model_output[:,9:,:,-1].unsqueeze(-1)),dim=1)
        terms['goal_pose'] = self.masked_l2(target_goal, output_goal, mask[:,:,:,-1].unsqueeze(-1))

        length = model_kwargs['y']['lengths']

        target_init = torch.cat((target_global[:,:9,:,0].unsqueeze(-1), target[:,9:,:,0].unsqueeze(-1)),dim=1)
        output_init = torch.cat((output_global[:,:9,:,0].unsqueeze(-1), model_output[:,9:,:,0].unsqueeze(-1)),dim=1)
        # target_init = torch.cat((target_global[torch.arange(bs),:9,:,-length].unsqueeze(-1), target[torch.arange(bs),9:,:,-length].unsqueeze(-1)),dim=1)
        # output_init = torch.cat((output_global[torch.arange(bs),:9,:,-length].unsqueeze(-1), model_output[torch.arange(bs),9:,:,-length].unsqueeze(-1)),dim=1)
        # print(target_init.shape)
        terms['init_pose'] = self.masked_l2(target_init, output_init, mask[:,:,:,-1].unsqueeze(-1))


        # print(target_goal.shape)
        # print(mask[:,:,:,-1])
        # terms['global_mse'] = self.masked_l2(target, model_output, mask)
        # terms['acc_mse'] = self.masked_l2(gt_acc[:,:9], pred_acc[:,:9], mask[:,:9,:,2:])

        # terms['time_smooth'] = self.masked_l2(model_output[:,:9], torch.zeros_like(model_output[:,:9]),mask)
        # 0123 调整后
        terms["loss"] = terms["rot_mse"] + terms['global_mse'] + terms['goal_pose'] + terms['init_pose']
        # terms["loss"] = terms["rot_mse"] + terms['global_mse'] + terms['time_smooth'] * 0.1 + terms['goal_pose'] + terms['init_pose']
        # terms["loss"] = terms["rot_mse"] + terms['time_smooth'] * 0.1

        return terms

    def training_losses_stage1_repair(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        # x_t = self.guide(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}
        # print(x_t.shape)
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        bs = target.shape[0]

        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)

        # 相对表示转为绝对表示
        target_global = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output_global = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()
        target_global = local2global_rot6d_by_matrix(target_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        output_global = local2global_rot6d_by_matrix(output_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        terms['global_mse'] = self.masked_l2(target_global, output_global, mask)

        goal_index = model_kwargs['y']['lengths'] - 1
        target_goal = torch.cat((target_global[torch.arange(bs),:9,:,goal_index].unsqueeze(-1), target[torch.arange(bs),9:,:,goal_index].unsqueeze(-1)),dim=1)
        output_goal = torch.cat((output_global[torch.arange(bs),:9,:,goal_index].unsqueeze(-1), model_output[torch.arange(bs),9:,:,goal_index].unsqueeze(-1)),dim=1)
        terms['goal_pose'] = self.masked_l2(target_goal, output_goal, mask)


        target_init = torch.cat((target_global[:,:9,:,0].unsqueeze(-1), target[:,9:,:,0].unsqueeze(-1)),dim=1)
        output_init = torch.cat((output_global[:,:9,:,0].unsqueeze(-1), model_output[:,9:,:,0].unsqueeze(-1)),dim=1)
        terms['init_pose'] = self.masked_l2(target_init, output_init, mask)


        terms['time_smooth'] = self.masked_l2(model_output[:,:9], torch.zeros_like(model_output[:,:9]),mask)


        terms["loss"] = terms["rot_mse"] + terms['global_mse'] * 10 + terms['goal_pose'] * 10 + terms['init_pose'] * 5 + terms['time_smooth'] * 0.01
        # terms["loss"] = terms["rot_mse"] + terms['global_mse'] * 20
        # terms["loss"] = terms["rot_mse"] + terms['global_mse'] * 20 + terms['time_smooth'] * 0.01
        # terms["loss"] = terms["rot_mse"]*5 + terms['global_mse'] * 20 + terms['goal_pose'] * 10 + terms['init_pose'] * 5 + terms['time_smooth'] * 0.01

        # print(terms["rot_mse"].mean() , terms['global_mse'].mean() , terms['goal_pose'].mean() , terms['init_pose'].mean() , terms['time_smooth'].mean())

        return terms

    def training_losses_stage1_simple(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        # x_t = self.guide(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}
        # print(x_t.shape)
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        bs = target.shape[0]

        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)

        # 相对表示转为绝对表示
        target_global = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output_global = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()
        target_global = local2global_rot6d_by_matrix(target_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        output_global = local2global_rot6d_by_matrix(output_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        terms['global_mse'] = self.masked_l2(target_global, output_global, mask)

        goal_index = model_kwargs['y']['lengths'] - 1
        target_goal = torch.cat((target_global[torch.arange(bs),:9,:,goal_index].unsqueeze(-1), target[torch.arange(bs),9:,:,goal_index].unsqueeze(-1)),dim=1)
        output_goal = torch.cat((output_global[torch.arange(bs),:9,:,goal_index].unsqueeze(-1), model_output[torch.arange(bs),9:,:,goal_index].unsqueeze(-1)),dim=1)
        terms['goal_pose'] = self.masked_l2(target_goal, output_goal, mask)


        target_init = torch.cat((target_global[:,:9,:,0].unsqueeze(-1), target[:,9:,:,0].unsqueeze(-1)),dim=1)
        output_init = torch.cat((output_global[:,:9,:,0].unsqueeze(-1), model_output[:,9:,:,0].unsqueeze(-1)),dim=1)
        terms['init_pose'] = self.masked_l2(target_init, output_init, mask)

        # 1
        # terms["loss"] = terms["rot_mse"] * 10

        # 2
        # terms['loss'] = terms['global_mse'] + terms["rot_mse"] * 10

        # 3
        # terms['loss'] = terms['global_mse'] + terms["rot_mse"] * 10 + terms['goal_pose'] + terms['init_pose']

        # 4

        terms['loss'] =  terms["rot_mse"] * 10 + terms['goal_pose'] + terms['init_pose']


        # print(terms["rot_mse"].mean() , terms['global_mse'].mean())
        # print(terms["rot_mse"].mean() , terms['global_mse'].mean() , terms['goal_pose'].mean() , terms['init_pose'].mean())

        return terms

    def training_losses_g2ho(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        # print(x_t.shape)
        terms = {}
        # print(x_t.shape)
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        bs = target.shape[0]

        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)

        gt_obj_T = target[:,99:102,:,:]
        pred_obj_T = model_output[:,99:102,:,:]
        terms['obj_T'] = self.masked_l2(gt_obj_T, pred_obj_T,mask)

        gt_hand_T = target[:,:3,:,:]
        pred_hand_T = model_output[:,:3,:,:]
        terms['hand_T'] = self.masked_l2(gt_hand_T, pred_hand_T,mask)

        gt_offset = gt_hand_T - gt_obj_T
        pred_offset = pred_hand_T - pred_obj_T
        terms['offset'] = self.masked_l2(gt_offset, pred_offset, mask)


        terms['loss'] =  terms["rot_mse"]*10 + terms['obj_T']*10 + terms['hand_T']*10 + terms['offset'] * 20

        # print(torch.mean(terms["rot_mse"]), torch.mean(terms['obj_T']) , torch.mean(terms['hand_T']) , torch.mean(terms['offset']))
        return terms

    def training_losses_o2h(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        # print(x_t.shape)
        terms = {}
        # print(x_t.shape)
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        bs = target.shape[0]

        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)



        gt_hand_T = target[:,:3,:,:]
        pred_hand_T = model_output[:,:3,:,:]
        terms['hand_T'] = self.masked_l2(gt_hand_T, pred_hand_T,mask)

        gt_obj_pose = model_kwargs['y']['obj_pose']
        gt_obj_T = gt_obj_pose[:,:,:3].permute(0,2,1).unsqueeze(2)

        gt_offset = gt_hand_T - gt_obj_T
        pred_offset = pred_hand_T - gt_obj_T
        terms['offset'] = self.masked_l2(gt_offset, pred_offset, mask)


        terms['loss'] =  terms["rot_mse"]*10 + terms['hand_T']*10  + terms['offset'] * 20

        # print(torch.mean(terms["rot_mse"]), torch.mean(terms['hand_T']) , torch.mean(terms['offset']))
        return terms



    def training_losses_stage2(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        
        # x_t = self.guide_stage2(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
        # print(target.shape)
        
        # 相对表示转为绝对表示
        target_global = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output_global = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()
        target_global = local2global_rot6d_by_matrix(target_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        output_global = local2global_rot6d_by_matrix(output_global)[:,:,:9].permute(0,2,1).unsqueeze(2) #global RT
        # model_output = torch.cumsum(model_output,dim=-1)
        terms['global_mse'] = self.masked_l2(target_global, output_global, mask)



        terms['time_smooth'] = self.masked_l2(model_output[:,:9], torch.zeros_like(model_output[:,:9]),mask)
        # 0123 调整后
        terms["loss"] = terms["rot_mse"] + terms['global_mse'] + terms['time_smooth'] * 0.1

        return terms

    def get_verts(self,o1_verts,o2_verts,o3_verts,o4_verts,target):
        bs = target.shape[0]
        frames = target.shape[1]
        tgt_matrix = obj_local2global_matrix(target) # bs,nframes,4,3,4
        tgt_R = tgt_matrix[:,:,:,:3,:3] #bs,nframes,4,3,3
        tgt_T = tgt_matrix[:,:,:,:3,3].unsqueeze(-2) #bs,nframes,4,1,3
        tgt_R = torch.einsum('...ij->...ji', [tgt_R]) # 对R的最后两个维度转置
        tgt_R1 = tgt_R[:,:,0,:,:]
        tgt_R2 = tgt_R[:,:,1,:,:]
        tgt_R3 = tgt_R[:,:,2,:,:]
        tgt_R4 = tgt_R[:,:,3,:,:]
        tgt_T1 = tgt_T[:,:,0,:,:]
        tgt_T2 = tgt_T[:,:,1,:,:]
        tgt_T3 = tgt_T[:,:,2,:,:]
        tgt_T4 = tgt_T[:,:,3,:,:]
        # print(o1_verts.shape,tgt_R1.shape,tgt_T1.shape)
        tgt_o1 = torch.einsum('bfpn,bfnk->bfpk',o1_verts,tgt_R1) + tgt_T1 #    b frames 500 3
        tgt_o2 = torch.einsum('bfpn,bfnk->bfpk',o2_verts,tgt_R2) + tgt_T2
        tgt_o3 = torch.einsum('bfpn,bfnk->bfpk',o3_verts,tgt_R3) + tgt_T3
        tgt_o4 = torch.einsum('bfpn,bfnk->bfpk',o4_verts,tgt_R4) + tgt_T4
        tgt_obj_verts = torch.cat((tgt_o1,tgt_o2,tgt_o3,tgt_o4),dim=2).reshape(bs,frames,-1).permute(0,2,1).unsqueeze(2)
        return tgt_obj_verts

    def get_verts_global(self,o1_verts,o2_verts,o3_verts,o4_verts,target):
        bs = target.shape[0]
        frames = target.shape[1]
        tgt_matrix = obj_rot6d2matrix(target) # bs,nframes,4,3,4
        tgt_R = tgt_matrix[:,:,:,:3,:3] #bs,nframes,4,3,3
        tgt_T = tgt_matrix[:,:,:,:3,3].unsqueeze(-2) #bs,nframes,4,1,3
        tgt_R = torch.einsum('...ij->...ji', [tgt_R]) # 对R的最后两个维度转置
        tgt_R1 = tgt_R[:,:,0,:,:]
        tgt_R2 = tgt_R[:,:,1,:,:]
        tgt_R3 = tgt_R[:,:,2,:,:]
        tgt_R4 = tgt_R[:,:,3,:,:]
        tgt_T1 = tgt_T[:,:,0,:,:]
        tgt_T2 = tgt_T[:,:,1,:,:]
        tgt_T3 = tgt_T[:,:,2,:,:]
        tgt_T4 = tgt_T[:,:,3,:,:]
        # print(o1_verts.shape,tgt_R1.shape,tgt_T1.shape)
        tgt_o1 = torch.einsum('bfpn,bfnk->bfpk',o1_verts,tgt_R1) + tgt_T1 #    b frames 500 3
        tgt_o2 = torch.einsum('bfpn,bfnk->bfpk',o2_verts,tgt_R2) + tgt_T2
        tgt_o3 = torch.einsum('bfpn,bfnk->bfpk',o3_verts,tgt_R3) + tgt_T3
        tgt_o4 = torch.einsum('bfpn,bfnk->bfpk',o4_verts,tgt_R4) + tgt_T4
        tgt_obj_verts = torch.cat((tgt_o1,tgt_o2,tgt_o3,tgt_o4),dim=2).reshape(bs,frames,-1).permute(0,2,1).unsqueeze(2)
        return tgt_obj_verts

    def get_1verts_global(self,o1_verts,target):
        bs = target.shape[0]
        frames = target.shape[1]
        tgt_R = rotation_6d_to_matrix(target[:,:,3:]) # bs,nframes,3,4
        tgt_T = target[:,:,:3].unsqueeze(2) #bs,nframes,1,3
        tgt_R = torch.einsum('...ij->...ji', [tgt_R]) # 对R的最后两个维度转置
        # print(o1_verts.shape,tgt_R.shape,tgt_T.shape)
        tgt_o1 = torch.einsum('bfpn,bfnk->bfpk',o1_verts,tgt_R) + tgt_T #    b frames 500 3
        tgt_obj_verts = tgt_o1.reshape(bs,frames,-1).permute(0,2,1).unsqueeze(2)
        return tgt_obj_verts

    def training_losses_stage0(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        
        # x_t = self.guide_stage0(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
        # print(target.shape)
        
        # 相对表示转为绝对表示
        target = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()
        target_global = obj_local2global_rot6d_by_matrix(target).permute(0,2,1).unsqueeze(2) #global RT
        output_global = obj_local2global_rot6d_by_matrix(output).permute(0,2,1).unsqueeze(2) #global RT
        # model_output = torch.cumsum(model_output,dim=-1)
        # print(target_global.shape)
        terms['global_mse'] = self.masked_l2(target_global, output_global, mask)

        # obj verts loss
        # print(model_kwargs['y']['obj_points'].shape)
        length = model.length
        print(length)
        obj_verts = model_kwargs['y']['obj_points'][:,:2000].reshape(-1,2000,3).unsqueeze(1).repeat(1,length,1,1) # (b, nframs,npoints,3) 只截取物体的部分
        o1_verts = obj_verts[:,:,:500,:]
        o2_verts = obj_verts[:,:,500:1000,:]
        o3_verts = obj_verts[:,:,1000:1500,:]
        o4_verts = obj_verts[:,:,1500:,:]
        
        # print(target.device)
        tgt_obj_verts = self.get_verts(o1_verts,o2_verts,o3_verts,o4_verts,target)
        out_obj_verts = self.get_verts(o1_verts,o2_verts,o3_verts,o4_verts,output)
        terms['obj_verts'] = self.masked_l2(tgt_obj_verts, out_obj_verts, mask)


        terms['time_smooth'] = self.masked_l2(model_output[:,:9], torch.zeros_like(model_output[:,:9]),mask)
        # 0123 调整后
        terms["loss"] = terms["rot_mse"] + terms['global_mse'] + terms['time_smooth'] * 0.1 + terms['obj_verts'] * 0.01

        return terms

    def training_losses_stage0_flag(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪

        # x_t = self.guide_stage0(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}

        model_output,pred_flag = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
        # print(target.shape)
        # print(model_output.shape)
        gt_flag = model_kwargs['y']['flag'] # b,4,nf
        # print(gt_flag.shape)
        gt_move_flag = gt_flag.sum(dim=-1)>0
        # print(gt_move_flag.shape)
        gt_move_index = torch.where(gt_move_flag==1)
        # print(gt_move_index[0].shape)
        gt_active_flag = gt_flag[gt_move_index]
        gt_class = gt_move_index[1].long()

        pred_flag = pred_flag.permute(0,2,1).contiguous() # bs frame 4 -- b 4 frames
        # pred_move = torch.sum(pred_flag,dim=-1)
        pred_move = torch.max(pred_flag,dim=-1)
        C = nn.CrossEntropyLoss()
        # print(pred_move.shape,gt_class.shape)
        terms['move_flag'] = C(pred_move,gt_class)

        pred_active = torch.sigmoid(pred_flag[gt_move_index])
        B = nn.BCELoss()
        terms['active_flag'] = B(pred_active, gt_active_flag)

        # 相对表示转为绝对表示
        target = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()
        target_global = obj_local2global_rot6d_by_matrix(target).permute(0,2,1).unsqueeze(2) #global RT
        output_global = obj_local2global_rot6d_by_matrix(output).permute(0,2,1).unsqueeze(2) #global RT
        # model_output = torch.cumsum(model_output,dim=-1)
        # print(target_global.shape)
        terms['global_mse'] = self.masked_l2(target_global, output_global, mask)

        # obj verts loss
        # print(model_kwargs['y']['obj_points'].shape)
        length = model.length
        # print(length)
        obj_verts = model_kwargs['y']['obj_points'][:,:2000].reshape(-1,2000,3).unsqueeze(1).repeat(1,length,1,1) # (b, nframs,npoints,3) 只截取物体的部分
        o1_verts = obj_verts[:,:,:500,:]
        o2_verts = obj_verts[:,:,500:1000,:]
        o3_verts = obj_verts[:,:,1000:1500,:]
        o4_verts = obj_verts[:,:,1500:,:]

        # print(target.device)
        tgt_obj_verts = self.get_verts(o1_verts,o2_verts,o3_verts,o4_verts,target)
        out_obj_verts = self.get_verts(o1_verts,o2_verts,o3_verts,o4_verts,output)
        terms['obj_verts'] = self.masked_l2(tgt_obj_verts, out_obj_verts, mask)


        terms['time_smooth'] = self.masked_l2(model_output[:,:9], torch.zeros_like(model_output[:,:9]),mask)
        # 0123 调整后
        terms["loss"] = terms["rot_mse"] + terms['global_mse'] + terms['time_smooth'] * 0.1 + terms['obj_verts'] + terms['move_flag'] + terms['active_flag']

        return terms

    def training_losses_stage0_flag_global(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪

        # x_t = self.guide_stage0(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}

        model_output,pred_flag = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        # print(target.shape) ([64, 36, 1, 69])
        bs = target.shape[0]
        nf = target.shape[-1]
        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
        # terms["rot_mse_init"] = self.masked_l2(target[:,:,:,0], model_output, mask) # mean_flat(rot_mse)
        # print(target.shape)
        # print(model_output.shape)
        # print(pred_flag)

        pred_active_flag = pred_flag[0]
        pred_move_flag = pred_flag[1]
        gt_flag = model_kwargs['y']['flag'] # b,4,nf
        # print(gt_flag.shape)
        gt_move_flag = gt_flag.sum(dim=-1)>0
        # print(gt_move_flag.shape)
        gt_move_index = torch.where(gt_move_flag==1)
        # print(gt_move_index[0].shape)
        gt_active_flag = gt_flag[gt_move_index]
        gt_class = gt_move_index[1].long()
        gt_move_index = torch.argmax(gt_active_flag,dim=-1)

        # pred_flag = pred_flag.permute(0,2,1).contiguous() # bs frame 4 -- b 4 frames
        # pred_move = torch.sum(pred_flag,dim=-1)
        C = nn.CrossEntropyLoss()
        # print(pred_move.shape,gt_class.shape)
        terms['active_flag'] = C(pred_active_flag,gt_class)
        # print(pred_move_flag.shape)
        # print(gt_move_index)
        terms['move_flag'] = C(pred_move_flag, gt_move_index)

        target = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()

        gt_T = target.reshape(bs,nf,4,9)[:,:,:,:3].reshape(bs,nf,-1).permute(0,2,1).unsqueeze(2)
        out_T = output.reshape(bs,nf,4,9)[:,:,:,:3].reshape(bs,nf,-1).permute(0,2,1).unsqueeze(2)
        # print(gt_T.shape)
        # print(mask.shape)
        terms['obj_T'] = self.masked_l2(gt_T, out_T, mask)


        # obj verts loss
        # print(model_kwargs['y']['obj_points'].shape)
        length = model.length
        # print(length)
        obj_verts = model_kwargs['y']['obj_points'][:,:2000].reshape(-1,2000,3).unsqueeze(1).repeat(1,length,1,1) # (b, nframs,npoints,3) 只截取物体的部分
        o1_verts = obj_verts[:,:,:500,:]
        o2_verts = obj_verts[:,:,500:1000,:]
        o3_verts = obj_verts[:,:,1000:1500,:]
        o4_verts = obj_verts[:,:,1500:,:]

        # print(target.device)
        tgt_obj_verts = self.get_verts_global(o1_verts,o2_verts,o3_verts,o4_verts,target)
        out_obj_verts = self.get_verts_global(o1_verts,o2_verts,o3_verts,o4_verts,output)
        terms['obj_verts'] = self.masked_l2(tgt_obj_verts, out_obj_verts, mask)

        # print(target.shape)[64, 69, 36]
        target = target.reshape(bs,nf,4,9)
        target_active = target[torch.arange(bs),:,gt_class].permute(0,2,1).unsqueeze(2)  # [64, 69, 9]
        output = output.reshape(bs,nf,4,9)
        output_active = output[torch.arange(bs),:,gt_class].permute(0,2,1).unsqueeze(2)  # [64, 69, 9]
        # print(target_active.shape)
        terms['rot_mse_active'] = self.masked_l2(target_active, output_active, mask)

        terms['time_smooth'] = self.masked_l2(model_output[:,:9], torch.zeros_like(model_output[:,:9]),mask)

        # 0226 global
        # terms["loss"] = terms["rot_mse"] + terms['time_smooth'] * 0.1 + terms['obj_verts'] + terms['move_flag'] + terms['active_flag'] + terms['rot_mse_active']*10
        # 0226_stage0_global2
        # terms["loss"] = terms["rot_mse"] + terms['time_smooth'] * 0.1 + terms['obj_verts'] + terms['active_flag'] + terms['rot_mse_active']*10

        ## CVPR loss
        terms["loss"] = 20*terms["rot_mse"] + 20* terms['obj_T'] + 10*terms['obj_verts']
        return terms

    def training_losses_stage0_1obj(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪

        # x_t = self.guide_stage0(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        # print(target.shape) ([64, 36, 1, 69])
        bs = target.shape[0]
        nf = target.shape[-1]
        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)

        target = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()

        gt_T = target[:,:,:3].reshape(bs,nf,-1).permute(0,2,1).unsqueeze(2)
        out_T = output[:,:,:3].reshape(bs,nf,-1).permute(0,2,1).unsqueeze(2)
        # print(gt_T.shape)
        # print(mask.shape)
        terms['obj_T'] = self.masked_l2(gt_T, out_T, mask)


        # obj verts loss
        # print(model_kwargs['y']['obj_points'].shape)
        length = model.length
        # print(length)
        obj_verts = model_kwargs['y']['obj_points'][:,:500].reshape(-1,500,3).unsqueeze(1).repeat(1,length,1,1) # (b, nframs,npoints,3) 只截取物体的部分

        tgt_obj_verts = self.get_1verts_global(obj_verts,target)
        out_obj_verts = self.get_1verts_global(obj_verts,output)
        terms['obj_verts'] = self.masked_l2(tgt_obj_verts, out_obj_verts, mask)

        vel = model_output[:,:,:,1:] -  model_output[:,:,:,:-1]
        # print(vel.shape)
        terms['time_smooth'] = self.masked_l2(vel, torch.zeros_like(vel),mask[:,:,:,1:])

        # terms["loss"] = 20*terms["rot_mse"] + 20* terms['obj_T'] + 10*terms['obj_verts'] + terms['time_smooth']
        terms["loss"] = 10*terms["rot_mse"] + 30* terms['obj_T'] + 10*terms['obj_verts'] + terms['time_smooth']
        return terms

    def training_losses_stage0_1(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # 对x0加噪
        
        # x_t = self.guide_stage0(x_t, t, model_kwargs=model_kwargs, train=True)
        # print(x_t.shape)
        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # model_output = self.guide(model_output,t,model_kwargs=model_kwargs, train=True)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
        # print(target.shape)
        bs = target.shape[0]
        frames = target.shape[-1]
        # 相对表示转为绝对表示
        target = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()
        target_global = obj_local2global_rot6d_by_matrix(target.unsqueeze(-2).repeat(1,1,4,1).reshape(bs,-1,36)).permute(0,2,1).unsqueeze(2) #global RT
        output_global = obj_local2global_rot6d_by_matrix(output.unsqueeze(-2).repeat(1,1,4,1).reshape(bs,-1,36)).permute(0,2,1).unsqueeze(2) #global RT
        # model_output = torch.cumsum(model_output,dim=-1)
        # print(target_global.shape)
        terms['global_mse'] = self.masked_l2(target_global, output_global, mask)

        # obj verts loss
        # print(model_kwargs['y']['obj_points'].shape)
        obj_verts = model_kwargs['y']['obj_points'][:,:500].unsqueeze(1).repeat(1,345,1,1) # (b, nframs,npoints,3) 只截取物体的部分
        
        tgt_matrix = obj_local2global_matrix(target.unsqueeze(-2).repeat(1,1,4,1).reshape(bs,-1,36)) # bs,nframes,4,3,4
        tgt_R = tgt_matrix[:,:,0,:3,:3] #bs,nframes,4,3,3
        tgt_T = tgt_matrix[:,:,0,:3,3].unsqueeze(-2) #bs,nframes,4,1,3
        tgt_R = torch.einsum('...ij->...ji', [tgt_R]) # 对R的最后两个维度转置
        # print(o1_verts.shape,tgt_R1.shape,tgt_T1.shape)
        tgt_o = torch.einsum('bfpn,bfnk->bfpk',obj_verts,tgt_R) + tgt_T #    b frames 500 3
        tgt_obj_verts = tgt_o.reshape(bs,frames,-1).permute(0,2,1).unsqueeze(2)

        tgt_matrix = obj_local2global_matrix(output.unsqueeze(-2).repeat(1,1,4,1).reshape(bs,-1,36)) # bs,nframes,4,3,4
        tgt_R = tgt_matrix[:,:,0,:3,:3] #bs,nframes,4,3,3
        tgt_T = tgt_matrix[:,:,0,:3,3].unsqueeze(-2) #bs,nframes,4,1,3
        tgt_R = torch.einsum('...ij->...ji', [tgt_R]) # 对R的最后两个维度转置
        # print(o1_verts.shape,tgt_R1.shape,tgt_T1.shape)
        out_o = torch.einsum('bfpn,bfnk->bfpk',obj_verts,tgt_R) + tgt_T #    b frames 500 3
        out_obj_verts = out_o.reshape(bs,frames,-1).permute(0,2,1).unsqueeze(2)
        
        # print(target.device)
        # tgt_obj_verts = self.get_verts(o1_verts,o2_verts,o3_verts,o4_verts,target)
        # out_obj_verts = self.get_verts(o1_verts,o2_verts,o3_verts,o4_verts,output)
        # print(tgt_obj_verts.shape, out_obj_verts.shape)
        terms['obj_verts'] = self.masked_l2(tgt_obj_verts, out_obj_verts, mask)


        terms['time_smooth'] = self.masked_l2(model_output[:,:9], torch.zeros_like(model_output[:,:9]),mask)
        # 0123 调整后
        terms["loss"] = terms["rot_mse"] + terms['global_mse'] + terms['time_smooth'] * 0.1 + terms['obj_verts']

        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
