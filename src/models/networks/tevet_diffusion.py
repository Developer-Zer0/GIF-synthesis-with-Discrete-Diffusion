from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor
from src.utils.torch_utils import remove_padding
from torch.distributions.distribution import Distribution

from src.utils.diffusion import gaussian_diffusion as gd
from src.utils.diffusion.resample import create_named_schedule_sampler
from src.utils.diffusion.respace import SpacedDiffusion, space_timesteps

import matplotlib.pyplot as plt

class TevetDiffusion(nn.Module):
    def __init__(self, textencoder, motionencoder, transforms, pose_dim, checkpoint_path, **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)
        self.motionencoder = instantiate(motionencoder, pose_dim=pose_dim)

        # Hardcoded
        self.diffusion_steps = 1000

        self.diffusion = self.create_gaussian_diffusion()
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

        state_dict = torch.load(checkpoint_path, map_location='cuda:0')
        text_state_dict = {}
        text_state_dict['weight'] = state_dict.pop('embed_text.weight')
        text_state_dict['bias'] = state_dict.pop('embed_text.bias')
        self.motionencoder.load_state_dict(state_dict)
        # for param_key in list(state_dict.keys()):
        #     if 'embed_text.weight' == param_key or "embed_text.bias" == param_key:
        #         print('###############################t', param_key)
        #         state_dict.pop(param_key)
        #         self.textencoder.embed_text.load_state_dict(state_dict)
        #     else:
        #         print('#################################3m', param_key)
        #         self.motionencoder.load_state_dict(state_dict)

    def forward(self, batch, do_inference = False):

        clip_denoised = False  # FIXME - hardcoded
        
        device = next(self.parameters()).device
        motion, cond = batch['model_input']

        # Set device for model inputs
        motion = motion.to(device)
        cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

        # Calculate random timesteps
        timesteps, weights = self.schedule_sampler.sample(motion.shape[0], device)

        if not do_inference:
            # Get Clip embeddings for text
            enc_text = self.text_encoder_forward(cond['y']['text'])

            # training        
            losses, model_out, x_t = self.diffusion.training_losses(self.motionencoder, motion, timesteps, model_kwargs={'enc_text': enc_text, 'y': cond['y']})
            
            # Due to different collate_fn, revert the dimensions of output as needed by render_animations
            datastruct_from_text = self.transforms.Datastruct(features=torch.transpose(model_out.squeeze(2), 1, 2))
            output_of_x_t = datastruct_from_text

        else:
            # inference

            # Set masking 0 for inference and get Clip embeddings for text
            train_cond_mask_prob = self.textencoder.cond_mask_prob
            self.textencoder.cond_mask_prob = 0
            enc_text = self.text_encoder_forward(cond['y']['text'])
            self.textencoder.cond_mask_prob = train_cond_mask_prob

            # Hard coded timesteps to the most noised step
            timesteps = torch.tensor([self.diffusion_steps-1]*len(enc_text), device=timesteps.device)

            # Currently done without ddim
            _, single_step_out, x_t = self.diffusion.training_losses(self.motionencoder, motion, timesteps, model_kwargs={'enc_text': enc_text, 'y': cond['y']})
            inference_out = self.diffusion.p_sample_loop(self.motionencoder,
                                            motion.shape,
                                            clip_denoised=clip_denoised,
                                            model_kwargs={'enc_text': enc_text, 'y': cond['y']},
                                            skip_timesteps=0,
                                            init_image=None,
                                            progress=True,
                                            dump_steps=None,
                                            noise=None,
                                            const_noise=False)
            losses = None

            output_of_x_t = self.transforms.Datastruct(features=torch.transpose(single_step_out.squeeze(2), 1, 2))
            datastruct_from_text = self.transforms.Datastruct(features=torch.transpose(inference_out.squeeze(2), 1, 2))

            # Plot graphs for checking input and output vector in model

            # t1 = x_t.detach().cpu().numpy().reshape(-1)
            # plt.hist(t1, bins=50)
            # plt.savefig('logs/' +cond['y']['text'][0] + 'hist.png')
            # plt.show()
            # t1 = []

            # t2 = motion.detach().cpu().numpy().reshape(-1)
            # plt.hist(t2, bins=50)
            # plt.savefig('logs/'+cond['y']['text'][0] + 'motion_hist.png')
            # plt.show()
            # t2 = []

        x_t = self.transforms.Datastruct(features=torch.transpose(x_t.squeeze(2), 1, 2))

        # GT data
        # datastruct_gt = batch['datastruct']
        datastruct_gt = self.transforms.Datastruct(features=torch.transpose(motion.squeeze(2), 1, 2))

        model_out = {
            'pred_data': datastruct_from_text,
            # 'pred_data': output_of_x_t,
            'gt_data': datastruct_gt,
            'losses': losses,
            'x_t': x_t,
            'output_of_x_t': output_of_x_t,
        }
        return model_out

    def get_motion_embeddings(self, autoencoder, features):
        pass

    def get_text_embeddings(self, features):
        pass

    def text_encoder_forward(self, captions):
        
        # Force_mask = true when unconditional generation
        force_mask = False
        enc_text = self.textencoder(captions, force_mask=force_mask)

        return enc_text


    # From tevet/utils/model_util
    def create_gaussian_diffusion(self):

        # default params set from the GuyTevet paper
        predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
        steps = self.diffusion_steps
        scale_beta = 1.  # no scaling
        timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
        learn_sigma = False
        rescale_timesteps = False
        noise_schedule = 'cosine'       # choices=['linear', 'cosine']
        sigma_small = True              # Use smaller sigma values
        lambda_vel = 0
        lambda_rcxyz = 0
        lambda_fc = 0

        betas = gd.get_named_beta_schedule(noise_schedule, steps, scale_beta)
        loss_type = gd.LossType.MSE

        if not timestep_respacing:
            timestep_respacing = [steps]

        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            # model_mean_type=(
            #     gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            # ),
            model_mean_type=gd.ModelMeanType.START_X,
            # model_var_type=(
            #     (
            #         gd.ModelVarType.FIXED_LARGE
            #         if not .sigma_small
            #         else gd.ModelVarType.FIXED_SMALL        # Exact code from GuyTevet repo giving syntaxError
            #     )
            #     if not learn_sigma
            #     else gd.ModelVarType.LEARNED_RANGE
            # ),

            model_var_type = gd.ModelVarType.FIXED_SMALL,      # To avoid syntaxError, taken the one we want.

            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            lambda_vel=lambda_vel,
            lambda_rcxyz=lambda_rcxyz,
            lambda_fc=lambda_fc,
        )
