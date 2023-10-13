import torch
import torch.nn as nn
from hydra.utils import instantiate
import pytorch_lightning as pl
from torch import Tensor


class VQVAEGuo(pl.LightningModule):
    def __init__(self, encoder, decoder, quantizer, pose_dim, transforms, checkpoint_path, **kwargs):
        super().__init__()
        self.motionencoder = instantiate(encoder, input_size=pose_dim)
        self.motiondecoder = instantiate(decoder, pose_dim=pose_dim)
        # self.discriminator = instantiate(discriminator, input_size=pose_dim)
        self.quantizer = instantiate(quantizer)

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

        checkpoint = torch.load(checkpoint_path, map_location=self.motionencoder.device)
        self.motionencoder.load_state_dict(checkpoint['vq_encoder'])
        self.motiondecoder.load_state_dict(checkpoint['vq_decoder'])
        self.quantizer.load_state_dict(checkpoint['quantizer'])

    def forward(self, batch, do_inference=False):

        motions = batch['datastruct'].features
        motions = motions.detach().to(self.motionencoder.device).float()
        # print(self.motions.shape)
        pre_latents = self.motionencoder(motions[..., :-4])

        # indices = self.quantizer.map2index(pre_latents)
        # vq_latents = self.quantizer.get_codebook_entry(indices)

        # print(self.pre_latents.shape)
        embedding_loss, vq_latents, quants, perplexity = self.quantizer(pre_latents)
        # print(self.vq_latents.shape)
        recon_motions = self.motiondecoder(vq_latents)

        # GT data
        datastruct_gt = batch['datastruct']
        datastruct = self.transforms.Datastruct(features=recon_motions)

        model_out = {
            'pred_data': datastruct,
            # 'pred_data': output_of_x_t,
            'gt_data': datastruct_gt,
            'losses': embedding_loss,
            'perplexity': perplexity,
        }
        return model_out
