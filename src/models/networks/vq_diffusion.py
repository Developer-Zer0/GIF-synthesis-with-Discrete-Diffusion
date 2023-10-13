from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate

import matplotlib.pyplot as plt

# Very simple Linear implementation. can refer to tevet_transformer_encoder later to improve
class LengthEmbedder(nn.Module):
    def __init__(self, latent_dim, len_embed_dim):
        super().__init__()

        # latent_dim = 50, len_embed_dim = 512
        self.len_embed = nn.Sequential(
            nn.Linear(latent_dim, len_embed_dim),
            nn.SiLU(),
            nn.Linear(len_embed_dim, len_embed_dim),
        )

    def forward(self, length):
        return self.len_embed(length)

class VQDiffusion(nn.Module):
    def __init__(self, textencoder, diffusion_model, transforms, **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)
        self.diffusion_model = instantiate(diffusion_model)

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct
        self.device = next(self.parameters()).device

        #self.length_embedder = LengthEmbedder(50, 512)

    def forward(self, batch, autoencoder, length_estimator, do_inference=False):

        # Shuhong's vq-vae
        # motion = batch['datastruct'].features.float()
        # quant = autoencoder.encode_feature(motion)
        # quant = quant.squeeze(2)

        # Guo's vq-vae
        motion = batch['datastruct'].features
        motion = motion.detach().to(autoencoder.motionencoder.device).float()
        pre_latents = autoencoder.motionencoder(motion[..., :-4])
        _, _, quant, _ = autoencoder.quantizer(pre_latents)
        quant = quant.view(motion.shape[0], -1)

        text_emb = self.textencoder(batch['caption'])
        text_emb = text_emb.unsqueeze(1)

        # len_est = length_estimator.text_encoder_forward(batch["cap_lens"], batch["word_embs"], batch["pos_onehot"])
        # len_emb = self.length_embedder(len_est)
        # len_emb = len_emb.unsqueeze(1)
        #
        # text_emb += len_emb

        diffusion_input = {'condition_embed_token': text_emb,
                           'content_token': quant,
                           }

        diffusion_out = self.diffusion_model(diffusion_input, return_loss=True)

        # Shuhong's vq-vae
        # single_step_out = diffusion_out['pred_data'].unsqueeze(2)
        # single_step_out = autoencoder.decode_rq_seq(single_step_out)
        # single_step_out = torch.transpose(single_step_out, 1, 2)

        # Guo's vq-vae
        single_step_out = autoencoder.quantizer.get_codebook_entry(diffusion_out['pred_data'])
        single_step_out = autoencoder.motiondecoder(single_step_out)

        single_step_out = self.transforms.Datastruct(features=single_step_out)

        if do_inference:

            cf_captions = [''] * len(batch['caption'])
            cf_text_emb = self.textencoder(cf_captions)
            cf_text_emb = cf_text_emb.unsqueeze(1)

            inference_out = self.diffusion_model.sample(
                batch['caption'],
                None,
                text_emb,
                cf_text_emb,
                content_token=None,
                filter_ratio=0,
            )

            # inference_out = inference_out['content_token'].unsqueeze(2)
            # inference_out = autoencoder.decode_rq_seq(inference_out)
            # inference_out = torch.transpose(inference_out, 1, 2).double()

            inference_out = autoencoder.quantizer.get_codebook_entry(inference_out['content_token'])
            inference_out = autoencoder.motiondecoder(inference_out)

            inference_out = self.transforms.Datastruct(features=inference_out)

        # test = autoencoder.decode_rq_seq(quant.unsqueeze(2))
        # test = torch.transpose(test, 1, 2)
        # datastruct_test = self.transforms.Datastruct(features=test)

        test = autoencoder.quantizer.get_codebook_entry(quant)
        test = autoencoder.motiondecoder(test)
        datastruct_test = self.transforms.Datastruct(features=test)

        # GT data
        datastruct_gt = batch['datastruct']

        if do_inference:
            model_out = {
                'pred_data': inference_out,
                # 'pred_data': single_step_out,
                'pred_single_step': single_step_out,
                'gt_data': datastruct_gt,
                'losses': diffusion_out['loss'],
                'datastruct_test': datastruct_test,
            }
        else:
            model_out = {
                'pred_data': single_step_out,
                'gt_data': datastruct_gt,
                'losses': diffusion_out['loss'],
                'datastruct_test': datastruct_test,
            }

        return model_out

    def get_motion_embeddings(self, autoencoder, features):
        # quant = autoencoder.encode_feature(features)
        poses = torch.transpose(features, 1, 2)
        quant, diff, metrics = autoencoder.encode(poses)
        # datastruct_test = self.transforms.Datastruct(features=test)
        return quant

    def get_text_embeddings(self, features):
        text_emb = self.textencoder(features)
        text_emb = text_emb.unsqueeze(1)
        return text_emb
