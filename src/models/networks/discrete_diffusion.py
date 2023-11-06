from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate

import matplotlib.pyplot as plt

class DiscreteDiffusion(nn.Module):
    def __init__(self, textencoder, diffusion_model, **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)
        self.diffusion_model = instantiate(diffusion_model)

        self.device = next(self.parameters()).device

        #self.length_embedder = LengthEmbedder(50, 512)

    def forward(self, batch, autoencoder, length_estimator, do_inference=False):

        x = batch['video'].to(autoencoder.device)
        quant = autoencoder.encode(x)
        quant_flat = quant.view(x.shape[0], -1)

        text_emb = self.textencoder(batch['text'])
        text_emb = text_emb.unsqueeze(1)

        # len_est = length_estimator.text_encoder_forward(batch["cap_lens"], batch["word_embs"], batch["pos_onehot"])
        # len_emb = self.length_embedder(len_est)
        # len_emb = len_emb.unsqueeze(1)
        #
        # text_emb += len_emb

        diffusion_input = {'condition_embed_token': text_emb,
                           'content_token': quant_flat,
                           }

        diffusion_out = self.diffusion_model(diffusion_input, return_loss=True)

        single_step_out = autoencoder.decode(diffusion_out['pred_data'].view(quant.shape))

        if do_inference:

            cf_captions = [''] * len(batch['text'])
            cf_text_emb = self.textencoder(cf_captions)
            cf_text_emb = cf_text_emb.unsqueeze(1)

            inference_out = self.diffusion_model.sample(
                batch['text'],
                None,
                text_emb,
                cf_text_emb,
                content_token=None,
                filter_ratio=0,
            )

            inference_out = autoencoder.decode(inference_out['content_token'].view(quant.shape))

        test = autoencoder.decode(quant)

        if do_inference:
            model_out = {
                'pred_data': inference_out,
                # 'pred_data': single_step_out,
                'pred_single_step': single_step_out,
                'gt_data': x,
                'losses': diffusion_out['loss'],
                'test': test,
            }
        else:
            model_out = {
                'pred_data': single_step_out,
                'gt_data': x,
                'losses': diffusion_out['loss'],
                'test': test,
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


# Very simple Linear implementation. can refer to tevet_transformer_encoder later to improve
# class LengthEmbedder(nn.Module):
#     def __init__(self, latent_dim, len_embed_dim):
#         super().__init__()

#         # latent_dim = 50, len_embed_dim = 512
#         self.len_embed = nn.Sequential(
#             nn.Linear(latent_dim, len_embed_dim),
#             nn.SiLU(),
#             nn.Linear(len_embed_dim, len_embed_dim),
#         )

#     def forward(self, length):
#         return self.len_embed(length)