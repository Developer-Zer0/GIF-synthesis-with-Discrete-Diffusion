import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch import Tensor

import clip

class TevetMLP(pl.LightningModule):

    def __init__(self, 
                clip_dim: int = 512,                      # CLIP embedding size
                latent_dim: int = 256,
                cond_mask_prob: float = 0.1,              # Probability of masking the text
                **kwargs) -> None:
        
        super(TevetMLP, self).__init__()

        self.embed_text = nn.Linear(clip_dim, latent_dim)
        print('Loading CLIP...')
        clip_version = 'ViT-B/32'                                 
        self.clip_model = self.load_and_freeze_clip(clip_version)
        self.cond_mask_prob = cond_mask_prob


    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device=next(self.parameters()).device,
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model


    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.cond_mask_prob > 0.:
            # Do this only during training
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    def forward(self, 
            raw_text: list,
            force_mask: bool = False,                   # Mask the conditional text
            ) -> Tensor:

        # encode_text from tevet
        max_text_len = 20  # if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        default_context_length = 77
        context_length = max_text_len + 2               # start_token + 20 + end_token
        assert context_length < default_context_length
        texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(next(self.parameters()).device)
        zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
        texts = torch.cat([texts, zero_pad], dim=1)
        enc_text = self.clip_model.encode_text(texts).float()

        emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        return emb