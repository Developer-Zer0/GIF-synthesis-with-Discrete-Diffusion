import torch
import torch.nn.functional as F
import numpy as np

def l1_loss(output):
    loss_fn = torch.nn.SmoothL1Loss()
    return loss_fn(pred_feats, gt_feats)

# Diffusion
def compute_dummy(output, loss_opts=None):
    try:
        return torch.mean(output['losses']['commitment_loss'] + output['losses']['recon_loss'])  # Total loss taken and mean over batch dimension
    except IndexError:
        return torch.mean(output['losses'])

# VQ_VAE losses
def compute_codebook_loss(output, loss_opts=None):
    codebook_loss = torch.sum(output['codebook_loss']) # codebook loss is already computed during forward process
    return codebook_loss

def compute_entropy_loss(output, loss_opts=None):
    compute_entropy_loss = torch.sum(output['entropy']) # codebook loss is already computed during forward process
    return compute_entropy_loss

def compute_perplexity_loss(output, loss_opts=None):
    perplexity_loss = torch.sum(output['perplexity']) # codebook loss is already computed during forward process
    return perplexity_loss

_matching_ = {"l_dummy": compute_dummy, "l_codebook": compute_codebook_loss, "l_entropy": compute_entropy_loss, "l_perplexity": compute_perplexity_loss,}

def get_loss_function(ltype):
    return _matching_[ltype]

def get_loss_names():
    return list(_matching_.keys())
