import torch
import torch.nn.functional as F
import numpy as np

def get_recon_data(output, loss_opts, pred_data_type='pred_data'):
    loss_type = loss_opts.get('loss_feat_type', 'jfeats')
    pred_feats = output[pred_data_type].get_feats(loss_type)
    gt_feats = output['gt_data'].get_feats(loss_type)
    return pred_feats, gt_feats

def l1_loss(output, loss_opts, pred_type):
    pred_feats, gt_feats = get_recon_data(output, loss_opts, pred_type)
    loss_fn = torch.nn.SmoothL1Loss()
    return loss_fn(pred_feats, gt_feats)

def compute_l1_data_loss(output, loss_opts):
    return l1_loss(output, loss_opts, 'pred_data')

def compute_l1_m2m_loss(output, loss_opts):
    return l1_loss(output, loss_opts, 'pred_m2m')

# seq2seq
def compute_mse_loss(output, loss_opts):
    pred_feats, gt_feats = get_recon_data(output, loss_opts)
    return F.mse_loss(pred_feats, gt_feats)

def compute_motion_continuous_loss(output, loss_opts):
    pred_feats, gt_feats = get_recon_data(output, loss_opts)
    n_element = pred_feats.numel()
    diff = [abs(pred_feats[:, n, :] - pred_feats[:, n-1, :]) for n in range(1, pred_feats.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    return cont_loss

# Guo
def compute_l_smooth_loss(output, loss_opts):
    pred_snippets = output['snippets_motion']
    loss_fn = torch.nn.L1Loss()
    return loss_fn(pred_snippets[:, 1:], pred_snippets[:, :-1])

def compute_l_sparsity_loss(output, loss_opts):
    pred_snippets = output['snippets_motion']
    return torch.mean(torch.abs(pred_snippets))

def compute_guo_kl_loss(output, loss_opts):
    sigma1 = output["logvars_post"].mul(0.5).exp()
    sigma2 = output["logvars_pri"].mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(output["logvars_post"]) + (output["mus_post"] - output["mus_pri"]) ** 2) / (
            2 * torch.exp(output["logvars_pri"])) - 1 / 2
    return kld.sum() / output["mus_post"].shape[0]

def compute_l1_snip2snip_loss(output, loss_opts):
    pred_snippets = output['pred_snippets']
    gt_snippets = output['gt_snippets']
    loss_fn = torch.nn.SmoothL1Loss()
    return loss_fn(pred_snippets, gt_snippets)

def compute_len_crossentropy_loss(output, loss_opts):
    pred_dis = output['pred_dis']
    gt_dis = output['gt_dis']
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(pred_dis, gt_dis)

def compute_matching_loss(output, loss_opts):
    output1 = output['pred_data']
    output2 = output['gt_data']
    batch_size = output2.shape[0]
    '''Positive pairs'''
    pos_labels = torch.zeros(batch_size).to(output2.device)
    loss_pos = compute_contrastive_loss(output2, output1, pos_labels)
    '''Negative Pairs, shifting index'''
    neg_labels = torch.ones(batch_size).to(output2.device)
    shift = np.random.randint(0, batch_size - 1)
    new_idx = np.arange(shift, batch_size + shift) % batch_size
    output1 = output1.clone()[new_idx]
    loss_neg = compute_contrastive_loss(output2, output1, neg_labels)
    return loss_pos + loss_neg

def compute_contrastive_loss(output1, output2, label):
    margin = 10
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

# GuyTevet Diffusion
def compute_dummy(output, loss_opts):
    try:
        return torch.mean(output['losses']['commitment_loss'] + output['losses']['recon_loss'])  # Total loss taken and mean over batch dimension
    except IndexError:
        return torch.mean(output['losses'])

# VQ_VAE losses
def compute_codebook_loss(output, loss_opts):
    codebook_loss = torch.sum(output['codebook_loss']) # codebook loss is already computed during forward process
    return codebook_loss

def compute_entropy_loss(output, loss_opts):
    compute_entropy_loss = torch.sum(output['entropy']) # codebook loss is already computed during forward process
    return compute_entropy_loss

def compute_perplexity_loss(output, loss_opts):
    perplexity_loss = torch.sum(output['perplexity']) # codebook loss is already computed during forward process
    return perplexity_loss

_matching_ = {"l1_data": compute_l1_data_loss, "l1_m2m": compute_l1_m2m_loss, "mse_data": compute_mse_loss, "cont_data": compute_motion_continuous_loss,
              "l_smooth": compute_l_smooth_loss, "l_sparsity": compute_l_sparsity_loss,
              "guo_kl": compute_guo_kl_loss, "l1_snip2snip": compute_l1_snip2snip_loss, "len_crossentropy": compute_len_crossentropy_loss, "l_matching": compute_matching_loss,
              "l_dummy": compute_dummy, "l_codebook": compute_codebook_loss, "l_entropy": compute_entropy_loss, "l_perplexity": compute_perplexity_loss,}

def get_loss_function(ltype):
    return _matching_[ltype]

def get_loss_names():
    return list(_matching_.keys())
