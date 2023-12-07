import torch
import torch.nn as nn
# from src.models.metrics.metrics import *
from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.datamodules.datasets.ucf101_dataset import preprocess
import numpy as np

class Evaluator:
    def __init__(self, device, videoencoder, checkpoint_paths):
        # self.generator = generator
        self.device = device
        self.videoencoder = instantiate(videoencoder, _recursive_=False)

        self.all_size = 0
        self.matching_score_sum = 0
        self.top_k_count = 0

        self.all_video_embeds_generated = []
        self.all_video_embeds_gt = []
        self.curr_video_embs_generated = None
        self.curr_video_embs_gt = None
        self.curr_text_embs_gt = None

        checkpoint = torch.load(checkpoint_paths,
                                map_location=self.device)
        self.videoencoder.load_state_dict(checkpoint)
        self.videoencoder.to(device)

    def reset(self):
        self.all_video_embeds_gt = []
        self.all_video_embeds_generated = []
        self.curr_video_embs_generated = None
        self.curr_video_embs_gt = None
        self.curr_text_embs_gt = None
        self.all_size = 0
        self.matching_score_sum = 0
        self.top_k_count = 0

    def push_vals(self, batch, batch_idx, outputs):
        target_resolution = 224

        gt_data = batch['video'].clone().permute(0, 2, 3, 4, 1).cpu().numpy()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        gt_data = gt_data * std + mean

        gt_data = (gt_data * 255).astype('uint8')
        gt_data = torch.from_numpy(gt_data)
        gt_data = torch.stack([preprocess(gt, target_resolution) for gt in gt_data]) * 2
        if gt_data.shape[2] == 8:
            gt_data = torch.repeat_interleave(gt_data, 2, dim=2)
        elif gt_data.shape[2] == 4:
            gt_data = torch.repeat_interleave(gt_data, 4, dim=2)

        pred_data = outputs.clone().permute(0, 2, 3, 4, 1).cpu().numpy()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        pred_data = pred_data * std + mean

        pred_data = (pred_data * 255).astype('uint8')
        pred_data = torch.from_numpy(pred_data)
        pred_data = torch.stack([preprocess(pred, target_resolution) for pred in pred_data]) * 2
        if pred_data.shape[2] == 8:
            pred_data = torch.repeat_interleave(pred_data, 2, dim=2)
        elif pred_data.shape[2] == 4:
            pred_data = torch.repeat_interleave(pred_data, 4, dim=2)

        self.push_generated_outputs(pred_data.to(self.device))
        self.push_gt(gt_data.to(self.device))
        # self.push_text(batch)
        # self.update_r_precision()
        # self.update_multimodality(batch, batch_idx, generator)

    def push_generated_outputs(self, outputs):
        self.curr_video_embs_generated = self.videoencoder(outputs)
        self.all_video_embeds_generated.append(self.curr_video_embs_generated.cpu())

    def push_gt(self, features):
        self.curr_video_embs_gt = self.videoencoder(features)
        self.all_video_embeds_gt.append(self.curr_video_embs_gt.cpu())

    def push_text(self, batch):
        self.curr_text_embs_gt = self.motionencoder.get_text_embeddings(batch["word_embs"],
                                                                        batch["pos_onehot"],
                                                                        batch["cap_lens"],
                                                                        batch["orig_length"])

    def evaluate_metrics(self, val_dataset, generator):
        generated_activations = np.concatenate(self.all_video_embeds_generated, axis=0)
        gt_activations = np.concatenate(self.all_video_embeds_gt, axis=0)
        generated_activations = np.reshape(generated_activations, (generated_activations.shape[0], -1))
        gt_activations = np.reshape(gt_activations, (gt_activations.shape[0], -1))
        fvd = self.evaluate_fvd(generated_activations, gt_activations)
        # diversity = self.evaluate_diversity(generated_activations)
        # matching_score, r_precision = self.evaluate_r_precision()
        # multimodality = self.evaluate_multimodality(val_dataset, generator)
        metrics = {
            'fvd': fvd,
            # 'diversity': diversity,
            # 'matching_score': matching_score,
            # 'R-precision-Top-1': r_precision[0],
            # 'R-precision-Top-2': r_precision[1],
            # 'R-precision-Top-3': r_precision[2],
            # 'multimodality': multimodality,
        }
        return metrics

    def evaluate_fvd(self, generated_activations, gt_activations):
        # gen_mu, gen_cov = calculate_activation_statistics(generated_activations)
        # gt_mu, gt_cov = calculate_activation_statistics(gt_activations)
        # fvd = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
        fvd = frechet_distance(generated_activations, gt_activations)
        return fvd


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1) # unbiased estimate
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd

    # def evaluate_diversity(self, activations):
    #     if activations.shape[0] > self.diversity_times:
    #         diversity = calculate_diversity(activations, self.diversity_times)
    #     else:
    #         diversity = 0
    #     return diversity

    # def update_r_precision(self):
    #     # Calculating Matching and R-precision in every step. Totaled at epoch_end
    #     t1 = self.curr_text_embs_gt.cpu().numpy()
    #     t2 = self.curr_motion_embs_gt.cpu().numpy()
    #     dist_mat = euclidean_distance_matrix(self.curr_text_embs_gt.cpu().numpy(),
    #                                          self.curr_motion_embs_generated.cpu().numpy())
    #     self.matching_score_sum += dist_mat.trace()

    #     argsmax = np.argsort(dist_mat, axis=1)
    #     top_k_mat = calculate_top_k(argsmax, top_k=3)
    #     self.top_k_count += top_k_mat.sum(axis=0)

    #     self.all_size += self.curr_text_embs_gt.shape[0]

    # def evaluate_r_precision(self):
    #     matching_score = self.matching_score_sum / self.all_size
    #     r_precision = self.top_k_count / self.all_size
    #     return matching_score, r_precision

    # # def update_multimodality(self, batch, batch_idx, generator):
    # #     ele_list = []
    # #     for i in range(self.batch_size):
    # #         if batch_idx*self.batch_size + i in self.mm_idxs:
    # #             ele_list.append(batch_idx*self.batch_size + i)
    # #
    # #         # if self.mm_idxs[self.mm_num_now] == batch_idx:
    # #             # sample = batch[i].unsqueeze(0)
    # #             # for key, val in batch[i].items():
    # #             #     sample[key] =
    # #     for t in range(self.mm_num_repeat):
    # #         outputs = generator(batch, do_inference=True)
    # #         for ele in ele_list:
    # #             self.mm_generated_motions.append(outputs['pred_data'].features[ele])
    # #             self.mm_m_lens.append(batch["length"][ele])
    # #         # self.mm_generated_motions.append({
    # #         #     'motion': outputs['pred_data'][0].cpu().numpy(),
    # #         #     'length': sample[0]["length"]
    # #         # })
    # #     self.mm_num_now = len(self.mm_generated_motions)

    # def evaluate_multimodality(self, datamodule, generator):
    #     val_dataset = datamodule.val_dataloader().dataset
    #     all_mm_generated_motions = []
    #     mm_idxs = np.random.choice(len(val_dataset), self.mm_num_samples, replace=False)
    #     mm_idxs = np.sort(mm_idxs)
    #     mm_num_now = 0
    #     dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=True, collate_fn=datamodule.dataloader_options['collate_fn'])
    #     for batch_idx, batch in tqdm(enumerate(dataloader)):
    #         if batch_idx == mm_idxs[mm_num_now]:
    #             batch_mm_motions = []
    #             for t in range(self.mm_num_repeat):
    #                 outputs = generator(batch, do_inference=True)
    #                 batch_mm_motions.append({
    #                     'motion': outputs['pred_data'].features[0].cpu().numpy(),
    #                     'length': batch["length"][0],
    #                 })
    #             all_mm_generated_motions.append(batch_mm_motions)
    #             mm_num_now = min(self.mm_num_samples-1, mm_num_now+1)

    #     all_mm_motion_embeddings = []
    #     for batch_mm_motion in all_mm_generated_motions:
    #         motions = []
    #         m_lens = []
    #         for mm_motion in batch_mm_motion:
    #             m_lens.append(mm_motion['length'])
    #             motion = mm_motion['motion']
    #             motion = motion[None, :]
    #             motions.append(motion)
    #         m_lens = np.array(m_lens, dtype=np.int)
    #         motions = np.concatenate(motions, axis=0)
    #         sort_indx = np.argsort(m_lens)[::-1].copy()
    #         m_lens = m_lens[sort_indx]
    #         motions = torch.tensor(motions[sort_indx]).to(self.device)
    #         all_mm_motion_embeddings.append(self.get_motion_embeddings(motions, m_lens).unsqueeze(0))

    #     mm_motion_embeddings = torch.cat(all_mm_motion_embeddings, dim=0).cpu().numpy()
    #     multimodality = calculate_multimodality(mm_motion_embeddings, self.mm_num_times)

    #     # for idx in range(len(self.mm_generated_motions)):
    #     #     mm_motion_embeddings.append(self.get_motion_embeddings(self.mm_generated_motions[idx], self.mm_m_lens[idx]))
    #     # mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
    #     # multimodality = calculate_multimodality(mm_motion_embeddings, self.mm_num_times)
    #     return multimodality
