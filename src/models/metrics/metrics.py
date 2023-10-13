from typing import List

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric
import numpy as np
from scipy import linalg

from DetUtil.rotation_conversions import matrix_of_angles
from src.datamodules.datasets.transforms.joints2jfeats import Rifke
from src.utils.torch_utils import remove_padding

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)

def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)

def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat

def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    t1 = diff.dot(diff)
    t2 = np.sum(mu1-mu2)
    t3 = np.sum(mu1)
    t4 = np.sum(mu2)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

class ComputeMetrics(Metric):
    def __init__(self, jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if jointstype != "mmm":
            raise NotImplementedError("This jointstype is not implemented.")

        super().__init__()
        self.jointstype = jointstype
        self.rifke = Rifke(jointstype=jointstype,
                           normalization=False)

        self.force_in_meter = force_in_meter
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # APE
        self.add_state("APE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        self.add_state("APE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.APE_metrics = ["APE_root", "APE_traj", "APE_pose", "APE_joints"]

        # AVE
        self.add_state("AVE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        self.add_state("AVE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.AVE_metrics = ["AVE_root", "AVE_traj", "AVE_pose", "AVE_joints"]

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics

        self.add_state("fid-val", default=torch.tensor(30.), dist_reduce_fx="sum")
        self.add_state("matching_score-val", default=torch.tensor(30.), dist_reduce_fx="sum")
        self.add_state("diversity-val", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("R-precision-Top-1-val", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("R-precision-Top-2-val", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("R-precision-Top-3-val", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.add_state("fid-test", default=torch.tensor(30.), dist_reduce_fx="sum")
        self.add_state("matching_score-test", default=torch.tensor(30.), dist_reduce_fx="sum")
        self.add_state("diversity-test", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("R-precision-Top-1-test", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("R-precision-Top-2-test", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("R-precision-Top-3-test", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self):
        count = self.count
        APE_metrics = {metric: getattr(self, metric) / count for metric in self.APE_metrics}

        # Compute average of APEs
        APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count

        count_seq = self.count_seq
        AVE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.AVE_metrics}

        # Compute average of AVEs
        AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq

        return {**APE_metrics, **AVE_metrics}

    def update(self, output):
        jts_text = output['pred_data'].joints
        jts_ref = output['gt_data'].joints
        lengths = output['length']

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        jts_text, poses_text, root_text, traj_text = self.transform(jts_text, lengths)
        jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref, lengths)

        for i in range(len(lengths)):
            self.APE_root += l2_norm(root_text[i], root_ref[i], dim=1).sum()
            self.APE_pose += l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
            self.APE_traj += l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
            self.APE_joints += l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

            root_sigma_text = variance(root_text[i], lengths[i], dim=0)
            root_sigma_ref = variance(root_ref[i], lengths[i], dim=0)
            self.AVE_root += l2_norm(root_sigma_text, root_sigma_ref, dim=0)

            traj_sigma_text = variance(traj_text[i], lengths[i], dim=0)
            traj_sigma_ref = variance(traj_ref[i], lengths[i], dim=0)
            self.AVE_traj += l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

            poses_sigma_text = variance(poses_text[i], lengths[i], dim=0)
            poses_sigma_ref = variance(poses_ref[i], lengths[i], dim=0)
            self.AVE_pose += l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

            jts_sigma_text = variance(jts_text[i], lengths[i], dim=0)
            jts_sigma_ref = variance(jts_ref[i], lengths[i], dim=0)
            self.AVE_joints += l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)

    def transform(self, joints: Tensor, lengths):
        features = self.rifke(joints)

        ret = self.rifke.extract(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = ret

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the local poses
        poses_local = rearrange(poses_features, "... (joints xyz) -> ... joints xyz", xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]], rotations)
        poses = torch.stack((poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local, rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # get the root joint
        root = torch.cat((trajectory[..., :, [0]],
                          root_y[..., None],
                          trajectory[..., :, [1]]), dim=-1)

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        # put back the root joint y
        poses[..., 0, 1] = root_y

        # Add the trajectory globally
        poses[..., [0, 2]] += trajectory[..., None, :]

        if self.force_in_meter:
            # return results in meters
            return (remove_padding(poses / 1000, lengths),
                    remove_padding(poses_local / 1000, lengths),
                    remove_padding(root / 1000, lengths),
                    remove_padding(trajectory / 1000, lengths))
        else:
            return (remove_padding(poses, lengths),
                    remove_padding(poses_local, lengths),
                    remove_padding(root, lengths),
                    remove_padding(trajectory, lengths))
