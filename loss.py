import torch
import itertools
from typing import Any, Callable, List, Optional, Tuple

# SNR loss function
def snr_loss(source, estimate):
    signal_power = torch.sum(source ** 2, dim=-1)  + 1e-6
    noise_power = torch.sum((source - estimate) ** 2, dim=-1)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-6))
    return -snr

def si_snr(source, estimate):
    alpha = torch.sum(estimate * source, dim=-1, keepdim=True) / (torch.sum(source**2, dim=-1, keepdim=True) + 1e-6)
    scaled_source = alpha * source + 1e-6
    si_snr = 10 * torch.log10(torch.norm(scaled_source, p=2, dim=-1)**2 / (torch.norm(scaled_source - estimate, p=2, dim=-1)**2 + 1e-6))
    return si_snr

def snr_thresholded(source, estimate, tau_db = 30):
    tau = 10 ** (-tau_db / 10)
    y_squared = torch.sum(estimate * source, dim=-1, keepdim=True)
    noise_squared = torch.sum((estimate - source)**2, dim=-1, keepdim=True)
    snr = 10 * torch.log10(y_squared / (noise_squared + tau * y_squared + 1e-5))
    return -snr

def apply_and_get_mix_matrix(loss_fn: Callable[..., torch.Tensor],
                             reference: torch.Tensor,
                             estimate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    batch, ref_num_sources = reference.shape[0], reference.shape[1]
    est_num_sources = estimate.shape[1]
    device = reference.device

    if ref_num_sources > est_num_sources:
        raise ValueError(
            f'ref_num_sources {ref_num_sources} should be <= est_num_sources {est_num_sources}.')

    idxs = list(itertools.product(range(ref_num_sources), repeat=est_num_sources))
    
    # Vectorized computation of mix matrices
    mix_matrixes = torch.nn.functional.one_hot(torch.tensor(idxs),ref_num_sources).permute(0, 2, 1).float().to(device)

    estimate_mixed = mix_matrixes @ estimate.unsqueeze(1).repeat(1,len(idxs), 1, 1)

    losses = torch.mean(loss_fn(reference.unsqueeze(1), estimate_mixed), dim=-1)

    idx_argmin = torch.argmin(losses, dim=1, keepdim=True)
    loss_best_mixture = losses.gather(1, idx_argmin)

    best_mix_matrixes = mix_matrixes[idx_argmin,...].squeeze(1)

    return loss_best_mixture, best_mix_matrixes

def apply_mixit(loss_fn: Callable[..., torch.Tensor],
                reference: torch.Tensor,
                estimate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    loss_best_mixture, best_mix_matrix = apply_and_get_mix_matrix(loss_fn, reference, estimate)
    estimate_mixed = torch.matmul(best_mix_matrix, estimate)

    return loss_best_mixture, estimate_mixed, best_mix_matrix


def contrastive_loss(x1, x2, label, margin: float = 1.0):

    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    # loss = torch.mean(loss)

    return loss



"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6) # numerical stability ok da chel

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
def labeled_contrastive_accuracy(embeddings, labels):
    """
    embeddings: tensor of shape [batch_size, embedding_dim]
    labels: tensor of shape [batch_size]
    """
    
    # Compute pairwise distances
    dists = torch.cdist(embeddings, embeddings)
    
    # Mask to ignore self-distances
    mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
    dists.masked_fill_(mask, float('inf'))
    
    # Get indices of the minimum distance for each row (excluding the diagonal)
    _, min_indices = dists.min(dim=1)

    # Check if the labels of nearest embeddings match the anchor labels
    matches = (labels == labels[min_indices]).float()

    # Calculate accuracy
    accuracy = matches.mean().item()

    return accuracy

import torch

import torch

def mean_distances(embeddings, labels):
    """
    Compute the mean distance between positive pairs and negative pairs.

    Parameters:
    - embeddings: tensor of shape [batch_size, embedding_dim]
    - labels: tensor of shape [batch_size]

    Returns:
    - mean_positive_distance: average distance between embeddings of the same label
    - mean_negative_distance: average distance between embeddings of different labels
    """
    
    # Compute pairwise distances
    dists = torch.cdist(embeddings, embeddings)

    # Create a matrix where entry (i, j) is 1 if labels[i] == labels[j], else 0
    label_matches = labels.unsqueeze(0) == labels.unsqueeze(1)
    # Set diagonal to False for self-distances
    label_matches.fill_diagonal_(0)
    # Create its negative
    label_non_matches = ~label_matches

    # Compute mean positive distance and mean negative distance
    mean_positive_distance = (dists * label_matches).sum() / (label_matches.sum() + 1e-6)
    mean_negative_distance = (dists * label_non_matches).sum() / (label_non_matches.sum() + 1e-6)

    return mean_positive_distance.item(), mean_negative_distance.item()
