"""Auxiliary loss functions for ADNU training.

This module provides a suite of advanced loss functions designed to 
regularize the latent space of visual prompts and enforce topological 
consistency in hypergraph learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Contrastive loss to maximize mutual information between related prompts.
    
    Used to pull representations of the same object (e.g., box vs point vs polygon)
    closer together in the latent space.
    """
    
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B*N, D) flattened features.
            labels: (B*N) object identity labels.
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Mask for positive pairs (same label)
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-contrast
        mask_diag = torch.eye(features.shape[0], device=features.device)
        mask = mask - mask_diag
        
        # Log-sum-exp for denominator
        exp_sim = torch.exp(similarity_matrix) * (1 - mask_diag)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        return loss


class TopologicalConsistencyLoss(nn.Module):
    """Enforces consistency between geometric overlap and feature similarity.
    
    If two prompts have high IoU (geometric overlap), their latent representations
    should be similar (high cosine similarity).
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, embeddings: torch.Tensor, iou_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, N, D)
            iou_matrix: (B, N, N)
        """
        # Feature similarity
        embeddings_norm = F.normalize(embeddings, dim=-1)
        sim_matrix = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2)) # (B, N, N)
        
        # We want sim_matrix to correlate with iou_matrix
        # MSE Loss between similarity and IoU
        loss = F.mse_loss(sim_matrix, iou_matrix)
        return loss


class SparsityLoss(nn.Module):
    """Encourages sparse activation of gating scores."""
    
    def __init__(self, target_sparsity: float = 0.1) -> None:
        super().__init__()
        self.target_sparsity = target_sparsity

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (B, N, 1) sigmoid outputs from gating network.
        """
        avg_activation = scores.mean()
        # KL Divergence between avg_activation and target_sparsity
        # KL(rho || rho_hat) = rho * log(rho/rho_hat) + (1-rho) * log((1-rho)/(1-rho_hat))
        rho = self.target_sparsity
        rho_hat = avg_activation
        
        # Clamp for numerical stability
        rho_hat = torch.clamp(rho_hat, min=1e-5, max=1-1e-5)
        
        loss = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return loss
