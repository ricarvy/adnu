"""Dynamic gating mechanism for visual prompts.

This module implements Innovation 2: adaptive importance weighting of
heterogeneous prompts to resolve box-versus-point performance inversion
in dense multi-prompt scenarios. It supports multiple gating strategies
including Top-K, Gumbel-Softmax, and hierarchical gating.
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BaseGating(nn.Module, abc.ABC):
    """Abstract base class for prompt gating strategies."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    @abc.abstractmethod
    def forward(self, prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply gating to embeddings.

        Args:
            prompt_embeddings: (B, N, D) input embeddings.

        Returns:
            Tuple (gated_embeddings, importance_scores).
        """
        pass


class PromptGating(BaseGating):
    """Standard content-aware gating with optional Top-K sparsification (Default).

    The gating network predicts a scalar importance score for each
    prompt embedding. Scores are used both for continuous scaling and
    for optional Top-K sparsification.
    """

    def __init__(self, embed_dim: int = 4096, top_k: Optional[int] = 5) -> None:
        super().__init__(embed_dim)
        self.top_k = top_k
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.gate_net(prompt_embeddings)  # (B, N, 1)
        weighted_embeddings = prompt_embeddings * scores

        if self.top_k is not None and self.top_k < prompt_embeddings.shape[1]:
            flat_scores = scores.squeeze(-1)
            # Use topk to find indices of the k largest values
            _, top_indices = torch.topk(flat_scores, k=self.top_k, dim=1)

            mask = torch.zeros_like(flat_scores)
            mask.scatter_(1, top_indices, 1.0)
            mask = mask.unsqueeze(-1)
            weighted_embeddings = weighted_embeddings * mask

        return weighted_embeddings, scores


class GumbelGating(BaseGating):
    """Differentiable discrete gating using Gumbel-Softmax trick.

    Useful for hard selection during training while maintaining differentiability.
    """

    def __init__(self, embed_dim: int = 4096, temperature: float = 1.0) -> None:
        super().__init__(embed_dim)
        self.temperature = temperature
        self.gate_net = nn.Linear(embed_dim, 2)  # logits for [reject, accept]

    def forward(self, prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B, N, 2)
        logits = self.gate_net(prompt_embeddings)
        
        if self.training:
            # (B, N, 2) - one hot approximation
            soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            # select the 'accept' probability (index 1)
            scores = soft_one_hot[..., 1:2]
        else:
            # Inference: hard argmax
            scores = torch.argmax(logits, dim=-1, keepdim=True).float()

        weighted_embeddings = prompt_embeddings * scores
        return weighted_embeddings, scores


class HierarchicalGating(BaseGating):
    def __init__(self, embed_dim: int = 4096) -> None:
        super().__init__(embed_dim)
        self.global_gate = nn.Linear(embed_dim, 1)
        self.local_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 8),
            nn.ReLU(),
            nn.Linear(embed_dim // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Assume prompt_embeddings contains a global token at index 0 or we pool it
        # Here we use mean pooling for global context
        global_feat = prompt_embeddings.mean(dim=1, keepdim=True) # (B, 1, D)
        global_score = torch.sigmoid(self.global_gate(global_feat)) # (B, 1, 1)
        
        local_scores = self.local_gate(prompt_embeddings) # (B, N, 1)
        
        # Combined score
        final_scores = global_score * local_scores
        
        weighted_embeddings = prompt_embeddings * final_scores
        return weighted_embeddings, final_scores


class MultiHeadPromptGating(BaseGating):
    def __init__(self, embed_dim: int = 4096, num_heads: int = 4) -> None:
        super().__init__(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_prompts, _ = prompt_embeddings.shape

        q = self.q_proj(prompt_embeddings)
        k = self.k_proj(prompt_embeddings)
        v = self.v_proj(prompt_embeddings)

        q = q.view(batch_size, num_prompts, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_prompts, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_prompts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).reshape(batch_size, num_prompts, self.embed_dim)
        raw_scores = self.out_proj(context)
        gate_scores = torch.sigmoid(raw_scores)

        weighted_embeddings = prompt_embeddings * gate_scores
        return weighted_embeddings, gate_scores
