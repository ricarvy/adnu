"""Dynamic gating mechanism for visual prompts.

This module implements Innovation 2: adaptive importance weighting of
heterogeneous prompts to resolve box-versus-point performance inversion
in dense multi-prompt scenarios.
"""

import torch
import torch.nn as nn


class PromptGating(nn.Module):
    """Compute content-aware and sparse importance weights over prompts.

    The gating network predicts a scalar importance score for each
    prompt embedding. Scores are used both for continuous scaling and
    for optional Top-K sparsification, which encourages the model to
    rely on a small subset of informative prompts and reduces visual
    hallucination.
    """

    def __init__(self, embed_dim: int = 4096, top_k: "int | None" = 5) -> None:
        """Initialize the prompt gating module.

        Args:
            embed_dim: Dimensionality of the input prompt embeddings.
            top_k: Optional number of most important prompts to retain
                per sample. If None, only soft gating is applied.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, prompt_embeddings: torch.Tensor):
        """Apply dynamic gating to a set of prompt embeddings.

        Args:
            prompt_embeddings: Float tensor of shape (B, N, D)
                containing per-prompt representations.

        Returns:
            A tuple `(gated_embeddings, scores)` where:
            - gated_embeddings has shape (B, N, D) and contains the
              scaled embeddings after soft and optional sparse gating.
            - scores has shape (B, N, 1) and contains raw importance
              scores in the range (0, 1).
        """
        scores = self.gate_net(prompt_embeddings)

        weighted_embeddings = prompt_embeddings * scores

        if self.top_k is not None and self.top_k < prompt_embeddings.shape[1]:
            flat_scores = scores.squeeze(-1)
            top_scores, top_indices = torch.topk(flat_scores, k=self.top_k, dim=1)

            mask = torch.zeros_like(flat_scores)
            mask.scatter_(1, top_indices, 1.0)
            mask = mask.unsqueeze(-1)

            weighted_embeddings = weighted_embeddings * mask

        return weighted_embeddings, scores
