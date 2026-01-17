"""High-level ADNU model wrapper around SPHINX-V components.

This module assembles all ADNU innovations on top of a SPHINX-V style
vision-language backbone:
- Innovation 1: visual prompt shape generalization.
- Innovation 2: dynamic prompt gating.
- Innovation 4: hypergraph-based multi-prompt reasoning.
"""

import torch
import torch.nn as nn
from typing import Optional

from .visual_prompt_encoder import VisualPromptEncoder
from .gating import PromptGating, BaseGating
from .hypergraph import HyperGraphPromptEncoder, BaseHyperGraphEncoder


class SphinxVAdvanced(nn.Module):
    """Composite model that injects ADNU modules into a SPHINX-V backbone.

    The class does not implement a full training loop; instead it
    provides reusable forward primitives that can be integrated with
    the training and inference pipelines of the original
    Draw-and-Understand codebase.
    """

    def __init__(
        self,
        llama_model: nn.Module,
        visual_encoder: nn.Module,
        embed_dim: int = 4096,
        gating: Optional[BaseGating] = None,
        hypergraph: Optional[BaseHyperGraphEncoder] = None,
        use_hypergraph: bool = True,
    ) -> None:
        """Initialize the advanced SPHINX-V model.

        Args:
            llama_model: Language model instance (e.g. LLaMA) providing
                a `config.hidden_size` attribute and a standard
                `forward(inputs_embeds, ...)` method.
            visual_encoder: Visual backbone mapping input images to a
                sequence of visual tokens.
            embed_dim: Internal embedding dimension of visual prompts.
        """
        super().__init__()
        self.llama = llama_model
        self.visual_encoder = visual_encoder

        self.vp_encoder = VisualPromptEncoder(embed_dim)

        if gating is None:
            self.gating = PromptGating(embed_dim)
        else:
            self.gating = gating

        if not use_hypergraph:
            self.hypergraph = None
        elif hypergraph is None:
            self.hypergraph = HyperGraphPromptEncoder(embed_dim)
        else:
            self.hypergraph = hypergraph

        self.prompt_to_llm = nn.Linear(embed_dim, llama_model.config.hidden_size)

    def encode_prompts(
        self,
        images: torch.Tensor,
        prompts: torch.Tensor,
        prompt_types: torch.Tensor,
        incidence_matrix=None,
    ):
        """Encode images and prompts into LLM-ready embeddings.

        Args:
            images: Float tensor of shape (B, C, H, W).
            prompts: Float tensor of shape (B, N, D) containing raw
                prompt descriptors.
            prompt_types: Long tensor of shape (B, N) with prompt
                type identifiers.
            incidence_matrix: Optional tensor of shape (B, N, E)
                defining the hypergraph structure used for multi-prompt
                reasoning. If None, only dynamic gating is applied.

        Returns:
            A tuple `(visual_tokens, prompt_tokens, importance_scores)`
            where:
            - visual_tokens: (B, T_v, H) encodes the global visual
              context from the backbone.
            - prompt_tokens: (B, N, H) are advanced prompt tokens mapped
              into the LLM hidden space.
            - importance_scores: (B, N, 1) are raw gating scores.
        """
        visual_feats = self.visual_encoder(images)

        prompt_feats = self.vp_encoder(prompts, prompt_types)

        gated_feats, importance_scores = self.gating(prompt_feats)

        if incidence_matrix is not None and self.hypergraph is not None:
            gated_feats = self.hypergraph(gated_feats, incidence_matrix)

        prompt_tokens = self.prompt_to_llm(gated_feats)

        return visual_feats, prompt_tokens, importance_scores

    def forward(
        self,
        images: torch.Tensor,
        prompts: torch.Tensor,
        prompt_types: torch.Tensor,
        incidence_matrix=None,
        text_embeds=None,
        labels=None,
    ):
        """Run a full multimodal forward pass through the LLM.

        Args:
            images: Float tensor of shape (B, C, H, W).
            prompts: Float tensor of shape (B, N, D).
            prompt_types: Long tensor of shape (B, N).
            incidence_matrix: Optional hypergraph incidence of shape
                (B, N, E). If omitted, no hypergraph reasoning is
                applied.
            text_embeds: Optional tensor of shape (B, T_text, H_llm)
                representing token embeddings of the textual prompt.
                If None, only visual and prompt tokens are used.
            labels: Optional tensor passed to the underlying LLM for
                supervised training.

        Returns:
            The output object returned by the underlying language model,
            typically containing `.loss` and `.logits` fields.
        """
        visual_tokens, prompt_tokens, importance_scores = self.encode_prompts(
            images, prompts, prompt_types, incidence_matrix
        )

        sequences = [visual_tokens, prompt_tokens]
        if text_embeds is not None:
            sequences.append(text_embeds)
        inputs_embeds = torch.cat(sequences, dim=1)

        output = self.llama(inputs_embeds=inputs_embeds, labels=labels)

        return output, importance_scores
