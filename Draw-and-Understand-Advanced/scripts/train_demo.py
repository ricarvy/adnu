"""Minimal training script for validating ADNU modules.

This script exercises the core components of the advanced model on a
synthetic dataset. It is intentionally lightweight and should be viewed
as a structural test rather than the final training pipeline.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from accessory.model.sphinx_v_advanced import SphinxVAdvanced
from accessory.model.mae_wrapper import MAEPromptReconstruction
from accessory.data.dataset_mock import MockDataset, collate_fn


class MockLLaMA(nn.Module):
    """Small stand-in for a LLaMA-like language model.

    The mock model implements the minimal interface required by
    SphinxVAdvanced so that we can verify tensor shapes and gradients
    without loading a full 13B checkpoint.
    """

    def __init__(self, hidden_size: int = 4096) -> None:
        """Initialize the mock LLaMA model.

        Args:
            hidden_size: Dimensionality of the hidden state and input
                embeddings.
        """
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.layers = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_embeds: torch.Tensor, labels=None):
        """Forward method mimicking transformers-style outputs.

        Args:
            inputs_embeds: Tensor of shape (B, T, H).
            labels: Optional tensor used to produce a dummy loss term.

        Returns:
            An object with `.loss` and `.logits` attributes.
        """
        out = self.layers(inputs_embeds)
        loss = None
        if labels is not None:
            loss = out.mean()
        return type("Output", (), {"loss": loss, "logits": out})()


class MockVisualEncoder(nn.Module):
    """Compact visual backbone used for structural testing only."""

    def __init__(self, embed_dim: int = 4096) -> None:
        """Initialize the mock visual encoder.

        Args:
            embed_dim: Dimensionality of the aggregated visual token.
        """
        super().__init__()
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=32, stride=32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input batch of images.

        Args:
            x: Tensor of shape (B, 3, 224, 224).

        Returns:
            Tensor of shape (B, 1, embed_dim) representing a single
            global visual token per image.
        """
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return x.unsqueeze(1)


def train() -> None:
    """Run a short synthetic training loop for regression testing."""
    print("Initializing Draw-and-Understand Advanced Model...")
    embed_dim = 128

    llama = MockLLaMA(hidden_size=embed_dim)
    visual_encoder = MockVisualEncoder(embed_dim=embed_dim)
    model = SphinxVAdvanced(llama, visual_encoder, embed_dim=embed_dim)

    mae_model = MAEPromptReconstruction(model.vp_encoder, decoder_dim=embed_dim // 2)

    dataset = MockDataset(length=10)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Training Loop Demo...")
    model.train()

    for i, (images, prompts, prompt_types, incidence_matrix) in enumerate(dataloader):
        optimizer.zero_grad()

        output, importance_scores = model(
            images=images,
            prompts=prompts,
            prompt_types=prompt_types,
            incidence_matrix=incidence_matrix,
        )

        llm_loss = output.loss if output.loss is not None else importance_scores.mean() * 0.01

        mae_loss, _ = mae_model(prompts, prompt_types)

        total_loss = llm_loss + mae_loss
        total_loss.backward()
        optimizer.step()

        print(f"Batch {i}: total_loss={total_loss.item():.4f}")


if __name__ == "__main__":
    train()
