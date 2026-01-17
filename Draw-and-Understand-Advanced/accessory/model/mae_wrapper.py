"""MAE-style prompt reconstruction for self-supervised pre-training.

This module implements Innovation 5: reducing label dependency by
learning to reconstruct masked visual prompts from their context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEPromptReconstruction(nn.Module):
    """Self-supervised learner that reconstructs masked prompt attributes.

    The module wraps a prompt encoder (e.g. VisualPromptEncoder) and
    applies a Masked Autoencoder objective on the prompt tokens. A
    subset of prompts is masked, and the model is trained to recover
    their coordinates from the latent representation.
    """

    def __init__(self, encoder: nn.Module, decoder_dim: int = 512, mask_ratio: float = 0.5) -> None:
        """Initialize the MAE wrapper.

        Args:
            encoder: Prompt encoder with attribute `embed_dim` and a
                forward signature `(prompts, prompt_types)`.
            decoder_dim: Hidden dimensionality of the reconstruction
                head.
            mask_ratio: Fraction of prompts to mask per sample.
        """
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = float(mask_ratio)

        self.decoder = nn.Sequential(
            nn.Linear(encoder.embed_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, 4),
        )

    def forward(self, prompts: torch.Tensor, prompt_types: torch.Tensor):
        """Compute MAE reconstruction loss for a batch of prompts.

        Args:
            prompts: Float tensor of shape (B, N, D) containing raw
                prompt descriptors (coordinates and optional extras).
            prompt_types: Long tensor of shape (B, N) indicating the
                type of each prompt.

        Returns:
            A tuple `(loss, reconstruction)` where:
            - loss is a scalar tensor representing the averaged MSE
              between reconstructed and ground-truth coordinates of
              masked prompts.
            - reconstruction is a tensor of shape (B, N_masked, 4)
              containing the reconstructed box-like coordinates for
              each sample in the batch.
        """
        device = prompts.device
        batch_size, num_prompts, _ = prompts.shape

        if num_prompts == 0:
            raise ValueError("MAEPromptReconstruction received zero prompts.")

        num_masked = max(1, int(num_prompts * self.mask_ratio))

        latent = self.encoder(prompts, prompt_types)

        all_reconstructions = []
        all_targets = []

        for b in range(batch_size):
            perm = torch.randperm(num_prompts, device=device)
            masked_idx = perm[:num_masked]

            masked_latent = latent[b, masked_idx, :]
            reconstruction_b = self.decoder(masked_latent)
            target_b = prompts[b, masked_idx, :4]

            all_reconstructions.append(reconstruction_b)
            all_targets.append(target_b)

        reconstruction = torch.stack(all_reconstructions, dim=0)
        target = torch.stack(all_targets, dim=0)

        loss = F.mse_loss(reconstruction, target)

        return loss, reconstruction
