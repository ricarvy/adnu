"""Visual prompt encoder for ADNU.

This module implements Innovation 1: visual prompt shape generalization.
It unifies point, box and free-form polygon prompts into a single token
space that can be consumed by the language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualPromptEncoder(nn.Module):
    """Encode heterogeneous visual prompts into a unified embedding space.

    The encoder supports three primary visual prompt types:
    - Point prompts: single 2D coordinates (x, y).
    - Box prompts: axis-aligned bounding boxes (x1, y1, x2, y2).
    - Polygon prompts: free-form shapes represented by a fixed number of
      sampled contour points (polygon_points).

    Each prompt type is first mapped into a type-specific latent, then
    projected into a common embedding space with a learned type embedding.
    """

    def __init__(self, embed_dim: int = 4096, polygon_points: int = 32) -> None:
        """Initialize the visual prompt encoder.

        Args:
            embed_dim: Dimension of the output token embeddings.
            polygon_points: Number of 2D points used to represent a polygon
                prompt; the effective polygon input dimension is
                `polygon_points * 2`.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.polygon_points = polygon_points

        self.point_encoder = nn.Linear(2, embed_dim)
        self.box_encoder = nn.Linear(4, embed_dim)

        self.polygon_encoder = nn.Sequential(
            nn.Linear(polygon_points * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 0: Point, 1: Box, 2: Polygon
        self.type_embedding = nn.Embedding(3, embed_dim)

    def forward(self, prompts: torch.Tensor, prompt_types: torch.Tensor) -> torch.Tensor:
        """Encode a batch of heterogeneous prompts.

        Args:
            prompts: Float tensor of shape (B, N, D_max), where D_max is
                the maximum prompt dimensionality in the batch. The
                leading dimensions for each type are interpreted as:
                - Point: first 2 values are (x, y).
                - Box: first 4 values are (x1, y1, x2, y2).
                - Polygon: first `polygon_points * 2` values are
                  flattened (x, y) coordinates.
            prompt_types: Long tensor of shape (B, N) with values in
                {0: point, 1: box, 2: polygon}.

        Returns:
            Tensor of shape (B, N, embed_dim) containing the encoded
            prompt tokens.
        """
        device = prompts.device
        batch_size, num_prompts, _ = prompts.shape

        embeddings = prompts.new_zeros(batch_size, num_prompts, self.embed_dim)

        point_mask = prompt_types == 0
        box_mask = prompt_types == 1
        polygon_mask = prompt_types == 2

        if point_mask.any():
            point_inputs = prompts[point_mask][..., :2]
            type_embed = self.type_embedding.weight[0].to(device)
            embeddings[point_mask] = self.point_encoder(point_inputs) + type_embed

        if box_mask.any():
            box_inputs = prompts[box_mask][..., :4]
            type_embed = self.type_embedding.weight[1].to(device)
            embeddings[box_mask] = self.box_encoder(box_inputs) + type_embed

        if polygon_mask.any():
            poly_dim = self.polygon_points * 2
            polygon_raw = prompts[polygon_mask]

            if polygon_raw.shape[-1] < poly_dim:
                padded = polygon_raw.new_zeros(polygon_raw.shape[0], poly_dim)
                padded[:, : polygon_raw.shape[-1]] = polygon_raw
                polygon_inputs = padded
            else:
                polygon_inputs = polygon_raw[..., :poly_dim]

            type_embed = self.type_embedding.weight[2].to(device)
            embeddings[polygon_mask] = self.polygon_encoder(polygon_inputs) + type_embed

        return embeddings

    @staticmethod
    def encode_fourier_descriptors(polygon_points: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Convert polygon points to truncated Fourier descriptors.

        This utility implements a classic shape descriptor that is
        translation- and scale-normalized and can be used to build
        more invariant prompt features.

        Args:
            polygon_points: Float tensor of shape (M, 2) or (B, M, 2)
                containing ordered polygon vertices in image coordinates.
            k: Number of low-frequency coefficients to retain for the
                descriptor (excluding the DC component).

        Returns:
            Tensor of shape (M, 4k) or (B, M, 4k) containing the
            concatenated real and imaginary parts of the first k
            non-DC Fourier coefficients.
        """
        if polygon_points.dim() == 2:
            polygon_points = polygon_points.unsqueeze(0)

        coords = polygon_points.to(torch.float32)

        center = coords.mean(dim=1, keepdim=True)
        centered = coords - center

        scale = centered.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True).values.clamp_min(1e-6)
        normalized = centered / scale

        complex_seq = torch.complex(normalized[..., 0], normalized[..., 1])

        fft_coeffs = torch.fft.fft(complex_seq, dim=1)

        coeffs = fft_coeffs[:, 1 : k + 1]
        real = coeffs.real
        imag = coeffs.imag
        descriptor = torch.cat([real, imag], dim=-1)

        if descriptor.shape[0] == 1:
            descriptor = descriptor.squeeze(0)

        return descriptor
