"""Synthetic dataset for validating ADNU modules in isolation.

This dataset is not used for final experiments, but provides a
deterministic way to exercise all components of the advanced model
without relying on the full MDVP data pipeline.
"""

import torch
from torch.utils.data import Dataset


class MockDataset(Dataset):
    """Generate random images and heterogeneous prompts for debugging.

    Each sample contains a fixed number of prompts with mixed types
    (point, box, polygon) and a trivial hypergraph structure. The
    distribution is intentionally simple but covers all code paths in
    the encoder and reasoning modules.
    """

    def __init__(self, length: int = 100, polygon_points: int = 32) -> None:
        """Initialize the synthetic dataset.

        Args:
            length: Number of synthetic samples.
            polygon_points: Number of points used to parameterize
                polygon prompts.
        """
        self.length = length
        self.polygon_points = polygon_points

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> dict:
        """Generate a single synthetic sample.

        Args:
            idx: Sample index (unused, generation is random).

        Returns:
            A dictionary with keys:
            - image: (3, 224, 224) tensor.
            - prompts: (N, D_max) tensor of prompt descriptors.
            - prompt_types: (N,) tensor of type identifiers.
            - incidence_matrix: (N, N) tensor encoding a simple
              hypergraph structure.
        """
        image = torch.randn(3, 224, 224)

        num_prompts = 5
        prompt_types = torch.randint(0, 3, (num_prompts,))

        d_max = self.polygon_points * 2
        prompts = torch.zeros(num_prompts, d_max)

        for i, p_type in enumerate(prompt_types):
            if p_type == 0:
                prompts[i, :2] = torch.rand(2)
            elif p_type == 1:
                x1y1 = torch.rand(2) * 0.5
                wh = torch.rand(2) * 0.5
                x2y2 = (x1y1 + wh).clamp(max=1.0)
                prompts[i, :4] = torch.cat([x1y1, x2y2], dim=0)
            else:
                prompts[i, : self.polygon_points * 2] = torch.rand(self.polygon_points * 2)

        incidence_matrix = torch.eye(num_prompts)

        return {
            "image": image,
            "prompts": prompts,
            "prompt_types": prompt_types,
            "incidence_matrix": incidence_matrix,
        }


def collate_fn(batch):
    """Collate a list of synthetic samples into a mini-batch.

    Args:
        batch: List of per-sample dictionaries as returned by
            MockDataset.__getitem__.

    Returns:
        A tuple `(images, prompts, prompt_types, incidence_matrix)`
        where each element is a batched tensor.
    """
    images = torch.stack([item["image"] for item in batch])
    prompts = torch.stack([item["prompts"] for item in batch])
    prompt_types = torch.stack([item["prompt_types"] for item in batch])
    incidence_matrix = torch.stack([item["incidence_matrix"] for item in batch])

    return images, prompts, prompt_types, incidence_matrix
