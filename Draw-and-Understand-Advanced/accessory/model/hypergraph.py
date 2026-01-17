"""Hypergraph-based relation encoder for visual prompts.

This module implements Innovation 4: multi-prompt and multi-target
relation modelling via a Hyper-Graph Prompt Encoder.
"""

import torch
import torch.nn as nn


class HyperGraphPromptEncoder(nn.Module):
    """Encode higher-order relations between prompts using a hypergraph.

    Nodes correspond to individual visual prompts. Hyperedges connect
    groups of prompts that are spatially or semantically related. The
    encoder performs HyperSAGE-style message passing between nodes and
    hyperedges to enrich each prompt representation with contextual
    information.
    """

    def __init__(self, embed_dim: int = 4096) -> None:
        """Initialize the hypergraph encoder.

        Args:
            embed_dim: Dimensionality of the prompt embeddings to be
                processed and returned.
        """
        super().__init__()
        self.embed_dim = embed_dim

        self.node_to_edge = nn.Linear(embed_dim, embed_dim)
        self.edge_to_node = nn.Linear(embed_dim, embed_dim)
        self.update_fn = nn.GRUCell(embed_dim, embed_dim)

    def forward(self, prompt_embeddings: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        """Apply one round of hypergraph message passing.

        Args:
            prompt_embeddings: Float tensor of shape (B, N, D)
                containing node-level prompt representations.
            incidence_matrix: Float or binary tensor of shape
                (B, N, E) defining membership of nodes in E
                hyperedges. Non-zero entries indicate that a node
                participates in a particular hyperedge.

        Returns:
            Tensor of shape (B, N, D) with updated node embeddings.
        """
        batch_size, num_nodes, _ = prompt_embeddings.shape

        edge_features = torch.bmm(
            incidence_matrix.transpose(1, 2), self.node_to_edge(prompt_embeddings)
        )

        node_updates = torch.bmm(incidence_matrix, self.edge_to_node(edge_features))

        prompt_embeddings_flat = prompt_embeddings.reshape(-1, self.embed_dim)
        node_updates_flat = node_updates.reshape(-1, self.embed_dim)

        new_embeddings_flat = self.update_fn(node_updates_flat, prompt_embeddings_flat)
        new_embeddings = new_embeddings_flat.view(batch_size, num_nodes, self.embed_dim)

        return new_embeddings

    @staticmethod
    def build_spatial_incidence_matrix(boxes: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Construct a hyperedge incidence matrix from box overlaps.

        Each node corresponds to a bounding box. For every node i, we
        create a hyperedge whose members are all nodes whose IoU with i
        exceeds the given threshold (including the node itself). This
        produces an incidence matrix where each column corresponds to a
        center node and captures its local neighborhood.

        Args:
            boxes: Float tensor of shape (B, N, 4) with coordinates in
                (x1, y1, x2, y2) format.
            threshold: IoU threshold used to decide whether two boxes
                should be connected in the same hyperedge.

        Returns:
            Float tensor of shape (B, N, N) where entry (b, i, j) is
            1.0 if node i participates in the hyperedge centered at
            node j, and 0.0 otherwise.
        """
        B, N, _ = boxes.shape
        device = boxes.device

        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        x1 = x1.unsqueeze(2)
        y1 = y1.unsqueeze(2)
        x2 = x2.unsqueeze(2)
        y2 = y2.unsqueeze(2)

        xx1 = torch.maximum(x1, x1.transpose(1, 2))
        yy1 = torch.maximum(y1, y1.transpose(1, 2))
        xx2 = torch.minimum(x2, x2.transpose(1, 2))
        yy2 = torch.minimum(y2, y2.transpose(1, 2))

        inter_w = (xx2 - xx1).clamp_min(0)
        inter_h = (yy2 - yy1).clamp_min(0)
        inter_area = inter_w * inter_h

        area = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
        area_i = area
        area_j = area.transpose(1, 2)
        union = area_i + area_j - inter_area
        iou = inter_area / union.clamp_min(1e-6)

        mask = iou >= threshold

        diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        mask = mask | diag_mask

        incidence = mask.to(torch.float32)

        return incidence
