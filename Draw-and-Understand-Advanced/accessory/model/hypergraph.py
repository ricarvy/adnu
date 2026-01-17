"""Hypergraph-based relation encoder for visual prompts.

This module implements Innovation 4: multi-prompt and multi-target
relation modelling via a Hyper-Graph Prompt Encoder. It supports both
HyperSAGE-style message passing and Attention-based HyperGAT.
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class BaseHyperGraphEncoder(nn.Module, abc.ABC):
    """Abstract base class for Hypergraph Encoders."""
    
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    @abc.abstractmethod
    def forward(self, prompt_embeddings: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        """Apply message passing."""
        pass

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

        x1 = x1.unsqueeze(2)  # (B, N, 1)
        y1 = y1.unsqueeze(2)
        x2 = x2.unsqueeze(2)
        y2 = y2.unsqueeze(2)

        # Intersection area
        xx1 = torch.maximum(x1, x1.transpose(1, 2))
        yy1 = torch.maximum(y1, y1.transpose(1, 2))
        xx2 = torch.minimum(x2, x2.transpose(1, 2))
        yy2 = torch.minimum(y2, y2.transpose(1, 2))

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        # Union area
        area = (x2 - x1) * (y2 - y1)  # (B, N, 1)
        area = area.squeeze(-1)       # (B, N)
        union = area.unsqueeze(1) + area.unsqueeze(2) - inter

        iou = inter / (union + 1e-6)

        incidence = (iou > threshold).float()
        return incidence


class HyperGraphPromptEncoder(BaseHyperGraphEncoder):
    """Standard HyperSAGE-style encoder (Default).
    
    Nodes correspond to individual visual prompts. Hyperedges connect
    groups of prompts that are spatially or semantically related. The
    encoder performs HyperSAGE-style message passing between nodes and
    hyperedges to enrich each prompt representation with contextual
    information.
    """

    def __init__(self, embed_dim: int = 4096) -> None:
        super().__init__(embed_dim)
        self.node_to_edge = nn.Linear(embed_dim, embed_dim)
        self.edge_to_node = nn.Linear(embed_dim, embed_dim)
        self.update_fn = nn.GRUCell(embed_dim, embed_dim)

    def forward(self, prompt_embeddings: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = prompt_embeddings.shape

        # Node -> Hyperedge Aggregation (Mean pooling implied by matrix mult if binary)
        # H = H^T * W_ne(X)
        # incidence_matrix is (B, N_nodes, N_edges) if strictly following math, 
        # but here we use square (B, N, N) where each node centers a hyperedge.
        edge_features = torch.bmm(
            incidence_matrix.transpose(1, 2), self.node_to_edge(prompt_embeddings)
        )

        # Hyperedge -> Node Aggregation
        # X' = H * W_en(E)
        node_updates = torch.bmm(incidence_matrix, self.edge_to_node(edge_features))

        prompt_embeddings_flat = prompt_embeddings.reshape(-1, self.embed_dim)
        node_updates_flat = node_updates.reshape(-1, self.embed_dim)

        # Gated Update
        new_embeddings_flat = self.update_fn(node_updates_flat, prompt_embeddings_flat)
        new_embeddings = new_embeddings_flat.view(batch_size, num_nodes, self.embed_dim)

        return new_embeddings


class HyperGATPromptEncoder(BaseHyperGraphEncoder):
    def __init__(self, embed_dim: int = 4096, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Attention projections for Node -> Edge
        self.n2e_q = nn.Linear(embed_dim, embed_dim)
        self.n2e_k = nn.Linear(embed_dim, embed_dim)
        self.n2e_v = nn.Linear(embed_dim, embed_dim)
        
        # Attention projections for Edge -> Node
        self.e2n_q = nn.Linear(embed_dim, embed_dim)
        self.e2n_k = nn.Linear(embed_dim, embed_dim)
        self.e2n_v = nn.Linear(embed_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q, k, v: (B, N, H, D)
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, N, H, H) - Wait, dimensions are wrong for standard attn
        # Here we are doing attention over the set defined by the incidence matrix.
        # This is tricky to vectorize efficiently without sparse operations.
        # We will use a simplified global attention modulated by incidence matrix for demonstration.
        
        # Correct approach for dense matrix implementation:
        # We want to aggregate K/V into Q buckets.
        # Q represents the target (Hyperedges), K/V represent the source (Nodes).
        
        # Reshape for multihead
        B, N_target, _ = q.shape
        B, N_source, _ = k.shape
        
        q = q.view(B, N_target, self.num_heads, self.head_dim).transpose(1, 2) # (B, Heads, N_target, HeadDim)
        k = k.view(B, N_source, self.num_heads, self.head_dim).transpose(1, 2) # (B, Heads, N_source, HeadDim)
        v = v.view(B, N_source, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) * self.scale # (B, Heads, N_target, N_source)
        
        if mask is not None:
            # mask is (B, N_target, N_source)
            mask = mask.unsqueeze(1) # (B, 1, N_target, N_source)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ v # (B, Heads, N_target, HeadDim)
        
        out = out.transpose(1, 2).reshape(B, N_target, -1)
        return out

    def forward(self, prompt_embeddings: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        # incidence_matrix: (B, Nodes, Edges). 
        # In our case, edges are centered at nodes, so (B, Nodes_as_nodes, Nodes_as_edges)
        
        # 1. Node -> Hyperedge Aggregation
        # Target: Edges (N_edges), Source: Nodes (N_nodes)
        # We want to compute Edge features.
        
        # Mask: For each edge, which nodes are connected?
        # incidence_matrix[b, i, j] = 1 means Node i is in Edge j.
        # So for Edge j, we attend to Nodes i where mask[b, j, i] is 1.
        # Transpose incidence to get (B, Edges, Nodes)
        n2e_mask = incidence_matrix.transpose(1, 2)
        
        # Edges don't have features yet, usually we initialize them or use the center node's feature.
        # Here we initialize Edge queries from the nodes that center them.
        edge_queries = self.n2e_q(prompt_embeddings)
        node_keys = self.n2e_k(prompt_embeddings)
        node_values = self.n2e_v(prompt_embeddings)
        
        edge_features = self._attention(edge_queries, node_keys, node_values, mask=n2e_mask)
        edge_features = self.norm1(edge_features + prompt_embeddings) # Residual from center node
        
        # 2. Hyperedge -> Node Aggregation
        # Target: Nodes, Source: Edges
        # We want to update Node features.
        # Mask: For each node, which edges is it part of?
        # incidence_matrix[b, i, j] = 1 means Node i is in Edge j.
        # So for Node i, we attend to Edges j where mask[b, i, j] is 1.
        e2n_mask = incidence_matrix
        
        node_queries = self.e2n_q(prompt_embeddings)
        edge_keys = self.e2n_k(edge_features)
        edge_values = self.e2n_v(edge_features)
        
        node_updates = self._attention(node_queries, edge_keys, edge_values, mask=e2n_mask)
        node_features = self.norm2(prompt_embeddings + node_updates)
        
        # 3. FFN
        out = self.norm3(node_features + self.ffn(node_features))
        
        return out


class StackedHyperGraphEncoder(BaseHyperGraphEncoder):
    def __init__(
        self,
        embed_dim: int = 4096,
        num_layers: int = 2,
        use_attention: bool = False,
    ) -> None:
        super().__init__(embed_dim)
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        layers: List[BaseHyperGraphEncoder] = []
        for _ in range(num_layers):
            if use_attention:
                layers.append(HyperGATPromptEncoder(embed_dim=embed_dim))
            else:
                layers.append(HyperGraphPromptEncoder(embed_dim=embed_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, prompt_embeddings: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        out = prompt_embeddings
        for layer in self.layers:
            out = layer(out, incidence_matrix)
        return out
