"""Visualization tools for ADNU analysis.

This module provides plotting utilities to visualize:
1. Hypergraph structures (nodes and hyperedges).
2. Attention maps from HyperGAT.
3. Gating score distributions.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

# Try importing networkx for graph plotting, handle if missing
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class Visualizer:
    """Analysis and plotting suite."""

    def __init__(self, save_dir: str = "vis_outputs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_hypergraph(self, incidence_matrix: torch.Tensor, save_name: str = "hypergraph.png"):
        """Visualize the hypergraph structure using NetworkX.
        
        Args:
            incidence_matrix: (N, E) binary matrix.
            save_name: Output filename.
        """
        if not HAS_NETWORKX:
            print("NetworkX not installed, skipping graph plot.")
            return

        incidence = incidence_matrix.cpu().numpy()
        num_nodes, num_edges = incidence.shape
        
        # Create a bipartite graph for visualization
        B = nx.Graph()
        node_ids = [f"N{i}" for i in range(num_nodes)]
        edge_ids = [f"E{j}" for j in range(num_edges)]
        
        B.add_nodes_from(node_ids, bipartite=0)
        B.add_nodes_from(edge_ids, bipartite=1)
        
        for i in range(num_nodes):
            for j in range(num_edges):
                if incidence[i, j] > 0.5:
                    B.add_edge(node_ids[i], edge_ids[j])
                    
        plt.figure(figsize=(10, 8))
        pos = nx.bipartite_layout(B, node_ids)
        
        nx.draw_networkx_nodes(B, pos, nodelist=node_ids, node_color='lightblue', node_size=500, label='Prompts')
        nx.draw_networkx_nodes(B, pos, nodelist=edge_ids, node_color='orange', node_size=500, label='Hyperedges')
        nx.draw_networkx_edges(B, pos)
        nx.draw_networkx_labels(B, pos)
        
        plt.title("Hypergraph Structure (Bipartite View)")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def plot_gating_distribution(self, scores: torch.Tensor, save_name: str = "gating_dist.png"):
        """Plot histogram of importance scores.
        
        Args:
            scores: (B, N) tensor of scores.
        """
        scores_np = scores.detach().cpu().numpy().flatten()
        
        plt.figure(figsize=(8, 6))
        plt.hist(scores_np, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.title("Distribution of Prompt Importance Scores")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.5)
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def plot_attention_map(self, attn_weights: torch.Tensor, save_name: str = "attn_map.png"):
        """Heatmap of attention weights.
        
        Args:
            attn_weights: (N, N) matrix.
        """
        w = attn_weights.detach().cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(w, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title("HyperGAT Attention Map")
        plt.xlabel("Source")
        plt.ylabel("Target")
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()
