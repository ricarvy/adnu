"""Reinforcement Learning based Prompt Optimizer (Experimental).

This module implements a Policy Gradient agent that learns to select
the optimal subset of visual prompts to maximize downstream task reward
(e.g., IoU or QA accuracy).

NOTE: This is an experimental module for exploration and is not part
of the main ADNU training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


class PromptPolicyNetwork(nn.Module):
    """Actor network that outputs selection probabilities for prompts."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, N, D) embedding of prompts.
        Returns:
            probs: (B, N, 1) probability of selecting each prompt.
        """
        return self.net(state)


class RLPromptSelector:
    """Agent that uses REINFORCE to optimize prompt selection."""
    
    def __init__(self, embed_dim: int, learning_rate: float = 1e-3):
        self.policy = PromptPolicyNetwork(embed_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = 0.99  # Discount factor (not strictly needed for one-step MDP)

    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions (selection masks) based on policy.
        
        Args:
            state: (B, N, D) prompt embeddings.
            
        Returns:
            action: (B, N) binary mask.
            log_probs: (B, N) log probabilities of the actions.
        """
        probs = self.policy(state).squeeze(-1)  # (B, N)
        
        # Bernoulli sampling
        m = torch.distributions.Bernoulli(probs)
        action = m.sample()
        
        log_probs = m.log_prob(action)
        return action, log_probs

    def update(self, log_probs: torch.Tensor, rewards: torch.Tensor):
        """Update policy using REINFORCE.
        
        Args:
            log_probs: (B, N) log probabilities of taken actions.
            rewards: (B,) scalar reward per sample.
        """
        # Expand rewards to match log_probs shape
        rewards = rewards.unsqueeze(1).expand_as(log_probs)
        
        # Policy Gradient Loss: - E[log(pi(a|s)) * R]
        loss = -(log_probs * rewards).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
