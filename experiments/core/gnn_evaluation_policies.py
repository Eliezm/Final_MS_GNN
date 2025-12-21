#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN EVALUATION POLICIES - Merge decision strategies
===================================================
Implements different merge selection policies for M&S evaluation.
"""

import numpy as np


class RandomMergePolicy:
    """
    Random merge strategy - selects merge edges randomly.

    Used as baseline comparison against GNN-guided strategy.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize random merge policy.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.merge_count = 0

    def select_merge(self, edge_logits: np.ndarray, num_edges: int) -> int:
        """
        Select a random merge edge.

        Args:
            edge_logits: Unused (for compatibility with GNN)
            num_edges: Number of available merge edges

        Returns:
            Random edge index to merge
        """
        if num_edges == 0:
            return 0
        action = self.rng.randint(0, num_edges)
        self.merge_count += 1
        return action

    def reset(self):
        """Reset merge counter for new problem."""
        self.merge_count = 0


class GNNMergePolicy:
    """
    GNN merge policy wrapper - encapsulates a trained GNN model.

    Provides interface-compatible predictions for merge selection.
    """

    def __init__(self, model):
        """
        Initialize GNN policy with trained model.

        Args:
            model: Loaded PPO model from stable_baselines3
        """
        self.model = model
        self.decision_count = 0

    def select_merge(self, obs: dict, deterministic: bool = True) -> int:
        """
        Select merge using GNN policy.

        Args:
            obs: Observation from environment
            deterministic: Whether to use deterministic policy

        Returns:
            Action index selected by GNN
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        self.decision_count += 1
        return int(action)

    def reset(self):
        """Reset decision counter."""
        self.decision_count = 0