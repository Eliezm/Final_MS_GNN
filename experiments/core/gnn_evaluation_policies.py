#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN EVALUATION POLICIES - Merge decision strategies (FIXED)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class RandomMergePolicy:
    """Random merge strategy - selects merge edges randomly."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.merge_count = 0

    def select_merge(self, obs, num_edges: int = None) -> int:
        """
        ✅ FIX: Accept observation dict, extract num_edges
        """
        # Extract num_edges from observation
        if obs is None:
            num_edges_to_use = 1
        elif isinstance(obs, dict):
            num_edges_to_use = int(obs.get('num_edges', 1))
        else:
            num_edges_to_use = int(num_edges) if num_edges else 1

        num_edges_to_use = max(1, num_edges_to_use)
        action = self.rng.randint(0, num_edges_to_use)
        self.merge_count += 1

        # ✅ FIX: Ensure return is Python int
        return int(action)

    def reset(self):
        """Reset merge counter for new problem."""
        self.merge_count = 0


class GNNMergePolicy:
    """GNN merge policy wrapper - encapsulates a trained GNN model."""

    def __init__(self, model):
        """
        Initialize GNN policy with trained model.

        Args:
            model: Loaded PPO model from stable_baselines3
        """
        self.model = model
        self.decision_count = 0

    def select_merge(self, obs, deterministic: bool = True) -> int:
        """
        ✅ FIX: Properly extract action from model.predict() output

        Args:
            obs: Observation dict from environment
            deterministic: Whether to use deterministic policy

        Returns:
            Action index (Python int, not numpy)
        """
        try:
            # model.predict returns (action_array, _states)
            action_array, _states = self.model.predict(obs, deterministic=deterministic)

            # ✅ FIX 1: Extract scalar from numpy array
            if isinstance(action_array, np.ndarray):
                action = int(action_array.flat[0])  # Use .flat[0] for any shape
            else:
                action = int(action_array)

            self.decision_count += 1

            # ✅ FIX 2: Validate action is Python int
            assert isinstance(action, int) and not isinstance(action, (np.integer, np.ndarray)), \
                f"Action must be Python int, got {type(action)}: {action}"

            return action

        except Exception as e:
            logger.error(f"[GNNPolicy] select_merge failed: {e}")
            logger.error(f"  obs type: {type(obs)}")
            logger.error(f"  obs keys: {obs.keys() if isinstance(obs, dict) else 'not dict'}")
            raise

    def reset(self):
        """Reset decision counter."""
        self.decision_count = 0