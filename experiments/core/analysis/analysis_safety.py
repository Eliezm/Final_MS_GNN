#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFETY ANALYSIS - Problem soundness validation
==============================================
Ensures the learned merge strategy maintains solvability.

Analyzes:
- Dead-end creation over time
- Solvability maintenance
- Problem state
"""

from typing import Dict, List, Any
import numpy as np

from experiments.core.logging import EpisodeMetrics


def analyze_dead_end_creation(
        training_log: List[EpisodeMetrics],
        output_dir=None,
) -> Dict[str, Any]:
    """
    Analyze dead-end prevention during training.

    Returns metrics on how well GNN avoids creating dead-ends.
    """

    if not training_log:
        return {}

    successful_log = [m for m in training_log if m.error is None]

    if not successful_log:
        return {}

    dead_end_penalties = [m.penalty_dead_end for m in successful_log]
    solvability_penalties = [m.penalty_solvability_loss for m in successful_log]

    return {
        'avg_dead_end_penalty': float(np.mean(dead_end_penalties)),
        'max_dead_end_penalty': float(np.max(dead_end_penalties)),
        'min_dead_end_penalty': float(np.min(dead_end_penalties)),
        'dead_end_episodes': sum(1 for p in dead_end_penalties if p > 0.1),
        'avg_solvability_penalty': float(np.mean(solvability_penalties)),
    }