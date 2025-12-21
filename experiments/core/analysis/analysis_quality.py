#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUALITY ANALYSIS - Heuristic quality metrics
============================================
Measures whether the learned abstraction preserves solution optimality.

Core metric: Bisimulation (h* preservation)
"""

from typing import Dict, List, Any
import numpy as np

from experiments.core.logging import EpisodeMetrics


def analyze_bisimulation_preservation(
        training_log: List[EpisodeMetrics],
        output_dir=None,
) -> Dict[str, Any]:
    """
    Analyze bisimulation preservation (h* protection).

    High h-preservation = abstraction is faithful to original problem.

    Returns:
        {
            'bisimulation_preservation_rate': float,
            'num_episodes_with_preservation': int,
            'avg_min_h_preservation': float,
            'learning_trend': 'improving' or 'declining',
        }
    """

    if not training_log:
        return {}

    successful_log = [m for m in training_log if m.error is None]

    if not successful_log:
        return {}

    # ====================================================================
    # ANALYZE PRESERVATION
    # ====================================================================

    h_pres_values = [m.h_star_preservation for m in successful_log]

    preservation_threshold = 0.95
    preserved = sum(1 for h in h_pres_values if h >= preservation_threshold)

    preservation_rate = preserved / len(h_pres_values)

    # ====================================================================
    # DETECT LEARNING TREND
    # ====================================================================

    if len(h_pres_values) > 50:
        early_avg = np.mean(h_pres_values[:len(h_pres_values) // 3])
        late_avg = np.mean(h_pres_values[-len(h_pres_values) // 3:])
        trend = 'improving' if late_avg > early_avg else 'declining'
    else:
        trend = 'insufficient_data'

    return {
        'bisimulation_preservation_rate': float(preservation_rate),
        'num_episodes_with_preservation': preserved,
        'total_episodes': len(h_pres_values),
        'min_h_preservation_per_episode': h_pres_values,
        'avg_min_h_preservation': float(np.mean(h_pres_values)),
        'learning_trend': trend,
    }