#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPONENT ANALYSIS - Reward component evolution
===============================================
Analyzes each reward component independently:
- H* Preservation
- Transition Control
- Operator Projection
- Label Combinability
- Bonus Signals
"""

from typing import List, Dict, Any
import numpy as np
import warnings

from experiments.core.logging import EpisodeMetrics


def analyze_component_trajectories(
        training_log: List[EpisodeMetrics],
        output_dir=None,
) -> Dict[str, Any]:
    """
    Analyze how each reward component evolves.

    Returns:
        {
            'component_trajectories': {component_name: [values]},
            'stability_metrics': {component_name: stability_score},
            'degradation_patterns': {phase: analysis},
        }
    """

    if not training_log:
        return {}

    component_names = [
        'component_h_preservation',
        'component_transition_control',
        'component_operator_projection',
        'component_label_combinability',
        'component_bonus_signals',
    ]

    # ====================================================================
    # EXTRACT TRAJECTORIES
    # ====================================================================

    component_trajectories = {name: [] for name in component_names}

    for metrics in training_log:
        for component_name in component_names:
            value = getattr(metrics, component_name, 0.0)
            component_trajectories[component_name].append(value)

    # ====================================================================
    # COMPUTE STABILITY
    # ====================================================================

    stability_metrics = {}
    for component_name, trajectory in component_trajectories.items():
        if len(trajectory) > 1:
            variance = np.var(trajectory)
            stability = 1.0 / (1.0 + variance)
            stability_metrics[component_name] = float(stability)
        else:
            stability_metrics[component_name] = 1.0

    # ====================================================================
    # ANALYZE DEGRADATION BY PHASE
    # ====================================================================

    def analyze_phase(phase_values, phase_name):
        """Analyze one training phase."""
        if len(phase_values) < 2:
            return None

        early_avg = np.mean(phase_values[:max(1, len(phase_values) // 3)])
        late_avg = np.mean(phase_values[-max(1, len(phase_values) // 3):])
        degradation = early_avg - late_avg
        degradation_pct = (degradation / (abs(early_avg) + 1e-6)) * 100

        return {
            'phase': phase_name,
            'early_avg': float(early_avg),
            'late_avg': float(late_avg),
            'absolute_degradation': float(degradation),
            'percent_degradation': float(degradation_pct),
            'is_degrading': degradation > 0.05,
        }

    n = len(training_log)
    third = n // 3

    degradation_patterns = {}
    overall_trajectory = [m.reward for m in training_log]

    if third > 0:
        degradation_patterns['early_degradation'] = analyze_phase(
            overall_trajectory[:third], 'early'
        )
        degradation_patterns['mid_run_degradation'] = analyze_phase(
            overall_trajectory[third:2 * third], 'mid'
        )
        degradation_patterns['late_run_degradation'] = analyze_phase(
            overall_trajectory[2 * third:], 'late'
        )

    return {
        'component_trajectories': {
            k: [float(v) for v in v_list]
            for k, v_list in component_trajectories.items()
        },
        'stability_metrics': stability_metrics,
        'degradation_patterns': degradation_patterns,
    }