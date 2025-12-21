#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKWARD COMPATIBILITY WRAPPER - FIXED
=======================================
Re-exports new modular visualization system with same public API.

✅ FIXED: Handles all edge cases, no silent failures
"""

import logging

logger = logging.getLogger(__name__)

# Re-export everything from new modular system
try:
    from experiments.core.visualization.orchestrator import generate_all_plots
    from experiments.core.visualization.plots_01_learning import plot_learning_curves
    from experiments.core.visualization.plots_02_components import (
        plot_component_trajectories,
        plot_component_stability,
        plot_merge_quality_heatmap,
    )
    from experiments.core.visualization.plots_03_features import plot_feature_importance
    from experiments.core.visualization.plots_04_quality import plot_bisimulation_preservation
    from experiments.core.visualization.plots_05_safety import plot_dead_end_analysis
    from experiments.core.visualization.plots_06_transitions import (
        plot_label_reduction_impact,
        plot_transition_explosion,
    )
    from experiments.core.visualization.plots_07_decisions import (
        plot_causal_alignment,
        plot_gnn_decision_quality,
    )
    from experiments.core.visualization.plots_08_baselines import (
        plot_merge_quality_distribution,
        plot_baseline_comparison,
    )
    from experiments.core.visualization.plots_09_literature import plot_literature_alignment
    from experiments.core.visualization.plotting_utils import (
        setup_matplotlib,
        validate_data,
        safe_mean_and_rolling,
        create_plots_directory,
        save_plot_safely,
    )
except ImportError as e:
    logger.error(f"Failed to import visualization modules: {e}")
    raise

__all__ = [
    "generate_all_plots",
    "plot_learning_curves",
    "plot_component_trajectories",
    "plot_component_stability",
    "plot_merge_quality_heatmap",
    "plot_feature_importance",
    "plot_bisimulation_preservation",
    "plot_dead_end_analysis",
    "plot_label_reduction_impact",
    "plot_transition_explosion",
    "plot_causal_alignment",
    "plot_gnn_decision_quality",
    "plot_merge_quality_distribution",
    "plot_baseline_comparison",
    "plot_literature_alignment",
    "setup_matplotlib",
    "validate_data",
    "safe_mean_and_rolling",
    "create_plots_directory",
    "save_plot_safely",
]