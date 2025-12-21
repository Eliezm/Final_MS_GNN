#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUALIZATION MODULE - Complete with Generalization & Curriculum plots
========================================================================
All plotting functionality, organized by research perspective.

Exports main public API for backward compatibility.
"""

# Import main orchestration function
from experiments.core.visualization.orchestrator import generate_all_plots

# Import individual plot generators (for direct use if needed)
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
)
from experiments.core.visualization.plots_09_literature import plot_literature_alignment

# Training diagnostics (plots_10)
from experiments.core.visualization.plots_10_training_diagnostics import (
    plot_policy_entropy_evolution,
    plot_value_loss_evolution,
    plot_gradient_health,
    plot_inference_performance,
    plot_graph_compression,
)

# Generalization analysis (plots_11)
from experiments.core.visualization.plots_11_generalization import (
    plot_performance_by_problem_size,
    plot_seen_vs_unseen_gap,
    plot_training_size_effect,
    plot_complexity_correlation,
    plot_generalization_heatmap,
)

# Curriculum learning analysis (plots_12)
from experiments.core.visualization.plots_12_curriculum import (
    plot_curriculum_phase_transitions,
    plot_knowledge_transfer_analysis,
    plot_domain_transfer_results,
    plot_curriculum_vs_direct_training,
    plot_phase_difficulty_progression,
)

# Utilities
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib,
    validate_data,
    safe_mean_and_rolling,
    create_plots_directory,
    save_plot_safely,
)

__all__ = [
    # Main entry point
    "generate_all_plots",

    # Training dynamics (plots 01-04)
    "plot_learning_curves",
    "plot_component_trajectories",
    "plot_component_stability",
    "plot_merge_quality_heatmap",

    # Feature analysis (plot 05)
    "plot_feature_importance",

    # Quality analysis (plot 06)
    "plot_bisimulation_preservation",

    # Safety (plot 07)
    "plot_dead_end_analysis",

    # Transitions (plots 08-09)
    "plot_label_reduction_impact",
    "plot_transition_explosion",

    # Decisions (plots 10-11)
    "plot_causal_alignment",
    "plot_gnn_decision_quality",

    # Baselines (plot 12)
    "plot_merge_quality_distribution",

    # Comparison (plots 13-16 in orchestrator)

    # Literature (plot 17)
    "plot_literature_alignment",

    # Training diagnostics (plots 18-22)
    "plot_policy_entropy_evolution",
    "plot_value_loss_evolution",
    "plot_gradient_health",
    "plot_inference_performance",
    "plot_graph_compression",

    # Generalization analysis (plots 23-27)
    "plot_performance_by_problem_size",
    "plot_seen_vs_unseen_gap",
    "plot_training_size_effect",
    "plot_complexity_correlation",
    "plot_generalization_heatmap",

    # Curriculum learning (plots 28-32)
    "plot_curriculum_phase_transitions",
    "plot_knowledge_transfer_analysis",
    "plot_domain_transfer_results",
    "plot_curriculum_vs_direct_training",
    "plot_phase_difficulty_progression",

    # Utilities
    "setup_matplotlib",
    "validate_data",
    "safe_mean_and_rolling",
    "create_plots_directory",
    "save_plot_safely",
]