#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSIS MODULE - Backward Compatibility Wrapper
=================================================
Imports all analysis components for unified public API.
"""

# Import orchestrator (main entry point)
from experiments.core.analysis.analysis_orchestrator import analyze_training_results

# Import data structures
from experiments.core.analysis.analysis_metrics import (
    ExperimentSummary,
    ProblemStats,
    AnalysisPhase,
)

# Import all specialized analyzers (for advanced users)
from experiments.core.analysis.analysis_training import (
    _analyze_per_problem_stats,
    _analyze_overall_stats,
    _detect_convergence,
)

from experiments.core.analysis.analysis_components import (
    analyze_component_trajectories,
)

from experiments.core.analysis.analysis_features import (
    analyze_feature_reward_correlation,
    analyze_feature_importance_from_decisions,
)

from experiments.core.analysis.analysis_quality import (
    analyze_bisimulation_preservation,
)

from experiments.core.analysis.analysis_safety import (
    analyze_dead_end_creation,
)

from experiments.core.analysis.analysis_decisions import (
    analyze_causal_alignment,
    analyze_transition_explosion_risk,
    analyze_gnn_decision_quality,
)

from experiments.core.analysis.analysis_validation import (
    generate_literature_alignment_report,
)

__all__ = [
    # Main entry point
    "analyze_training_results",
    "ExperimentSummary",

    # Specialized analyzers
    "analyze_component_trajectories",
    "analyze_feature_reward_correlation",
    "analyze_feature_importance_from_decisions",
    "analyze_bisimulation_preservation",
    "analyze_dead_end_creation",
    "analyze_causal_alignment",
    "analyze_transition_explosion_risk",
    "analyze_gnn_decision_quality",
    "generate_literature_alignment_report",
]