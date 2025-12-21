#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSIS MODULE - COMPLETE RE-EXPORT
=====================================
Makes ALL analysis functions available from single import.

✅ FIXED: All analysis methods now properly exported and documented
"""



import logging

logger = logging.getLogger(__name__)

# ============================================================================
# MAIN ENTRY POINT - ORCHESTRATOR
# ============================================================================

from experiments.core.analysis.analysis_orchestrator import (
    analyze_training_results,
)

# ============================================================================
# METRICS & DATA STRUCTURES
# ============================================================================

from experiments.core.analysis.analysis_metrics import (
    ExperimentSummary,
    ProblemStats,
    AnalysisPhase,
)

# ============================================================================
# INDIVIDUAL ANALYSIS FUNCTIONS - NOW ALL EXPORTED
# ============================================================================

# Training Analysis
from experiments.core.analysis.analysis_training import (
    _analyze_per_problem_stats,
    _analyze_overall_stats,
    _detect_convergence,
)

# Component Analysis
from experiments.core.analysis.analysis_components import (
    analyze_component_trajectories,
)

# Feature Analysis
from experiments.core.analysis.analysis_features import (
    analyze_feature_reward_correlation,
    analyze_feature_importance_from_decisions,
)

# Quality Analysis
from experiments.core.analysis.analysis_quality import (
    analyze_bisimulation_preservation,
)

# Decision Analysis
from experiments.core.analysis.analysis_decisions import (
    analyze_causal_alignment,
    analyze_transition_explosion_risk,
    analyze_gnn_decision_quality,
)

# Safety Analysis
from experiments.core.analysis.analysis_safety import (
    analyze_dead_end_creation,
)

# Validation
from experiments.core.analysis.analysis_validation import (
    generate_literature_alignment_report,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Main orchestrator
    "analyze_training_results",

    # Metrics
    "ExperimentSummary",
    "ProblemStats",
    "AnalysisPhase",

    # Training Analysis
    "_analyze_per_problem_stats",
    "_analyze_overall_stats",
    "_detect_convergence",

    # Component Analysis
    "analyze_component_trajectories",

    # Feature Analysis
    "analyze_feature_reward_correlation",
    "analyze_feature_importance_from_decisions",

    # Quality Analysis
    "analyze_bisimulation_preservation",

    # Decision Analysis
    "analyze_causal_alignment",
    "analyze_transition_explosion_risk",
    "analyze_gnn_decision_quality",

    # Safety Analysis
    "analyze_dead_end_creation",

    # Validation
    "generate_literature_alignment_report",
]


def list_available_analyses():
    """Print all available analysis functions."""
    analyses = [
        ("Core Training", [
            "analyze_training_results - Main orchestrator",
            "_analyze_per_problem_stats - Per-problem breakdown",
            "_analyze_overall_stats - Aggregate statistics",
            "_detect_convergence - Training convergence detection",
        ]),
        ("Component Analysis", [
            "analyze_component_trajectories - Reward component evolution",
        ]),
        ("Feature Analysis", [
            "analyze_feature_reward_correlation - Feature-reward relationships",
            "analyze_feature_importance_from_decisions - GNN feature importance",
        ]),
        ("Quality Analysis", [
            "analyze_bisimulation_preservation - H* preservation (Greedy algorithm)",
        ]),
        ("Decision Analysis", [
            "analyze_causal_alignment - Merge ordering strategy",
            "analyze_transition_explosion_risk - Transition control learning",
            "analyze_gnn_decision_quality - GNN decision accuracy",
        ]),
        ("Safety Analysis", [
            "analyze_dead_end_creation - Dead-end prevention",
        ]),
        ("Validation", [
            "generate_literature_alignment_report - Helmert et al. 2014 & Nissim et al. 2011 checklist",
        ]),
    ]

    print("\n" + "=" * 80)
    print("AVAILABLE ANALYSIS FUNCTIONS")
    print("=" * 80 + "\n")

    for category, functions in analyses:
        print(f"📊 {category}:")
        for func in functions:
            print(f"   • {func}")
        print()

    print("=" * 80 + "\n")