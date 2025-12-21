#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORCHESTRATOR - Main analysis entry point
=========================================
Coordinates all analysis components in research-logical order.

This is the ONLY function called from the experiment pipeline.
All other functions are internal to the analysis module.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict
import numpy as np

from experiments.core.logging import EpisodeMetrics
from experiments.core.analysis.analysis_metrics import (
    ProblemStats, ExperimentSummary
)
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
from experiments.core.analysis.analysis_decisions import (
    analyze_causal_alignment,
    analyze_transition_explosion_risk,
    analyze_gnn_decision_quality,
)
from experiments.core.analysis.analysis_validation import (
    generate_literature_alignment_report,
)
from experiments.core.analysis.analysis_utils import (
    _create_empty_summary,
    _get_start_time,
    _get_end_time,
    _get_duration,
)


def analyze_training_results(
        training_log: List[EpisodeMetrics],
        eval_results: Optional[Union[Dict, List]],
        problem_names: List[str],
        benchmarks: List[tuple],
        experiment_id: str,
        timesteps_per_episode: int = 50,
) -> ExperimentSummary:
    """
    ✅ MAIN ENTRY POINT - Comprehensive training analysis

    Flow (Research Perspective):
    1. Validate inputs
    2. Analyze per-problem performance
    3. Compute overall statistics
    4. Failure taxonomy
    5. Coverage validation
    6. Compute summary

    Args:
        training_log: Episode metrics from training
        eval_results: Evaluation results (dict or list)
        problem_names: Names of training problems
        benchmarks: (domain, problem_file) tuples
        experiment_id: Unique experiment identifier
        timesteps_per_episode: Timesteps per episode

    Returns:
        ExperimentSummary with all analysis results
    """

    # ====================================================================
    # VALIDATION & EARLY RETURN
    # ====================================================================

    if not training_log:
        return _create_empty_summary(experiment_id, len(benchmarks))

    # ====================================================================
    # STEP 1: PER-PROBLEM ANALYSIS
    # ====================================================================

    per_problem_stats, coverage_info = _analyze_per_problem_stats(
        training_log, benchmarks, problem_names
    )

    # ====================================================================
    # STEP 2: OVERALL STATISTICS
    # ====================================================================

    overall_stats = _analyze_overall_stats(
        training_log, per_problem_stats, timesteps_per_episode
    )

    # ====================================================================
    # STEP 3: FAILURE TAXONOMY
    # ====================================================================

    failure_taxonomy = _build_failure_taxonomy(training_log)

    # ====================================================================
    # STEP 4: EVALUATION COMPARISON
    # ====================================================================

    eval_stats = _analyze_evaluation_results(training_log, eval_results)

    # ====================================================================
    # STEP 5: CREATE SUMMARY
    # ====================================================================

    summary = ExperimentSummary(
        # Scale
        num_problems=len(benchmarks),
        num_train_episodes=len(training_log),
        num_failed_episodes=len([m for m in training_log if m.error is not None]),
        total_timesteps=len([m for m in training_log if m.error is None]) * timesteps_per_episode,

        # Timing
        start_time=_get_start_time(training_log),
        end_time=_get_end_time(training_log),
        duration_seconds=_get_duration(training_log),

        # Main results
        avg_reward_over_all=overall_stats['avg_reward'],
        best_reward_over_all=overall_stats['best_reward'],
        worst_reward_over_all=overall_stats['worst_reward'],

        # Per-problem
        per_problem_stats=[s.to_dict() for s in per_problem_stats],

        # Learning quality
        reward_variance=overall_stats['reward_variance'],
        h_preservation_improvement_ratio=overall_stats['h_pres_improvement'],
        solve_rate_improvement=overall_stats['solve_rate_improvement'],
        early_convergence_episodes=min(
            [s.episodes_to_convergence for s in per_problem_stats
             if s.episodes_to_convergence],
            default=0
        ) or 0,

        # Coverage
        problem_coverage_valid=coverage_info['valid'],
        min_problem_coverage_pct=coverage_info['min_coverage'],
        max_problem_coverage_pct=coverage_info['max_coverage'],
        all_problems_trained=coverage_info['all_trained'],

        # Failures
        failure_taxonomy=failure_taxonomy,

        # Resources
        avg_step_time_ms=overall_stats['avg_step_time_ms'],
        avg_peak_memory_mb=overall_stats['avg_peak_memory_mb'],

        # Metadata
        experiment_id=experiment_id,
        training_final_avg=eval_stats['training_final_avg'],
        evaluation_avg=eval_stats['evaluation_avg'],
        overfitting_ratio=eval_stats['overfitting_ratio'],
    )

    return summary


# ============================================================================
# INTERNAL HELPER FUNCTIONS
# ============================================================================

def _build_failure_taxonomy(training_log: List[EpisodeMetrics]) -> Dict[str, int]:
    """Categorize and count failures by type."""
    taxonomy = defaultdict(int)
    for m in training_log:
        if m.failure_type:
            taxonomy[m.failure_type] += 1
    return dict(taxonomy)


def _analyze_evaluation_results(
        training_log: List[EpisodeMetrics],
        eval_results: Optional[Union[Dict, List]],
) -> Dict[str, float]:
    """Extract evaluation statistics."""

    successful_log = [m for m in training_log if m.error is None]

    # Training final average
    training_final_avg = np.mean(
        [m.reward for m in successful_log[-50:]]
    ) if len(successful_log) >= 50 else (
        np.mean([m.reward for m in successful_log]) if successful_log else 0.0
    )

    # Evaluation average
    evaluation_avg = 0.0
    if eval_results:
        try:
            if isinstance(eval_results, dict):
                eval_rewards = [
                    r.get('avg_reward', 0) if isinstance(r, dict) else 0
                    for r in eval_results.values()
                ]
                evaluation_avg = np.mean(eval_rewards) if eval_rewards else 0.0
            elif isinstance(eval_results, list):
                solved_count = sum(
                    1 for r in eval_results
                    if hasattr(r, 'solved') and r.solved
                )
                evaluation_avg = (
                    solved_count / max(1, len(eval_results))
                    if eval_results else 0.0
                )
        except (AttributeError, TypeError):
            evaluation_avg = 0.0

    # Overfitting ratio
    overfitting_ratio = (
        training_final_avg / evaluation_avg
        if evaluation_avg > 0 else 1.0
    )

    return {
        'training_final_avg': float(training_final_avg),
        'evaluation_avg': float(evaluation_avg),
        'overfitting_ratio': float(overfitting_ratio),
    }