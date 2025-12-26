#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTILITIES - Shared helper functions
===================================
Common utilities used across analysis modules.
"""

from typing import Optional, List
from datetime import datetime

from experiments.core.logging import EpisodeMetrics
from experiments.core.analysis.analysis_metrics import ExperimentSummary


def _create_empty_summary(
        experiment_id: str,
        num_problems: int,
) -> ExperimentSummary:
    """Create empty summary when no training occurred."""
    return ExperimentSummary(
        # Experiment scale
        num_problems=num_problems,
        num_train_episodes=0,
        num_failed_episodes=0,
        total_timesteps=0,

        # Timing
        start_time="",
        end_time="",
        duration_seconds=0.0,

        # Main results
        avg_reward_over_all=0.0,
        best_reward_over_all=0.0,
        worst_reward_over_all=0.0,

        # Per-problem breakdown
        per_problem_stats=[],

        # Learning quality
        reward_variance=0.0,
        h_preservation_improvement_ratio=1.0,
        solve_rate_improvement=0.0,
        early_convergence_episodes=0,

        # ✅ FIX: Add the 4 missing required fields
        problem_coverage_valid=False,      # No training = invalid coverage
        min_problem_coverage_pct=0.0,      # No coverage
        max_problem_coverage_pct=0.0,      # No coverage
        all_problems_trained=False,        # No problems trained

        # Metadata
        experiment_id=experiment_id,
    )


def _get_start_time(training_log: List[EpisodeMetrics]) -> str:
    """Extract start time from first episode."""
    if training_log:
        return datetime.fromtimestamp(training_log[0].timestamp).isoformat()
    return ""


def _get_end_time(training_log: List[EpisodeMetrics]) -> str:
    """Extract end time from last episode."""
    if training_log:
        return datetime.fromtimestamp(training_log[-1].timestamp).isoformat()
    return ""


def _get_duration(training_log: List[EpisodeMetrics]) -> float:
    """Calculate training duration in seconds."""
    if len(training_log) > 1:
        return training_log[-1].timestamp - training_log[0].timestamp
    return 0.0