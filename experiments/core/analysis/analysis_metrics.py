#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATA STRUCTURES - Unified metrics containers
=============================================
Defines all result data structures for experiment analysis.

Purpose:
- Type safety for analysis results
- Self-documenting code
- Easy serialization for papers
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import Enum


class AnalysisPhase(Enum):
    """Training phases for analysis."""
    EARLY = "early"  # 0-33% of training
    MID = "mid"  # 33-67% of training
    LATE = "late"  # 67-100% of training


@dataclass
class ProblemStats:
    """
    Statistics for a single problem across entire training.

    Answers: How did the model perform on THIS problem?
    """
    problem_name: str
    num_episodes: int
    num_failed: int
    coverage_percent: float

    # Reward statistics
    avg_reward: float
    best_reward: float
    worst_reward: float
    final_reward: float
    improvement_ratio: float

    # Quality metrics
    avg_h_preservation: float
    solve_rate: float

    # Learning dynamics
    episodes_to_convergence: Optional[int] = None
    avg_step_time_ms: float = 0.0
    avg_memory_mb: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentSummary:
    """
    Complete experiment statistics.

    Answers: Did the overall training work?
    What are the key paper results?
    """

    # Experiment scale
    num_problems: int
    num_train_episodes: int
    num_failed_episodes: int
    total_timesteps: int

    # Timing
    start_time: str
    end_time: str
    duration_seconds: float

    # Main results (FOR PAPER)
    avg_reward_over_all: float
    best_reward_over_all: float
    worst_reward_over_all: float

    # Per-problem breakdown
    per_problem_stats: List[Dict]

    # Learning quality
    reward_variance: float
    h_preservation_improvement_ratio: float
    solve_rate_improvement: float
    early_convergence_episodes: int

    # Coverage validation
    problem_coverage_valid: bool
    min_problem_coverage_pct: float
    max_problem_coverage_pct: float
    all_problems_trained: bool

    # Failure analysis
    failure_taxonomy: Dict[str, int] = field(default_factory=dict)

    # Resource usage
    avg_step_time_ms: float = 0.0
    avg_peak_memory_mb: float = 0.0

    # Metadata
    experiment_id: str = ''
    checkpoints_saved: int = 0
    best_model_path: str = ''

    # Plateau detection
    plateau_episode: Optional[int] = None
    convergence_threshold: float = 0.05
    overfitting_ratio: float = 1.0

    # Evaluation metrics
    training_final_avg: float = 0.0
    evaluation_avg: float = 0.0

    def __post_init__(self):
        if self.failure_taxonomy is None:
            self.failure_taxonomy = {}

    def to_dict(self) -> Dict:
        return asdict(self)