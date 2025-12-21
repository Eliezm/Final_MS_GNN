#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION METRICS - Data structures for evaluation results
============================================================
Defines all metric containers and helpers for evaluation.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


# ============================================================================
# METRICS DATACLASSES
# ============================================================================

@dataclass
class ExtendedMetrics:
    """Comprehensive metrics for planners (especially M&S)."""
    # Solution
    solved: bool = False
    cost: Optional[int] = None

    # Timing breakdown (seconds)
    time_total: float = 0.0
    time_translation: float = 0.0
    time_search: float = 0.0
    time_initialization: float = 0.0
    time_ms_main_loop: float = 0.0

    # Search metrics
    expansions: Optional[int] = None
    generated_states: Optional[int] = None
    evaluated_states: Optional[int] = None

    # M&S specific metrics
    max_abstraction_size: Optional[int] = None
    final_abstraction_size: Optional[int] = None
    num_merges: Optional[int] = None
    num_shrinks: Optional[int] = None
    num_label_reductions: Optional[int] = None
    max_states_before_merge: Optional[int] = None

    # Error tracking
    error_reason: Optional[str] = None
    error_details: Optional[str] = None

    # Metadata
    planner: str = ""
    domain: str = ""
    problem_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetailedMetrics:
    """Complete metrics for a single evaluation run."""

    problem_name: str
    planner_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Solve status
    solved: bool = False
    wall_clock_time: float = 0.0

    # Planning quality
    plan_cost: int = 0
    plan_length: int = 0
    solution_cost: int = 0

    # Search metrics
    nodes_expanded: int = 0
    nodes_generated: int = 0
    search_depth: int = 0
    branching_factor: float = 1.0

    # Memory
    peak_memory_kb: int = 0

    # Time breakdown
    search_time: float = 0.0
    translate_time: float = 0.0
    preprocess_time: float = 0.0
    total_time: float = 0.0

    # GNN-specific metrics
    h_star_preservation: float = 1.0
    num_active_systems: int = 0
    merge_episodes: int = 0
    solvable: bool = True

    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timeout_occurred: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateStatistics:
    """Summary statistics across all problems."""

    planner_name: str
    num_problems_total: int
    num_problems_solved: int
    solve_rate_pct: float

    # Time statistics
    mean_time_sec: float
    median_time_sec: float
    std_time_sec: float
    min_time_sec: float
    max_time_sec: float
    q1_time_sec: float
    q3_time_sec: float
    iqr_time_sec: float

    # Expansion statistics
    mean_expansions: int
    median_expansions: int
    std_expansions: int

    # Plan quality
    mean_plan_cost: int
    median_plan_cost: int
    std_plan_cost: int

    # GNN-specific
    mean_h_preservation: float = 1.0
    median_h_preservation: float = 1.0

    # Counts
    unsolved_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    total_wall_clock_time_sec: float = 0.0

    # M&S specific
    mean_max_abstraction_size: int = 0
    mean_final_abstraction_size: int = 0
    mean_num_merges: int = 0
    mean_num_shrinks: int = 0
    mean_num_label_reductions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)