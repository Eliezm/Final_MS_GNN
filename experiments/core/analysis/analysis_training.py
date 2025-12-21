#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAINING ANALYSIS - Per-problem and overall training statistics
===============================================================
Analyzes the fundamentals: Did training work? How much did each problem improve?
"""

from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np

from experiments.core.logging import EpisodeMetrics
from experiments.core.analysis.analysis_metrics import ProblemStats


def _analyze_per_problem_stats(
        training_log: List[EpisodeMetrics],
        benchmarks: List[tuple],
        problem_names: List[str],
) -> Tuple[List[ProblemStats], Dict]:
    """
    Analyze performance on each training problem.

    Returns:
        (list of ProblemStats, coverage_info dict)
    """

    # Group episodes by problem
    by_problem = defaultdict(list)
    for metrics in training_log:
        by_problem[metrics.problem_name].append(metrics)

    per_problem_stats = []
    min_coverage = 100.0
    max_coverage = 0.0
    all_trained = True

    for domain, problem_file in benchmarks:
        problem_name = problem_names[len(per_problem_stats)]

        if problem_name not in by_problem:
            # Problem was never trained
            all_trained = False
            per_problem_stats.append(ProblemStats(
                problem_name=problem_name,
                num_episodes=0,
                num_failed=0,
                coverage_percent=0.0,
                avg_reward=0.0,
                best_reward=0.0,
                worst_reward=0.0,
                final_reward=0.0,
                improvement_ratio=0.0,
                avg_h_preservation=0.0,
                solve_rate=0.0,
            ))
            continue

        episodes = by_problem[problem_name]

        # Coverage
        coverage = (len(episodes) / len(training_log) * 100) if training_log else 0
        min_coverage = min(min_coverage, coverage)
        max_coverage = max(max_coverage, coverage)

        # Separate successes from failures
        successful = [e for e in episodes if e.error is None]
        failed = [e for e in episodes if e.error is not None]

        if successful:
            rewards = [e.reward for e in successful]
            h_pres = [e.h_star_preservation for e in successful]
            step_times = [e.step_time_ms for e in successful if e.step_time_ms > 0]
            peak_mems = [e.peak_memory_mb for e in successful if e.peak_memory_mb > 0]
        else:
            rewards = []
            h_pres = []
            step_times = []
            peak_mems = []

        # Convergence
        episodes_to_convergence = _detect_convergence(rewards)

        # Improvement ratio
        if len(set(rewards)) > 1:
            best = max(rewards) if rewards else 0
            worst = min(rewards) if rewards else 0
            final = rewards[-1] if rewards else 0
            improvement = (final - worst) / (best - worst) if (best != worst) else 0.0
        else:
            improvement = 0.0

        stats = ProblemStats(
            problem_name=problem_name,
            num_episodes=len(episodes),
            num_failed=len(failed),
            coverage_percent=coverage,
            avg_reward=float(np.mean(rewards)) if rewards else 0.0,
            best_reward=float(max(rewards)) if rewards else 0.0,
            worst_reward=float(min(rewards)) if rewards else 0.0,
            final_reward=float(rewards[-1]) if rewards else 0.0,
            improvement_ratio=float(improvement),
            avg_h_preservation=float(np.mean(h_pres)) if h_pres else 0.0,
            solve_rate=len(successful) / len(episodes) if episodes else 0.0,
            episodes_to_convergence=episodes_to_convergence,
            avg_step_time_ms=float(np.mean(step_times)) if step_times else 0.0,
            avg_memory_mb=float(np.mean(peak_mems)) if peak_mems else 0.0,
        )

        per_problem_stats.append(stats)

    coverage_valid = all_trained and min_coverage >= 5.0

    return per_problem_stats, {
        'valid': coverage_valid,
        'min_coverage': min_coverage if training_log else 0.0,
        'max_coverage': max_coverage if training_log else 0.0,
        'all_trained': all_trained,
    }


def _analyze_overall_stats(
        training_log: List[EpisodeMetrics],
        per_problem_stats: List[ProblemStats],
        timesteps_per_episode: int,
) -> Dict:
    """Compute overall training statistics."""

    successful_log = [m for m in training_log if m.error is None]

    if not successful_log:
        return {
            'avg_reward': 0.0,
            'best_reward': 0.0,
            'worst_reward': 0.0,
            'reward_variance': 0.0,
            'h_pres_improvement': 1.0,
            'solve_rate_improvement': 0.0,
            'avg_step_time_ms': 0.0,
            'avg_peak_memory_mb': 0.0,
        }

    # Reward statistics
    all_rewards = [m.reward for m in successful_log]
    avg_reward = float(np.mean(all_rewards))
    best_reward = float(np.max(all_rewards))
    worst_reward = float(np.min(all_rewards))
    reward_var = float(np.var(all_rewards))

    # H* preservation improvement
    all_h_pres = [m.h_star_preservation for m in successful_log]
    if len(all_h_pres) > 10:
        early_h = np.mean(all_h_pres[:10])
        late_h = np.mean(all_h_pres[-10:])
        h_pres_improvement = late_h / early_h if early_h > 0 else 1.0
    else:
        h_pres_improvement = 1.0

    # Solve rate improvement
    if len(successful_log) > 10:
        early_solve = sum(1 for m in successful_log[:10] if m.is_solvable) / 10
        late_solve = sum(1 for m in successful_log[-10:] if m.is_solvable) / 10
        solve_improvement = late_solve - early_solve
    else:
        solve_improvement = 0.0

    # Resources
    step_times = [m.step_time_ms for m in successful_log if m.step_time_ms > 0]
    peak_mems = [m.peak_memory_mb for m in successful_log if m.peak_memory_mb > 0]

    return {
        'avg_reward': avg_reward,
        'best_reward': best_reward,
        'worst_reward': worst_reward,
        'reward_variance': reward_var,
        'h_pres_improvement': float(h_pres_improvement),
        'solve_rate_improvement': float(solve_improvement),
        'avg_step_time_ms': float(np.mean(step_times)) if step_times else 0.0,
        'avg_peak_memory_mb': float(np.mean(peak_mems)) if peak_mems else 0.0,
    }


def _detect_convergence(
        rewards: List[float],
        window: int = 10,
        threshold: float = 0.05,
) -> Optional[int]:
    """
    Detect episode where training converges.

    Convergence = window where improvement < threshold%.
    """
    if len(rewards) <= window * 2:
        return None

    for i in range(window, len(rewards) - window):
        recent = np.mean(rewards[i:i + window])
        older = np.mean(rewards[i - window:i])

        if older != 0 and abs(recent - older) / abs(older) < threshold:
            return i

    return None