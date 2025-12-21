#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION ANALYZER - Results analysis and comparison
===================================================
Analyzes and compares evaluation results across planners.
"""

import logging
from typing import Dict, List, Optional
from collections import defaultdict, Counter

import numpy as np

from experiments.core.evaluation_metrics import DetailedMetrics, AggregateStatistics

logger = logging.getLogger(__name__)


# ============================================================================
# COMPARISON ANALYZER
# ============================================================================

class ComparisonAnalyzer:
    """Analyze and compare evaluation results."""

    def __init__(self, results: List[DetailedMetrics]):
        self.results = results
        self.by_planner = defaultdict(list)
        for result in results:
            self.by_planner[result.planner_name].append(result)

    def get_aggregate_statistics(self, planner_name: str) -> AggregateStatistics:
        """
        Compute statistics for a planner.

        Aggregates:
        - Solve rate
        - Time statistics
        - Expansion statistics
        - Plan quality
        - Error counts
        """
        results = self.by_planner[planner_name]
        solved = [r for r in results if r.solved]
        num_solved = len(solved)
        num_total = len(results)

        times = [r.wall_clock_time for r in solved if r.wall_clock_time > 0]
        expansions = [r.nodes_expanded for r in solved if r.nodes_expanded > 0]
        costs = [r.plan_cost for r in solved if r.plan_cost > 0]
        h_preservations = [r.h_star_preservation for r in solved]

        def safe_percentile(data, q):
            if not data:
                return 0.0
            return float(np.percentile(data, q))

        return AggregateStatistics(
            planner_name=planner_name,
            num_problems_total=num_total,
            num_problems_solved=num_solved,
            solve_rate_pct=(num_solved / max(num_total, 1)) * 100,
            mean_time_sec=float(np.mean(times)) if times else 0.0,
            median_time_sec=float(np.median(times)) if times else 0.0,
            std_time_sec=float(np.std(times)) if times else 0.0,
            min_time_sec=float(np.min(times)) if times else 0.0,
            max_time_sec=float(np.max(times)) if times else 0.0,
            q1_time_sec=safe_percentile(times, 25),
            q3_time_sec=safe_percentile(times, 75),
            iqr_time_sec=safe_percentile(times, 75) - safe_percentile(times, 25),
            mean_expansions=int(np.mean(expansions)) if expansions else 0,
            median_expansions=int(np.median(expansions)) if expansions else 0,
            std_expansions=int(np.std(expansions)) if expansions else 0,
            mean_plan_cost=int(np.mean(costs)) if costs else 0,
            median_plan_cost=int(np.median(costs)) if costs else 0,
            std_plan_cost=int(np.std(costs)) if costs else 0,
            mean_h_preservation=float(np.mean(h_preservations)) if h_preservations else 1.0,
            median_h_preservation=float(np.median(h_preservations)) if h_preservations else 1.0,
            unsolved_count=len([r for r in results if not r.solved]),
            timeout_count=len([r for r in results if r.timeout_occurred]),
            error_count=len([r for r in results if r.error_type]),
            total_wall_clock_time_sec=float(sum(times)) if times else 0.0,
        )

    def get_per_problem_winners(self) -> Dict[str, str]:
        """
        Get best planner for each problem.

        Returns:
            Dict mapping problem_name -> best_planner_name
        """
        problems = set(r.problem_name for r in self.results)
        winners = {}

        for problem in sorted(problems):
            problem_results = [r for r in self.results if r.problem_name == problem]
            solved_results = [r for r in problem_results if r.solved]

            if solved_results:
                best = min(solved_results, key=lambda r: r.wall_clock_time)
                winners[problem] = best.planner_name
            else:
                winners[problem] = "NONE"

        return winners

    def get_speedup_analysis(self) -> Dict[str, any]:
        """
        Analyze GNN speedup vs baselines.

        Returns:
            Dict with speedup statistics
        """
        if "GNN" not in self.by_planner:
            return {}

        gnn_results = self.by_planner["GNN"]
        speedups = []

        for gnn_result in gnn_results:
            if not gnn_result.solved:
                continue

            baseline_results = [
                r for r in self.results
                if r.problem_name == gnn_result.problem_name
                   and r.planner_name != "GNN"
                   and r.solved
            ]

            if baseline_results:
                best_baseline = min(baseline_results, key=lambda r: r.wall_clock_time)
                if gnn_result.wall_clock_time > 0:
                    speedup = best_baseline.wall_clock_time / gnn_result.wall_clock_time
                    speedups.append(speedup)

        return {
            'mean_speedup': float(np.mean(speedups)) if speedups else 0.0,
            'geometric_mean_speedup': float(np.exp(np.mean(np.log(speedups)))) if speedups else 0.0,
            'num_speedup_comparisons': len(speedups),
        }


# ============================================================================
# BASELINE EXECUTION
# ============================================================================

def run_baseline_on_benchmark_set(
        benchmark_set: List[tuple],
        baseline_name: str,
        search_config: str,
        max_problems: Optional[int] = None,
        baseline_runner=None,
) -> tuple:
    """
    Run one baseline on a benchmark set.

    Orchestrates:
    1. Problem subset selection
    2. Individual problem runs
    3. Metrics aggregation
    4. Error summarization

    Args:
        benchmark_set: List of (domain, problem, problem_id) tuples
        baseline_name: Human-readable name
        search_config: FD search configuration
        max_problems: Limit number of problems (for testing)
        baseline_runner: BaselineRunner instance

    Returns:
        Tuple of (aggregate_results_dict, detailed_metrics_list)
    """
    from experiments.core.baseline_runner import BaselineRunner

    if baseline_runner is None:
        baseline_runner = BaselineRunner(timeout_sec=300)

    logger.info(f"\n  Running: {baseline_name}")

    if max_problems:
        benchmark_set = benchmark_set[:max_problems]

    logger.info(f"    Problems: {len(benchmark_set)}")

    detailed_results = []
    solved_count = 0

    for i, (domain_file, problem_file, problem_id) in enumerate(benchmark_set, 1):
        logger.debug(f"    [{i}/{len(benchmark_set)}] {problem_id}...")

        # Run problem
        result = baseline_runner.run(
            domain_file=domain_file,
            problem_file=problem_file,
            search_config=search_config,
            baseline_name=baseline_name
        )

        detailed_results.append(result)

        if result.solved:
            solved_count += 1
            logger.info(
                f"    [{i}/{len(benchmark_set)}] {problem_id:<25} ✓ "
                f"cost={result.plan_cost} time={result.wall_clock_time:.2f}s"
            )
        else:
            logger.warning(
                f"    [{i}/{len(benchmark_set)}] {problem_id:<25} ✗ {result.error_type}"
            )

    # Aggregate results from DetailedMetrics
    aggregate = _aggregate_detailed_metrics(baseline_name, detailed_results)

    return aggregate, detailed_results


def _aggregate_detailed_metrics(
        baseline_name: str,
        metrics_list: List[DetailedMetrics]
) -> Dict:
    """Aggregate DetailedMetrics across problems."""

    total_problems = len(metrics_list)
    solved_list = [m for m in metrics_list if m.solved]
    solved_count = len(solved_list)

    solve_rate = (solved_count / total_problems * 100) if total_problems > 0 else 0

    def safe_avg(values: List) -> float:
        valid = [v for v in values if v is not None and isinstance(v, (int, float)) and v > 0]
        return sum(valid) / len(valid) if valid else 0.0

    result = {
        "name": baseline_name,
        "set_size": total_problems,
        "solved": solved_count,
        "solve_rate_%": round(solve_rate, 1),

        # Timing (averaged over solved)
        "avg_time_total_s": round(safe_avg([m.wall_clock_time for m in solved_list]), 3),
        "avg_time_translation_s": round(safe_avg([m.translate_time for m in solved_list]), 3),
        "avg_time_search_s": round(safe_avg([m.search_time for m in solved_list]), 3),
        "avg_time_initialization_s": round(safe_avg([m.preprocess_time for m in solved_list]), 3),
        "avg_time_ms_main_loop_s": 0,

        # Search metrics
        "avg_expansions": int(safe_avg([m.nodes_expanded for m in solved_list])),
        "avg_cost": int(safe_avg([m.plan_cost for m in solved_list])),
        "avg_generated_states": int(safe_avg([m.nodes_generated for m in solved_list])),
        "avg_evaluated_states": 0,

        # M&S metrics (0 for baseline FD)
        "avg_max_abstraction_size": 0,
        "avg_final_abstraction_size": 0,
        "avg_num_merges": 0,
        "avg_num_shrinks": 0,
        "avg_num_label_reductions": 0,

        # Error summary
        "errors": _summarize_errors([m.error_type for m in metrics_list if not m.solved])
    }

    logger.info(
        f"    ✓ Solved {solved_count}/{total_problems} ({solve_rate:.1f}%) | "
        f"Avg time: {result['avg_time_total_s']:.3f}s"
    )

    return result


def _summarize_errors(error_reasons: List[str]) -> str:
    """Summarize error reasons."""
    if not error_reasons:
        return "None"

    counts = Counter(error_reasons)
    return ", ".join(f"{e}({c})" for e, c in sorted(counts.items()))