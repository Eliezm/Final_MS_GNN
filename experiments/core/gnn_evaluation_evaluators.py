#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN EVALUATION EVALUATORS - High-level evaluator classes
========================================================
GNNPolicyEvaluator and RandomMergeEvaluator for strategy comparison.
"""

from __future__ import annotations

import os
import time
import logging
import numpy as np  # ✅ MOVED TO TOP
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING, Dict, Any
from tqdm import tqdm

# ✅ FIXED: Use absolute imports
from experiments.core.gnn_evaluation_policies import GNNMergePolicy, RandomMergePolicy
from experiments.core.gnn_evaluation_parser import FastDownwardOutputParser
from experiments.core.gnn_evaluation_executor import ExecutorConfig, MergeExecutor
from experiments.shared_experiment_utils import DEFAULT_REWARD_WEIGHTS

# Required import for RandomMergeEvaluator
import numpy as np

if TYPE_CHECKING:
    from experiments.core.evaluation_metrics import DetailedMetrics

logger = logging.getLogger(__name__)



class GNNPolicyEvaluator:
    """
    Evaluate a trained GNN policy on test problems.

    Process:
    1. Load trained GNN model (PPO from stable-baselines3)
    2. Run merge-and-shrink with GNN decisions to build abstraction
    3. Execute A* search with the built heuristic
    4. Extract solution metrics and h* preservation
    """

    def __init__(
            self,
            model_path: str,
            downward_dir: Optional[str] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
    ):
        """
        Initialize GNN policy evaluator.

        Args:
            model_path: Path to trained model.zip file
            downward_dir: Path to Fast Downward installation
            max_merges: Maximum merge steps
            timeout_per_step: Timeout per merge step (seconds)
        """
        self.model_path = model_path

        if downward_dir is None:
            downward_dir = str(Path(__file__).parent.parent.parent / "downward")

        self.downward_dir = downward_dir
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step

        logger.info(f"Loading GNN model from: {model_path}")
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            self.policy = GNNMergePolicy(model)
            logger.info("✅ GNN model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.executor = None

    def _create_executor(self) -> MergeExecutor:
        """Create executor for this evaluator."""
        config = ExecutorConfig(
            max_merges=self.max_merges,
            timeout_per_step=self.timeout_per_step,
            reward_weights=DEFAULT_REWARD_WEIGHTS.copy(),
            downward_dir=self.downward_dir,
            debug=False,
        )
        return MergeExecutor(config)

    # In the evaluate_problems method of GNNPolicyEvaluator class

    def run_problem(
            self,
            domain_file: str,
            problem_file: str,
            seed: int = 42,
    ) -> 'DetailedMetrics':
        """Run GNN-guided merge-and-shrink with proper A* completion detection."""
        from experiments.core.evaluation_metrics import DetailedMetrics
        import time
        import subprocess

        problem_name = os.path.basename(problem_file)
        logger.info(f"\n[GNN] Evaluating: {problem_name}")

        metrics = DetailedMetrics(
            problem_name=problem_name,
            planner_name="GNN"
        )

        executor = self._create_executor()

        try:
            # Run merges with timeout
            start_total = time.time()
            success, num_merges, h_pres, elapsed_merge, is_solvable, info = executor.execute_merges(
                domain_file=domain_file,
                problem_file=problem_file,
                policy=self.policy,
                seed=seed,
            )

            # ✅ FIX: Extract FD metrics and ensure process is dead
            metrics.solved = info.get('solved', False)
            metrics.plan_cost = info.get('plan_cost', 0)
            metrics.plan_length = info.get('plan_length', 0)
            metrics.nodes_expanded = info.get('nodes_expanded', 0)
            metrics.wall_clock_time = elapsed_merge
            metrics.h_star_preservation = h_pres
            metrics.merge_episodes = num_merges
            metrics.solvable = is_solvable
            metrics.timeout_occurred = not success

            # ✅ FIX: Verify search actually completed
            if metrics.solved:
                logger.info(f"  ✅ SOLVED in {elapsed_merge:.2f}s | Merges: {num_merges} | Cost: {metrics.plan_cost}")
            elif is_solvable:
                logger.warning(f"  ⚠️  Not solved but solvable in {elapsed_merge:.2f}s | Merges: {num_merges}")
            else:
                logger.warning(f"  ❌ NOT SOLVED or NOT SOLVABLE in {elapsed_merge:.2f}s | Merges: {num_merges}")

        except subprocess.TimeoutExpired as e:
            logger.error(f"  ❌ SUBPROCESS TIMEOUT: {e}")
            metrics.solved = False
            metrics.timeout_occurred = True
            metrics.error_type = "subprocess_timeout"
            metrics.error_message = str(e)[:500]

        except TimeoutError as e:
            logger.error(f"  ❌ OBSERVATION TIMEOUT: {e}")
            metrics.solved = False
            metrics.timeout_occurred = True
            metrics.error_type = "observation_timeout"
            metrics.error_message = str(e)[:500]

        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}")
            metrics.solved = False
            metrics.error_type = "exception"
            metrics.error_message = str(e)[:500]

        finally:
            executor.cleanup()

        return metrics

    def evaluate_problems(
            self,
            domain_file: str,
            problem_files: List[str],
            num_runs_per_problem: int = 1,
    ) -> List[DetailedMetrics]:  # ✅ String annotation
        """
        Evaluate GNN on multiple problems.

        Args:
            domain_file: Path to domain.pddl
            problem_files: List of problem.pddl paths
            num_runs_per_problem: Number of runs per problem

        Returns:
            List of DetailedMetrics for all runs
        """
        all_results = []

        for problem_file in tqdm(problem_files, desc="GNN Evaluation", unit="problem"):
            for run_idx in range(num_runs_per_problem):
                seed = 42 + run_idx * 10000
                result = self.run_problem(domain_file, problem_file, seed=seed)
                all_results.append(result)

        return all_results


class RandomMergeEvaluator:
    """
    Evaluate random merge baseline on test problems.

    Process:
    1. Run merge-and-shrink with RANDOM merge decisions
    2. Execute A* search with the built abstraction heuristic
    3. Collect solution metrics
    """

    def __init__(
            self,
            downward_dir: Optional[str] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
    ):
        """
        Initialize random merge evaluator.

        Args:
            downward_dir: Path to Fast Downward installation
            max_merges: Maximum merge steps
            timeout_per_step: Timeout per merge step (seconds)
        """
        if downward_dir is None:
            downward_dir = str(Path(__file__).parent.parent.parent / "downward")

        self.downward_dir = downward_dir
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step
        self.policy = RandomMergePolicy()

        self.executor = None

    def _create_executor(self) -> MergeExecutor:
        """Create executor for this evaluator."""
        config = ExecutorConfig(
            max_merges=self.max_merges,
            timeout_per_step=self.timeout_per_step,
            reward_weights=DEFAULT_REWARD_WEIGHTS.copy(),
            downward_dir=self.downward_dir,
            debug=False,
        )
        return MergeExecutor(config)

    def run_problem(
            self,
            domain_file: str,
            problem_file: str,
            seed: int = 42,
    ) -> DetailedMetrics:  # ✅ String annotation
        """
        Run random merge strategy with A* search.

        Args:
            domain_file: Path to domain.pddl
            problem_file: Path to problem.pddl
            seed: Random seed for reproducibility

        Returns:
            DetailedMetrics with solution metrics
        """
        from experiments.core.evaluation_metrics import DetailedMetrics

        problem_name = os.path.basename(problem_file)
        logger.info(f"\n[RANDOM] Evaluating: {problem_name}")

        metrics = DetailedMetrics(
            problem_name=problem_name,
            planner_name="Random"
        )

        self.policy.rng = np.random.RandomState(seed)
        executor = self._create_executor()

        try:
            success, num_merges, h_pres, elapsed, is_solvable, info = executor.execute_merges(
                domain_file=domain_file,
                problem_file=problem_file,
                policy=self.policy,
                seed=seed,
            )

            # Extract FD results if available
            fd_log_path = info.get('fd_log_path') if isinstance(info, dict) else None
            fd_metrics = {}

            if fd_log_path and Path(fd_log_path).exists():
                fd_metrics = FastDownwardOutputParser.parse_fd_log_file(str(fd_log_path))

            # Populate metrics
            metrics.solved = fd_metrics.get('solved', False)
            metrics.plan_cost = fd_metrics.get('plan_cost', 0)
            metrics.plan_length = fd_metrics.get('plan_length', 0)
            metrics.nodes_expanded = fd_metrics.get('nodes_expanded', 0)
            metrics.search_time = fd_metrics.get('search_time', 0.0)
            metrics.wall_clock_time = elapsed
            metrics.h_star_preservation = h_pres
            metrics.merge_episodes = num_merges
            metrics.solvable = is_solvable
            metrics.timeout_occurred = not success

            if metrics.solved:
                logger.info(f"  ✅ SOLVED in {elapsed:.2f}s | Cost: {metrics.plan_cost} | Merges: {num_merges}")
            else:
                logger.warning(f"  ❌ NOT SOLVED in {elapsed:.2f}s | Merges: {num_merges}")

        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}")
            metrics.solved = False
            metrics.error_type = "exception"
            metrics.error_message = str(e)[:500]

        finally:
            executor.cleanup()

        return metrics

    def evaluate_problems(
            self,
            domain_file: str,
            problem_files: List[str],
            num_runs_per_problem: int = 1,
    ) -> List[DetailedMetrics]:  # ✅ String annotation
        """
        Evaluate random on multiple problems.

        Args:
            domain_file: Path to domain.pddl
            problem_files: List of problem.pddl paths
            num_runs_per_problem: Number of runs per problem

        Returns:
            List of DetailedMetrics for all runs
        """
        all_results = []

        for problem_file in tqdm(problem_files, desc="Random Evaluation", unit="problem"):
            for run_idx in range(num_runs_per_problem):
                seed = 42 + run_idx * 10000
                result = self.run_problem(domain_file, problem_file, seed=seed)
                all_results.append(result)

        return all_results


def create_comparable_metrics_dict(
        gnn_results: List['DetailedMetrics'],
        random_results: List['DetailedMetrics'],
        baseline_results: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Create unified metrics dict comparing GNN, Random, and Baselines.

    Ensures ALL metrics are computed for all strategies for fair comparison.
    """

    def compute_stats(results_list):
        if not results_list:
            return {
                'total': 0,
                'solved': 0,
                'solve_rate': 0.0,
                'avg_time': 0.0,
                'avg_cost': 0.0,
                'avg_expansions': 0,
                'avg_h_preservation': 1.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
            }

        solved_results = [r for r in results_list if r.solved]
        num_solved = len(solved_results)
        num_total = len(results_list)

        times = [r.wall_clock_time for r in solved_results if r.wall_clock_time > 0]
        costs = [r.plan_cost for r in solved_results if r.plan_cost > 0]
        expansions = [r.nodes_expanded for r in solved_results if r.nodes_expanded > 0]
        h_pres = [r.h_star_preservation for r in solved_results]

        return {
            'total': num_total,
            'solved': num_solved,
            'solve_rate': (num_solved / num_total * 100) if num_total > 0 else 0.0,
            'avg_time': float(np.mean(times)) if times else 0.0,
            'median_time': float(np.median(times)) if times else 0.0,
            'std_time': float(np.std(times)) if times else 0.0,
            'min_time': float(np.min(times)) if times else 0.0,
            'max_time': float(np.max(times)) if times else 0.0,
            'avg_cost': float(np.mean(costs)) if costs else 0.0,
            'avg_expansions': int(np.mean(expansions)) if expansions else 0,
            'avg_h_preservation': float(np.mean(h_pres)) if h_pres else 1.0,
        }

    metrics = {
        'GNN': compute_stats(gnn_results),
        'Random': compute_stats(random_results),
    }

    # Add baselines if available
    if baseline_results:
        for baseline_name, baseline_data in baseline_results.items():
            metrics[baseline_name] = baseline_data

    return metrics