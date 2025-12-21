#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN EVALUATION FRAMEWORK - Main orchestrator
============================================
Orchestrates evaluation of both GNN and Random merge strategies.

Returns results in format compatible with evaluation.py framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

# ✅ FIXED: Use absolute imports
from experiments.core.gnn_evaluation_evaluators import GNNPolicyEvaluator, RandomMergeEvaluator

if TYPE_CHECKING:
    from experiments.core.evaluation_metrics import DetailedMetrics

logger = logging.getLogger(__name__)

class GNNRandomEvaluationFramework:
    """
    Orchestrate evaluation of both GNN and Random merge strategies.

    Returns results in format compatible with evaluation.py framework.
    """

    def __init__(
            self,
            model_path: str,
            domain_file: str,
            problem_files: List[str],
            output_dir: str = "evaluation_results",
            num_runs_per_problem: int = 1,
            downward_dir: Optional[str] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
    ):
        """
        Initialize evaluation framework.

        Args:
            model_path: Path to trained model.zip
            domain_file: Path to domain.pddl
            problem_files: List of problem.pddl paths
            output_dir: Output directory for results
            num_runs_per_problem: Number of runs per problem
            downward_dir: Path to Fast Downward
            max_merges: Maximum merge steps
            timeout_per_step: Timeout per merge step
        """
        self.model_path = model_path
        self.domain_file = domain_file
        self.problem_files = problem_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_runs_per_problem = num_runs_per_problem
        self.downward_dir = downward_dir
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step

        self.all_results: List[DetailedMetrics] = []

    def evaluate(
            self,
            include_gnn: bool = True,
            include_random: bool = True,
    ) -> Tuple[List['DetailedMetrics'], List['DetailedMetrics']]:

        """
        Execute full evaluation pipeline.

        Args:
            include_gnn: Whether to evaluate GNN policy
            include_random: Whether to evaluate random baseline

        Returns:
            Tuple of (gnn_results, random_results)
        """
        logger.info("\n" + "=" * 100)
        logger.info("GNN vs RANDOM MERGE STRATEGY EVALUATION")
        logger.info("=" * 100)
        logger.info(f"\n📋 Configuration:")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Domain: {self.domain_file}")
        logger.info(f"   Test problems: {len(self.problem_files)}")
        logger.info(f"   Runs per problem: {self.num_runs_per_problem}")

        gnn_results = []
        random_results = []

        # Evaluate GNN
        if include_gnn:
            logger.info("\n" + "-" * 100)
            logger.info("PHASE 1: GNN-GUIDED MERGE-AND-SHRINK + A* SEARCH")
            logger.info("-" * 100 + "\n")

            gnn_evaluator = GNNPolicyEvaluator(
                model_path=self.model_path,
                downward_dir=self.downward_dir,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
            )

            gnn_results = gnn_evaluator.evaluate_problems(
                domain_file=self.domain_file,
                problem_files=self.problem_files,
                num_runs_per_problem=self.num_runs_per_problem,
            )

            self.all_results.extend(gnn_results)

        # Evaluate Random
        if include_random:
            logger.info("\n" + "-" * 100)
            logger.info("PHASE 2: RANDOM MERGE-AND-SHRINK + A* SEARCH")
            logger.info("-" * 100 + "\n")

            random_evaluator = RandomMergeEvaluator(
                downward_dir=self.downward_dir,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
            )

            random_results = random_evaluator.evaluate_problems(
                domain_file=self.domain_file,
                problem_files=self.problem_files,
                num_runs_per_problem=self.num_runs_per_problem,
            )

            self.all_results.extend(random_results)

        logger.info("\n" + "=" * 100)
        logger.info("✅ EVALUATION COMPLETE")
        logger.info("=" * 100)

        return gnn_results, random_results

    def get_all_results(self) -> List[DetailedMetrics]:  # ✅ String annotation
        """
        Get all evaluation results combined.

        Returns:
            List of all DetailedMetrics
        """
        return self.all_results

    def to_summary(self) -> Dict[str, any]:  # ✅ Type hints
        """
        Convert results to summary format.

        Returns:
            Summary dictionary with statistics
        """
        from experiments.core.evaluation_analyzer import ComparisonAnalyzer

        if not self.all_results:
            return {}

        analyzer = ComparisonAnalyzer(self.all_results)

        summary = {}
        for planner_name in set(r.planner_name for r in self.all_results):
            stats = analyzer.get_aggregate_statistics(planner_name)
            summary[planner_name] = stats.to_dict()

        return summary