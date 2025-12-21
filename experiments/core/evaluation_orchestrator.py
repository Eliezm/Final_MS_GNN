#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION ORCHESTRATOR - Main framework orchestration
===================================================
High-level orchestration of complete evaluation pipeline.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.core.evaluation_metrics import DetailedMetrics
from experiments.core.evaluation_config import EvaluationConfig
from experiments.core.benchmark_discovery import (
    verify_fd_setup, discover_benchmarks, filter_benchmarks
)
from experiments.core.evaluation_analyzer import (
    ComparisonAnalyzer, run_baseline_on_benchmark_set
)
from experiments.core.evaluation_export import (
    export_results_to_csv, export_statistics_to_json,
    export_summary_report, generate_reports
)
from experiments.core.gnn_random_evaluation import GNNRandomEvaluationFramework
from experiments.core.evaluation_plots import GenerateEvaluationPlots

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN EVALUATION FRAMEWORK
# ============================================================================

class EvaluationFramework:
    """Main evaluation orchestrator."""

    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize evaluation framework.

        Args:
            output_dir: Base output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: List[DetailedMetrics] = []

        logger.info(f"Evaluation framework initialized: {self.output_dir}")

    def run_gnn_and_random_evaluation(
            self,
            model_path: str,
            domain_file: str,
            problem_files: List[str],
            num_runs_per_problem: int = 1,
    ) -> Dict[str, Any]:
        """..."""
        try:
            # ✅ VALIDATION: Check inputs
            if not Path(model_path).exists():
                logger.error(f"Model not found: {model_path}")
                return {"status": "failed", "error": "Model file not found"}

            if not Path(domain_file).exists():
                logger.error(f"Domain not found: {domain_file}")
                return {"status": "failed", "error": "Domain file not found"}

            if not problem_files:
                logger.error("No problem files provided")
                return {"status": "failed", "error": "Empty problem list"}

            # ✅ VERIFICATION: Check files exist
            missing_problems = [p for p in problem_files if not Path(p).exists()]
            if missing_problems:
                logger.error(f"Missing problem files: {missing_problems[:3]}")
                return {"status": "failed", "error": f"{len(missing_problems)} problem files not found"}

            eval_framework = GNNRandomEvaluationFramework(
                model_path=model_path,
                domain_file=domain_file,
                problem_files=problem_files,
                output_dir=str(self.output_dir / "gnn_random_evaluation"),
                num_runs_per_problem=num_runs_per_problem,
                downward_dir=str(PROJECT_ROOT / "downward"),
            )

            gnn_results, random_results = eval_framework.evaluate()

            # ✅ VALIDATION: Results are DetailedMetrics objects
            if not gnn_results and not random_results:
                logger.warning("No evaluation results returned")
                return {"status": "partial", "error": "No results from evaluation"}

            all_detailed_results = gnn_results + random_results
            self.all_results.extend(all_detailed_results)

            summary = eval_framework.to_summary()

            # ✅ VALIDATION: Summary has required keys
            if not summary or "status" not in str(summary):
                logger.warning("Summary is empty or incomplete")

            logger.info("\n📊 GNN vs Random Comparison:")
            if "GNN" in summary and isinstance(summary["GNN"], dict):
                gnn_solve = summary["GNN"].get("solve_rate_pct", 0)
                logger.info(f"   GNN solve rate: {gnn_solve:.1f}%")
            if "Random" in summary and isinstance(summary["Random"], dict):
                random_solve = summary["Random"].get("solve_rate_pct", 0)
                logger.info(f"   Random solve rate: {random_solve:.1f}%")

            return {
                "status": "success",
                "gnn_results": [r.to_dict() for r in gnn_results] if gnn_results else [],
                "random_results": [r.to_dict() for r in random_results] if random_results else [],
                "summary": summary or {},
                "num_gnn_solved": sum(1 for r in gnn_results if r.solved) if gnn_results else 0,
                "num_random_solved": sum(1 for r in random_results if r.solved) if random_results else 0,
                "total_evaluated": len(problem_files),
            }

        except FileNotFoundError as e:
            logger.error(f"File not found during evaluation: {e}")
            return {"status": "failed", "error": f"File not found: {str(e)[:100]}"}
        except Exception as e:
            logger.error(f"GNN and Random evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "failed", "error": str(e)[:200]}

    def run_baseline_evaluation(
            self,
            timeout_sec: int = 300,
            max_problems: Optional[int] = None,
            domains: Optional[str] = None,
            sizes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run baseline evaluation on discovered benchmarks.

        Orchestrates:
        1. FD setup verification
        2. Benchmark discovery
        3. Benchmark filtering
        4. Baseline runs
        5. Result reporting
        6. Plot generation
        """
        logger.info("\n" + "=" * 100)
        logger.info("BASELINE EVALUATION FRAMEWORK")
        logger.info("=" * 100 + "\n")

        # Step 1: Verify setup
        if not self._verify_setup():
            return {"status": "failed", "phase": "verification"}

        # Step 2: Discover benchmarks
        all_benchmarks = self._discover_problems()
        if not all_benchmarks:
            return {"status": "failed", "phase": "discovery"}

        # Step 3: Filter if requested
        if domains or sizes:
            all_benchmarks = self._filter_problems(all_benchmarks, domains, sizes)
            if not all_benchmarks:
                logger.error("No benchmarks match filter criteria")
                return {"status": "failed", "phase": "filtering"}
            logger.info(f"Filtered to {len(all_benchmarks)} benchmark set(s)")

        # Step 4: Run baselines
        all_results, all_detailed = self._run_all_baselines(all_benchmarks, max_problems)

        # Step 5: Generate reports
        self._generate_all_reports(all_results, all_detailed)

        # Step 6: Generate plots
        self._generate_evaluation_plots(all_detailed)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ BASELINE EVALUATION COMPLETE")
        logger.info("=" * 80)

        return {
            "status": "success",
            "num_baselines": len(EvaluationConfig.BASELINE_CONFIGS),
            "num_benchmark_sets": len(all_benchmarks),
            "num_results": len(all_detailed),
            "output_dir": str(self.output_dir),
        }

    # ========================================================================
    # INTERNAL ORCHESTRATION METHODS
    # ========================================================================

    def _verify_setup(self) -> bool:
        """Verify Fast Downward setup."""
        return verify_fd_setup()

    def _discover_problems(self) -> Dict[str, List[tuple]]:
        """Discover benchmark problems."""
        return discover_benchmarks()

    def _filter_problems(
            self,
            all_benchmarks: Dict[str, List[tuple]],
            domains: Optional[str],
            sizes: Optional[str]
    ) -> Dict[str, List[tuple]]:
        """Filter benchmarks by domain and size."""
        return filter_benchmarks(all_benchmarks, domains, sizes)

    def _run_all_baselines(
            self,
            all_benchmarks: Dict[str, List[tuple]],
            max_problems: Optional[int]
    ) -> tuple:
        """
        Run all baseline configurations.

        Returns:
            Tuple of (aggregate_results_list, detailed_metrics_list)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: RUN BASELINE PLANNERS")
        logger.info("=" * 80)

        all_aggregate_results = []
        all_detailed_metrics = []
        baselines = EvaluationConfig.get_baselines(EvaluationConfig.RANDOM_SEED)

        for benchmark_name, benchmark_set in all_benchmarks.items():
            logger.info(f"\n{benchmark_name} Benchmark Set")
            logger.info("-" * 80)

            for baseline_config in baselines:
                aggregate, detailed = run_baseline_on_benchmark_set(
                    benchmark_set=benchmark_set,
                    baseline_name=baseline_config["name"],
                    search_config=baseline_config["search_config"],
                    max_problems=max_problems
                )

                all_aggregate_results.append(aggregate)
                all_detailed_metrics.extend(detailed)

        return all_aggregate_results, all_detailed_metrics

    def _generate_all_reports(
            self,
            all_results: List[Dict[str, Any]],
            all_detailed: List[DetailedMetrics]
    ) -> None:
        """Generate CSV, JSON, and text reports."""
        generate_reports(all_results, all_detailed)

    def _generate_evaluation_plots(
            self,
            detailed_metrics: List[DetailedMetrics]
    ) -> None:
        """Generate visualization plots from evaluation results."""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING EVALUATION PLOTS")
        logger.info("=" * 80)

        try:
            # Compute statistics for all planners
            analyzer = ComparisonAnalyzer(detailed_metrics)
            planners = set(r.planner_name for r in detailed_metrics)

            stats_dicts = {}
            for planner_name in planners:
                stats = analyzer.get_aggregate_statistics(planner_name)
                stats_dicts[planner_name] = stats.to_dict()

            # Generate plots
            plotter = GenerateEvaluationPlots(output_dir=str(self.output_dir))
            results_dicts = [r.to_dict() for r in detailed_metrics]

            plotter.generate_all_plots(
                statistics=stats_dicts,
                results=results_dicts,
                gnn_results={}
            )

            logger.info("\n✓ All evaluation plots generated successfully")
            logger.info(f"  Plots saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to generate evaluation plots: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def run_comprehensive_evaluation(
            self,
            domain_file: str,
            problem_pattern: str,
            model_path: str,
            timeout_sec: int = 300,
            include_baselines: bool = True,
            include_gnn_random: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation: baselines + GNN + Random + analysis.

        Orchestrates:
        1. Baseline evaluation
        2. GNN and Random evaluation
        3. Comprehensive analysis
        """
        logger.info("\n" + "=" * 100)
        logger.info("COMPREHENSIVE EVALUATION FRAMEWORK")
        logger.info("=" * 100 + "\n")

        import glob

        # Load problems
        problems = sorted(glob.glob(problem_pattern))
        if not problems:
            logger.error("No problems found!")
            return {}

        logger.info(f"Found {len(problems)} problem(s)")

        # Run baselines
        if include_baselines:
            self.run_baseline_evaluation(timeout_sec=timeout_sec)

        # Run GNN and Random
        if include_gnn_random:
            try:
                gnn_random_result = self.run_gnn_and_random_evaluation(
                    model_path=model_path,
                    domain_file=domain_file,
                    problem_files=problems,
                    num_runs_per_problem=1,
                )
                if gnn_random_result.get("status") != "success":
                    logger.warning(f"GNN and Random evaluation failed: {gnn_random_result.get('error')}")
            except Exception as e:
                logger.warning(f"GNN and Random evaluation skipped: {e}")

        return {"status": "complete", "output_dir": str(self.output_dir)}