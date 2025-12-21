#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION MODULE - Refactored Main Entry Point
===============================================
UNIFIED: Baseline comparison, GNN evaluation, comprehensive metrics,
statistical analysis, and visualization.

This module maintains FULL BACKWARD COMPATIBILITY with the previous
monolithic evaluation.py while providing cleaner internal structure.

Features:
✓ Baseline Fast Downward comparison (7 configurations)
✓ Automatic benchmark discovery (recursively scan benchmarks/)
✓ FD setup verification before running
✓ GNN policy evaluation with ThinMergeEnv
✓ Detailed metrics tracking (25+ per run)
✓ Statistical significance testing
✓ Comprehensive reporting (CSV, JSON, TXT)
✓ Visualization and analysis plots
✓ H* preservation tracking (primary metric)
✓ Per-problem and aggregate statistics
✓ Speedup analysis vs baselines
✓ Scaling analysis
✓ Extended M&S-specific metrics
✓ Robust error handling and validation

REFACTORED STRUCTURE:
- evaluation_metrics.py:     Data structures
- evaluation_config.py:      Configuration
- baseline_runner.py:        FD execution
- benchmark_discovery.py:    Problem discovery
- evaluation_analyzer.py:    Analysis
- evaluation_export.py:      Reporting
- evaluation_orchestrator.py:Main framework
- evaluation.py:             Entry point (this file)
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# IMPORTS - Expose public API for backward compatibility
# ============================================================================

# Metrics
from experiments.core.evaluation_metrics import (
    ExtendedMetrics,
    DetailedMetrics,
    AggregateStatistics,
)

# Config
from experiments.core.evaluation_config import EvaluationConfig

# Baseline runner
from experiments.core.baseline_runner import (
    FastDownwardOutputParser,
    BaselineRunner,
)

# Benchmark discovery
from experiments.core.benchmark_discovery import (
    verify_fd_setup,
    discover_benchmarks,
    filter_benchmarks,
)

# Analyzer
from experiments.core.evaluation_analyzer import (
    ComparisonAnalyzer,
    run_baseline_on_benchmark_set,
)

# Export
from experiments.core.evaluation_export import (
    export_results_to_csv,
    export_statistics_to_json,
    export_summary_report,
    generate_reports,
)

# Main framework
from experiments.core.evaluation_orchestrator import EvaluationFramework

# GNN evaluation
from experiments.core.gnn_random_evaluation import (
    RandomMergePolicy,
    GNNPolicyEvaluator,
    RandomMergeEvaluator,
    GNNRandomEvaluationFramework,
)

# Plots
from experiments.core.evaluation_plots import GenerateEvaluationPlots

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(filename)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)

# ============================================================================
# BACKWARD COMPATIBILITY - Expose all public classes/functions
# ============================================================================

__all__ = [
    # Metrics
    "ExtendedMetrics",
    "DetailedMetrics",
    "AggregateStatistics",
    # Config
    "EvaluationConfig",
    # Baseline runner
    "FastDownwardOutputParser",
    "BaselineRunner",
    # Benchmark discovery
    "verify_fd_setup",
    "discover_benchmarks",
    "filter_benchmarks",
    # Analyzer
    "ComparisonAnalyzer",
    "run_baseline_on_benchmark_set",
    # Export
    "export_results_to_csv",
    "export_statistics_to_json",
    "export_summary_report",
    "generate_reports",
    # Main framework
    "EvaluationFramework",
    # GNN evaluation
    "RandomMergePolicy",
    "GNNPolicyEvaluator",
    "RandomMergeEvaluator",
    "GNNRandomEvaluationFramework",
    # Plots
    "GenerateEvaluationPlots",
]


# ============================================================================
# MAIN ENTRY POINT & CLI
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation Framework for GNN Merge Strategy Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluation.py --baselines --seed 42\n"
            "  python evaluation.py --baselines --max-problems 3\n"
            "  python evaluation.py --baselines --domains blocksworld --sizes small,medium\n"
        )
    )

    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Run baseline evaluation on discovered benchmarks"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        dest="time_limit",
        help="Time limit per problem in seconds (default: 300)"
    )

    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        dest="max_problems",
        help="Limit problems per benchmark set (for quick testing)"
    )

    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated domains: blocksworld,logistics (default: all)"
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated sizes: small,medium,large (default: all)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        dest="output_dir",
        help="Output directory (default: evaluation_results)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main execution orchestration."""
    args = parse_arguments()

    # Configure
    EvaluationConfig.TIME_LIMIT_PER_RUN_S = args.time_limit
    EvaluationConfig.RANDOM_SEED = args.seed

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Banner
    logger.info("\n" + "=" * 80)
    logger.info("GNN MERGE STRATEGY LEARNING - EVALUATION FRAMEWORK")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Seed:              {EvaluationConfig.RANDOM_SEED}")
    logger.info(f"  Time limit:        {EvaluationConfig.TIME_LIMIT_PER_RUN_S}s per problem")
    if args.max_problems:
        logger.info(f"  Max problems:      {args.max_problems} per set")
    if args.domains:
        logger.info(f"  Domains:           {args.domains}")
    if args.sizes:
        logger.info(f"  Sizes:             {args.sizes}")

    if args.baselines:
        # Run baseline evaluation
        framework = EvaluationFramework(output_dir=args.output_dir)
        result = framework.run_baseline_evaluation(
            timeout_sec=args.time_limit,
            max_problems=args.max_problems,
            domains=args.domains,
            sizes=args.sizes,
        )

        if result.get("status") == "success":
            return 0
        else:
            return 1
    else:
        logger.info("\n--baselines flag not provided")
        logger.info("Usage: python evaluation.py --baselines [options]")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)