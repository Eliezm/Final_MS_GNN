#!/usr/bin/env python3
"""Quick evaluation on minimal test set."""

import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quick_evaluation():
    """Evaluate model on 2-3 test problems."""

    logger.info("\n" + "=" * 80)
    logger.info("QUICK EVALUATION TEST (2-3 minutes)")
    logger.info("=" * 80 + "\n")

    model_path = "test_model_output/gnn_model_test.zip"

    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        logger.error("   Run test_run_quick_train.py first")
        return 1

    logger.info(f"[1/2] Loading model: {model_path}")

    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        logger.info("✅ Model loaded successfully\n")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return 1

    # ====================================================================
    # Test on small benchmark
    # ====================================================================

    logger.info("[2/2] QUICK EVALUATION")

    from shared_experiment_utils import (
        load_and_validate_benchmarks,
        get_benchmarks_for_sizes,
        evaluate_model_on_problems
    )

    all_benchmarks = load_and_validate_benchmarks(
        benchmark_dir="benchmarks",
        logger=logger
    )

    test_benchmarks = get_benchmarks_for_sizes(
        all_benchmarks,
        sizes=["small"],
        max_problems_per_combination=2  # ✅ Only 2 problems
    )

    logger.info(f"  Evaluating on {len(test_benchmarks)} test problems...\n")

    results = evaluate_model_on_problems(
        model_path=model_path,
        benchmarks=test_benchmarks,
        reward_variant="astar_search",
        logger=logger
    )

    # ====================================================================
    # Report
    # ====================================================================

    logger.info("\n" + "-" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("-" * 80)
    logger.info(f"  Total problems: {results.get('total_problems', 0)}")
    logger.info(f"  Solved: {results.get('solved_count', 0)}")
    logger.info(f"  Solve rate: {results.get('solve_rate', 0):.1f}%")
    logger.info(f"  Avg reward: {results.get('avg_reward', 0):.4f}")
    logger.info(f"  Avg time: {results.get('avg_time', 0):.2f}s\n")

    # Save results
    output_dir = "test_evaluation_output"
    Path(output_dir).mkdir(exist_ok=True)

    import json
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"✅ Results saved: {results_file}\n")

    logger.info("=" * 80)
    logger.info("✅ QUICK EVALUATION TEST PASSED")
    logger.info("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(test_quick_evaluation())
