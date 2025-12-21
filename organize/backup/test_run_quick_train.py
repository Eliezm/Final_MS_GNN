#!/usr/bin/env python3
"""Quick 1-minute model training to validate everything works."""

import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quick_training():
    """Train model for 1 minute on minimal data."""

    logger.info("\n" + "=" * 80)
    logger.info("QUICK TRAINING TEST (1-2 minutes)")
    logger.info("=" * 80 + "\n")

    # ====================================================================
    # STEP 1: Setup
    # ====================================================================

    logger.info("[1/3] SETUP")

    from shared_experiment_utils import (
        setup_logging, print_section, print_subsection,
        load_and_validate_benchmarks, get_benchmarks_for_sizes,
        ensure_directories_exist, train_gnn_model
    )

    ensure_directories_exist()

    # Load test benchmarks (only 2-3 problems per size)
    logger.info("  Loading test benchmarks...")
    all_benchmarks = load_and_validate_benchmarks(
        benchmark_dir="benchmarks",
        logger=logger
    )

    if not all_benchmarks:
        logger.error("❌ No benchmarks loaded!")
        return 1

    # Get ONLY small problems for speed
    train_benchmarks = get_benchmarks_for_sizes(
        all_benchmarks,
        sizes=["small"],
        max_problems_per_combination=3
    )

    logger.info(f"  Using {len(train_benchmarks)} test problems\n")

    # ====================================================================
    # STEP 2: Train
    # ====================================================================

    logger.info("[2/3] TRAINING")
    logger.info("  Training for 500 timesteps (very quick)...\n")

    output_dir = "test_model_output"
    Path(output_dir).mkdir(exist_ok=True)

    model_path = train_gnn_model(
        benchmarks=train_benchmarks,
        reward_variant="astar_search",
        total_timesteps=500,  # ✅ QUICK: Only 500 timesteps
        timesteps_per_problem=100,
        model_output_path=os.path.join(output_dir, "gnn_model_test.zip"),
        logger=logger,
        tb_log_name="quick_test_training"
    )

    if not model_path:
        logger.error("❌ Training failed!")
        return 1

    logger.info(f"✅ Model saved: {model_path}\n")

    # ====================================================================
    # STEP 3: Verify
    # ====================================================================

    logger.info("[3/3] VERIFICATION")

    if not os.path.exists(model_path):
        logger.error("❌ Model file doesn't exist after training!")
        return 1

    model_size_mb = os.path.getsize(model_path) / 1024 / 1024
    logger.info(f"  Model file size: {model_size_mb:.1f} MB")

    if model_size_mb < 0.1:
        logger.error("❌ Model file too small (likely corrupt)")
        return 1

    logger.info("✅ Model file valid\n")

    # ====================================================================
    # Summary
    # ====================================================================

    logger.info("=" * 80)
    logger.info("✅ QUICK TRAINING TEST PASSED")
    logger.info("=" * 80)
    logger.info(f"\nModel ready for evaluation: {model_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(test_quick_training())
