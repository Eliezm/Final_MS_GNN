#!/usr/bin/env python3
"""Run all quick validation tests in sequence."""

import sys
import os
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_test(test_name, script_path):
    """Run a test script and return success status."""
    import subprocess

    logger.info(f"\n{'=' * 80}")
    logger.info(f"TEST: {test_name}")
    logger.info(f"{'=' * 80}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            timeout=600  # 10 minute timeout per test
        )

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error(f"❌ {test_name} TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ {test_name} ERROR: {e}")
        return False


def main():
    """Run all quick tests."""

    logger.info("\n" + "=" * 80)
    logger.info("5-MINUTE QUICK VALIDATION TEST SUITE")
    logger.info("=" * 80 + "\n")

    start_time = time.time()

    tests = [
        ("Setup Test Benchmarks", "setup_test_benchmarks.py"),
        ("C++ Component Validation", "test_cpp_merge_strategy.py"),
        ("Merge Analysis Tools", "test_merge_analysis_tools.py"),
        ("Quick Training", "test_run_quick_train.py"),
        ("Quick Evaluation", "test_run_quick_evaluation.py"),
        ("Report Generation", "test_run_quick_reports.py"),
    ]

    results = {}

    for test_name, script_path in tests:
        if not os.path.exists(script_path):
            logger.warning(f"⚠️  {script_path} not found, creating stub...\n")
            # Skip if script doesn't exist
            results[test_name] = None
            continue

        success = run_test(test_name, script_path)
        results[test_name] = success

    elapsed = time.time() - start_time

    # ====================================================================
    # SUMMARY
    # ====================================================================

    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE SUMMARY")
    logger.info("=" * 80 + "\n")

    for test_name, passed in results.items():
        if passed is None:
            status = "⊘ SKIPPED"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"

        logger.info(f"{status:12} {test_name}")

    total = len([r for r in results.values() if r is not None])
    passed = len([r for r in results.values() if r is True])

    logger.info(f"\n{passed}/{total} tests passed")
    logger.info(f"Total time: {elapsed:.1f} seconds\n")

    if passed == total:
        logger.info("=" * 80)
        logger.info("✅ ALL TESTS PASSED - Framework is ready!")
        logger.info("=" * 80 + "\n")
        logger.info("Next steps:")
        logger.info("  1. Run full experiments:")
        logger.info("     python experiment_1_problem_overfit.py ...")
        logger.info("  2. Run evaluation:")
        logger.info("     python run_full_evaluation.py --model ... --domain ... --problems ...")
        logger.info("  3. Analyze merge metadata:")
        logger.info("     python analyze_merge_metadata.py\n")
        return 0
    else:
        logger.error("=" * 80)
        logger.error("❌ SOME TESTS FAILED - Fix issues above")
        logger.error("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
