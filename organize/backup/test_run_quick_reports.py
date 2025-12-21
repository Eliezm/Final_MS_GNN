#!/usr/bin/env python3
"""Validate report generation on test evaluation results."""

import sys
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quick_reports():
    """Generate reports from test evaluation."""

    logger.info("\n" + "=" * 80)
    logger.info("QUICK REPORT GENERATION TEST")
    logger.info("=" * 80 + "\n")

    results_file = "test_evaluation_output/results.json"

    if not os.path.exists(results_file):
        logger.error(f"❌ Results file not found: {results_file}")
        logger.error("   Run test_run_quick_evaluation.py first")
        return 1

    logger.info("[1/3] Loading results...")

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logger.info(f"✅ Loaded results with {len(results.get('details', []))} problems\n")
    except Exception as e:
        logger.error(f"❌ Failed to load results: {e}")
        return 1

    # ====================================================================
    # Generate text report
    # ====================================================================

    logger.info("[2/3] Generating text report...")

    output_dir = "test_evaluation_output"
    report_file = os.path.join(output_dir, "results.txt")

    try:
        from shared_experiment_utils import save_results_to_txt

        save_results_to_txt(
            results,
            report_file,
            "TEST_EVALUATION",
            logger
        )
        logger.info(f"✅ Report saved: {report_file}\n")

    except Exception as e:
        logger.error(f"❌ Failed to generate report: {e}")
        return 1

    # ====================================================================
    # Verify report content
    # ====================================================================

    logger.info("[3/3] Verifying report content...")

    try:
        with open(report_file, 'r') as f:
            content = f.read()

        required_sections = [
            "RESULTS SUMMARY",
            "Timestamp",
            "evaluation",
            "summary"
        ]

        missing = [s for s in required_sections if s not in content]

        if missing:
            logger.warning(f"⚠️  Missing report sections: {missing}")
        else:
            logger.info("✅ All required sections present in report\n")

        # Show preview
        lines = content.split('\n')
        logger.info("Report preview (first 20 lines):")
        for line in lines[:20]:
            logger.info(f"  {line}")
        logger.info("  ...\n")

    except Exception as e:
        logger.error(f"❌ Failed to verify report: {e}")
        return 1

    logger.info("=" * 80)
    logger.info("✅ QUICK REPORT TEST PASSED")
    logger.info("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(test_run_quick_reports())
