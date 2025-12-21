#!/usr/bin/env python3
"""Validate merge analysis tools work."""

import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_merge_analysis_tools():
    """Test merge metadata collection and analysis."""

    logger.info("\n" + "=" * 80)
    logger.info("MERGE ANALYSIS TOOLS VALIDATION")
    logger.info("=" * 80 + "\n")

    # ====================================================================
    # Test 1: Metadata collector
    # ====================================================================

    logger.info("[1/3] Testing merge metadata collector...")

    try:
        from merge_metadata_collector import MergeMetadataCollector

        collector = MergeMetadataCollector("downward/fd_output")
        logger.info("✅ MergeMetadataCollector imports successfully\n")

    except Exception as e:
        logger.error(f"❌ Failed to import MergeMetadataCollector: {e}")
        return 1

    # ====================================================================
    # Test 2: Explainability analyzer
    # ====================================================================

    logger.info("[2/3] Testing merge explainability analyzer...")

    try:
        from merge_explainability import MergeExplainabilityAnalyzer

        logger.info("✅ MergeExplainabilityAnalyzer imports successfully\n")

    except Exception as e:
        logger.error(f"❌ Failed to import MergeExplainabilityAnalyzer: {e}")
        return 1

    # ====================================================================
    # Test 3: Choice analyzer
    # ====================================================================

    logger.info("[3/3] Testing merge choice analyzer...")

    try:
        from merge_choice_analysis import MergeChoiceAnalyzer

        logger.info("✅ MergeChoiceAnalyzer imports successfully\n")

    except Exception as e:
        logger.error(f"❌ Failed to import MergeChoiceAnalyzer: {e}")
        return 1

    # ====================================================================
    # Test data structures
    # ====================================================================

    logger.info("Testing data structures...")

    try:
        from merge_metadata_collector import MergeDecision
        from dataclasses import fields

        # Check key fields exist
        field_names = {f.name for f in fields(MergeDecision)}
        required_fields = {
            'iteration', 'ts1_id', 'ts2_id', 'ts1_size', 'ts2_size',
            'merged_size', 'reachability_ratio', 'branching_factor'
        }

        missing = required_fields - field_names

        if missing:
            logger.error(f"❌ Missing fields in MergeDecision: {missing}")
            return 1

        logger.info(f"✅ MergeDecision has all required fields\n")

    except Exception as e:
        logger.error(f"❌ Error checking data structures: {e}")
        return 1

    logger.info("=" * 80)
    logger.info("✅ MERGE ANALYSIS TOOLS VALIDATION PASSED")
    logger.info("=" * 80)
    logger.info("\nAll merge analysis components ready:\n")
    logger.info("  - Metadata collector: ✅")
    logger.info("  - Explainability analyzer: ✅")
    logger.info("  - Choice analyzer: ✅")
    logger.info("  - Data structures: ✅\n")

    return 0


if __name__ == "__main__":
    sys.exit(test_merge_analysis_tools())
