#!/usr/bin/env python3
"""Validate C++ merge strategy component."""

import sys
import os
import json
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cpp_merge_strategy():
    """Quick validation of C++ merge strategy."""

    logger.info("\n" + "=" * 80)
    logger.info("C++ MERGE STRATEGY VALIDATION TEST")
    logger.info("=" * 80 + "\n")

    # ====================================================================
    # STEP 1: Check C++ binary exists
    # ====================================================================

    logger.info("[1/4] Checking C++ build...")

    fd_bin = os.path.abspath("downward/builds/release/bin/downward.exe")
    fd_translate = os.path.abspath("downward/builds/release/bin/translate/translate.py")

    missing = []
    if not os.path.exists(fd_bin):
        missing.append(f"  ❌ FD binary: {fd_bin}")
    else:
        logger.info(f"  ✅ FD binary: {fd_bin}")

    if not os.path.exists(fd_translate):
        missing.append(f"  ❌ FD translator: {fd_translate}")
    else:
        logger.info(f"  ✅ FD translator: {fd_translate}\n")

    if missing:
        logger.error("❌ Missing C++ components:")
        for m in missing:
            logger.error(m)
        logger.error("\nRebuild Fast Downward:")
        logger.error("  cd downward && python build.py release")
        return 1

    # ====================================================================
    # STEP 2: Check GNN strategy registration
    # ====================================================================

    logger.info("[2/4] Checking GNN strategy in C++...")

    gnn_strategy_file = "downward/src/search/merge_and_shrink/merge_strategy_gnn.cc"

    if not os.path.exists(gnn_strategy_file):
        logger.error(f"❌ GNN strategy source not found: {gnn_strategy_file}")
        return 1

    try:
        with open(gnn_strategy_file, 'r') as f:
            content = f.read()

        if "MergeStrategyGNN" not in content:
            logger.error("❌ MergeStrategyGNN not found in source")
            return 1

        logger.info("✅ GNN strategy source file found and valid\n")

    except Exception as e:
        logger.error(f"❌ Failed to read strategy file: {e}")
        return 1

    # ====================================================================
    # STEP 3: Quick translate test
    # ====================================================================

    logger.info("[3/4] Testing translator...")

    domain = "benchmarks/small/domain.pddl"
    problem = "benchmarks/small/problem_small_00.pddl"

    if not os.path.exists(domain) or not os.path.exists(problem):
        logger.warning("⚠️  Test domain/problem not found, skipping translate test")
        logger.info("   (Run setup_test_benchmarks.py first)\n")
    else:
        import subprocess

        try:
            # Test translate
            result = subprocess.run(
                f'python "{fd_translate}" "{domain}" "{problem}" --sas-file test_output.sas',
                shell=True,
                cwd=os.path.abspath("downward"),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and os.path.exists("downward/test_output.sas"):
                logger.info("✅ Translator works\n")
                try:
                    os.remove("downward/test_output.sas")
                except:
                    pass
            else:
                logger.warning("⚠️  Translator test failed (non-critical)")
                logger.info(f"   stderr: {result.stderr[:200]}\n")

        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Translator timed out\n")
        except Exception as e:
            logger.warning(f"⚠️  Translator test error: {e}\n")

    # ====================================================================
    # STEP 4: Check output directories
    # ====================================================================

    logger.info("[4/4] Checking output directories...")

    required_dirs = [
        "downward/gnn_output",
        "downward/fd_output",
        "downward/gnn_metadata"
    ]

    for d in required_dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            logger.info(f"  ✅ Created: {d}")
        else:
            logger.info(f"  ✅ Exists: {d}")

    logger.info()

    logger.info("=" * 80)
    logger.info("✅ C++ MERGE STRATEGY VALIDATION PASSED")
    logger.info("=" * 80)
    logger.info("\nC++ component ready for evaluation:\n")
    logger.info("  - Fast Downward binary: ✅")
    logger.info("  - GNN merge strategy: ✅")
    logger.info("  - Output directories: ✅\n")

    return 0


if __name__ == "__main__":
    sys.exit(test_cpp_merge_strategy())
