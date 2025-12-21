#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BENCHMARK DISCOVERY - Problem discovery and validation
====================================================
Discovers and filters benchmarks from directory structure.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from experiments.core.evaluation_config import EvaluationConfig

logger = logging.getLogger(__name__)


# ============================================================================
# SETUP VERIFICATION
# ============================================================================

def verify_fd_setup() -> bool:
    """
    Verify Fast Downward installation and configuration.

    Checks:
    - FD Translate script exists
    - FD Downward binary exists
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: VERIFY FAST DOWNWARD SETUP")
    logger.info("=" * 80)

    checks = [
        ("FD Translate Script", EvaluationConfig.FD_TRANSLATE_BIN),
        ("FD Downward Binary", EvaluationConfig.FD_DOWNWARD_BIN),
    ]

    all_ok = True
    for name, path in checks:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {name:<30} {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        logger.error("\n❌ FD setup verification FAILED")
        logger.error("   Please ensure Fast Downward is built in downward/builds/release/")
        return False

    logger.info("\n✅ FD setup verified")
    return True


# ============================================================================
# BENCHMARK DISCOVERY
# ============================================================================

def discover_benchmarks() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Discover benchmarks from directory structure.

    Expected structure:
        benchmarks/
            {domain}/
                {size}/
                    domain.pddl
                    problem_{size}_00.pddl
                    problem_{size}_01.pddl
                    ...

    Returns:
        Dictionary mapping "Domain Size" -> [(domain, problem, problem_id), ...]
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: DISCOVER BENCHMARK PROBLEMS")
    logger.info("=" * 80)

    all_benchmarks = {}

    for domain_name, domain_config in sorted(EvaluationConfig.BENCHMARK_DOMAINS.items()):
        domain_base_path = domain_config["path"]
        problem_prefix = domain_config["problem_prefix"]

        if not os.path.exists(domain_base_path):
            logger.warning(f"  ⚠️  Domain directory not found: {domain_base_path}")
            continue

        logger.info(f"\nDiscovering domain: {domain_name.upper()}")

        for size in EvaluationConfig.PROBLEM_SIZES:
            size_dir = os.path.join(domain_base_path, size)

            if not os.path.exists(size_dir):
                logger.debug(f"    ⚠️  Size directory not found: {size_dir}")
                continue

            domain_file = os.path.join(size_dir, "domain.pddl")
            if not os.path.exists(domain_file):
                logger.warning(f"    ⚠️  Domain file not found: {domain_file}")
                continue

            # Find all problems
            problem_pattern = os.path.join(size_dir, f"{problem_prefix}{size}_*.pddl")
            problems = sorted(glob.glob(problem_pattern))

            if not problems:
                logger.debug(f"    ⚠️  No problems found matching: {problem_pattern}")
                continue

            benchmark_id = f"{domain_name.capitalize()} {size.capitalize()}"
            logger.info(f"  {size.capitalize():<8} {len(problems):3d} problems")

            benchmarks = [
                (
                    os.path.abspath(domain_file),
                    os.path.abspath(prob),
                    os.path.basename(prob).replace(".pddl", "")
                )
                for prob in problems
            ]

            all_benchmarks[benchmark_id] = benchmarks

    if not all_benchmarks:
        logger.error("\n❌ No benchmarks discovered")
        return {}

    logger.info(f"\n✅ Discovered {len(all_benchmarks)} benchmark set(s)")
    total_problems = sum(len(probs) for probs in all_benchmarks.values())
    logger.info(f"   Total problems: {total_problems}")

    return all_benchmarks


# ============================================================================
# BENCHMARK FILTERING
# ============================================================================

def filter_benchmarks(
        all_benchmarks: Dict[str, List[Tuple[str, str, str]]],
        domains: Optional[str],
        sizes: Optional[str]
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Filter benchmarks by domain and size.

    Args:
        all_benchmarks: Dict from discover_benchmarks()
        domains: Comma-separated domains: blocksworld,logistics
        sizes: Comma-separated sizes: small,medium,large

    Returns:
        Filtered benchmarks dict with only matching keys
    """
    if not domains and not sizes:
        return all_benchmarks

    domain_set = set(d.lower() for d in domains.split(",")) if domains else None
    size_set = set(s.lower() for s in sizes.split(",")) if sizes else None

    filtered = {}
    for key, value in all_benchmarks.items():
        parts = key.split()
        if len(parts) >= 2:
            domain = parts[0].lower()
            size = parts[1].lower()

            if domain_set and domain not in domain_set:
                continue
            if size_set and size not in size_set:
                continue

            filtered[key] = value

    logger.info(f"Filtered benchmarks by domain {domains} / size {sizes}: {len(filtered)} combinations")
    return filtered