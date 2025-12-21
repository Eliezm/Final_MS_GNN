#!/usr/bin/env python3
"""Setup minimal test benchmarks for quick validation."""

import os
import shutil
from pathlib import Path


def setup_test_benchmarks():
    """Create test benchmarks with just 2-3 problems per size."""

    test_dir = Path("../benchmarks")
    test_dir.mkdir(exist_ok=True)

    # Copy only FIRST 2-3 problems from each size
    for size in ["small", "medium", "large"]:
        size_dir = Path("../misc/benchmarks") / size

        if not size_dir.exists():
            print(f"⚠️  Source {size_dir} doesn't exist, skipping")
            continue

        test_size_dir = test_dir / size
        test_size_dir.mkdir(exist_ok=True)

        # Copy domain
        domain_src = size_dir / "domain.pddl"
        domain_dst = test_size_dir / "domain.pddl"
        if domain_src.exists():
            shutil.copy(domain_src, domain_dst)
            print(f"✓ Copied {size} domain")

        # Copy ONLY first 2-3 problems
        max_problems = 2 if size != "small" else 3
        problems = sorted(size_dir.glob("problem_*.pddl"))[:max_problems]

        for prob in problems:
            shutil.copy(prob, test_size_dir / prob.name)
            print(f"  ✓ Copied {prob.name}")

    print("\n✅ Test benchmarks ready in ./benchmarks/")


if __name__ == "__main__":
    setup_test_benchmarks()
