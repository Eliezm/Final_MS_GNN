#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROBLEM GENERATION FRAMEWORK - MAIN COORDINATOR
==============================================

Main interface for generating and validating benchmark problems.

Usage:
    python problem_generation_framework.py \
        --domain blocksworld \
        --output-dir ./benchmarks \
        --num-per-difficulty 10 \
        --fd-build-dir ./fast-downward
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import time

from blocksworld_generator import BlocksworldGenerator
from baseline_validator import BaselineValidator
from generator_utils import export_metadata, get_benchmark_statistics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProblemGenerationFramework:
    """Main coordinator for problem generation and validation."""

    def __init__(
            self,
            domain: str,
            output_dir: str,
            fd_build_dir: str = None
    ):
        """
        Initialize framework.

        Args:
            domain: "blocksworld", "logistics", or "parking"
            output_dir: Output directory for generated problems
            fd_build_dir: Path to Fast Downward build
        """
        self.domain = domain
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize generator based on domain
        if domain == "blocksworld":
            self.generator = BlocksworldGenerator()
        else:
            logger.error(f"Domain not yet supported: {domain}")
            raise ValueError(f"Unsupported domain: {domain}")

        # Initialize validator
        self.validator = BaselineValidator(fd_build_dir=fd_build_dir)

        # Statistics
        self.generated_problems = defaultdict(lambda: defaultdict(list))
        self.validation_results = defaultdict(lambda: defaultdict(list))

    def generate_benchmark(
            self,
            num_per_difficulty: int = 10,
            max_attempts_per_problem: int = 5,
            characteristics: List[str] = None
    ) -> bool:
        """
        Generate complete benchmark with problems at each difficulty level.

        Args:
            num_per_difficulty: Target number of problems per difficulty
            max_attempts_per_problem: Max generation attempts per problem
            characteristics: List of characteristics to sample from (None = all)

        Returns:
            True if successful
        """
        logger.info(f"=" * 70)
        logger.info(f"GENERATING BENCHMARK FOR: {self.domain.upper()}")
        logger.info(f"=" * 70)
        logger.info(f"Target: {num_per_difficulty} problems per difficulty")
        logger.info(f"Max attempts per problem: {max_attempts_per_problem}")

        difficulties = ["small", "medium", "large"]

        if characteristics is None:
            characteristics = list(self.generator.CHARACTERISTICS.keys())

        logger.info(f"Using characteristics: {len(characteristics)}")

        # Generate problems for each difficulty
        for difficulty in difficulties:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Generating {difficulty.upper()} problems")
            logger.info(f"{'=' * 70}")

            generated_count = 0
            valid_count = 0
            attempt = 0

            while valid_count < num_per_difficulty:
                attempt += 1
                if attempt > max_attempts_per_problem * num_per_difficulty:
                    logger.warning(f"Exceeded max attempts for {difficulty}")
                    break

                # Select random characteristic
                import random
                characteristic = random.choice(characteristics)

                logger.info(f"\nAttempt {attempt}: {characteristic}")

                # Generate problem
                problem_file, metadata = self.generator.generate_problem(
                    characteristic=characteristic,
                    difficulty=difficulty
                )

                if problem_file is None:
                    logger.warning("Generation failed")
                    continue

                generated_count += 1

                # Validate problem
                domain_file = self._get_domain_file()
                is_valid, classification = self.validator.validate_problem(
                    domain_file, problem_file
                )

                # Update metadata with validation results
                if metadata is not None:
                    metadata.update(classification)

                    # Organize output
                    problem_dir = self.output_dir / difficulty / f"problem_{valid_count + 1:03d}"
                    problem_dir.mkdir(parents=True, exist_ok=True)

                    # Copy problem file
                    import shutil
                    dst_problem = problem_dir / "problem.pddl"
                    shutil.copy(problem_file, dst_problem)

                    # Export metadata
                    metadata_file = problem_dir / "metadata.json"
                    export_metadata(metadata, str(metadata_file))

                    logger.info(f"✓ Saved to {problem_dir}")

                    if is_valid:
                        valid_count += 1
                        self.generated_problems[difficulty][characteristic].append({
                            "file": str(dst_problem),
                            "metadata": metadata
                        })

                        logger.info(f"✓ Valid! ({valid_count}/{num_per_difficulty})")

                    else:
                        logger.warning(f"✗ Invalid or extreme difficulty")
                        logger.warning(f"  Reason: {classification.get('error', 'unknown')}")

                # Clean up temp file
                if problem_file and os.path.exists(problem_file):
                    os.unlink(problem_file)

        # Report results
        self._report_results()
        return True

    def validate_specific_problem(
            self,
            problem_size: str,
            problem_num: int
    ) -> Dict[str, Any]:
        """
        Validate a specific generated problem.

        Args:
            problem_size: "small", "medium", or "large"
            problem_num: Problem number (1-indexed)

        Returns:
            Validation results dictionary
        """
        problem_dir = self.output_dir / problem_size / f"problem_{problem_num:03d}"
        problem_file = problem_dir / "problem.pddl"

        if not problem_file.exists():
            logger.error(f"Problem not found: {problem_file}")
            return {}

        logger.info(f"Validating: {problem_file}")

        domain_file = self._get_domain_file()
        is_valid, classification = self.validator.validate_problem(
            domain_file, str(problem_file)
        )

        logger.info(f"Result: {classification}")

        return classification

    def _get_domain_file(self) -> str:
        """Get path to domain PDDL file."""
        domain_file = Path(__file__).parent / f"{self.domain}.pddl"
        if not domain_file.exists():
            # Try current directory
            domain_file = Path(self.domain).with_suffix(".pddl")
        return str(domain_file)

    def _report_results(self):
        """Report generation and validation results."""
        logger.info(f"\n{'=' * 70}")
        logger.info(f"BENCHMARK GENERATION COMPLETE")
        logger.info(f"{'=' * 70}")

        for difficulty in ["small", "medium", "large"]:
            probs = self.generated_problems[difficulty]
            total = sum(len(v) for v in probs.values())
            logger.info(f"{difficulty.upper():8} : {total:3} problems")

            for characteristic, problems in probs.items():
                logger.info(f"  - {characteristic:30} : {len(problems):3}")

        # Statistics
        logger.info(f"\n{'=' * 70}")
        logger.info(f"BENCHMARK STATISTICS")
        logger.info(f"{'=' * 70}")

        stats = get_benchmark_statistics(str(self.output_dir))

        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v}")
            else:
                logger.info(f"{key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate and validate PDDL benchmark problems"
    )

    parser.add_argument(
        "--domain",
        default="blocksworld",
        choices=["blocksworld", "logistics", "parking"],
        help="Domain to generate problems for"
    )

    parser.add_argument(
        "--output-dir",
        default="./benchmarks",
        help="Output directory for generated problems"
    )

    parser.add_argument(
        "--num-per-difficulty",
        type=int,
        default=10,
        help="Target number of problems per difficulty level"
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="Maximum generation attempts per problem"
    )

    parser.add_argument(
        "--characteristics",
        help="Comma-separated list of characteristics to use (default: all)"
    )

    parser.add_argument(
        "--fd-build-dir",
        help="Path to Fast Downward build directory"
    )

    parser.add_argument(
        "--validate-only",
        nargs=2,
        metavar=("SIZE", "NUM"),
        help="Validate specific problem: --validate-only small 1"
    )

    args = parser.parse_args()

    try:
        framework = ProblemGenerationFramework(
            domain=args.domain,
            output_dir=args.output_dir,
            fd_build_dir=args.fd_build_dir
        )

        if args.validate_only:
            # Validate specific problem
            size, num = args.validate_only
            framework.validate_specific_problem(size, int(num))

        else:
            # Generate benchmark
            characteristics = None
            if args.characteristics:
                characteristics = [c.strip() for c in args.characteristics.split(",")]

            framework.generate_benchmark(
                num_per_difficulty=args.num_per_difficulty,
                max_attempts_per_problem=args.max_attempts,
                characteristics=characteristics
            )

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()