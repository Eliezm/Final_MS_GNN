# time_based_generator.py
"""
Time-based problem generation with automatic calibration.

Generates problems targeting specific solving time ranges.
Uses iterative calibration to discover working configurations.
"""

import logging
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import os

from config import (
    TIME_DIFFICULTY_TIERS,
    CALIBRATION_CONFIG,
    DIFFICULTY_TIERS,
)
from backward_generator import BackwardProblemGenerator
from pddl_writer import PDDLWriter
from baseline_planner import FastDownwardRunner
from metadata_store import MetadataStore, ProblemMetadata
from validator import PDDLValidator
from goal_archetypes import GoalArchetype
from actions import ActionExecutor

logger = logging.getLogger(__name__)


@dataclass
class ConfigPerformance:
    """Tracks performance of a (num_blocks, plan_length) configuration."""
    num_blocks: int
    plan_length: int
    test_count: int = 0
    total_time: float = 0.0
    success_count: int = 0
    times: List[float] = None  # For statistics

    def __post_init__(self):
        if self.times is None:
            self.times = []

    @property
    def avg_time(self) -> Optional[float]:
        """Average solving time for this config."""
        return self.total_time / self.test_count if self.test_count > 0 else None

    @property
    def success_rate(self) -> float:
        """Fraction of attempts that succeeded."""
        return self.success_count / self.test_count if self.test_count > 0 else 0.0

    def record_result(self, solve_time: float, success: bool):
        """Record a test result."""
        self.test_count += 1
        if success:
            self.success_count += 1
            self.total_time += solve_time
            self.times.append(solve_time)

    def is_viable(self, time_tier) -> bool:
        """Check if this config is in acceptable time range."""
        if self.success_rate < 0.5:
            return False  # Too many failures
        return (self.avg_time is not None and
                time_tier.min_time <= self.avg_time <= time_tier.max_time)

    def is_too_hard(self, time_tier) -> bool:
        """Check if this config is consistently too hard."""
        if self.avg_time is None:
            return False
        return self.avg_time > time_tier.max_rejects

    def __repr__(self):
        return (f"Config(blocks={self.num_blocks}, "
                f"plan_len={self.plan_length}, "
                f"avg_time={self.avg_time:.1f}s, "
                f"success_rate={self.success_rate:.0%})")


class TimeBasedProblemGenerator:
    """
    Generate problems targeting specific solving time ranges.

    Uses automated calibration to discover which configurations
    (num_blocks, plan_length) produce the desired solving times.
    """

    def __init__(self,
                 difficulty: str,
                 domain_dir: str,
                 problems_dir: str,
                 metadata_dir: str,
                 random_seed: int = None):
        """
        Initialize the time-based generator.

        Args:
            difficulty: 'small', 'medium', or 'large'
            domain_dir: Directory for domain files
            problems_dir: Directory for problem files
            metadata_dir: Directory for metadata
            random_seed: Random seed for reproducibility
        """
        self.difficulty = difficulty
        self.domain_dir = domain_dir
        self.problems_dir = problems_dir
        self.metadata_dir = metadata_dir

        if random_seed is not None:
            random.seed(random_seed)

        self.time_tier = TIME_DIFFICULTY_TIERS[difficulty]
        self.backward_gen = BackwardProblemGenerator(random_seed=random_seed)
        self.pddl_writer = PDDLWriter()
        self.fd_runner = FastDownwardRunner()
        self.validator = PDDLValidator()
        self.metadata_store = MetadataStore(metadata_dir)

        # Configuration performance tracking
        self.config_performance: Dict[Tuple[int, int], ConfigPerformance] = {}
        self.viable_configs: List[ConfigPerformance] = []

        # Statistics
        self.problems_generated = 0
        self.problems_accepted = 0
        self.calibration_attempts = 0

    def _generate_test_problem(self,
                               num_blocks: int,
                               target_plan_length: int,
                               archetype: Optional[GoalArchetype] = None
                               ) -> Tuple[Optional[Dict], float]:
        """
        Generate and test a single problem.

        Returns:
            (result_dict, solve_time) or (None, timeout)
        """
        self.problems_generated += 1

        try:
            # Generate the problem
            initial_state, goal_state, plan, arch = (
                self.backward_gen.generate_problem(
                    num_blocks=num_blocks,
                    target_plan_length=target_plan_length,
                    archetype=archetype,
                    tolerance=2
                )
            )

            problem_name = (f"{self.difficulty}-test-"
                            f"b{num_blocks}_l{target_plan_length}-"
                            f"{self.problems_generated}")

            # Write PDDL
            domain_file = os.path.join(self.domain_dir, "blocksworld.pddl")
            problem_file = os.path.join(self.problems_dir, f"{problem_name}.pddl")

            if self.problems_generated == 1:
                self.pddl_writer.write_domain(domain_file)

            self.pddl_writer.write_problem(
                problem_file, problem_name, initial_state, goal_state
            )

            # Validate PDDL
            is_valid, error = self.validator.validate_problem(domain_file, problem_file)
            if not is_valid:
                logger.debug(f"PDDL validation failed: {error}")
                return None, 0

            # Run planner
            planner_result = self.fd_runner.run_problem(
                domain_file,
                problem_file,
                search_config="astar(lmcut())",
                timeout=int(self.time_tier.max_rejects)
            )

            solve_time = planner_result.get('time', 0)
            success = planner_result.get('success', False)

            # Clean up test file (optional)
            try:
                os.remove(problem_file)
            except:
                pass

            if success:
                return {
                    'initial': initial_state,
                    'goal': goal_state,
                    'plan': plan,
                    'archetype': arch,
                    'plan_length': len(plan),
                }, solve_time
            else:
                return None, solve_time

        except Exception as e:
            logger.debug(f"Problem generation failed: {e}")
            return None, 0

    def calibrate(self) -> bool:
        """
        Calibration phase: discover which configurations work.

        Returns:
            True if viable configurations found, False otherwise
        """
        print(f"\n{'=' * 70}")
        print(f"CALIBRATION PHASE: {self.difficulty.upper()}")
        print(f"Target time: {self.time_tier.min_time:.0f}s - "
              f"{self.time_tier.max_time:.0f}s")
        print(f"{'=' * 70}\n")

        configs_to_try = []

        # Generate candidate configurations
        plan_length_tier = DIFFICULTY_TIERS[self.difficulty]

        for num_blocks in CALIBRATION_CONFIG['block_range']:
            for plan_len in range(
                    plan_length_tier.min_length,
                    plan_length_tier.max_length + 1,
                    CALIBRATION_CONFIG['plan_length_step']
            ):
                configs_to_try.append((num_blocks, plan_len))

        # Test configurations
        for num_blocks, plan_len in configs_to_try:
            config_key = (num_blocks, plan_len)
            config = ConfigPerformance(num_blocks, plan_len)

            # Test this config a few times
            for attempt in range(CALIBRATION_CONFIG['initial_samples_per_config']):
                problem_result, solve_time = self._generate_test_problem(
                    num_blocks, plan_len
                )

                success = problem_result is not None
                config.record_result(solve_time, success)

                if config.is_too_hard(self.time_tier):
                    logger.debug(f"Config too hard, stopping tests")
                    break

            self.config_performance[config_key] = config

            # Check if viable
            if config.is_viable(self.time_tier):
                self.viable_configs.append(config)
                status = "✓ VIABLE"
            elif config.is_too_hard(self.time_tier):
                status = "✗ TOO HARD"
            else:
                status = "✗ Out of range"

            print(f"  {config}  {status}")

        # Summary
        print(f"\n{'=' * 70}")
        print(f"Calibration Results:")
        print(f"  Tested: {len(self.config_performance)} configurations")
        print(f"  Viable: {len(self.viable_configs)} configurations")
        print(f"{'=' * 70}\n")

        if self.viable_configs:
            print("VIABLE CONFIGURATIONS:")
            for config in sorted(self.viable_configs,
                                 key=lambda c: c.avg_time or 0):
                print(f"  • {config}")
            print()
            return True
        else:
            print("⚠ No viable configurations found!")
            print("Consider adjusting TIME_DIFFICULTY_TIERS in config.py\n")
            return False

    def generate_batch(self, target_count: int) -> List[str]:
        """
        Generation phase: create N problems fitting the difficulty tier.

        Args:
            target_count: Generate this many problems

        Returns:
            List of generated problem names
        """
        if not self.viable_configs:
            print("✗ No viable configurations. Run calibration first!")
            return []

        print(f"\n{'=' * 70}")
        print(f"GENERATION PHASE: {self.difficulty.upper()}")
        print(f"Target: {target_count} problems in range "
              f"{self.time_tier.min_time:.0f}s - {self.time_tier.max_time:.0f}s")
        print(f"Max reject threshold: {self.time_tier.max_rejects:.0f}s")
        print(f"{'=' * 70}\n")

        # Cycle through archetypes for variety
        archetypes = list(GoalArchetype)
        archetype_idx = 0

        problem_names = []
        attempt = 0
        max_attempts = target_count * 5  # Safety limit

        domain_file = os.path.join(self.domain_dir, "blocksworld.pddl")
        self.pddl_writer.write_domain(domain_file)

        while (len(problem_names) < target_count and
               attempt < max_attempts):
            attempt += 1

            # Select config (random from viable)
            config = random.choice(self.viable_configs)
            archetype = archetypes[archetype_idx % len(archetypes)]
            archetype_idx += 1

            # Generate problem
            problem_result, solve_time = self._generate_test_problem(
                config.num_blocks,
                config.plan_length,
                archetype
            )

            if problem_result is None:
                print(f"  Attempt {attempt}: Failed to generate", end='\r',
                      flush=True)
                continue

            # Check if in acceptable range
            in_range = (self.time_tier.min_time <= solve_time <=
                        self.time_tier.max_time)
            too_hard = solve_time > self.time_tier.max_rejects

            if too_hard:
                status = "✗ TOO HARD (rejected)"
                accepted = False
            elif in_range:
                status = f"✓ ACCEPTED ({solve_time:.1f}s)"
                accepted = True
            else:
                status = f"⚠ Out of range ({solve_time:.1f}s)"
                accepted = False

            # Generate permanent problem file if accepted
            if accepted:
                self.problems_accepted += 1
                problem_name = (
                    f"{self.difficulty}-{self.problems_accepted}-"
                    f"{archetype.value}-b{config.num_blocks}"
                )
                problem_names.append(problem_name)

                problem_file = os.path.join(
                    self.problems_dir,
                    f"{problem_name}.pddl"
                )

                self.pddl_writer.write_problem(
                    problem_file,
                    problem_name,
                    problem_result['initial'],
                    problem_result['goal']
                )

                # Store metadata
                metadata = ProblemMetadata(
                    problem_name=problem_name,
                    domain="blocksworld",
                    difficulty=self.difficulty,
                    num_blocks=config.num_blocks,
                    goal_archetype=archetype.value,
                    plan_length=problem_result['plan_length'],
                    optimal_plan_cost=problem_result['plan_length'],
                    planner_time=solve_time,
                    planner_success=True,
                    nodes_expanded=0,
                    plan_cost=problem_result['plan_length'],
                    domain_file=domain_file,
                    problem_file=problem_file
                )
                self.metadata_store.save_metadata(metadata)

                progress = len(problem_names)
                print(f"  [{progress}/{target_count}] Problem: {problem_name:50s} "
                      f"{status}")
            else:
                print(f"  Attempt {attempt}: {status}", end='\r', flush=True)

        print(f"\n{'=' * 70}")
        print(f"Generation Complete!")
        print(f"  Generated: {len(problem_names)}/{target_count} problems")
        print(f"  Attempts: {attempt}")
        print(f"{'=' * 70}\n")

        return problem_names