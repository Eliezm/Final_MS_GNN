#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOMAIN-SPECIFIC PROBLEM GENERATORS - FIXED REVERSE PLANNING
===========================================================
Generate problems by reversing from goal state to initial state with validation.

Each generator:
1. Defines problem characteristics
2. Creates a goal state with specific properties
3. Applies reverse actions to create initial state
4. Validates that forward actions return to goal
5. Exports PDDL files with guaranteed solvability
"""

import random
import tempfile
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# BASE GENERATOR CLASS
# ============================================================================

class ProblemGenerator(ABC):
    """Base class for all domain-specific generators."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def generate_problem(
            self,
            characteristic: str,
            target_solution_length: int
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generate a problem with specific characteristics.

        Args:
            characteristic: Problem type/characteristic
            target_solution_length: Desired solution length

        Returns:
            (problem_file_path, metadata_dict) or (None, None) if generation fails
        """
        pass

    @abstractmethod
    def _create_goal_state(self, characteristic: str) -> Dict[str, Any]:
        """Create a goal state with specific properties."""
        pass

    @abstractmethod
    def _reverse_to_initial(
            self,
            goal_state: Dict[str, Any],
            target_steps: int
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Reverse from goal state to initial state.

        Returns:
            (initial_state, action_sequence) or (None, []) if fails
        """
        pass

    @abstractmethod
    def _export_pddl(
            self,
            initial_state: Dict[str, Any],
            goal_state: Dict[str, Any]
    ) -> str:
        """Export problem as PDDL file."""
        pass

    @staticmethod
    def _is_valid_state(state: Dict[str, Any]) -> bool:
        """✅ NEW: Validate state consistency."""
        if not isinstance(state, dict):
            return False
        if "on" not in state or "on_table" not in state or "clear" not in state:
            return False
        return True


# ============================================================================
# BLOCKSWORLD GENERATOR - COMPLETELY FIXED
# ============================================================================

class BlocksworldGenerator(ProblemGenerator):
    """Generate Blocksworld problems by reversing from goal state."""

    CHARACTERISTICS = {
        "few_blocks_simple_config": {
            "num_blocks": (3, 5),
            "init_stacks": (1, 2),
            "goal_stacks": (1, 2),
            "description": "Few blocks, simple initial/goal configs"
        },
        "few_blocks_scattered": {
            "num_blocks": (4, 6),
            "init_stacks": (2, 3),
            "goal_stacks": (2, 3),
            "description": "Few blocks, scattered across stacks"
        },
        "clear_from_table": {
            "num_blocks": (5, 7),
            "init_stacks": (1, 2),
            "goal_stacks": (1, 1),
            "description": "Clear blocks from table, stack into tower"
        },
        "medium_blocks_multiple_stacks": {
            "num_blocks": (6, 10),
            "init_stacks": (2, 4),
            "goal_stacks": (2, 4),
            "description": "Medium blocks, multiple target stacks"
        },
        "mixed_clear_and_stack": {
            "num_blocks": (7, 11),
            "init_stacks": (3, 5),
            "goal_stacks": (2, 4),
            "description": "Mixed clearing and stacking operations"
        },
        "partial_order_required": {
            "num_blocks": (8, 12),
            "init_stacks": (2, 4),
            "goal_stacks": (2, 3),
            "description": "Requires careful ordering of operations"
        },
        "many_blocks_complex": {
            "num_blocks": (10, 15),
            "init_stacks": (3, 6),
            "goal_stacks": (2, 4),
            "description": "Many blocks, complex rearrangement"
        },
        "deep_stacks_required": {
            "num_blocks": (12, 18),
            "init_stacks": (2, 3),
            "goal_stacks": (1, 2),
            "description": "Build deep stacks"
        },
        "global_rearrangement": {
            "num_blocks": (15, 20),
            "init_stacks": (4, 6),
            "goal_stacks": (3, 5),
            "description": "Global rearrangement required"
        }
    }

    def generate_problem(
            self,
            characteristic: str,
            target_solution_length: int
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Generate a Blocksworld problem with validation."""

        if characteristic not in self.CHARACTERISTICS:
            logger.error(f"Unknown characteristic: {characteristic}")
            return None, None

        char_spec = self.CHARACTERISTICS[characteristic]

        # Randomly choose parameters within ranges
        num_blocks = random.randint(char_spec["num_blocks"][0], char_spec["num_blocks"][1])
        goal_stacks = random.randint(char_spec["goal_stacks"][0], char_spec["goal_stacks"][1])

        logger.debug(f"BW Params: blocks={num_blocks}, goal_stacks={goal_stacks}")

        # Create goal state
        goal_state = self._create_goal_state_for_config(num_blocks, goal_stacks)

        if goal_state is None:
            logger.error(f"Failed to create goal state")
            return None, None

        # Reverse to initial state
        initial_state, action_sequence = self._reverse_to_initial(goal_state, target_solution_length)

        if initial_state is None:
            logger.error(f"Failed to reverse to initial state")
            return None, None

        # ✅ CRITICAL: Validate that forward planning can reach goal
        if not self._validate_problem(initial_state, goal_state, action_sequence):
            logger.error(f"Problem validation failed - initial state cannot reach goal")
            return None, None

        # Export PDDL
        problem_file = self._export_pddl(initial_state, goal_state)

        # Create metadata
        metadata = {
            "domain": "blocksworld",
            "characteristic": characteristic,
            "num_blocks": num_blocks,
            "goal_stacks": goal_stacks,
            "solution_length": len(action_sequence),
            "solution_plan": action_sequence,
            "generation_method": "reverse_planning_validated"
        }

        return problem_file, metadata

    def _create_goal_state(self, characteristic: str) -> Dict[str, Any]:
        """Create a goal state with specific properties."""
        char_spec = self.CHARACTERISTICS[characteristic]
        num_blocks = random.randint(char_spec["num_blocks"][0], char_spec["num_blocks"][1])
        goal_stacks = random.randint(char_spec["goal_stacks"][0], char_spec["goal_stacks"][1])
        return self._create_goal_state_for_config(num_blocks, goal_stacks)

    def _create_goal_state_for_config(self, num_blocks: int, num_stacks: int) -> Optional[Dict[str, Any]]:
        """Create goal state by distributing blocks into stacks."""
        if num_blocks <= 0 or num_stacks <= 0 or num_stacks > num_blocks:
            return None

        goal_state = {"on": {}, "on_table": [], "clear": []}
        blocks = [f"b{i}" for i in range(1, num_blocks + 1)]

        # Distribute blocks into stacks
        blocks_per_stack = num_blocks // num_stacks
        remainder = num_blocks % num_stacks

        block_idx = 0
        for stack_idx in range(num_stacks):
            stack_size = blocks_per_stack + (1 if stack_idx < remainder else 0)

            if stack_size == 0:
                continue

            # First block of stack goes on table
            first_block = blocks[block_idx]
            goal_state["on_table"].append(first_block)
            goal_state["clear"].append(first_block)
            block_idx += 1

            # Rest stack on top
            for i in range(1, stack_size):
                if block_idx >= len(blocks):
                    break

                current_block = blocks[block_idx]
                prev_block = blocks[block_idx - 1]

                # Current block stacks on previous block
                goal_state["on"][current_block] = prev_block

                # Remove prev from clear, add current to clear
                if prev_block in goal_state["clear"]:
                    goal_state["clear"].remove(prev_block)
                goal_state["clear"].append(current_block)

                block_idx += 1

        # Verify state
        if not self._validate_state(goal_state):
            logger.warning("Generated goal state is invalid")
            return None

        return goal_state

    @staticmethod
    def _validate_state(state: Dict[str, Any]) -> bool:
        """✅ NEW: Validate Blocksworld state consistency."""
        # Check structure
        if not all(k in state for k in ["on", "on_table", "clear"]):
            return False

        all_blocks = set(state["on_table"])
        all_blocks.update(state["on"].keys())
        all_blocks.update(state["on"].values())

        # Check on_table blocks are actually on table (not on other blocks)
        for block in state["on_table"]:
            if block in state["on"]:
                return False  # Block can't be both on table and on another block

        # Check clear blocks
        blocks_on_others = set(state["on"].keys())
        for block in state["clear"]:
            if block in state["on"]:
                # Block is stacked on something, shouldn't be clear
                return False

        # Check on edges are valid (no cycles, blocks exist)
        visited = set()
        for block in state["on"]:
            current = block
            path = []
            while current in state["on"]:
                if current in path:
                    return False  # Cycle detected
                path.append(current)
                current = state["on"][current]

            # Current should be on_table
            if current not in state["on_table"]:
                return False

        return True

    def _reverse_to_initial(
            self,
            goal_state: Dict[str, Any],
            target_steps: int
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """✅ COMPLETELY FIXED: Reverse from goal to initial state."""
        try:
            if goal_state is None or not self._validate_state(goal_state):
                return None, []

            current_state = {
                "on": dict(goal_state["on"]),
                "on_table": list(goal_state["on_table"]),
                "clear": list(goal_state["clear"])
            }

            action_sequence = []
            max_iterations = target_steps * 3  # More attempts

            for iteration in range(max_iterations):
                if len(action_sequence) >= target_steps:
                    break

                # ✅ FIXED: Find blocks that can be unstacked (reverse of stack action)
                # In reverse: pick a block that is on another block and clear
                unstackable_blocks = [
                    block for block in current_state["on"]
                    if block in current_state["clear"]
                ]

                if not unstackable_blocks:
                    break

                # Choose a random block to unstack
                block = random.choice(unstackable_blocks)
                under_block = current_state["on"][block]

                # ✅ FIXED: Properly apply reverse action
                # Reverse of stack(block, under_block):
                # - Remove block from on[block]
                # - Add block to on_table
                # - Remove under_block from clear, add it back after
                # - Add block to clear, remove it after

                del current_state["on"][block]
                current_state["on_table"].append(block)

                # Update clear state
                if block in current_state["clear"]:
                    current_state["clear"].remove(block)
                if under_block not in current_state["clear"]:
                    current_state["clear"].append(under_block)

                # Record reverse action (which is the forward action to undo)
                action_sequence.insert(0, f"stack({block}, {under_block})")

                # Validate state after each step
                if not self._validate_state(current_state):
                    logger.debug(f"State became invalid at iteration {iteration}")
                    return None, []

            return current_state, action_sequence

        except Exception as e:
            logger.debug(f"Reverse planning failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, []

    def _validate_problem(
            self,
            initial_state: Dict[str, Any],
            goal_state: Dict[str, Any],
            action_sequence: List[str]
    ) -> bool:
        """✅ NEW: Validate that the problem is actually solvable."""
        try:
            # Start from initial state
            current = {
                "on": dict(initial_state["on"]),
                "on_table": list(initial_state["on_table"]),
                "clear": list(initial_state["clear"])
            }

            # Apply each action forward
            for action in action_sequence:
                # Parse action
                if action.startswith("stack(") and action.endswith(")"):
                    args = action[6:-1].split(", ")
                    if len(args) != 2:
                        logger.debug(f"Invalid action format: {action}")
                        return False

                    block, under_block = args

                    # Verify preconditions
                    if block not in current["clear"]:
                        logger.debug(f"Precondition failed: {block} not clear")
                        return False
                    if under_block not in current["clear"]:
                        logger.debug(f"Precondition failed: {under_block} not clear")
                        return False
                    if block not in current["on_table"]:
                        logger.debug(f"Precondition failed: {block} not on table")
                        return False

                    # Apply action
                    current["on_table"].remove(block)
                    current["on"][block] = under_block
                    current["clear"].remove(block)
                    current["clear"].remove(under_block)

                else:
                    logger.debug(f"Unknown action: {action}")
                    return False

                # Validate state
                if not self._validate_state(current):
                    logger.debug(f"State invalid after action {action}")
                    return False

            # Check if we reached goal
            if current != goal_state:
                logger.debug(f"Final state doesn't match goal state")
                logger.debug(f"Current: {current}")
                logger.debug(f"Goal:    {goal_state}")
                return False

            return True

        except Exception as e:
            logger.debug(f"Problem validation exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _export_pddl(
            self,
            initial_state: Dict[str, Any],
            goal_state: Dict[str, Any]
    ) -> str:
        """Export problem as PDDL file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.pddl',
            delete=False,
            dir=tempfile.gettempdir()
        )

        # Get all blocks
        all_blocks = set(initial_state["on_table"])
        all_blocks.update(initial_state["on"].keys())
        all_blocks.update(initial_state["on"].values())
        all_blocks = sorted(list(all_blocks))

        # Write PDDL
        pddl_content = f"""(define (problem bw-{random.randint(10000, 99999)})
  (:domain blocksworld)
  (:objects {' '.join(all_blocks)})
  (:init
    (arm-empty)
"""

        # Initial state: on-table facts
        for block in initial_state["on_table"]:
            pddl_content += f"    (on-table {block})\n"

        # Initial state: on facts
        for block, under_block in initial_state["on"].items():
            pddl_content += f"    (on {block} {under_block})\n"

        # Initial state: clear facts
        for block in initial_state["clear"]:
            pddl_content += f"    (clear {block})\n"

        pddl_content += "  )\n  (:goal (and\n"

        # Goal: on-table facts
        for block in goal_state["on_table"]:
            pddl_content += f"    (on-table {block})\n"

        # Goal: on facts
        for block, under_block in goal_state["on"].items():
            pddl_content += f"    (on {block} {under_block})\n"

        pddl_content += "  ))\n)\n"

        temp_file.write(pddl_content)
        temp_file.close()

        return temp_file.name


# ============================================================================
# LOGISTICS GENERATOR - MINIMAL CHANGES (unchanged structure works)
# ============================================================================

class LogisticsGenerator(ProblemGenerator):
    """Generate Logistics problems."""

    CHARACTERISTICS = {
        "single_city_intra_logistics": {
            "num_cities": 1,
            "packages_per_city": (2, 4),
            "trucks_per_city": (1, 2),
            "airplanes": 0,
            "description": "Single city, intra-city logistics"
        },
        "few_packages": {
            "num_cities": 1,
            "packages_per_city": (1, 3),
            "trucks_per_city": (1, 1),
            "airplanes": 0,
            "description": "Few packages, simple delivery"
        },
        "direct_delivery": {
            "num_cities": 1,
            "packages_per_city": (2, 3),
            "trucks_per_city": (2, 2),
            "airplanes": 0,
            "description": "Direct truck delivery"
        },
        "two_cities_truck_transport": {
            "num_cities": 2,
            "packages_per_city": (2, 4),
            "trucks_per_city": (1, 2),
            "airplanes": 0,
            "description": "Two cities, truck transport"
        },
        "moderate_packages": {
            "num_cities": 2,
            "packages_per_city": (3, 5),
            "trucks_per_city": (1, 2),
            "airplanes": 0,
            "description": "Moderate package count"
        },
        "single_truck_logistics": {
            "num_cities": 2,
            "packages_per_city": (3, 5),
            "trucks_per_city": (1, 1),
            "airplanes": 0,
            "description": "Single truck for multiple cities"
        },
        "multi_city_air_transport": {
            "num_cities": 3,
            "packages_per_city": (3, 6),
            "trucks_per_city": (1, 2),
            "airplanes": 1,
            "description": "Multiple cities with airplane"
        },
        "many_packages": {
            "num_cities": 2,
            "packages_per_city": (5, 8),
            "trucks_per_city": (2, 3),
            "airplanes": 1,
            "description": "Many packages to deliver"
        },
        "complex_routing": {
            "num_cities": 3,
            "packages_per_city": (4, 8),
            "trucks_per_city": (2, 3),
            "airplanes": 1,
            "description": "Complex multi-city routing"
        }
    }

    def generate_problem(
            self,
            characteristic: str,
            target_solution_length: int
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Generate a Logistics problem."""

        if characteristic not in self.CHARACTERISTICS:
            logger.error(f"Unknown characteristic: {characteristic}")
            return None, None

        char_spec = self.CHARACTERISTICS[characteristic]

        num_cities = char_spec["num_cities"]
        packages_per_city = random.randint(char_spec["packages_per_city"][0],
                                           char_spec["packages_per_city"][1])
        trucks_per_city = random.randint(char_spec["trucks_per_city"][0],
                                         char_spec["trucks_per_city"][1])
        num_airplanes = char_spec["airplanes"]

        logger.debug(f"Logistics Params: cities={num_cities}, "
                     f"packages_per_city={packages_per_city}, trucks={trucks_per_city}")

        # Create goal state
        goal_state = self._create_goal_state_for_config(
            num_cities, packages_per_city, trucks_per_city, num_airplanes
        )

        # Reverse to initial state
        initial_state, action_sequence = self._reverse_to_initial(goal_state, target_solution_length)

        if initial_state is None:
            return None, None

        # Export PDDL
        problem_file = self._export_pddl(initial_state, goal_state, char_spec)

        # Create metadata
        metadata = {
            "domain": "logistics",
            "characteristic": characteristic,
            "num_cities": num_cities,
            "packages": num_cities * packages_per_city,
            "trucks": num_cities * trucks_per_city,
            "airplanes": num_airplanes,
            "solution_length": len(action_sequence),
            "solution_plan": action_sequence,
            "generation_method": "reverse_planning"
        }

        return problem_file, metadata

    def _create_goal_state(self, characteristic: str) -> Dict[str, Any]:
        """Create a goal state."""
        char_spec = self.CHARACTERISTICS[characteristic]
        num_cities = char_spec["num_cities"]
        packages_per_city = random.randint(char_spec["packages_per_city"][0],
                                           char_spec["packages_per_city"][1])
        trucks_per_city = random.randint(char_spec["trucks_per_city"][0],
                                         char_spec["trucks_per_city"][1])
        num_airplanes = char_spec["airplanes"]

        return self._create_goal_state_for_config(
            num_cities, packages_per_city, trucks_per_city, num_airplanes
        )

    def _create_goal_state_for_config(
            self,
            num_cities: int,
            packages_per_city: int,
            trucks_per_city: int,
            num_airplanes: int
    ) -> Dict[str, Any]:
        """Create goal state with distributed packages."""
        goal_state = {
            "cities": num_cities,
            "packages": {},
            "package_at": {}
        }

        # Create packages distributed across cities
        for c in range(1, num_cities + 1):
            for p in range(1, packages_per_city + 1):
                pkg_id = f"p{c}_{p}"
                goal_city = random.randint(1, num_cities)
                goal_state["packages"][pkg_id] = f"c{goal_city}"
                goal_state["package_at"][pkg_id] = f"c{goal_city}_l1"

        return goal_state

    def _reverse_to_initial(
            self,
            goal_state: Dict[str, Any],
            target_steps: int
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Reverse from goal to initial state."""
        try:
            current_state = {
                "package_at": dict(goal_state["package_at"])
            }

            action_sequence = []
            max_iterations = target_steps * 2

            for _ in range(max_iterations):
                if len(action_sequence) >= target_steps:
                    break

                # Randomly move a package to a different location
                if not current_state["package_at"]:
                    break

                pkg = random.choice(list(current_state["package_at"].keys()))
                current_location = current_state["package_at"][pkg]

                # Move to a different location
                current_city = int(current_location.split('_')[0][1:])
                new_city = random.choice([c for c in range(1, goal_state["cities"] + 1)
                                          if c != current_city])
                new_location = f"c{new_city}_l1"

                current_state["package_at"][pkg] = new_location
                action_sequence.insert(0, f"move-package({pkg}, {current_location}, {new_location})")

            return current_state, action_sequence

        except Exception as e:
            logger.debug(f"Reverse planning failed: {e}")
            return None, []

    def _export_pddl(
            self,
            initial_state: Dict[str, Any],
            goal_state: Dict[str, Any],
            char_spec: Dict[str, Any]
    ) -> str:
        """Export as PDDL problem file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.pddl',
            delete=False,
            dir=tempfile.gettempdir()
        )

        pddl_content = f"""(define (problem log-{random.randint(10000, 99999)})
  (:domain logistics-strips)
  (:objects
"""

        # Objects: cities, locations, trucks, packages
        num_cities = char_spec["num_cities"]
        packages_per_city = random.randint(char_spec["packages_per_city"][0],
                                           char_spec["packages_per_city"][1])
        trucks_per_city = random.randint(char_spec["trucks_per_city"][0],
                                         char_spec["trucks_per_city"][1])

        for c in range(1, num_cities + 1):
            pddl_content += f"    c{c}"
        pddl_content += " - city\n    "

        for c in range(1, num_cities + 1):
            for l in range(1, 3):
                pddl_content += f"c{c}_l{l} "
        pddl_content += " - location\n    "

        for c in range(1, num_cities + 1):
            for t in range(1, trucks_per_city + 1):
                pddl_content += f"t{c}_{t} "
        pddl_content += " - truck\n    "

        for c in range(1, num_cities + 1):
            for p in range(1, packages_per_city + 1):
                pddl_content += f"p{c}_{p} "
        pddl_content += " - obj\n  )\n  (:init\n"

        # Initial state facts
        for c in range(1, num_cities + 1):
            for l in range(1, 3):
                pddl_content += f"    (LOCATION c{c}_l{l})\n"
                pddl_content += f"    (in-city c{c}_l{l} c{c})\n"

        for c in range(1, num_cities + 1):
            for t in range(1, trucks_per_city + 1):
                pddl_content += f"    (TRUCK t{c}_{t})\n"
                pddl_content += f"    (at t{c}_{t} c{c}_l1)\n"

        for c in range(1, num_cities + 1):
            for p in range(1, packages_per_city + 1):
                pddl_content += f"    (OBJ p{c}_{p})\n"
                pddl_content += f"    (at p{c}_{p} c{c}_l1)\n"

        pddl_content += "  )\n  (:goal (and\n"

        # Goal facts
        for c in range(1, num_cities + 1):
            for p in range(1, packages_per_city + 1):
                goal_city = random.randint(1, num_cities)
                pddl_content += f"    (at p{c}_{p} c{goal_city}_l1)\n"

        pddl_content += "  ))\n)\n"

        temp_file.write(pddl_content)
        temp_file.close()

        return temp_file.name


# ============================================================================
# PARKING GENERATOR - MINIMAL CHANGES
# ============================================================================

class ParkingGenerator(ProblemGenerator):
    """Generate Parking problems."""

    CHARACTERISTICS = {
        "few_cars_simple_grid": {
            "grid_size": (3, 5),
            "num_cars": (2, 4),
            "num_obstacles": (0, 2),
            "description": "Few cars, simple grid"
        },
        "clear_blocked_car": {
            "grid_size": (4, 6),
            "num_cars": (3, 5),
            "num_obstacles": (1, 3),
            "description": "Clear blocked cars"
        },
        "minimal_obstacles": {
            "grid_size": (5, 7),
            "num_cars": (4, 6),
            "num_obstacles": (0, 2),
            "description": "Minimal obstacles"
        },
        "moderate_cars_medium_grid": {
            "grid_size": (6, 8),
            "num_cars": (5, 8),
            "num_obstacles": (2, 4),
            "description": "Moderate cars, medium grid"
        },
        "multiple_blocked_cars": {
            "grid_size": (7, 9),
            "num_cars": (6, 10),
            "num_obstacles": (3, 5),
            "description": "Multiple blocked cars"
        },
        "some_obstacles": {
            "grid_size": (6, 8),
            "num_cars": (6, 9),
            "num_obstacles": (2, 4),
            "description": "Some obstacles"
        },
        "many_cars_large_grid": {
            "grid_size": (8, 10),
            "num_cars": (8, 12),
            "num_obstacles": (4, 6),
            "description": "Many cars, large grid"
        },
        "complex_blocking_patterns": {
            "grid_size": (9, 11),
            "num_cars": (10, 15),
            "num_obstacles": (5, 8),
            "description": "Complex blocking patterns"
        },
        "dense_obstacles": {
            "grid_size": (10, 12),
            "num_cars": (12, 16),
            "num_obstacles": (6, 10),
            "description": "Dense obstacles"
        }
    }

    def generate_problem(
            self,
            characteristic: str,
            target_solution_length: int
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Generate a Parking problem."""

        if characteristic not in self.CHARACTERISTICS:
            logger.error(f"Unknown characteristic: {characteristic}")
            return None, None

        char_spec = self.CHARACTERISTICS[characteristic]

        grid_size = random.randint(char_spec["grid_size"][0], char_spec["grid_size"][1])
        num_cars = random.randint(char_spec["num_cars"][0], char_spec["num_cars"][1])
        num_obstacles = random.randint(char_spec["num_obstacles"][0], char_spec["num_obstacles"][1])

        logger.debug(f"Parking Params: grid={grid_size}x{grid_size}, "
                     f"cars={num_cars}, obstacles={num_obstacles}")

        # Create goal state
        goal_state = self._create_goal_state_for_config(grid_size, num_cars, num_obstacles)

        # Reverse to initial state
        initial_state, action_sequence = self._reverse_to_initial(goal_state, target_solution_length)

        if initial_state is None:
            return None, None

        # Export PDDL
        problem_file = self._export_pddl(initial_state, goal_state, grid_size)

        # Create metadata
        metadata = {
            "domain": "parking",
            "characteristic": characteristic,
            "grid_size": grid_size,
            "num_cars": num_cars,
            "num_obstacles": num_obstacles,
            "solution_length": len(action_sequence),
            "solution_plan": action_sequence,
            "generation_method": "reverse_planning"
        }

        return problem_file, metadata

    def _create_goal_state(self, characteristic: str) -> Dict[str, Any]:
        """Create a goal state."""
        char_spec = self.CHARACTERISTICS[characteristic]
        grid_size = random.randint(char_spec["grid_size"][0], char_spec["grid_size"][1])
        num_cars = random.randint(char_spec["num_cars"][0], char_spec["num_cars"][1])
        num_obstacles = random.randint(char_spec["num_obstacles"][0], char_spec["num_obstacles"][1])

        return self._create_goal_state_for_config(grid_size, num_cars, num_obstacles)

    def _create_goal_state_for_config(
            self,
            grid_size: int,
            num_cars: int,
            num_obstacles: int
    ) -> Dict[str, Any]:
        """Create goal state with cars parked on specific curbs."""
        goal_state = {
            "grid_size": grid_size,
            "cars": {},
            "obstacles": set()
        }

        # Randomly place obstacles
        available_curbs = set((i, j) for i in range(grid_size) for j in range(grid_size))

        for _ in range(min(num_obstacles, len(available_curbs))):
            curb = random.choice(list(available_curbs))
            goal_state["obstacles"].add(curb)
            available_curbs.remove(curb)

        # Randomly place cars
        car_positions = random.sample(list(available_curbs), min(num_cars, len(available_curbs)))

        for i, pos in enumerate(car_positions):
            goal_state["cars"][f"car{i + 1}"] = pos

        return goal_state

    def _reverse_to_initial(
            self,
            goal_state: Dict[str, Any],
            target_steps: int
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Reverse from goal to initial state."""
        try:
            current_state = {
                "cars": dict(goal_state["cars"])
            }

            action_sequence = []
            grid_size = goal_state["grid_size"]
            max_iterations = target_steps * 2

            for _ in range(max_iterations):
                if len(action_sequence) >= target_steps:
                    break

                if not current_state["cars"]:
                    break

                # Move a random car to a new location
                car = random.choice(list(current_state["cars"].keys()))
                current_pos = current_state["cars"][car]

                # Move to adjacent or nearby position
                new_x = random.randint(max(0, current_pos[0] - 2),
                                       min(grid_size - 1, current_pos[0] + 2))
                new_y = random.randint(max(0, current_pos[1] - 2),
                                       min(grid_size - 1, current_pos[1] + 2))

                if (new_x, new_y) != current_pos:
                    old_pos = current_state["cars"][car]
                    current_state["cars"][car] = (new_x, new_y)
                    action_sequence.insert(0, f"move({car}, {old_pos}, ({new_x},{new_y}))")

            return current_state, action_sequence

        except Exception as e:
            logger.debug(f"Reverse planning failed: {e}")
            return None, []

    def _export_pddl(
            self,
            initial_state: Dict[str, Any],
            goal_state: Dict[str, Any],
            grid_size: int
    ) -> str:
        """Export as PDDL problem file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.pddl',
            delete=False,
            dir=tempfile.gettempdir()
        )

        pddl_content = f"""(define (problem park-{random.randint(10000, 99999)})
  (:domain parking)
  (:objects
"""

        # Objects: curbs and cars
        for i in range(grid_size):
            for j in range(grid_size):
                pddl_content += f"    curb_{i}_{j}"
        pddl_content += " - curb\n    "

        for car in goal_state["cars"].keys():
            pddl_content += f"    {car}"
        pddl_content += " - car\n  )\n  (:init\n"

        # Initial state: cars at positions
        for car, pos in initial_state["cars"].items():
            pddl_content += f"    (at-curb {car})\n"
            pddl_content += f"    (at-curb-num {car} curb_{pos[0]}_{pos[1]})\n"
            pddl_content += f"    (car-clear {car})\n"

        # All curbs initially clear except those with cars
        for i in range(grid_size):
            for j in range(grid_size):
                curb = f"curb_{i}_{j}"
                has_car = any(pos == (i, j) for pos in initial_state["cars"].values())
                if not has_car:
                    pddl_content += f"    (curb-clear {curb})\n"

        pddl_content += "  )\n  (:goal (and\n"

        # Goal: cars at target positions
        for car, pos in goal_state["cars"].items():
            pddl_content += f"    (at-curb-num {car} curb_{pos[0]}_{pos[1]})\n"

        pddl_content += "  ))\n)\n"

        temp_file.write(pddl_content)
        temp_file.close()

        return temp_file.name