#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLOCKSWORLD PROBLEM GENERATOR - REVERSE PLANNING APPROACH
=========================================================

Generates Blocksworld problems by:
1. Creating a goal state with specific structural characteristics
2. Applying random valid actions in reverse to create initial state
3. Guaranteeing solution length = number of reverse steps
4. Ensuring solvability before validation

Key insight: If we reverse N actions from goal → initial,
then initial --[N actions]--> goal is guaranteed solvable.
"""

import random
import tempfile
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from copy import deepcopy

logger = logging.getLogger(__name__)


class BlocksworldGenerator:
    """Generate Blocksworld problems via reverse planning."""

    # Problem characteristics define structural variations
    CHARACTERISTICS = {
        # Small problems: few blocks, simple configurations
        "small_simple_stacks": {
            "num_blocks": (3, 5),
            "num_stacks_goal": (1, 2),
            "stack_heights": "varied",
            "description": "Few blocks, simple stacking"
        },
        "small_scattered": {
            "num_blocks": (4, 6),
            "num_stacks_goal": (2, 3),
            "stack_heights": "flat",
            "description": "Blocks scattered, multiple stacks"
        },
        "small_single_tower": {
            "num_blocks": (3, 6),
            "num_stacks_goal": (1, 1),
            "stack_heights": "tall",
            "description": "Build single tower"
        },

        # Medium problems: moderate blocks, mixed configurations
        "medium_multiple_towers": {
            "num_blocks": (6, 10),
            "num_stacks_goal": (2, 3),
            "stack_heights": "mixed",
            "description": "Multiple towers to build"
        },
        "medium_scattered_complex": {
            "num_blocks": (7, 11),
            "num_stacks_goal": (3, 4),
            "stack_heights": "varied",
            "description": "Complex scattered arrangement"
        },
        "medium_partial_order": {
            "num_blocks": (8, 12),
            "num_stacks_goal": (2, 4),
            "stack_heights": "mixed",
            "description": "Requires careful partial ordering"
        },

        # Large problems: many blocks, complex rearrangement
        "large_complex_towers": {
            "num_blocks": (10, 15),
            "num_stacks_goal": (2, 4),
            "stack_heights": "tall",
            "description": "Large complex tower building"
        },
        "large_scattered": {
            "num_blocks": (12, 16),
            "num_stacks_goal": (4, 6),
            "stack_heights": "varied",
            "description": "Large scattered rearrangement"
        },
        "large_mixed": {
            "num_blocks": (14, 18),
            "num_stacks_goal": (3, 5),
            "stack_heights": "mixed",
            "description": "Large complex mixed arrangement"
        },
    }

    # Solution length targets (in reverse steps)
    # Actual solve time depends on planner heuristic
    SOLUTION_LENGTHS = {
        "small": (6, 7, 8),      # 6-8 steps
        "medium": (11, 12, 13),  # 11-13 steps
        "large": (16, 17, 18),   # 16-18 steps
    }

    def __init__(self, seed: int = 42):
        """Initialize generator."""
        self.seed = seed
        random.seed(seed)

    def generate_problem(
            self,
            characteristic: str,
            difficulty: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generate a Blocksworld problem.

        Args:
            characteristic: One of CHARACTERISTICS keys
            difficulty: "small", "medium", or "large"

        Returns:
            (pddl_file_path, metadata_dict) or (None, None)
        """
        logger.info(f"Generating {difficulty} problem: {characteristic}")

        if characteristic not in self.CHARACTERISTICS:
            logger.error(f"Unknown characteristic: {characteristic}")
            return None, None

        if difficulty not in self.SOLUTION_LENGTHS:
            logger.error(f"Unknown difficulty: {difficulty}")
            return None, None

        try:
            char_spec = self.CHARACTERISTICS[characteristic]
            target_steps = random.choice(self.SOLUTION_LENGTHS[difficulty])

            # Step 1: Create goal state
            goal_state = self._create_goal_state(char_spec)
            if goal_state is None:
                logger.error("Failed to create goal state")
                return None, None

            logger.debug(f"Goal state: {len(goal_state['all_blocks'])} blocks, "
                        f"{self._count_stacks(goal_state)} stacks")

            # Step 2: Reverse to initial state
            initial_state, action_sequence = self._reverse_to_initial(goal_state, target_steps)
            if initial_state is None or not action_sequence:
                logger.error(f"Failed to generate {target_steps} reverse steps")
                return None, None

            logger.debug(f"Generated {len(action_sequence)} reverse steps")

            # Step 3: Validate that forward path works
            if not self._validate_forward_path(initial_state, goal_state, action_sequence):
                logger.error("Forward validation failed")
                return None, None

            # Step 4: Export PDDL
            pddl_file = self._export_pddl(initial_state, goal_state)
            if pddl_file is None:
                logger.error("Failed to export PDDL")
                return None, None

            # Step 5: Create metadata
            metadata = {
                "domain": "blocksworld",
                "characteristic": characteristic,
                "difficulty": difficulty,
                "generation_method": "reverse_planning",
                "num_blocks": len(goal_state['all_blocks']),
                "goal_stacks": self._count_stacks(goal_state),
                "initial_stacks": self._count_stacks(initial_state),
                "guaranteed_solution_length": len(action_sequence),
                "guaranteed_solution_plan": action_sequence,
                "seed": self.seed,
            }

            logger.info(f"✓ Generated problem with {len(action_sequence)} step solution")
            return pddl_file, metadata

        except Exception as e:
            logger.error(f"Exception during generation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None

    def _create_goal_state(self, char_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a goal state with specific characteristics."""
        try:
            num_blocks = random.randint(char_spec["num_blocks"][0],
                                       char_spec["num_blocks"][1])
            num_stacks = random.randint(char_spec["num_stacks_goal"][0],
                                       char_spec["num_stacks_goal"][1])
            stack_type = char_spec["stack_heights"]

            if num_stacks > num_blocks:
                num_stacks = num_blocks

            # Create block names
            blocks = [f"b{i}" for i in range(1, num_blocks + 1)]
            all_blocks = set(blocks)

            # Distribute blocks into stacks based on configuration type
            stacks = self._create_stack_distribution(
                blocks, num_stacks, stack_type
            )

            # Convert stacks to state representation
            state = {
                "on_table": [],
                "on": {},
                "clear": [],
                "all_blocks": list(all_blocks),
                "stacks": stacks  # For tracking
            }

            # Process each stack
            for stack in stacks:
                if not stack:
                    continue

                # First block goes on table
                first_block = stack[0]
                state["on_table"].append(first_block)
                state["clear"].append(first_block)

                # Rest of blocks stack on previous
                for i in range(1, len(stack)):
                    current = stack[i]
                    previous = stack[i - 1]
                    state["on"][current] = previous

                    # Only top block is clear
                    if previous in state["clear"]:
                        state["clear"].remove(previous)

                # Add top block to clear
                top_block = stack[-1]
                if top_block not in state["clear"]:
                    state["clear"].append(top_block)

            # Validate state
            if not self._is_valid_state(state):
                logger.warning("Generated state is invalid")
                return None

            return state

        except Exception as e:
            logger.debug(f"Goal state creation failed: {e}")
            return None

    def _create_stack_distribution(
            self,
            blocks: List[str],
            num_stacks: int,
            stack_type: str
    ) -> List[List[str]]:
        """Distribute blocks into stacks based on configuration type."""
        if num_stacks > len(blocks):
            num_stacks = len(blocks)

        random.shuffle(blocks)

        if stack_type == "tall":
            # Concentrated: few tall stacks
            stacks = [[] for _ in range(num_stacks)]
            for i, block in enumerate(blocks):
                stacks[i % num_stacks].append(block)
            # Sort by length descending
            stacks.sort(key=len, reverse=True)

        elif stack_type == "flat":
            # Distributed: many flat stacks
            stacks = [[] for _ in range(num_stacks)]
            for i, block in enumerate(blocks):
                stacks[i % num_stacks].append(block)

        elif stack_type == "mixed":
            # Random heights
            stacks = [[] for _ in range(num_stacks)]
            for i, block in enumerate(blocks):
                stacks[i % num_stacks].append(block)
            # Randomly shuffle within stacks to vary heights
            for stack in stacks:
                random.shuffle(stack)

        else:  # varied
            # Some tall, some flat
            stacks = [[] for _ in range(num_stacks)]
            for i, block in enumerate(blocks):
                stacks[i % num_stacks].append(block)

        return [s for s in stacks if s]  # Remove empty stacks

    def _reverse_to_initial(
            self,
            goal_state: Dict[str, Any],
            target_steps: int
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Reverse from goal to initial state by applying random valid actions.

        Strategy:
        1. Start with goal state
        2. Find all applicable actions
        3. Pick random action
        4. Apply it (moving away from goal)
        5. Record action for forward path
        6. Repeat target_steps times
        """
        try:
            current_state = deepcopy(goal_state)
            action_sequence = []
            max_attempts = target_steps * 5  # Allow some failures

            for attempt in range(max_attempts):
                if len(action_sequence) >= target_steps:
                    break

                # Find applicable actions in current state
                applicable = self._get_applicable_actions(current_state)
                if not applicable:
                    logger.debug(f"No applicable actions at step {len(action_sequence)}")
                    break

                # Pick random action
                action = random.choice(applicable)
                action_name, params = action

                # Apply action
                new_state = self._apply_action(current_state, action_name, params)
                if new_state is None:
                    continue

                # Validate new state
                if not self._is_valid_state(new_state):
                    logger.debug(f"Invalid state after {action_name}")
                    continue

                current_state = new_state
                action_sequence.insert(0, f"{action_name}({', '.join(params)})")

            if not action_sequence:
                logger.warning("Failed to generate any reverse steps")
                return None, []

            if len(action_sequence) < target_steps:
                logger.warning(f"Only generated {len(action_sequence)}/{target_steps} steps")

            return current_state, action_sequence

        except Exception as e:
            logger.debug(f"Reverse planning exception: {e}")
            return None, []

    def _get_applicable_actions(self, state: Dict[str, Any]) -> List[Tuple[str, Tuple[str, ...]]]:
        """Find all applicable actions in current state."""
        actions = []
        blocks = state["all_blocks"]
        arm_empty = len(state["on_table"]) + len(state["on"]) == len(blocks)

        # Action 1: pickup(X) - pick up X from table if arm empty
        if arm_empty:
            for block in state["on_table"]:
                if block in state["clear"]:
                    actions.append(("pickup", (block,)))

        # Action 2: putdown(X) - put down X to table if holding
        if not arm_empty:
            # Find which block is being held (not on table, not on another block)
            for block in blocks:
                if block not in state["on_table"] and block not in state["on"]:
                    # This block is held
                    actions.append(("putdown", (block,)))

        # Action 3: stack(X, Y) - stack X on Y if both clear and X on table, arm empty
        if arm_empty:
            for x in state["on_table"]:
                if x in state["clear"]:
                    for y in blocks:
                        if y != x and y in state["clear"] and y not in state["on"]:
                            actions.append(("stack", (x, y)))

        # Action 4: unstack(X, Y) - unstack X from Y if X clear and arm empty
        if arm_empty:
            for x, y in state["on"].items():
                if x in state["clear"]:
                    actions.append(("unstack", (x, y)))

        return actions

    def _apply_action(
            self,
            state: Dict[str, Any],
            action_name: str,
            params: Tuple[str, ...]
    ) -> Optional[Dict[str, Any]]:
        """Apply action to state, return new state or None if preconditions fail."""
        try:
            new_state = deepcopy(state)

            if action_name == "pickup":
                x = params[0]
                if x not in new_state["on_table"] or x not in new_state["clear"]:
                    return None
                if len(new_state["on_table"]) + len(new_state["on"]) != len(new_state["all_blocks"]):
                    return None  # Arm not empty

                new_state["on_table"].remove(x)
                new_state["clear"].remove(x)
                # x is now held (implicit: not on_table, not in on)

            elif action_name == "putdown":
                x = params[0]
                if x in new_state["on_table"] or x in new_state["on"]:
                    return None  # x should be held
                if len(new_state["on_table"]) + len(new_state["on"]) != len(new_state["all_blocks"]) - 1:
                    return None  # x should be held

                new_state["on_table"].append(x)
                new_state["clear"].append(x)

            elif action_name == "stack":
                x, y = params
                if x not in new_state["on_table"] or x not in new_state["clear"]:
                    return None
                if y not in new_state["clear"] or y in new_state["on"]:
                    return None
                if len(new_state["on_table"]) + len(new_state["on"]) != len(new_state["all_blocks"]):
                    return None  # Arm not empty

                new_state["on_table"].remove(x)
                new_state["on"][x] = y
                new_state["clear"].remove(x)
                if y in new_state["clear"]:
                    new_state["clear"].remove(y)

            elif action_name == "unstack":
                x, y = params
                if x not in new_state["on"] or new_state["on"][x] != y:
                    return None
                if x not in new_state["clear"]:
                    return None
                if len(new_state["on_table"]) + len(new_state["on"]) != len(new_state["all_blocks"]):
                    return None  # Arm not empty

                del new_state["on"][x]
                new_state["on_table"].append(x)
                new_state["clear"].append(y)
                # x was already clear, stays clear

            else:
                return None

            return new_state

        except Exception as e:
            logger.debug(f"Action application failed: {e}")
            return None

    def _is_valid_state(self, state: Dict[str, Any]) -> bool:
        """Validate state consistency."""
        try:
            required_keys = {"on_table", "on", "clear", "all_blocks"}
            if not all(k in state for k in required_keys):
                return False

            blocks = set(state["all_blocks"])

            # Check: every block is either on-table or on another block (not both)
            on_blocks = set(state["on"].keys())
            if on_blocks & set(state["on_table"]):
                return False

            # Check: no block is on-table and on another block
            for b in on_blocks:
                if b in state["on_table"]:
                    return False

            # Check: on edges point to valid blocks
            for x, y in state["on"].items():
                if y not in blocks or x not in blocks:
                    return False

            # Check: on_table blocks are valid
            for b in state["on_table"]:
                if b not in blocks:
                    return False

            # Check: clear blocks are valid
            for b in state["clear"]:
                if b not in blocks:
                    return False

            # Check: exactly one block is held (implicit)
            on_table_or_stacked = set(state["on_table"]) | set(state["on"].keys())
            held = blocks - on_table_or_stacked
            if len(held) > 1:
                return False

            # Check: block that's on another block has that block as its base
            for x, y in state["on"].items():
                if y in state["on"]:  # y is on something
                    pass  # This is fine, it's a stack
                # Make sure we don't have cycles
                visited = set()
                current = x
                while current in state["on"]:
                    if current in visited:
                        return False  # Cycle
                    visited.add(current)
                    current = state["on"][current]

            return True

        except Exception as e:
            logger.debug(f"State validation exception: {e}")
            return False

    def _validate_forward_path(
            self,
            initial_state: Dict[str, Any],
            goal_state: Dict[str, Any],
            action_sequence: List[str]
    ) -> bool:
        """Validate that applying actions forward reaches goal."""
        try:
            current = deepcopy(initial_state)

            for action_str in action_sequence:
                # Parse action
                if not (action_str.startswith("pickup(") or
                       action_str.startswith("putdown(") or
                       action_str.startswith("stack(") or
                       action_str.startswith("unstack(")):
                    logger.debug(f"Invalid action format: {action_str}")
                    return False

                # Extract name and params
                action_name = action_str.split("(")[0]
                params_str = action_str[len(action_name) + 1:-1]
                params = tuple(p.strip() for p in params_str.split(","))

                current = self._apply_action(current, action_name, params)
                if current is None:
                    logger.debug(f"Action preconditions failed: {action_str}")
                    return False

                if not self._is_valid_state(current):
                    logger.debug(f"Invalid state after: {action_str}")
                    return False

            # Check if we reached goal
            if current != goal_state:
                logger.debug("Final state doesn't match goal")
                logger.debug(f"Expected: {goal_state}")
                logger.debug(f"Got: {current}")
                return False

            return True

        except Exception as e:
            logger.debug(f"Forward validation exception: {e}")
            return False

    def _count_stacks(self, state: Dict[str, Any]) -> int:
        """Count number of stacks in state."""
        if "stacks" in state:
            return len(state["stacks"])
        # Count by finding all base blocks
        base_blocks = set(state["on_table"]) - set(state["on"].values())
        return len(base_blocks)

    def _export_pddl(
            self,
            initial_state: Dict[str, Any],
            goal_state: Dict[str, Any]
    ) -> Optional[str]:
        """Export problem as PDDL file."""
        try:
            blocks = sorted(initial_state["all_blocks"])
            problem_id = random.randint(100000, 999999)

            pddl_content = f"""(define (problem bw-{problem_id})
  (:domain blocksworld)
  (:objects {' '.join(blocks)})
  (:init
    (arm-empty)
"""

            # Initial state: on-table
            for block in initial_state["on_table"]:
                pddl_content += f"    (on-table {block})\n"

            # Initial state: on
            for block, under in initial_state["on"].items():
                pddl_content += f"    (on {block} {under})\n"

            # Initial state: clear
            for block in initial_state["clear"]:
                pddl_content += f"    (clear {block})\n"

            pddl_content += "  )\n  (:goal (and\n"

            # Goal: on-table
            for block in goal_state["on_table"]:
                pddl_content += f"    (on-table {block})\n"

            # Goal: on
            for block, under in goal_state["on"].items():
                pddl_content += f"    (on {block} {under})\n"

            pddl_content += "  ))\n)\n"

            # Write to temp file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.pddl',
                delete=False,
                dir=tempfile.gettempdir()
            )
            temp_file.write(pddl_content)
            temp_file.close()

            return temp_file.name

        except Exception as e:
            logger.error(f"PDDL export failed: {e}")
            return None