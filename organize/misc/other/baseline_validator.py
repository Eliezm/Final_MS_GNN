#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASELINE VALIDATOR - DIFFICULTY CLASSIFICATION
==============================================
Run baseline planner (Fast Downward with astar(lmcut()))
to measure actual solve time and classify difficulty.
"""

import subprocess
import os
import json
import time
import logging
import re
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaselineValidator:
    """Validate and classify problem difficulty using baseline planner."""

    # Difficulty thresholds (in seconds)
    DIFFICULTY_THRESHOLDS = {
        "small": (0, 60),        # 0-1 minute
        "medium": (60, 180),     # 1-3 minutes
        "large": (180, 420),     # 3-7 minutes
        "extreme": (420, float('inf'))  # > 7 minutes (reject)
    }

    def __init__(
            self,
            timeout_sec: int = 480,
            fd_build_dir: str = None
    ):
        """
        Initialize validator.

        Args:
            timeout_sec: Timeout for planner (default 8 min)
            fd_build_dir: Path to Fast Downward build directory
        """
        self.timeout_sec = timeout_sec
        self.fd_build_dir = fd_build_dir or os.environ.get("FD_BUILD_DIR", "./fast-downward")

        self.fd_exe = os.path.join(self.fd_build_dir, "builds/release/bin/downward")
        self.translator_py = os.path.join(
            self.fd_build_dir, "builds/release/bin/translate/translate.py"
        )

        # Check if FD is available
        self._check_fd_availability()

    def _check_fd_availability(self):
        """Check if Fast Downward is available."""
        if not os.path.exists(self.fd_build_dir):
            logger.warning(f"FD build dir not found: {self.fd_build_dir}")
            logger.warning("Set FD_BUILD_DIR environment variable or pass fd_build_dir")
            return

        if not os.path.exists(self.fd_exe):
            logger.warning(f"FD executable not found: {self.fd_exe}")
            logger.warning("Make sure Fast Downward is built")

    def validate_problem(
            self,
            domain_file: str,
            problem_file: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate and classify problem difficulty.

        Args:
            domain_file: Path to domain PDDL
            problem_file: Path to problem PDDL

        Returns:
            (is_solvable, classification_dict) where classification_dict includes:
              - difficulty: "small", "medium", "large", or "extreme"
              - time: time to solve (seconds)
              - plan_cost: solution cost
              - nodes_expanded: nodes expanded by A*
              - search_depth: max search depth
              - error: error message if applicable
        """
        logger.info(f"Validating: {os.path.basename(problem_file)}")

        try:
            # Run baseline planner
            result = self._run_baseline(domain_file, problem_file)

            if result is None:
                logger.warning("  ✗ Baseline execution failed")
                return False, {"difficulty": "unknown", "error": "execution_failed"}

            # Check if solved
            if not result.get("solved", False):
                logger.warning(f"  ✗ Problem unsolvable or timed out")
                logger.warning(f"    Time: {result.get('time', 'unknown')}s")
                return False, {
                    "difficulty": "unsolvable",
                    "time": result.get("time", 0),
                    "error": "unsolvable_or_timeout"
                }

            # Classify difficulty
            difficulty = self._classify_difficulty(result["time"])
            is_valid = difficulty != "extreme"

            classification = {
                "difficulty": difficulty,
                "time": result["time"],
                "plan_cost": result.get("plan_cost", 0),
                "nodes_expanded": result.get("nodes_expanded", 0),
                "search_depth": result.get("search_depth", 0),
                "branching_factor": result.get("branching_factor", 1.0),
            }

            status = "✓" if is_valid else "✗"
            logger.info(f"  {status} Classified as: {difficulty}")
            logger.info(f"    Time: {result['time']:.2f}s, Cost: {result.get('plan_cost', 0)}")

            return is_valid, classification

        except Exception as e:
            logger.error(f"  ✗ Exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, {"difficulty": "error", "error": str(e)}

    def _run_baseline(
            self,
            domain_file: str,
            problem_file: str
    ) -> Optional[Dict[str, Any]]:
        """Run baseline planner on problem."""
        try:
            temp_dir = tempfile.mkdtemp()

            try:
                # Step 1: Translate PDDL to SAS
                sas_file = os.path.join(temp_dir, "output.sas")

                translate_cmd = [
                    "python",
                    self.translator_py,
                    os.path.abspath(domain_file),
                    os.path.abspath(problem_file),
                ]

                logger.debug(f"Running translator...")
                translate_result = subprocess.run(
                    translate_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=temp_dir
                )

                if translate_result.returncode != 0:
                    logger.debug(f"Translation failed:")
                    logger.debug(translate_result.stderr[:500])
                    return None

                # Step 2: Run search with astar(lmcut())
                logger.debug(f"Running search with astar(lmcut())...")
                start_time = time.time()

                search_cmd = [
                    self.fd_exe,
                    "--search", "astar(lmcut())"
                ]

                search_result = subprocess.run(
                    search_cmd,
                    stdin=open(sas_file, 'r'),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    cwd=temp_dir
                )

                elapsed_time = time.time() - start_time

                # Parse output
                output = search_result.stdout + search_result.stderr

                # Extract metrics
                metrics = self._parse_fd_output(output)
                metrics["time"] = elapsed_time
                metrics["solved"] = "Solution found" in output

                logger.debug(f"Solved: {metrics['solved']}, Time: {elapsed_time:.2f}s")

                return metrics

            finally:
                # Cleanup
                shutil.rmtree(temp_dir, ignore_errors=True)

        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout after {self.timeout_sec}s")
            return {
                "time": self.timeout_sec,
                "solved": False,
                "error": "timeout"
            }

        except Exception as e:
            logger.debug(f"Baseline execution error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _parse_fd_output(self, output: str) -> Dict[str, Any]:
        """Extract metrics from Fast Downward output."""
        metrics = {
            "plan_cost": 0,
            "nodes_expanded": 0,
            "search_depth": 0,
            "branching_factor": 1.0,
        }

        try:
            # Plan cost
            match = re.search(r"Plan length:\s*(\d+)", output)
            if match:
                metrics["plan_cost"] = int(match.group(1))

            # Nodes expanded (last occurrence)
            matches = list(re.finditer(r"Expanded\s+(\d+)\s+state", output))
            if matches:
                metrics["nodes_expanded"] = int(matches[-1].group(1))

            # Search depth
            match = re.search(r"Search depth:\s*(\d+)", output)
            if match:
                metrics["search_depth"] = int(match.group(1))

            # Branching factor
            match = re.search(r"Branching factor:\s*([\d.]+)", output)
            if match:
                metrics["branching_factor"] = float(match.group(1))

        except Exception as e:
            logger.debug(f"Metric parsing error: {e}")

        return metrics

    def _classify_difficulty(self, time_sec: float) -> str:
        """Classify difficulty based on solve time."""
        for difficulty, (min_t, max_t) in self.DIFFICULTY_THRESHOLDS.items():
            if min_t <= time_sec < max_t:
                return difficulty
        return "extreme"