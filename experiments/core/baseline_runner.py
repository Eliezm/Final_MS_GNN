#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASELINE RUNNER - Fast Downward Execution (FIXED)
================================================
Handles FD translation, search, and output parsing.

✅ FIXED: Proper translate script detection and error handling
"""

import sys
import os
import subprocess
import time
import re
import logging
import shutil
import tempfile
import platform  # ✅ ADD THIS LINE
from pathlib import Path
from typing import Dict, Optional, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.core.evaluation_metrics import DetailedMetrics
from experiments.core.evaluation_config import EvaluationConfig

logger = logging.getLogger(__name__)





class FastDownwardOutputParser:
    """Parse Fast Downward output to extract comprehensive metrics."""

    @staticmethod
    def parse_search_output(output_text: str) -> Dict[str, Any]:
        """Extract comprehensive metrics from FD search output."""
        metrics = {
            'solved': False,
            'plan_cost': 0,
            'plan_length': 0,
            'nodes_expanded': 0,
            'nodes_generated': 0,
            'search_depth': 0,
            'branching_factor': 1.0,
            'search_time': 0.0,
            'peak_memory_kb': 0,
            'plan_found': False,
        }

        try:
            if any(indicator in output_text for indicator in [
                "Solution found",
                "Plan length:",
                "Evaluating",
                "Preferred operators used",
            ]):
                metrics['solved'] = True
                metrics['plan_found'] = True

            match = re.search(r'Plan length:\s*(\d+)', output_text)
            if match:
                metrics['plan_length'] = int(match.group(1))
                metrics['plan_cost'] = int(match.group(1))

            matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', output_text))
            if matches:
                metrics['nodes_expanded'] = int(matches[-1].group(1))

            match = re.search(r'Generated\s+(\d+)\s+state', output_text)
            if match:
                metrics['nodes_generated'] = int(match.group(1))

            match = re.search(r'Search depth:\s*(\d+)', output_text)
            if match:
                metrics['search_depth'] = int(match.group(1))

            match = re.search(r'Branching factor:\s*([\d.]+)', output_text)
            if match:
                metrics['branching_factor'] = float(match.group(1))

            match = re.search(r'Search time:\s*([\d.]+)s', output_text)
            if match:
                metrics['search_time'] = float(match.group(1))

            match = re.search(r'Peak memory:\s*(\d+)\s*KB', output_text)
            if match:
                metrics['peak_memory_kb'] = int(match.group(1))

        except Exception as e:
            logger.warning(f"Error parsing FD output: {e}")

        return metrics

    @staticmethod
    def parse_fd_log_file(log_path: str) -> Dict[str, Any]:
        """Parse complete FD log file."""
        metrics = {
            'solved': False,
            'plan_cost': 0,
            'plan_length': 0,
            'nodes_expanded': 0,
            'search_time': 0.0,
        }

        try:
            if Path(log_path).exists():
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                metrics.update(FastDownwardOutputParser.parse_search_output(content))
        except Exception as e:
            logger.warning(f"Failed to parse log file {log_path}: {e}")

        return metrics


def _find_fd_binary(downward_dir: Path) -> Path:
    """
    Find Fast Downward binary with platform-specific handling.

    ✅ FIXED: Properly handles Windows .exe extension
    """
    base_path = downward_dir / "builds" / "release" / "bin"

    # ✅ FIX: Check multiple possible binary names
    candidates = [
        base_path / "downward.exe",  # Windows
        base_path / "downward",  # Linux/Mac
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # ✅ FIX: On Windows, prefer .exe
    if platform.system() == 'Windows':
        return base_path / "downward.exe"

    return base_path / "downward"




class BaselineRunner:
    """Runs baseline Fast Downward planners."""

    # NEW CODE - REPLACE WITH THIS:
    def __init__(self, timeout_sec: int = 300, downward_dir: Optional[str] = None):
        self.timeout_sec = timeout_sec

        if downward_dir:
            self.downward_dir = Path(downward_dir).absolute()
        else:
            self.downward_dir = Path(EvaluationConfig.DOWNWARD_DIR)

        # ✅ FIX: Use helper function to find binary
        self.fd_bin = _find_fd_binary(self.downward_dir)
        self.fd_translate = self.downward_dir / "builds" / "release" / "bin" / "translate" / "translate.py"

        # ✅ FIX: Only warn if NEITHER exists (more helpful message)
        if not self.fd_bin.exists():
            alt_bin = self.fd_bin.with_suffix('.exe') if not str(self.fd_bin).endswith(
                '.exe') else self.fd_bin.with_suffix('')
            if alt_bin.exists():
                self.fd_bin = alt_bin
            else:
                logger.warning(f"FD binary not found. Checked:\n  - {self.fd_bin}\n  - {alt_bin}")

        if not self.fd_translate.exists():
            logger.warning(f"FD translate not found at: {self.fd_translate}")

    def run(
            self,
            domain_file: str,
            problem_file: str,
            search_config: str,
            baseline_name: str = "FD"
    ) -> DetailedMetrics:
        """
        Run baseline planner on a problem.

        ✅ FIXED: Proper translate script invocation and error handling
        """
        problem_name = os.path.basename(problem_file)
        logger.info(f"[BASELINE] {baseline_name}: {problem_name}")

        # Create temporary work directory
        work_dir = Path(tempfile.mkdtemp(prefix="fd_eval_"))
        sas_file = work_dir / "output.sas"

        try:
            # PHASE 1: Translate (✅ FIXED: use sys.executable for translate.py)
            translate_start = time.time()

            # ✅ FIX: Invoke Python translate.py correctly
            translate_cmd = [
                sys.executable,
                str(self.fd_translate),
                domain_file,
                problem_file,
                "--sas-file", str(sas_file)
            ]

            logger.debug(f"Translate command: {' '.join(translate_cmd)}")

            translate_result = subprocess.run(
                translate_cmd,
                cwd=str(self.downward_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            translate_time = time.time() - translate_start

            if translate_result.returncode != 0:
                logger.debug(f"Translate failed: {translate_result.stderr[:200]}")
                error_msg = translate_result.stderr[:500] if translate_result.stderr else "Unknown translate error"
                return DetailedMetrics(
                    problem_name=problem_name,
                    planner_name=baseline_name,
                    solved=False,
                    wall_clock_time=translate_time,
                    translate_time=translate_time,
                    error_type="translate_error",
                    error_message=error_msg,
                    timeout_occurred=False
                )

            if not sas_file.exists():
                logger.debug("Translate produced no output file")
                return DetailedMetrics(
                    problem_name=problem_name,
                    planner_name=baseline_name,
                    solved=False,
                    wall_clock_time=translate_time,
                    translate_time=translate_time,
                    error_type="translate_error",
                    error_message="No SAS file generated",
                    timeout_occurred=False
                )

            # PHASE 2: Search (✅ FIX: proper FD binary invocation)
            search_start = time.time()

            # ✅ FIX: Use downward binary with correct search config
            search_cmd = [str(self.fd_bin), "--search", search_config]

            logger.debug(f"Search command: {' '.join(search_cmd)}")

            with open(sas_file, 'r') as f_in:
                search_result = subprocess.run(
                    search_cmd,
                    stdin=f_in,
                    cwd=str(self.downward_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec
                )

            search_time = time.time() - search_start
            total_time = translate_time + search_time
            output_text = search_result.stdout + search_result.stderr

            logger.debug(f"FD output: {output_text[:200]}")

            # PHASE 3: Check if solved
            if "Solution found" not in output_text and "Plan length:" not in output_text:
                return DetailedMetrics(
                    problem_name=problem_name,
                    planner_name=baseline_name,
                    solved=False,
                    wall_clock_time=total_time,
                    translate_time=translate_time,
                    search_time=search_time,
                    error_type="no_solution",
                    timeout_occurred=False
                )

            # PHASE 4: Parse metrics
            metrics = FastDownwardOutputParser.parse_search_output(output_text)

            result = DetailedMetrics(
                problem_name=problem_name,
                planner_name=baseline_name,
                solved=metrics.get('solved', False),
                wall_clock_time=total_time,
                translate_time=translate_time,
                search_time=search_time,
                plan_cost=metrics.get('plan_cost', 0),
                plan_length=metrics.get('plan_length', 0),
                nodes_expanded=metrics.get('nodes_expanded', 0),
                nodes_generated=metrics.get('nodes_generated', 0),
                search_depth=metrics.get('search_depth', 0),
                branching_factor=metrics.get('branching_factor', 1.0),
                peak_memory_kb=metrics.get('peak_memory_kb', 0),
                timeout_occurred=False
            )

            logger.debug(f"Success: cost={result.plan_cost}, exp={result.nodes_expanded}")
            return result

        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout after {self.timeout_sec}s")
            return DetailedMetrics(
                problem_name=problem_name,
                planner_name=baseline_name,
                solved=False,
                wall_clock_time=self.timeout_sec,
                error_type="timeout",
                timeout_occurred=True
            )

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return DetailedMetrics(
                problem_name=problem_name,
                planner_name=baseline_name,
                solved=False,
                wall_clock_time=0,
                error_type="exception",
                error_message=str(e)[:500],
                timeout_occurred=False
            )

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except:
                pass