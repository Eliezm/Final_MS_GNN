#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION MODULE - Complete Framework for Training & Testing
==============================================================
UNIFIED: Baseline comparison, GNN evaluation, comprehensive metrics,
statistical analysis, and visualization.

Features:
✓ Baseline Fast Downward comparison (7 configurations)
✓ Automatic benchmark discovery (recursively scan benchmarks/domain/size/)
✓ FD setup verification before running
✓ GNN policy evaluation with ThinMergeEnv
✓ Detailed metrics tracking (25+ per run)
✓ Statistical significance testing
✓ Comprehensive reporting (CSV, JSON, TXT)
✓ Visualization and analysis plots
✓ H* preservation tracking (primary metric)
✓ Per-problem and aggregate statistics
✓ Speedup analysis vs baselines
✓ Scaling analysis
✓ Extended M&S-specific metrics
✓ Robust error handling and validation
"""

import sys
import os
import json
import glob
import subprocess
import time
import re
import csv
import logging
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.core.evaluation_plots import GenerateEvaluationPlots
from experiments.core.gnn_random_evaluation import (
    GNNRandomEvaluationFramework,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(filename)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES - METRICS
# ============================================================================

@dataclass
class ExtendedMetrics:
    """Comprehensive metrics for planners (especially M&S)."""
    # Solution
    solved: bool = False
    cost: Optional[int] = None

    # Timing breakdown (seconds)
    time_total: float = 0.0
    time_translation: float = 0.0
    time_search: float = 0.0
    time_initialization: float = 0.0
    time_ms_main_loop: float = 0.0

    # Search metrics
    expansions: Optional[int] = None
    generated_states: Optional[int] = None
    evaluated_states: Optional[int] = None

    # M&S specific metrics
    max_abstraction_size: Optional[int] = None
    final_abstraction_size: Optional[int] = None
    num_merges: Optional[int] = None
    num_shrinks: Optional[int] = None
    num_label_reductions: Optional[int] = None
    max_states_before_merge: Optional[int] = None

    # Error tracking
    error_reason: Optional[str] = None
    error_details: Optional[str] = None

    # Metadata
    planner: str = ""
    domain: str = ""
    problem_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetailedMetrics:
    """Complete metrics for a single evaluation run."""

    problem_name: str
    planner_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Solve status
    solved: bool = False
    wall_clock_time: float = 0.0

    # Planning quality
    plan_cost: int = 0
    plan_length: int = 0
    solution_cost: int = 0

    # Search metrics
    nodes_expanded: int = 0
    nodes_generated: int = 0
    search_depth: int = 0
    branching_factor: float = 1.0

    # Memory
    peak_memory_kb: int = 0

    # Time breakdown
    search_time: float = 0.0
    translate_time: float = 0.0
    preprocess_time: float = 0.0
    total_time: float = 0.0

    # GNN-specific metrics
    h_star_preservation: float = 1.0
    num_active_systems: int = 0
    merge_episodes: int = 0
    solvable: bool = True

    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timeout_occurred: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateStatistics:
    """Summary statistics across all problems."""

    planner_name: str
    num_problems_total: int
    num_problems_solved: int
    solve_rate_pct: float

    # Time statistics
    mean_time_sec: float
    median_time_sec: float
    std_time_sec: float
    min_time_sec: float
    max_time_sec: float
    q1_time_sec: float
    q3_time_sec: float
    iqr_time_sec: float

    # Expansion statistics
    mean_expansions: int
    median_expansions: int
    std_expansions: int

    # Plan quality
    mean_plan_cost: int
    median_plan_cost: int
    std_plan_cost: int

    # GNN-specific
    mean_h_preservation: float = 1.0
    median_h_preservation: float = 1.0

    # Counts
    unsolved_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    total_wall_clock_time_sec: float = 0.0

    # M&S specific
    mean_max_abstraction_size: int = 0
    mean_final_abstraction_size: int = 0
    mean_num_merges: int = 0
    mean_num_shrinks: int = 0
    mean_num_label_reductions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# CONFIGURATION
# ============================================================================

class EvaluationConfig:
    """Configuration for evaluation framework."""

    # Feature dimensions (match ThinMergeEnv)
    NODE_FEATURE_DIM = 9
    EDGE_FEATURE_DIM = 11

    # Reward weights (match thin_merge_env)
    REWARD_WEIGHTS = {
        'w_h_preservation': 0.40,
        'w_shrinkability': 0.25,
        'w_state_control': 0.20,
        'w_solvability': 0.15,
    }

    # Timeouts
    FD_TIMEOUT_SEC = 300
    GNN_TIMEOUT_SEC = 300

    # Time limits
    TIME_LIMIT_PER_RUN_S: int = 300  # 5 minutes per problem

    # Reproducibility
    RANDOM_SEED: int = 42

    # Fast Downward paths
    DOWNWARD_DIR = os.path.abspath("downward")
    FD_TRANSLATE_BIN = os.path.join(DOWNWARD_DIR, "builds/release/bin/translate/translate.py")
    FD_DOWNWARD_BIN = os.path.join(DOWNWARD_DIR, "builds/release/bin/downward.exe")

    # Output configuration
    OUTPUT_DIR = "evaluation_results"
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "baseline_performance_summary.csv")
    OUTPUT_DETAILED = os.path.join(OUTPUT_DIR, "detailed_results.json")

    # Working directories
    FD_TEMP_DIR = "evaluation_temp"

    # Problem discovery configuration
    BENCHMARK_DOMAINS = {
        "blocksworld": {"path": "benchmarks/blocksworld", "problem_prefix": "problem_"},
        "logistics": {"path": "benchmarks/logistics", "problem_prefix": "problem_"},
    }

    PROBLEM_SIZES = ["small", "medium", "large"]

    # Baseline Fast Downward configurations
    BASELINE_CONFIGS = [
        {
            "name": "FD_LM-Cut",
            "search": "astar(lmcut())",
            "description": "Fast Downward with LM-Cut heuristic",
            "category": "heuristic"
        },
        {
            "name": "FD_Add",
            "search": "astar(add())",
            "description": "Fast Downward with additive heuristic",
            "category": "heuristic"
        },
        {
            "name": "FD_Max",
            "search": "astar(max())",
            "description": "Fast Downward with max heuristic",
            "category": "heuristic"
        },
        {
            "name": "FD_Blind",
            "search": "astar(blind())",
            "description": "Fast Downward with blind heuristic",
            "category": "heuristic"
        },
        {
            "name": "FD_M&S_DFP",
            "search": (
                "astar(merge_and_shrink("
                "merge_strategy=merge_stateless("
                "merge_selector=score_based_filtering("
                "scoring_functions=[goal_relevance(),dfp(),total_order()])),"
                "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
                "label_reduction=exact(before_shrinking=true,before_merging=false),"
                "max_states=50000,threshold_before_merge=1))"
            ),
            "description": "Merge-and-shrink with DFP scoring",
            "category": "m&s"
        },
        {
            "name": "FD_M&S_SCC",
            "search": (
                "astar(merge_and_shrink("
                "merge_strategy=merge_sccs("
                "order_of_sccs=topological,"
                "merge_selector=score_based_filtering("
                "scoring_functions=[goal_relevance(),dfp(),total_order()])),"
                "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
                "label_reduction=exact(before_shrinking=true,before_merging=false),"
                "max_states=50000,threshold_before_merge=1))"
            ),
            "description": "Merge-and-shrink with SCC merging",
            "category": "m&s"
        },
        {
            "name": "FD_FF",
            "search": "astar(ff())",
            "description": "Fast Downward with FF heuristic",
            "category": "heuristic"
        },
    ]

    @staticmethod
    def get_baselines(seed: int) -> List[Dict[str, Any]]:
        """Get baseline configurations with seed support."""
        return EvaluationConfig.BASELINE_CONFIGS


# ============================================================================
# STEP 1: VERIFY SETUP
# ============================================================================

def verify_fd_setup() -> bool:
    """Verify Fast Downward installation and configuration."""
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
# STEP 2: DISCOVER BENCHMARKS
# ============================================================================

def discover_benchmarks() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Discover benchmarks from directory structure:
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
# STEP 3: RUN SINGLE PROBLEM
# ============================================================================

def run_single_fd_problem(
        domain_file: str,
        problem_file: str,
        search_config: str,
        planner_name: str,
        time_limit: int,
        temp_dir: str
) -> ExtendedMetrics:
    """
    Run Fast Downward on a single problem with comprehensive metrics capture.

    Args:
        domain_file: Path to domain.pddl
        problem_file: Path to problem.pddl
        search_config: FD search configuration string
        planner_name: Human-readable planner name
        time_limit: Time limit in seconds
        temp_dir: Temporary directory for SAS files

    Returns:
        ExtendedMetrics object with all captured metrics
    """
    problem_name = os.path.basename(problem_file)
    domain_name = problem_file.split("/")[-3] if "/" in problem_file else "unknown"

    metrics = ExtendedMetrics(
        planner=planner_name,
        domain=domain_name,
        problem_id=problem_name.replace(".pddl", "")
    )

    try:
        # ====== PHASE 1: TRANSLATE ======
        os.makedirs(temp_dir, exist_ok=True)
        sas_file = os.path.join(temp_dir, "output.sas")

        abs_domain = os.path.abspath(domain_file)
        abs_problem = os.path.abspath(problem_file)
        abs_sas = os.path.abspath(sas_file)
        abs_translate = os.path.abspath(EvaluationConfig.FD_TRANSLATE_BIN)

        translate_cmd = (
            f'python "{abs_translate}" '
            f'"{abs_domain}" "{abs_problem}" '
            f'--sas-file "{abs_sas}"'
        )

        translate_start = time.time()
        result = subprocess.run(
            translate_cmd,
            shell=True,
            cwd=os.path.abspath("."),
            capture_output=True,
            text=True,
            timeout=time_limit
        )
        metrics.time_translation = time.time() - translate_start

        if result.returncode != 0:
            metrics.solved = False
            metrics.error_reason = "translate_error"
            metrics.error_details = (result.stderr or result.stdout)[:500]
            logger.debug(f"      Translate failed: {metrics.error_reason}")
            return metrics

        if not os.path.exists(abs_sas) or os.path.getsize(abs_sas) == 0:
            metrics.solved = False
            metrics.error_reason = "translate_no_output"
            logger.debug(f"      No SAS file generated")
            return metrics

        logger.debug(f"      Translate: {metrics.time_translation:.2f}s")

        # ====== PHASE 2: SEARCH ======
        abs_downward = os.path.abspath(EvaluationConfig.FD_DOWNWARD_BIN)
        search_cmd = (
            f'"{abs_downward}" '
            f'--search "{search_config}" '
            f'< "{abs_sas}"'
        )

        search_start = time.time()
        result = subprocess.run(
            search_cmd,
            shell=True,
            cwd=os.path.dirname(abs_downward),
            capture_output=True,
            text=True,
            timeout=time_limit
        )
        metrics.time_search = time.time() - search_start

        output_text = result.stdout + result.stderr

        # ====== PHASE 3: PARSE OUTPUT ======
        if "Solution found" not in output_text and "Plan length:" not in output_text:
            metrics.solved = False
            metrics.error_reason = "no_solution"
            logger.debug(f"      No solution found")
            return metrics

        # Parse metrics from output
        _parse_fd_output(output_text, metrics)
        metrics.solved = True
        metrics.time_total = metrics.time_translation + metrics.time_search

        logger.debug(
            f"      Solved: cost={metrics.cost}, "
            f"exp={metrics.expansions}, "
            f"time={metrics.time_total:.2f}s"
        )

        return metrics

    except subprocess.TimeoutExpired:
        metrics.solved = False
        metrics.error_reason = "timeout"
        metrics.time_total = time_limit
        logger.debug(f"      Timeout after {time_limit}s")
        return metrics

    except Exception as e:
        metrics.solved = False
        metrics.error_reason = "exception"
        metrics.error_details = str(e)[:200]
        logger.debug(f"      Exception: {e}")
        return metrics


def _parse_fd_output(output_text: str, metrics: ExtendedMetrics) -> None:
    """
    Parse Fast Downward output to extract comprehensive metrics.

    Modifies the metrics object in-place.
    """

    # ====== PLAN METRICS ======
    match = re.search(r"Plan length:\s*(\d+)", output_text)
    if match:
        metrics.cost = int(match.group(1))

    # ====== SEARCH METRICS ======

    # Expanded states
    matches = list(re.finditer(r"Expanded\s+(\d+)\s+states?", output_text))
    if matches:
        metrics.expansions = int(matches[-1].group(1))

    # Generated states
    matches = list(re.finditer(r"Generated\s+(\d+)\s+states?", output_text))
    if matches:
        metrics.generated_states = int(matches[-1].group(1))

    # Evaluated states
    matches = list(re.finditer(r"Evaluated\s+(\d+)\s+states?", output_text))
    if matches:
        metrics.evaluated_states = int(matches[-1].group(1))

    # ====== TIMING METRICS ======

    # Search time from FD output
    matches = list(re.finditer(r"Search time:\s+([\d.]+)s", output_text))
    if matches:
        metrics.time_search = float(matches[-1].group(1))

    # Main loop time
    matches = list(re.finditer(r"Main loop runtime:\s+([\d.]+)", output_text))
    if matches:
        metrics.time_ms_main_loop = float(matches[-1].group(1))

    # Initialization time (if available)
    matches = list(re.finditer(r"Initialization time:\s+([\d.]+)", output_text))
    if matches:
        metrics.time_initialization = float(matches[-1].group(1))

    # ====== M&S SPECIFIC METRICS ======

    # Maximum intermediate abstraction size
    matches = list(re.finditer(
        r"Maximum intermediate abstraction size:\s*(\d+)",
        output_text
    ))
    if matches:
        metrics.max_abstraction_size = int(matches[-1].group(1))

    # Final abstraction size
    matches = list(re.finditer(
        r"Final abstraction size:\s*(\d+)",
        output_text
    ))
    if matches:
        metrics.final_abstraction_size = int(matches[-1].group(1))

    # Number of merges
    merge_count = len(re.findall(r"Next pair of indices:", output_text))
    if merge_count > 0:
        metrics.num_merges = merge_count

    # Shrinking operations
    shrink_count = len(re.findall(
        r"Shrinking|Transition system.*shrink",
        output_text,
        re.IGNORECASE
    ))
    if shrink_count > 0:
        metrics.num_shrinks = shrink_count

    # Label reduction operations
    label_reduction_count = len(re.findall(r"Label reduction", output_text))
    if label_reduction_count > 0:
        metrics.num_label_reductions = label_reduction_count

    # Max states before merge
    matches = list(re.finditer(
        r"max_states_before_merge.*?(\d+)",
        output_text
    ))
    if matches:
        metrics.max_states_before_merge = int(matches[-1].group(1))


# ============================================================================
# STEP 4: RUN BASELINE ON BENCHMARK SET
# ============================================================================

def run_baseline_on_benchmark_set(
        benchmark_set: List[Tuple[str, str, str]],
        baseline_name: str,
        search_config: str,
        max_problems: Optional[int] = None
) -> Tuple[Dict[str, Any], List[DetailedMetrics]]:
    """
    Run one baseline on a benchmark set, return DetailedMetrics.

    Args:
        benchmark_set: List of (domain, problem, problem_id) tuples
        baseline_name: Human-readable name
        search_config: FD search configuration
        max_problems: Limit number of problems (for testing)

    Returns:
        Tuple of (aggregate_results, detailed_metrics_list of DetailedMetrics)
    """
    logger.info(f"\n  Running: {baseline_name}")

    if max_problems:
        benchmark_set = benchmark_set[:max_problems]

    logger.info(f"    Problems: {len(benchmark_set)}")

    temp_dir = os.path.join(
        EvaluationConfig.FD_TEMP_DIR,
        baseline_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    )

    os.makedirs(temp_dir, exist_ok=True)

    baseline_runner = BaselineRunner(
        timeout_sec=EvaluationConfig.TIME_LIMIT_PER_RUN_S,
        downward_dir=str(PROJECT_ROOT / "downward")
    )

    try:
        detailed_results = []
        solved_count = 0

        for i, (domain_file, problem_file, problem_id) in enumerate(benchmark_set, 1):
            logger.debug(f"    [{i}/{len(benchmark_set)}] {problem_id}...")

            # Use BaselineRunner to get DetailedMetrics
            result = baseline_runner.run(
                domain_file=domain_file,
                problem_file=problem_file,
                search_config=search_config,
                baseline_name=baseline_name
            )

            detailed_results.append(result)

            if result.solved:
                solved_count += 1
                logger.info(
                    f"    [{i}/{len(benchmark_set)}] {problem_id:<25} ✓ "
                    f"cost={result.plan_cost} time={result.wall_clock_time:.2f}s"
                )
            else:
                logger.warning(
                    f"    [{i}/{len(benchmark_set)}] {problem_id:<25} ✗ {result.error_type}"
                )

        # Aggregate results from DetailedMetrics
        aggregate = _aggregate_detailed_metrics(baseline_name, detailed_results)

        return aggregate, detailed_results

    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def _aggregate_detailed_metrics(
        baseline_name: str,
        metrics_list: List[DetailedMetrics]
) -> Dict[str, Any]:
    """Aggregate DetailedMetrics across problems."""

    total_problems = len(metrics_list)
    solved_list = [m for m in metrics_list if m.solved]
    solved_count = len(solved_list)

    solve_rate = (solved_count / total_problems * 100) if total_problems > 0 else 0

    def safe_avg(values: List[Any]) -> float:
        valid = [v for v in values if v is not None and isinstance(v, (int, float)) and v > 0]
        return sum(valid) / len(valid) if valid else 0.0

    result = {
        "name": baseline_name,
        "set_size": total_problems,
        "solved": solved_count,
        "solve_rate_%": round(solve_rate, 1),

        # Timing (averaged over solved)
        "avg_time_total_s": round(safe_avg([m.wall_clock_time for m in solved_list]), 3),
        "avg_time_translation_s": round(safe_avg([m.translate_time for m in solved_list]), 3),
        "avg_time_search_s": round(safe_avg([m.search_time for m in solved_list]), 3),
        "avg_time_initialization_s": round(safe_avg([m.preprocess_time for m in solved_list]), 3),
        "avg_time_ms_main_loop_s": 0,  # Not available from BaselineRunner

        # Search metrics
        "avg_expansions": int(safe_avg([m.nodes_expanded for m in solved_list])),
        "avg_cost": int(safe_avg([m.plan_cost for m in solved_list])),
        "avg_generated_states": int(safe_avg([m.nodes_generated for m in solved_list])),
        "avg_evaluated_states": 0,  # Not available from BaselineRunner

        # M&S metrics (0 for baseline FD)
        "avg_max_abstraction_size": 0,
        "avg_final_abstraction_size": 0,
        "avg_num_merges": 0,
        "avg_num_shrinks": 0,
        "avg_num_label_reductions": 0,

        # Error summary
        "errors": _summarize_errors([m.error_type for m in metrics_list if not m.solved])
    }

    logger.info(
        f"    ✓ Solved {solved_count}/{total_problems} ({solve_rate:.1f}%) | "
        f"Avg time: {result['avg_time_total_s']:.3f}s "
        f"(trans: {result['avg_time_translation_s']:.3f}s, "
        f"search: {result['avg_time_search_s']:.3f}s)"
    )

    return result


def _summarize_errors(error_reasons: List[str]) -> str:
    """Summarize error reasons."""
    if not error_reasons:
        return "None"

    counts = Counter(error_reasons)
    return ", ".join(f"{e}({c})" for e, c in sorted(counts.items()))


# ============================================================================
# BASELINE RUNNER - FAST DOWNWARD
# ============================================================================

class BaselineRunner:
    """Runs baseline Fast Downward planners."""

    def __init__(self, timeout_sec: int = 300, downward_dir: Optional[str] = None):
        self.timeout_sec = timeout_sec

        if downward_dir:
            self.downward_dir = Path(downward_dir).absolute()
        else:
            self.downward_dir = PROJECT_ROOT / "downward"

        self.fd_bin = self.downward_dir / "builds" / "release" / "bin" / "downward.exe"
        self.fd_translate = self.downward_dir / "builds" / "release" / "bin" / "translate" / "translate.py"

        if not self.fd_bin.exists():
            logger.warning(f"FD binary not found at: {self.fd_bin}")
        if not self.fd_translate.exists():
            logger.warning(f"FD translate not found at: {self.fd_translate}")

    def run(
            self,
            domain_file: str,
            problem_file: str,
            search_config: str,
            baseline_name: str = "FD"
    ) -> DetailedMetrics:
        """Run baseline planner on a problem."""
        problem_name = os.path.basename(problem_file)
        logger.info(f"[BASELINE] {baseline_name}: {problem_name}")

        try:
            # PHASE 1: Translate
            translate_start = time.time()
            work_dir = Path("evaluation_temp")
            work_dir.mkdir(exist_ok=True)
            sas_file = work_dir / "output.sas"

            translate_result = subprocess.run(
                [
                    sys.executable,
                    str(self.fd_translate),
                    domain_file,
                    problem_file,
                    "--sas-file", str(sas_file)
                ],
                cwd=str(self.downward_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            translate_time = time.time() - translate_start

            if translate_result.returncode != 0 or not sas_file.exists():
                logger.debug(f"  [TRANSLATE] Failed: {translate_result.stderr[:200]}")
                return DetailedMetrics(
                    problem_name=problem_name,
                    planner_name=baseline_name,
                    solved=False,
                    wall_clock_time=translate_time,
                    translate_time=translate_time,
                    error_type="translate_error",
                    error_message=translate_result.stderr[:500],
                    timeout_occurred=False
                )

            # PHASE 2: Search
            search_start = time.time()
            search_result = subprocess.run(
                [str(self.fd_bin), "--search", search_config],
                stdin=open(sas_file, 'r'),
                cwd=str(self.downward_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            search_time = time.time() - search_start
            total_time = translate_time + search_time
            output_text = search_result.stdout + search_result.stderr

            # PHASE 3: Parse output
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

            metrics = self._parse_fd_output(output_text)
            if metrics is None:
                return DetailedMetrics(
                    problem_name=problem_name,
                    planner_name=baseline_name,
                    solved=True,
                    wall_clock_time=total_time,
                    translate_time=translate_time,
                    search_time=search_time,
                    error_type="parse_error",
                    timeout_occurred=False
                )

            result = DetailedMetrics(
                problem_name=problem_name,
                planner_name=baseline_name,
                solved=True,
                wall_clock_time=total_time,
                translate_time=translate_time,
                search_time=search_time,
                plan_cost=metrics.get('cost', 0),
                plan_length=metrics.get('cost', 0),
                nodes_expanded=metrics.get('expansions', 0),
                nodes_generated=metrics.get('generated_states', 0),
                search_depth=metrics.get('search_depth', 0),
                branching_factor=metrics.get('branching_factor', 1.0),
                peak_memory_kb=metrics.get('memory', 0),
                timeout_occurred=False
            )

            logger.debug(f"  [SUCCESS] cost={result.plan_cost}, exp={result.nodes_expanded}")
            return result

        except subprocess.TimeoutExpired:
            logger.debug(f"  [TIMEOUT] Exceeded {self.timeout_sec}s")
            return DetailedMetrics(
                problem_name=problem_name,
                planner_name=baseline_name,
                solved=False,
                wall_clock_time=self.timeout_sec,
                error_type="timeout",
                timeout_occurred=True
            )

        except Exception as e:
            logger.error(f"  [ERROR] {e}")
            return DetailedMetrics(
                problem_name=problem_name,
                planner_name=baseline_name,
                solved=False,
                wall_clock_time=0,
                error_type="exception",
                error_message=str(e)[:500],
                timeout_occurred=False
            )

    @staticmethod
    def _parse_fd_output(output_text: str) -> Optional[Dict[str, Any]]:
        """Extract metrics from FD output."""
        metrics = {}

        match = re.search(r'Plan length:\s*(\d+)', output_text)
        if match:
            metrics['cost'] = int(match.group(1))

        matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', output_text))
        if matches:
            metrics['expansions'] = int(matches[-1].group(1))

        matches = list(re.finditer(r'Generated\s+(\d+)\s+state', output_text))
        if matches:
            metrics['generated_states'] = int(matches[-1].group(1))

        match = re.search(r'Search depth:\s*(\d+)', output_text)
        if match:
            metrics['search_depth'] = int(match.group(1))

        match = re.search(r'Branching factor:\s*([\d.]+)', output_text)
        if match:
            metrics['branching_factor'] = float(match.group(1))

        match = re.search(r'Peak memory:\s*(\d+)\s*KB', output_text)
        if match:
            metrics['memory'] = int(match.group(1))

        if 'cost' not in metrics or 'expansions' not in metrics:
            return None

        return metrics


# ============================================================================
# COMPARISON ANALYZER
# ============================================================================

class ComparisonAnalyzer:
    """Analyze and compare evaluation results."""

    def __init__(self, results: List[DetailedMetrics]):
        self.results = results
        self.by_planner = defaultdict(list)
        for result in results:
            self.by_planner[result.planner_name].append(result)

    def get_aggregate_statistics(self, planner_name: str) -> AggregateStatistics:
        """Compute statistics for a planner."""
        results = self.by_planner[planner_name]
        solved = [r for r in results if r.solved]
        num_solved = len(solved)
        num_total = len(results)

        times = [r.wall_clock_time for r in solved if r.wall_clock_time > 0]
        expansions = [r.nodes_expanded for r in solved if r.nodes_expanded > 0]
        costs = [r.plan_cost for r in solved if r.plan_cost > 0]
        h_preservations = [r.h_star_preservation for r in solved]

        def safe_percentile(data, q):
            if not data:
                return 0.0
            return float(np.percentile(data, q))

        return AggregateStatistics(
            planner_name=planner_name,
            num_problems_total=num_total,
            num_problems_solved=num_solved,
            solve_rate_pct=(num_solved / max(num_total, 1)) * 100,
            mean_time_sec=float(np.mean(times)) if times else 0.0,
            median_time_sec=float(np.median(times)) if times else 0.0,
            std_time_sec=float(np.std(times)) if times else 0.0,
            min_time_sec=float(np.min(times)) if times else 0.0,
            max_time_sec=float(np.max(times)) if times else 0.0,
            q1_time_sec=safe_percentile(times, 25),
            q3_time_sec=safe_percentile(times, 75),
            iqr_time_sec=safe_percentile(times, 75) - safe_percentile(times, 25),
            mean_expansions=int(np.mean(expansions)) if expansions else 0,
            median_expansions=int(np.median(expansions)) if expansions else 0,
            std_expansions=int(np.std(expansions)) if expansions else 0,
            mean_plan_cost=int(np.mean(costs)) if costs else 0,
            median_plan_cost=int(np.median(costs)) if costs else 0,
            std_plan_cost=int(np.std(costs)) if costs else 0,
            mean_h_preservation=float(np.mean(h_preservations)) if h_preservations else 1.0,
            median_h_preservation=float(np.median(h_preservations)) if h_preservations else 1.0,
            unsolved_count=len([r for r in results if not r.solved]),
            timeout_count=len([r for r in results if r.timeout_occurred]),
            error_count=len([r for r in results if r.error_type]),
            total_wall_clock_time_sec=float(sum(times)) if times else 0.0,
        )

    def get_per_problem_winners(self) -> Dict[str, str]:
        """Get best planner for each problem."""
        problems = set(r.problem_name for r in self.results)
        winners = {}

        for problem in sorted(problems):
            problem_results = [r for r in self.results if r.problem_name == problem]
            solved_results = [r for r in problem_results if r.solved]

            if solved_results:
                best = min(solved_results, key=lambda r: r.wall_clock_time)
                winners[problem] = best.planner_name
            else:
                winners[problem] = "NONE"

        return winners

    def get_speedup_analysis(self) -> Dict[str, Any]:
        """Analyze GNN speedup vs baselines."""
        if "GNN" not in self.by_planner:
            return {}

        gnn_results = self.by_planner["GNN"]
        speedups = []

        for gnn_result in gnn_results:
            if not gnn_result.solved:
                continue

            baseline_results = [
                r for r in self.results
                if r.problem_name == gnn_result.problem_name
                and r.planner_name != "GNN"
                and r.solved
            ]

            if baseline_results:
                best_baseline = min(baseline_results, key=lambda r: r.wall_clock_time)
                if gnn_result.wall_clock_time > 0:
                    speedup = best_baseline.wall_clock_time / gnn_result.wall_clock_time
                    speedups.append(speedup)

        return {
            'mean_speedup': float(np.mean(speedups)) if speedups else 0.0,
            'geometric_mean_speedup': float(np.exp(np.mean(np.log(speedups)))) if speedups else 0.0,
            'num_speedup_comparisons': len(speedups),
        }


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results_to_csv(
        results: List[DetailedMetrics],
        output_path: str
) -> None:
    """Export detailed results to CSV (handles DetailedMetrics)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to export")
        return

    fieldnames = [
        "planner_name",
        "problem_name",
        "solved",
        "plan_cost",
        "wall_clock_time",
        "translate_time",
        "search_time",
        "nodes_expanded",
        "nodes_generated",
        "search_depth",
        "branching_factor",
        "peak_memory_kb",
        "h_star_preservation",
        "error_type",
        "timestamp"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            # Ensure all fields exist
            for field in fieldnames:
                if field not in row:
                    row[field] = None
            writer.writerow(row)

    logger.info(f"✓ CSV exported: {output_path}")


def export_statistics_to_json(
        statistics: Dict[str, AggregateStatistics],
        output_path: str
) -> None:
    """Export statistics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {name: stats.to_dict() for name, stats in statistics.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"✓ JSON exported: {output_path}")


def export_summary_report(
        statistics: Dict[str, AggregateStatistics],
        speedup_analysis: Dict[str, Any],
        winners: Dict[str, str],
        output_path: str
) -> None:
    """Export comprehensive text report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        # Statistics section
        f.write("PLANNER STATISTICS\n")
        f.write("-" * 100 + "\n")
        for planner_name in sorted(statistics.keys()):
            stats = statistics[planner_name]
            f.write(f"\n{planner_name}:\n")
            f.write(f"  Solved: {stats.num_problems_solved}/{stats.num_problems_total} "
                   f"({stats.solve_rate_pct:.1f}%)\n")
            f.write(f"  Time: {stats.mean_time_sec:.2f}s (±{stats.std_time_sec:.2f}s)\n")
            f.write(f"  Expansions: {stats.mean_expansions:,}\n")
            if stats.mean_h_preservation < 1.0:
                f.write(f"  H* Preservation: {stats.mean_h_preservation:.3f}\n")

        # Winners
        if winners:
            f.write("\n\nPER-PROBLEM WINNERS\n")
            f.write("-" * 100 + "\n")
            for problem in sorted(winners.keys()):
                f.write(f"  {problem}: {winners[problem]}\n")

        # Speedup
        if speedup_analysis:
            f.write("\n\nGNN SPEEDUP ANALYSIS\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Mean Speedup: {speedup_analysis.get('mean_speedup', 0):.2f}x\n")
            f.write(f"  Geometric Mean Speedup: {speedup_analysis.get('geometric_mean_speedup', 0):.2f}x\n")
            f.write(f"  Comparisons: {speedup_analysis.get('num_speedup_comparisons', 0)}\n")

        f.write("\n" + "=" * 100 + "\n")

    logger.info(f"✓ Report exported: {output_path}")


# ============================================================================
# STEP 5: GENERATE REPORTS
# ============================================================================

def generate_reports(
        all_results: List[Dict[str, Any]],
        all_detailed: List[ExtendedMetrics]
) -> None:
    """Generate CSV summary and detailed JSON reports."""

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: GENERATE REPORTS")
    logger.info("=" * 80)

    if not all_results:
        logger.error("No results to report")
        return

    os.makedirs(EvaluationConfig.OUTPUT_DIR, exist_ok=True)

    # ====== CSV SUMMARY ======
    fieldnames = [
        "name",
        "set_size",
        "solved",
        "solve_rate_%",
        "avg_time_total_s",
        "avg_time_translation_s",
        "avg_time_search_s",
        "avg_time_initialization_s",
        "avg_time_ms_main_loop_s",
        "avg_expansions",
        "avg_cost",
        "avg_generated_states",
        "avg_evaluated_states",
        "avg_max_abstraction_size",
        "avg_final_abstraction_size",
        "avg_num_merges",
        "avg_num_shrinks",
        "avg_num_label_reductions",
        "errors"
    ]

    csv_path = EvaluationConfig.OUTPUT_CSV
    logger.info(f"\nWriting CSV report: {csv_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"  ✓ {len(all_results)} baseline configurations")

    # ====== JSON DETAILED ======
    json_path = EvaluationConfig.OUTPUT_DETAILED
    logger.info(f"\nWriting detailed JSON: {json_path}")

    detailed_dicts = [m.to_dict() for m in all_detailed]
    with open(json_path, "w") as f:
        json.dump(detailed_dicts, f, indent=2)

    logger.info(f"  ✓ {len(all_detailed)} individual problem results")

    # ====== PRINT SUMMARY TABLE ======
    logger.info("\n" + "=" * 100)
    logger.info("BASELINE PERFORMANCE SUMMARY")
    logger.info("=" * 100)

    header = (
        f"{'Planner':<25} "
        f"{'Solved':<10} "
        f"{'Total':<8} "
        f"{'Time':<10} "
        f"{'Trans':<10} "
        f"{'Search':<10} "
        f"{'Exp':<10} "
        f"{'Cost':<8}"
    )
    logger.info(header)
    logger.info("-" * 100)

    for result in all_results:
        name = result["name"][:23]
        solved = f"{result['solved']}/{result['set_size']}"
        rate = f"({result['solve_rate_%']:.0f}%)"
        time_str = f"{result['avg_time_total_s']:.3f}s"
        trans_str = f"{result['avg_time_translation_s']:.3f}s"
        search_str = f"{result['avg_time_search_s']:.3f}s"
        exp_str = f"{result['avg_expansions']:,}"
        cost_str = f"{result['avg_cost']}"

        logger.info(
            f"{name:<25} "
            f"{solved:<10} {rate:<8} "
            f"{time_str:<10} "
            f"{trans_str:<10} "
            f"{search_str:<10} "
            f"{exp_str:<10} "
            f"{cost_str:<8}"
        )

    logger.info("-" * 100)
    logger.info(f"\nFull reports:")
    logger.info(f"  CSV:      {os.path.abspath(csv_path)}")
    logger.info(f"  Detailed: {os.path.abspath(json_path)}")


# ============================================================================
# UTILITIES
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation Framework for GNN Merge Strategy Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluation.py --baselines --seed 42\n"
            "  python evaluation.py --baselines --max-problems 3\n"
            "  python evaluation.py --baselines --domains blocksworld --sizes small,medium\n"
        )
    )

    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Run baseline evaluation on discovered benchmarks"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        dest="time_limit",
        help="Time limit per problem in seconds (default: 300)"
    )

    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        dest="max_problems",
        help="Limit problems per benchmark set (for quick testing)"
    )

    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated domains: blocksworld,logistics (default: all)"
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated sizes: small,medium,large (default: all)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def filter_benchmarks(
        all_benchmarks: Dict[str, List[Tuple[str, str, str]]],
        domains: Optional[str],
        sizes: Optional[str]
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Filter benchmarks by domain and size."""

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

    return filtered


# ============================================================================
# MAIN EVALUATION ORCHESTRATOR
# ============================================================================

class EvaluationFramework:
    """Main evaluation orchestrator."""

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: List[ExtendedMetrics] = []

    def run_gnn_and_random_evaluation(
            self,
            model_path: str,
            domain_file: str,
            problem_files: List[str],
            num_runs_per_problem: int = 1,
    ) -> Dict[str, Any]:
        """
        Run GNN and Random merge strategy evaluation.

        Uses the dedicated gnn_random_evaluation module for cleaner separation
        of concerns and better code organization.

        Args:
            model_path: Path to trained GNN model
            domain_file: Path to domain PDDL file
            problem_files: List of problem PDDL files
            num_runs_per_problem: Number of runs per problem

        Returns:
            Dictionary with evaluation results and statistics
        """
        logger.info("\n" + "=" * 100)
        logger.info("GNN AND RANDOM MERGE STRATEGY EVALUATION")
        logger.info("=" * 100 + "\n")

        try:
            # Create evaluation framework
            eval_framework = GNNRandomEvaluationFramework(
                model_path=model_path,
                domain_file=domain_file,
                problem_files=problem_files,
                output_dir=str(self.output_dir / "gnn_random_evaluation"),
                num_runs_per_problem=num_runs_per_problem,
                downward_dir=str(PROJECT_ROOT / "downward"),
            )

            # Run evaluation
            gnn_results, random_results = eval_framework.evaluate()

            # All results are DetailedMetrics
            all_detailed_results = gnn_results + random_results
            self.all_results.extend(all_detailed_results)

            # Generate summary
            summary = eval_framework.to_summary()

            logger.info("\n📊 GNN vs Random Comparison:")
            if "GNN" in summary and "Random" in summary:
                gnn_solve = summary["GNN"].get("solve_rate_pct", 0)
                random_solve = summary["Random"].get("solve_rate_pct", 0)
                logger.info(f"   GNN solve rate: {gnn_solve:.1f}%")
                logger.info(f"   Random solve rate: {random_solve:.1f}%")
                logger.info(f"   Improvement: +{gnn_solve - random_solve:.1f}%")

            return {
                "status": "success",
                "gnn_results": [r.to_dict() for r in gnn_results],
                "random_results": [r.to_dict() for r in random_results],
                "summary": summary,
                "num_gnn_solved": sum(1 for r in gnn_results if r.solved),
                "num_random_solved": sum(1 for r in random_results if r.solved),
            }

        except Exception as e:
            logger.error(f"GNN and Random evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "failed", "error": str(e)}

    def run_baseline_evaluation(
            self,
            timeout_sec: int = 300,
            max_problems: Optional[int] = None,
            domains: Optional[str] = None,
            sizes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run baseline evaluation on discovered benchmarks."""
        logger.info("\n" + "=" * 100)
        logger.info("BASELINE EVALUATION FRAMEWORK")
        logger.info("=" * 100 + "\n")

        # Step 1: Verify setup
        if not verify_fd_setup():
            return {"status": "failed", "phase": "verification"}

        # Step 2: Discover benchmarks
        all_benchmarks = discover_benchmarks()
        if not all_benchmarks:
            return {"status": "failed", "phase": "discovery"}

        # Step 3: Filter if requested
        if domains or sizes:
            all_benchmarks = filter_benchmarks(all_benchmarks, domains, sizes)
            if not all_benchmarks:
                logger.error("No benchmarks match filter criteria")
                return {"status": "failed", "phase": "filtering"}
            logger.info(f"Filtered to {len(all_benchmarks)} benchmark set(s)")

        # Step 4: Run baselines
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: RUN BASELINE PLANNERS")
        logger.info("=" * 80)

        all_aggregate_results = []
        all_detailed_metrics = []  # This will be DetailedMetrics
        baselines = EvaluationConfig.get_baselines(EvaluationConfig.RANDOM_SEED)

        for benchmark_name, benchmark_set in all_benchmarks.items():
            logger.info(f"\n{benchmark_name} Benchmark Set")
            logger.info("-" * 80)

            for baseline_config in baselines:
                aggregate, detailed = run_baseline_on_benchmark_set(
                    benchmark_set=benchmark_set,
                    baseline_name=baseline_config["name"],
                    search_config=baseline_config["search"],
                    max_problems=max_problems
                )

                all_aggregate_results.append(aggregate)
                all_detailed_metrics.extend(detailed)  # These are DetailedMetrics

        # Step 5: Generate reports
        generate_reports(all_aggregate_results, all_detailed_metrics)

        # Step 6: Generate plots (NEW)
        try:
            plotter = GenerateEvaluationPlots(
                output_dir=str(self.output_dir)
            )

            # Convert results to format for plotter
            stats_dicts = {result['name']: result for result in all_aggregate_results}
            results_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r
                             for r in all_detailed_metrics]

            plotter.generate_all_plots(
                statistics=stats_dicts,
                results=results_dicts,
                gnn_results={}
            )
        except Exception as e:
            logger.warning(f"Plot generation failed (non-critical): {e}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ BASELINE EVALUATION COMPLETE")
        logger.info("=" * 80)

        return {
            "status": "success",
            "num_baselines": len(baselines),
            "num_benchmark_sets": len(all_benchmarks),
            "num_results": len(all_detailed_metrics),
            "output_dir": str(self.output_dir),
        }

    def run_comprehensive_evaluation(
            self,
            domain_file: str,
            problem_pattern: str,
            model_path: str,
            timeout_sec: int = 300,
            include_baselines: bool = True,
            include_gnn_random: bool = True,
            baseline_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run complete evaluation: baselines + GNN + Random + analysis."""
        logger.info("\n" + "=" * 100)
        logger.info("COMPREHENSIVE EVALUATION FRAMEWORK")
        logger.info("=" * 100 + "\n")

        # Load problems
        problems = sorted(glob.glob(problem_pattern))
        if not problems:
            logger.error("No problems found!")
            return {}

        problem_names = [Path(p).name for p in problems]
        logger.info(f"Found {len(problems)} problem(s)")

        # Run baselines
        if include_baselines:
            self._run_all_baselines(domain_file, problems, problem_names, timeout_sec, baseline_names)

        # Run GNN and Random using new unified module
        if include_gnn_random:
            try:
                gnn_random_result = self.run_gnn_and_random_evaluation(
                    model_path=model_path,
                    domain_file=domain_file,
                    problem_files=problems,
                    num_runs_per_problem=1,
                )
                if gnn_random_result.get("status") != "success":
                    logger.warning(f"GNN and Random evaluation failed: {gnn_random_result.get('error')}")
            except Exception as e:
                logger.warning(f"GNN and Random evaluation skipped: {e}")

        # Generate reports
        return self._generate_comprehensive_report()

    def _run_all_baselines(
            self,
            domain_file: str,
            problems: List[str],
            problem_names: List[str],
            timeout_sec: int,
            baseline_names: Optional[List[str]] = None
    ):
        """Run all baseline configurations."""
        logger.info("\n" + "-" * 100)
        logger.info("RUNNING BASELINE PLANNERS")
        logger.info("-" * 100 + "\n")

        baseline_runner = BaselineRunner(timeout_sec)
        configs = EvaluationConfig.BASELINE_CONFIGS

        if baseline_names:
            configs = [c for c in configs if c['name'] in baseline_names]

        for config in configs:
            logger.info(f"\n{config['name']}: {config['description']}")
            logger.info("-" * 80)

            for problem, problem_name in zip(problems, problem_names):
                result = baseline_runner.run(
                    domain_file,
                    problem,
                    config['search'],
                    baseline_name=config['name']
                )
                self.all_results.append(result)

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive reports and statistics."""
        logger.info("\n" + "=" * 100)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 100 + "\n")

        if not self.all_results:
            logger.error("No results to report!")
            return {}

        # Compute statistics
        by_planner = defaultdict(list)
        for result in self.all_results:
            by_planner[result.planner_name].append(result)

        statistics = {}
        for planner_name in sorted(by_planner.keys()):
            analyzer = ComparisonAnalyzer(by_planner[planner_name])
            statistics[planner_name] = analyzer.get_aggregate_statistics(planner_name)

            logger.info(f"\n{planner_name}:")
            logger.info(f"  Solve rate: {statistics[planner_name].solve_rate_pct:.1f}%")
            logger.info(f"  Avg time: {statistics[planner_name].mean_time_sec:.2f}s")
            logger.info(f"  Avg expansions: {statistics[planner_name].mean_expansions:,}")

        # Comparison analysis
        analyzer = ComparisonAnalyzer(self.all_results)
        winners = analyzer.get_per_problem_winners()
        speedup = analyzer.get_speedup_analysis()

        # Export results
        self._export_all_results(statistics, winners, speedup)

        # ===== NEW: GENERATE VISUALIZATION PLOTS =====
        self._generate_evaluation_plots(statistics)

        return {
            "statistics": {name: stats.to_dict() for name, stats in statistics.items()},
            "speedup_analysis": speedup,
            "num_problems": len(set(r.problem_name for r in self.all_results)),
            "num_planners": len(statistics),
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_evaluation_plots(self, statistics: Dict[str, AggregateStatistics]) -> None:
        """Generate visualization plots from evaluation results."""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING EVALUATION PLOTS")
        logger.info("=" * 80)

        try:
            plotter = GenerateEvaluationPlots(output_dir=str(self.output_dir))

            # Convert AggregateStatistics to dicts for compatibility
            stats_dicts = {}
            for planner_name, stats in statistics.items():
                stats_dicts[planner_name] = stats.to_dict()

            # Convert DetailedMetrics results to dict format
            results_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r
                             for r in self.all_results]

            plotter.generate_all_plots(
                statistics=stats_dicts,
                results=results_dicts,
                gnn_results={}
            )

            logger.info("\n✓ All evaluation plots generated successfully")
            logger.info(f"  Plots saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to generate evaluation plots: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _export_all_results(
            self,
            statistics: Dict[str, AggregateStatistics],
            winners: Dict[str, str],
            speedup: Dict[str, Any]
    ):
        """Export results in all formats."""
        # CSV
        export_results_to_csv(
            self.all_results,
            str(self.output_dir / "evaluation_results.csv")
        )

        # JSON
        export_statistics_to_json(
            statistics,
            str(self.output_dir / "evaluation_summary.json")
        )

        # Text report
        export_summary_report(
            statistics,
            speedup,
            winners,
            str(self.output_dir / "evaluation_report.txt")
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main execution orchestration."""
    args = parse_arguments()

    # Configure
    EvaluationConfig.TIME_LIMIT_PER_RUN_S = args.time_limit
    EvaluationConfig.RANDOM_SEED = args.seed

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Banner
    logger.info("\n" + "=" * 80)
    logger.info("GNN MERGE STRATEGY LEARNING - EVALUATION FRAMEWORK")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Seed:              {EvaluationConfig.RANDOM_SEED}")
    logger.info(f"  Time limit:        {EvaluationConfig.TIME_LIMIT_PER_RUN_S}s per problem")
    if args.max_problems:
        logger.info(f"  Max problems:      {args.max_problems} per set")
    if args.domains:
        logger.info(f"  Domains:           {args.domains}")
    if args.sizes:
        logger.info(f"  Sizes:             {args.sizes}")

    if args.baselines:
        # Run baseline evaluation
        framework = EvaluationFramework()
        result = framework.run_baseline_evaluation(
            timeout_sec=args.time_limit,
            max_problems=args.max_problems,
            domains=args.domains,
            sizes=args.sizes,
        )

        if result.get("status") == "success":
            return 0
        else:
            return 1
    else:
        logger.info("\n--baselines flag not provided")
        logger.info("Usage: python evaluation.py --baselines [options]")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)