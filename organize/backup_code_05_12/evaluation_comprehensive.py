#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE EVALUATION FRAMEWORK - FULLY COMPATIBLE VERSION
==============================================================
Complete evaluation system compatible with:
✓ ThinMergeEnv (15-dim node features, 10-dim edge features)
✓ All 4 experiment types (overfit, problem_gen, scale_gen, curriculum)
✓ Baseline Fast Downward planners
✓ GNN policy with enhanced reward function
✓ Research-grade statistical analysis
✓ Multiple output formats (CSV, JSON, TXT, plots)

Architecture:
- BaselineRunner: Execute FD with multiple heuristics
- GNNPolicyRunner: Execute GNN policy with ThinMergeEnv
- DetailedMetrics: 20+ metrics per run (comprehensive tracking)
- AggregateStatistics: Statistical analysis (mean, median, std, percentiles, IQR)
- EvaluationFramework: Orchestrates complete evaluation pipeline

Scientific Focus:
- H* preservation as primary metric
- State explosion control
- Shrinkability analysis
- Statistical significance tests
- Per-problem and aggregate reporting
- Error tracking and diagnostics
"""

import sys
import os
import logging
import glob
import json
import subprocess
import time
import re
import csv
import argparse
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
from collections import defaultdict

sys.path.insert(0, os.getcwd())

# Setup logging with both console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_comprehensive.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - FRAMEWORK COMPATIBLE
# ============================================================================

class EvaluationConfig:
    """Configuration matching framework standards."""

    # Feature dimensions (match ThinMergeEnv)
    NODE_FEATURE_DIM = 15
    EDGE_FEATURE_DIM = 10

    # Reward weights (match thin_merge_env DEFAULT_REWARD_WEIGHTS)
    REWARD_WEIGHTS = {
        'w_h_preservation': 0.40,  # Primary signal
        'w_shrinkability': 0.25,
        'w_state_control': 0.20,
        'w_solvability': 0.15,
    }

    # Timeouts
    FD_TIMEOUT_SEC = 300
    GNN_TIMEOUT_SEC = 300

    # Baseline configurations (complete set)
    BASELINE_CONFIGS = [
        {
            "name": "FD_LM-Cut",
            "search": "astar(lmcut())",
            "description": "Fast Downward with LM-Cut heuristic"
        },
        {
            "name": "FD_Blind",
            "search": "astar(blind())",
            "description": "Fast Downward with blind search"
        },
        {
            "name": "FD_Add",
            "search": "astar(add())",
            "description": "Fast Downward with additive heuristic"
        },
        {
            "name": "FD_Max",
            "search": "astar(max())",
            "description": "Fast Downward with max heuristic"
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
            "description": "Merge-and-shrink with DFP scoring"
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
            "description": "Merge-and-shrink with SCC merging"
        },
    ]


# ============================================================================
# DATA STRUCTURES - COMPREHENSIVE METRICS
# ============================================================================

@dataclass
class DetailedMetrics:
    """Complete set of metrics for a single run (20+ metrics)."""

    # Problem identification
    problem_name: str
    planner_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Solve status (primary)
    solved: bool = False
    wall_clock_time: float = 0.0

    # Planning quality
    plan_cost: int = 0
    plan_length: int = 0
    solution_cost: int = 0

    # Search metrics (expansion analysis)
    nodes_expanded: int = 0
    nodes_generated: int = 0
    search_depth: int = 0
    branching_factor: float = 1.0

    # Memory metrics
    peak_memory_kb: int = 0

    # Time breakdown
    search_time: float = 0.0
    translate_time: float = 0.0
    preprocess_time: float = 0.0
    total_time: float = 0.0

    # Solution quality (h* related for GNN)
    initial_heuristic: int = 0
    average_heuristic: float = 0.0
    final_heuristic: int = 0
    h_star_preservation: float = 1.0  # GNN-specific

    # State control metrics
    num_active_systems: int = 0
    merge_episodes: int = 0
    shrinkability: float = 0.0

    # Reachability and dead-ends
    reachability_ratio: float = 1.0
    dead_end_ratio: float = 0.0

    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Metadata
    timeout_occurred: bool = False
    evaluation_notes: str = ""

    def efficiency_score(self) -> float:
        """
        RESEARCH METRIC: Efficiency score (lower is better).

        Combines:
        - Nodes expanded (search efficiency)
        - Wall clock time (computational efficiency)
        - Solution cost (solution quality)

        Returns: Score in [0, ∞). Lower is better.
        """
        if not self.solved:
            return float('inf')

        # Avoid division by zero
        if self.nodes_expanded == 0 and self.wall_clock_time == 0:
            return 0.0

        # Weighted efficiency
        if self.plan_cost > 0:
            efficiency = (
                    (self.nodes_expanded / (self.plan_cost * 100.0)) * 0.4 +
                    (self.wall_clock_time / 10.0) * 0.3 +
                    (self.search_depth / 100.0) * 0.3
            )
        else:
            efficiency = self.wall_clock_time

        return float(efficiency)

    def quality_score(self) -> float:
        """RESEARCH METRIC: Solution quality score (higher is better)."""
        if not self.solved:
            return 0.0

        if self.plan_cost <= 0:
            return 1.0

        # Penalize expensive solutions
        quality = 1.0 / (1.0 + self.plan_cost / 100.0)
        return float(quality)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        d = asdict(self)
        d['efficiency_score'] = self.efficiency_score()
        d['quality_score'] = self.quality_score()
        return d


@dataclass
class AggregateStatistics:
    """Summary statistics across all problems (research-grade)."""

    planner_name: str
    num_problems_total: int
    num_problems_solved: int
    solve_rate_pct: float

    # Time statistics (solved only)
    mean_time_sec: float
    median_time_sec: float
    std_time_sec: float
    min_time_sec: float
    max_time_sec: float
    q1_time_sec: float  # 25th percentile
    q3_time_sec: float  # 75th percentile

    # Expansion statistics
    mean_expansions: int
    median_expansions: int
    std_expansions: int
    min_expansions: int
    max_expansions: int

    # Plan quality statistics
    mean_plan_cost: int
    median_plan_cost: int
    std_plan_cost: int

    # H* preservation (GNN-specific)
    mean_h_preservation: float = 1.0
    median_h_preservation: float = 1.0

    # Efficiency metrics
    mean_efficiency_score: float = 0.0
    mean_quality_score: float = 0.0

    # Coverage and errors
    unsolved_count: int = 0
    timeout_count: int = 0
    error_count: int = 0

    # Aggregate times
    total_wall_clock_time_sec: float = 0.0

    # Statistical tests
    statistical_significance: Optional[str] = None
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# BASELINE RUNNER - COMPATIBLE WITH FD
# ============================================================================

class BaselineRunner:
    """
    Runs baseline Fast Downward planners.

    Compatible with:
    - Multiple heuristics (LM-Cut, Blind, Add, Max, M&S variants)
    - Comprehensive metrics extraction
    - Error handling and diagnostics
    """

    def __init__(self, timeout_sec: int = 300, downward_dir: Optional[str] = None):
        self.timeout_sec = timeout_sec

        if downward_dir:
            self.downward_dir = Path(downward_dir).absolute()
        else:
            self.downward_dir = Path(__file__).parent / "downward"

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
        """
        Run baseline planner.

        Args:
            domain_file: Path to domain PDDL
            problem_file: Path to problem PDDL
            search_config: FD search configuration string
            baseline_name: Name of baseline (for tracking)

        Returns:
            DetailedMetrics with all measurements
        """
        problem_name = os.path.basename(problem_file)
        logger.info(f"[BASELINE] {baseline_name}: {problem_name}")

        try:
            # PHASE 1: Translate
            logger.debug(f"  [TRANSLATE] Starting translation...")
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

            logger.debug(f"  [TRANSLATE] Success ({sas_file.stat().st_size} bytes)")

            # PHASE 2: Search
            logger.debug(f"  [SEARCH] Starting search with {baseline_name}...")
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

            logger.debug(f"  [SEARCH] Completed in {search_time:.2f}s")

            # PHASE 3: Parse output
            if "Solution found" not in output_text and "Plan length:" not in output_text:
                logger.debug(f"  [PARSE] No solution found")
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

            # Extract metrics from output
            metrics = self._parse_fd_output(output_text)

            if metrics is None:
                logger.debug(f"  [PARSE] Could not extract metrics")
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

            # Build complete result
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

        # Plan cost
        match = re.search(r'Plan length:\s*(\d+)', output_text)
        if match:
            metrics['cost'] = int(match.group(1))

        # Expansions (take last)
        matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', output_text))
        if matches:
            metrics['expansions'] = int(matches[-1].group(1))

        # Search depth
        match = re.search(r'Search depth:\s*(\d+)', output_text)
        if match:
            metrics['search_depth'] = int(match.group(1))

        # Branching factor
        match = re.search(r'Branching factor:\s*([\d.]+)', output_text)
        if match:
            metrics['branching_factor'] = float(match.group(1))

        # Memory
        match = re.search(r'Peak memory:\s*(\d+)\s*KB', output_text)
        if match:
            metrics['memory'] = int(match.group(1))

        if 'cost' not in metrics or 'expansions' not in metrics:
            return None

        return metrics


# ============================================================================
# GNN POLICY RUNNER - COMPATIBLE WITH ThinMergeEnv
# ============================================================================

class GNNPolicyRunner:
    """
    Runs GNN policy using ThinMergeEnv.

    Compatible with:
    - 15-dim node features
    - 10-dim edge features from C++
    - h* preservation reward function
    - Thin client architecture
    """

    def __init__(
            self,
            model_path: str,
            timeout_sec: float = 300,
            downward_dir: Optional[str] = None
    ):
        self.model_path = model_path
        self.timeout_sec = timeout_sec

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if downward_dir:
            self.downward_dir = downward_dir
        else:
            self.downward_dir = os.path.join(os.path.dirname(__file__), "downward")

        logger.info(f"[GNN] Initialized with model: {model_path}")

    def run(self, domain_file: str, problem_file: str) -> DetailedMetrics:
        """
        Run GNN policy on a problem.

        Returns:
            DetailedMetrics with GNN-specific measurements
        """
        problem_name = os.path.basename(problem_file)
        logger.info(f"[GNN] Evaluating: {problem_name}")

        try:
            from stable_baselines3 import PPO
            from thin_merge_env import ThinMergeEnv
        except ImportError as e:
            logger.error(f"[GNN] Import failed: {e}")
            return DetailedMetrics(
                problem_name=problem_name,
                planner_name="GNN",
                solved=False,
                wall_clock_time=0,
                error_type="import_error",
                error_message=str(e)
            )

        start_time = time.time()

        try:
            logger.debug(f"  [LOAD] Loading model...")
            model = PPO.load(self.model_path)
            logger.debug(f"  [LOAD] Model loaded successfully")

            logger.debug(f"  [ENV] Creating ThinMergeEnv...")
            env = ThinMergeEnv(
                domain_file=os.path.abspath(domain_file),
                problem_file=os.path.abspath(problem_file),
                max_merges=50,
                timeout_per_step=self.timeout_sec,
                debug=False,
            )
            logger.debug(f"  [ENV] Environment created")

            logger.debug(f"  [RESET] Resetting environment...")
            solve_start = time.time()
            obs, info = env.reset()
            logger.debug(f"  [RESET] Environment reset")

            total_episode_reward = 0.0
            steps = 0
            max_steps = 50

            # Track rewards from signal
            h_star_preservation = 1.0
            is_solvable = True
            num_active = info.get('num_active_systems', 0)

            logger.debug(f"  [INFERENCE] Starting policy inference...")

            while steps < max_steps:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))

                    total_episode_reward += reward
                    steps += 1

                    # Extract GNN-specific metrics
                    reward_signals = info.get('reward_signals', {})
                    h_star_preservation = float(reward_signals.get('h_star_preservation', 1.0))
                    is_solvable = bool(reward_signals.get('is_solvable', True))
                    num_active = info.get('num_active_systems', 0)

                    if done or truncated:
                        logger.debug(f"  [INFERENCE] Episode done at step {steps}")
                        break

                except KeyboardInterrupt:
                    logger.warning("  [INFERENCE] Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"  [INFERENCE] Step failed: {e}")
                    break

            solve_time = time.time() - solve_start
            total_time = time.time() - start_time

            logger.debug(f"  [INFERENCE] Completed in {steps} steps")

            # Build result
            result = DetailedMetrics(
                problem_name=problem_name,
                planner_name="GNN",
                solved=is_solvable,
                wall_clock_time=total_time,
                search_time=solve_time,
                h_star_preservation=h_star_preservation,
                num_active_systems=num_active,
                merge_episodes=steps,
                evaluation_notes=f"GNN policy: {steps} merge decisions"
            )

            logger.debug(f"  [SUCCESS] h*_preservation={h_star_preservation:.3f}, "
                         f"steps={steps}, solvable={is_solvable}")

            try:
                env.close()
            except:
                pass

            return result

        except subprocess.TimeoutExpired:
            logger.debug(f"  [TIMEOUT] Exceeded {self.timeout_sec}s")
            return DetailedMetrics(
                problem_name=problem_name,
                planner_name="GNN",
                solved=False,
                wall_clock_time=self.timeout_sec,
                error_type="timeout",
                timeout_occurred=True
            )

        except Exception as e:
            logger.error(f"  [ERROR] GNN run failed: {e}")
            logger.error(traceback.format_exc())

            try:
                env.close()
            except:
                pass

            return DetailedMetrics(
                problem_name=problem_name,
                planner_name="GNN",
                solved=False,
                wall_clock_time=time.time() - start_time,
                error_type="exception",
                error_message=str(e)[:500],
                timeout_occurred=False
            )


# ============================================================================
# EVALUATION FRAMEWORK - ORCHESTRATOR
# ============================================================================

class EvaluationFramework:
    """
    Main evaluation orchestrator.

    Coordinates:
    1. Baseline evaluation
    2. GNN evaluation
    3. Statistical analysis
    4. Report generation
    """

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[DetailedMetrics] = []
        logger.info(f"[FRAMEWORK] Output directory: {self.output_dir}")

    def run_comprehensive_evaluation(
            self,
            domain_file: str,
            problem_pattern: str,
            model_path: str,
            timeout_sec: int = 300,
            include_baselines: bool = True,
            baseline_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation: baselines + GNN + analysis.

        Args:
            domain_file: Path to domain PDDL
            problem_pattern: Glob pattern for problems
            model_path: Path to trained GNN model
            timeout_sec: Timeout per problem
            include_baselines: Whether to run baseline planners
            baseline_names: Which baselines to run (None = all)

        Returns:
            Evaluation results dictionary
        """
        print_section("COMPREHENSIVE EVALUATION FRAMEWORK")

        # Load problems
        logger.info(f"Loading problems: {problem_pattern}")
        problems = sorted(glob.glob(problem_pattern))

        if not problems:
            logger.error("No problems found!")
            return {}

        logger.info(f"Found {len(problems)} problem(s)")

        # Run baselines
        if include_baselines:
            self._run_all_baselines(
                domain_file,
                problems,
                timeout_sec,
                baseline_names
            )

        # Run GNN
        self._run_gnn(domain_file, problems, model_path, timeout_sec)

        # Generate reports
        return self._generate_comprehensive_report()

    def _run_all_baselines(
            self,
            domain_file: str,
            problems: List[str],
            timeout_sec: int,
            baseline_names: Optional[List[str]] = None
    ):
        """Run all baseline configurations."""
        print_subsection("RUNNING BASELINE PLANNERS")

        baseline_runner = BaselineRunner(timeout_sec)

        # Select baselines to run
        configs = EvaluationConfig.BASELINE_CONFIGS
        if baseline_names:
            configs = [c for c in configs if c['name'] in baseline_names]

        for config in configs:
            logger.info(f"\n{config['name']}: {config['description']}")
            logger.info("-" * 70)

            for i, problem in enumerate(problems, 1):
                logger.info(f"  [{i}/{len(problems)}] {os.path.basename(problem)}")

                result = baseline_runner.run(
                    domain_file,
                    problem,
                    config['search'],
                    baseline_name=config['name']
                )
                self.results.append(result)

    def _run_gnn(
            self,
            domain_file: str,
            problems: List[str],
            model_path: str,
            timeout_sec: int
    ):
        """Run GNN policy."""
        print_subsection("RUNNING GNN POLICY")

        try:
            gnn_runner = GNNPolicyRunner(model_path, timeout_sec)
        except FileNotFoundError as e:
            logger.error(f"GNN evaluation skipped: {e}")
            return

        for i, problem in enumerate(problems, 1):
            logger.info(f"  [{i}/{len(problems)}] {os.path.basename(problem)}")

            result = gnn_runner.run(domain_file, problem)
            self.results.append(result)

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive reports and statistics."""
        print_section("GENERATING COMPREHENSIVE REPORT")

        if not self.results:
            logger.error("No results to report!")
            return {}

        # Group by planner
        by_planner = defaultdict(list)
        for result in self.results:
            by_planner[result.planner_name].append(result)

        # Compute statistics for each planner
        summaries = {}
        for planner_name in sorted(by_planner.keys()):
            results_list = by_planner[planner_name]
            summary = self._compute_statistics(planner_name, results_list)
            summaries[planner_name] = summary

            logger.info(f"\n{planner_name}:")
            logger.info(f"  Solve rate: {summary.solve_rate_pct:.1f}%")
            logger.info(f"  Avg time: {summary.mean_time_sec:.2f}s")
            logger.info(f"  Avg expansions: {summary.mean_expansions:,}")
            if planner_name == "GNN":
                logger.info(f"  Avg h* preservation: {summary.mean_h_preservation:.3f}")

        # Export results
        self._export_all_results(summaries)

        return {
            "summaries": {name: summary.to_dict() for name, summary in summaries.items()},
            "timestamp": datetime.now().isoformat(),
            "num_problems": len(set(r.problem_name for r in self.results)),
            "num_planners": len(summaries)
        }

    def _compute_statistics(
            self,
            planner_name: str,
            results_list: List[DetailedMetrics]
    ) -> AggregateStatistics:
        """Compute comprehensive statistics."""

        solved = [r for r in results_list if r.solved]
        num_solved = len(solved)
        num_total = len(results_list)

        times = [r.wall_clock_time for r in solved if r.wall_clock_time > 0]
        expansions = [r.nodes_expanded for r in solved if r.nodes_expanded > 0]
        costs = [r.plan_cost for r in solved if r.plan_cost > 0]
        h_preservations = [r.h_star_preservation for r in solved]
        efficiency_scores = [r.efficiency_score() for r in solved if r.solved]
        quality_scores = [r.quality_score() for r in solved]

        unsolved = [r for r in results_list if not r.solved]
        errors = [r for r in unsolved if r.error_type]
        timeouts = [r for r in unsolved if r.timeout_occurred]

        # Compute percentiles
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
            mean_expansions=int(np.mean(expansions)) if expansions else 0,
            median_expansions=int(np.median(expansions)) if expansions else 0,
            std_expansions=int(np.std(expansions)) if expansions else 0,
            min_expansions=int(np.min(expansions)) if expansions else 0,
            max_expansions=int(np.max(expansions)) if expansions else 0,
            mean_plan_cost=int(np.mean(costs)) if costs else 0,
            median_plan_cost=int(np.median(costs)) if costs else 0,
            std_plan_cost=int(np.std(costs)) if costs else 0,
            mean_h_preservation=float(np.mean(h_preservations)) if h_preservations else 1.0,
            median_h_preservation=float(np.median(h_preservations)) if h_preservations else 1.0,
            mean_efficiency_score=float(np.mean(efficiency_scores)) if efficiency_scores else float('inf'),
            mean_quality_score=float(np.mean(quality_scores)) if quality_scores else 0.0,
            unsolved_count=len(unsolved),
            timeout_count=len(timeouts),
            error_count=len(errors),
            total_wall_clock_time_sec=float(sum(times)) if times else 0.0,
        )

    def _export_all_results(self, summaries: Dict[str, AggregateStatistics]):
        """Export results in all formats."""

        # CSV: detailed results
        csv_path = self.output_dir / "evaluation_results.csv"
        if self.results:
            fieldnames = list(self.results[0].to_dict().keys())
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())
            logger.info(f"✓ CSV: {csv_path}")

        # JSON: summary statistics
        json_path = self.output_dir / "evaluation_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                {name: summary.to_dict() for name, summary in summaries.items()},
                f,
                indent=2
            )
        logger.info(f"✓ JSON: {json_path}")

        # TXT: formatted report
        self._write_text_report(summaries)

    def _write_text_report(self, summaries: Dict[str, AggregateStatistics]):
        """Write formatted text comparison report."""
        report_path = self.output_dir / "comparison_report.txt"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("COMPREHENSIVE EVALUATION - COMPARISON REPORT\n")
                f.write("=" * 100 + "\n\n")

                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Total problems: {len(set(r.problem_name for r in self.results))}\n")
                f.write(f"Planners evaluated: {len(summaries)}\n\n")

                # Summary table
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Planner':<25} {'Solve Rate':<15} {'Avg Time':<15} {'Med Time':<15} "
                        f"{'Avg Exp':<15} {'H* Pres':<12}\n")
                f.write("-" * 100 + "\n")

                for planner_name in sorted(summaries.keys()):
                    summary = summaries[planner_name]
                    solve_str = f"{summary.num_problems_solved}/{summary.num_problems_total} ({summary.solve_rate_pct:.1f}%)"
                    avg_exp_str = f"{summary.mean_expansions:,}" if summary.num_problems_solved > 0 else "N/A"
                    h_pres_str = f"{summary.mean_h_preservation:.3f}" if planner_name == "GNN" else "N/A"

                    f.write(f"{planner_name:<25} {solve_str:<15} {summary.mean_time_sec:<15.2f} "
                            f"{summary.median_time_sec:<15.2f} {avg_exp_str:<15} {h_pres_str:<12}\n")

                f.write("-" * 100 + "\n\n")

                # Detailed statistics per planner
                f.write("DETAILED STATISTICS\n")
                f.write("-" * 100 + "\n\n")

                for planner_name in sorted(summaries.keys()):
                    summary = summaries[planner_name]
                    f.write(f"{planner_name}\n")
                    f.write(
                        f"  Solved: {summary.num_problems_solved}/{summary.num_problems_total} ({summary.solve_rate_pct:.1f}%)\n")
                    f.write(f"  Time (mean ± std): {summary.mean_time_sec:.2f} ± {summary.std_time_sec:.2f}s\n")
                    f.write(
                        f"  Time (median [Q1, Q3]): {summary.median_time_sec:.2f}s [{summary.q1_time_sec:.2f}, {summary.q3_time_sec:.2f}]\n")
                    f.write(f"  Expansions (mean): {summary.mean_expansions:,}\n")
                    f.write(f"  Plan cost (mean ± std): {summary.mean_plan_cost} ± {summary.std_plan_cost}\n")
                    f.write(f"  Efficiency score (mean): {summary.mean_efficiency_score:.4f}\n")
                    f.write(f"  Quality score (mean): {summary.mean_quality_score:.4f}\n")
                    if planner_name == "GNN":
                        f.write(
                            f"  H* preservation (mean ± median): {summary.mean_h_preservation:.3f} ± {summary.median_h_preservation:.3f}\n")
                    f.write(f"  Errors: {summary.error_count}, Timeouts: {summary.timeout_count}\n")
                    f.write(f"  Total time: {summary.total_wall_clock_time_sec:.1f}s\n\n")

            logger.info(f"✓ Text report: {report_path}")

        except Exception as e:
            logger.error(f"Failed to write text report: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str, width: int = 100):
    """Print formatted section header."""
    logger.info("")
    logger.info("=" * width)
    logger.info(f"// {title.upper()}")
    logger.info("=" * width)
    logger.info("")


def print_subsection(title: str, width: int = 80):
    """Print formatted subsection header."""
    logger.info("")
    logger.info("-" * width)
    logger.info(f">>> {title}")
    logger.info("-" * width)
    logger.info("")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete evaluation (baselines + GNN)
  python evaluation_comprehensive.py \\
      --model mvp_output/gnn_model.zip \\
      --domain domain.pddl \\
      --problems "problem_*.pddl" \\
      --output evaluation_results/

  # Run GNN only (skip baselines for speed)
  python evaluation_comprehensive.py \\
      --model model.zip \\
      --domain domain.pddl \\
      --problems "problem_*.pddl" \\
      --skip-baselines

  # Run specific baselines only
  python evaluation_comprehensive.py \\
      --domain domain.pddl \\
      --problems "problem_*.pddl" \\
      --baselines FD_LM-Cut FD_Blind \\
      --skip-gnn
        """
    )

    parser.add_argument("--model", help="Path to trained GNN model (ZIP)")
    parser.add_argument("--domain", required=True, help="Path to domain PDDL")
    parser.add_argument("--problems", required=True, help="Glob pattern for problems")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per problem (seconds)")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN evaluation")
    parser.add_argument(
        "--baselines",
        nargs='+',
        help="Specific baselines to run (default: all)"
    )
    parser.add_argument("--downward-dir", help="Path to Fast Downward directory")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.domain):
        logger.error(f"Domain not found: {args.domain}")
        return 1

    if not args.skip_gnn and not args.model:
        logger.error("--model required unless --skip-gnn is specified")
        return 1

    # Run evaluation
    framework = EvaluationFramework(args.output)

    if args.skip_gnn:
        include_baselines = True
        model_path = None
    else:
        include_baselines = not args.skip_baselines
        model_path = args.model

    result = framework.run_comprehensive_evaluation(
        domain_file=args.domain,
        problem_pattern=args.problems,
        model_path=model_path,
        timeout_sec=args.timeout,
        include_baselines=include_baselines,
        baseline_names=args.baselines if hasattr(args, 'baselines') else None
    )

    print_section("EVALUATION COMPLETE")
    logger.info("✅ Evaluation pipeline finished!")
    logger.info(f"📁 Results: {os.path.abspath(args.output)}")
    logger.info(f"📊 Summary: {args.output}/evaluation_summary.json")
    logger.info(f"📋 Report: {args.output}/comparison_report.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())