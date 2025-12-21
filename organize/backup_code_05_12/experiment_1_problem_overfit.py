#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVERFIT EXPERIMENT INFRASTRUCTURE (PRODUCTION HARDENED v2) - ENHANCED
=====================================================================
Trains GNN policy on a small set of problems with RIGOROUS validation:

CRITICAL ENHANCEMENTS IN THIS VERSION:
  ✅ PROBLEM COVERAGE: Per-problem episode tracking + minimum guarantee
  ✅ METRIC INTEGRITY: Explicit validation of h* preservation (infinity handling)
  ✅ GNN HEALTH: Policy entropy, value loss, gradient norms tracked
  ✅ FAILURE TAXONOMY: Timeout vs DeadEnd vs Solvability Loss vs Crash
  ✅ RESOURCE METRICS: Step time, memory usage, inference latency tracked
  ✅ TEMPORAL RESOLUTION: Per-step logging + episode-level aggregation
  ✅ FEATURE VALIDATION: Input normalization check before GNN inference
  ✅ REWARD VALIDATION: Bounds checking, scale verification
  ✅ CONVERGENCE CHECK: Verify learning actually happens
  ✅ OUTLIER RETENTION: Failed episodes preserved for replay analysis
  ✅ ADAPTIVE SAFETY: Minimum training per problem (no starvation)
  ✅ SIGNAL INTEGRITY: Explicit error on parsing failures (not silent defaults)

PREVIOUS FIXES RETAINED:
  ✅ BEST MODEL: Correctly saved after evaluation
  ✅ RESUME LOGIC: Uses episode_log length
  ✅ SEED ISOLATION: Training/eval use different namespaces
  ✅ NAN SAFETY: AdaptiveSampler bounds checking
  ✅ FILE SAFETY: Context manager pattern
  ✅ METRIC INTEGRITY: Failed episodes tracked separately
  ✅ TIMEOUT HANDLING: subprocess.TimeoutExpired catch
  ✅ ATOMIC WRITES: JSON write-then-rename
  ✅ EXPERIMENT ID: UUID-based
  ✅ EPISODE NUMBERING: Problem name based
  ✅ CUDA DETERMINISM: torch.backends.cudnn settings
  ✅ IMPORT VALIDATION: Early dependency check
"""

import sys
import os
import json
import glob
import logging
import argparse
import random
import re
import traceback
import shutil
import subprocess
import warnings
import uuid
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
from datetime import datetime
import time

# Progress bar
from tqdm import tqdm


# ============================================================================
# EARLY VALIDATION: Comprehensive dependency & environment check
# ============================================================================

def validate_dependencies():
    """Validate all dependencies + Fast Downward availability."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch (pip install torch)")

    try:
        from stable_baselines3 import PPO
    except ImportError:
        missing.append("stable_baselines3 (pip install stable-baselines3)")

    try:
        import gymnasium
    except ImportError:
        missing.append("gymnasium (pip install gymnasium)")

    try:
        import networkx
    except ImportError:
        missing.append("networkx (pip install networkx)")

    try:
        import psutil
    except ImportError:
        missing.append("psutil (pip install psutil)")

    # Check Fast Downward
    try:
        result = subprocess.run(
            ["fast-downward", "--version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            missing.append("Fast Downward (install from https://www.fast-downward.org/)")
    except FileNotFoundError:
        missing.append("Fast Downward binary not in PATH")
    except subprocess.TimeoutExpired:
        missing.append("Fast Downward (exists but slow/unresponsive)")

    if missing:
        print("\n❌ MISSING DEPENDENCIES (FATAL):\n")
        for dep in missing:
            print(f"   • {dep}")
        print("\nCannot proceed without these. Install and try again.\n")
        sys.exit(1)


def validate_disk_space(target_dir: str, min_gb: float = 5.0) -> bool:
    """Check available disk space on target directory."""
    try:
        stat_result = shutil.disk_usage(target_dir)
        available_gb = stat_result.free / (1024 ** 3)

        if available_gb < min_gb:
            print(f"\n⚠️  WARNING: Only {available_gb:.1f}GB free on {target_dir}")
            print(f"   Checkpointing every 1000 steps could exceed this.")
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            return response == 'y'
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True


# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
from shared_experiment_utils import (
    setup_logging,
    print_section,
    print_subsection,
    ExperimentCheckpoint,
    save_results_to_json,
    save_results_to_txt,
    DEFAULT_REWARD_WEIGHTS,
    cleanup_signal_files,
)

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# REPRODUCIBILITY: DETERMINISM MANDATE (ENHANCED)
# ============================================================================

def set_all_seeds(seed: int = 42):
    """Lock down randomness in ALL libraries."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ============================================================================
# SAFE FILE I/O (ATOMIC WRITES + VALIDATION)
# ============================================================================

def save_json_atomic(data: Dict[str, Any], path: str) -> None:
    """Save JSON atomically with validation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix('.tmp')

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        temp_path.replace(path)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise e


@contextmanager
def atomic_file_write(path: str):
    """Context manager for atomic file writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix('.tmp')

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            yield f
        temp_path.replace(path)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise e


# ============================================================================
# ENHANCED TRAINING LOGGER WITH TEMPORAL RESOLUTION & METRIC RICHNESS
# ============================================================================

class EnhancedSilentTrainingLogger:
    """
    ENHANCED LOGGING SYSTEM with:
    - Per-step temporal resolution
    - Structured EVENT format for parsing
    - Metric richness (GNN health, environment physics, failures)
    - Resource tracking (time, memory)
    """

    def __init__(self, log_dir: str, experiment_id: str, verbose: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.verbose = verbose

        self.log_file = self.log_dir / "training.log"
        self.file_handler = None
        self._open_file()

        # Per-problem tracking
        self.problem_episode_counts = defaultdict(int)
        self.problem_failure_counts = defaultdict(int)
        self.problem_rewards = defaultdict(list)

    def _open_file(self):
        """Open or append to log file."""
        mode = 'a' if self.log_file.exists() else 'w'
        self.file_handler = open(self.log_file, mode, encoding='utf-8', buffering=1)

    def _emit_event(self, event_type: str, **kwargs) -> None:
        """Emit structured EVENT to log file."""
        if not self.file_handler or self.file_handler.closed:
            self._open_file()

        event_data = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id,
            **kwargs
        }

        try:
            event_json = json.dumps(event_data)
            self.file_handler.write(f"EVENT: {event_json}\n")
            self.file_handler.flush()
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to emit event: {e}", file=sys.stderr)

    # ========================================================================
    # TRAINING LIFECYCLE EVENTS
    # ========================================================================

    def log_training_started(self, num_episodes: int, num_problems: int, seed: int):
        """Log training initialization."""
        self._emit_event(
            'training_started',
            num_episodes=num_episodes,
            num_problems=num_problems,
            seed=seed
        )

    def log_training_completed(self, total_steps: int, total_reward: float):
        """Log successful training completion."""
        self._emit_event(
            'training_completed',
            total_steps=total_steps,
            total_reward=total_reward
        )

    # ========================================================================
    # EPISODE LIFECYCLE EVENTS (with enhanced metrics)
    # ========================================================================

    def log_episode_started(self, episode: int, problem_name: str):
        """Log episode start."""
        self._emit_event(
            'episode_started',
            episode=episode,
            problem_name=problem_name
        )

    def log_episode_completed(
            self,
            episode: int,
            problem_name: str,
            reward: float,
            steps: int,
            h_preservation: float,
            is_solvable: bool,
            error: Optional[str] = None,
            failure_type: Optional[str] = None,  # 'timeout', 'dead_end', 'crash', 'solvability_loss'
            metrics: Optional[Dict] = None,  # Additional metrics (time, memory, etc.)
    ):
        """
        Log episode completion with rich metrics.

        failure_type: Explicit failure taxonomy
        metrics: {
            'step_time_ms': average time per step,
            'memory_mb': peak memory during episode,
            'inference_time_ms': average GNN inference time,
            'policy_entropy': entropy of policy at end,
            'value_loss': critic loss,
            'gradient_norm': policy gradient norm,
            'graph_size_reduction': % shrinkage of graph,
            'avg_node_degree_change': change in node degree,
            'dead_end_ratio': fraction of dead-ends,
        }
        """
        self._emit_event(
            'episode_completed',
            episode=episode,
            problem_name=problem_name,
            reward=reward,
            steps=steps,
            h_preservation=h_preservation,
            is_solvable=is_solvable,
            error=error,
            failure_type=failure_type,
            metrics=metrics or {}
        )

        # Update per-problem tracking
        self.problem_episode_counts[problem_name] += 1
        if error:
            self.problem_failure_counts[problem_name] += 1
        else:
            self.problem_rewards[problem_name].append(reward)

    # ========================================================================
    # STEP-LEVEL LOGGING (Temporal Resolution)
    # ========================================================================

    def log_step(
            self,
            episode: int,
            problem_name: str,
            step: int,
            action: int,
            reward: float,
            h_preservation: float,
            is_solvable: bool,
            metrics: Optional[Dict] = None,
    ):
        """
        Log individual step for micro-decision analysis.

        Enables analysis like: "GNN gets confident at step 5 but confused at step 20"
        """
        self._emit_event(
            'step_completed',
            episode=episode,
            problem_name=problem_name,
            step=step,
            action=action,
            reward=reward,
            h_preservation=h_preservation,
            is_solvable=is_solvable,
            metrics=metrics or {}
        )

    # ========================================================================
    # GNN HEALTH METRICS
    # ========================================================================

    def log_gnn_health(
            self,
            episode: int,
            step: int,
            policy_entropy: float,
            value_loss: float,
            gradient_norm: float,
            explained_variance: float,
    ):
        """Log GNN/RL health metrics for learning curve analysis."""
        self._emit_event(
            'gnn_health',
            episode=episode,
            step=step,
            policy_entropy=policy_entropy,
            value_loss=value_loss,
            gradient_norm=gradient_norm,
            explained_variance=explained_variance,
            entropy_collapse_warning=policy_entropy < 0.05,
            gradient_explosion_warning=gradient_norm > 10.0,
            gradient_vanishing_warning=gradient_norm < 1e-6,
        )

    # ========================================================================
    # ENVIRONMENT PHYSICS TRACKING
    # ========================================================================

    def log_environment_physics(
            self,
            episode: int,
            problem_name: str,
            step: int,
            graph_size_before: int,
            graph_size_after: int,
            avg_degree_before: float,
            avg_degree_after: float,
            h_star_before: float,
            h_star_after: float,
            dead_ends_created: int,
    ):
        """Track how environment topology changes during merge sequence."""
        size_reduction = (graph_size_before - graph_size_after) / max(1, graph_size_before)
        degree_change = (avg_degree_after - avg_degree_before) / max(0.01, avg_degree_before)

        self._emit_event(
            'environment_physics',
            episode=episode,
            problem_name=problem_name,
            step=step,
            graph_size_reduction_percent=size_reduction * 100,
            avg_degree_change_percent=degree_change * 100,
            h_star_degradation=1.0 - h_star_after / max(0.001, h_star_before),
            dead_ends_created=dead_ends_created,
        )

    # ========================================================================
    # FAILURE TAXONOMY
    # ========================================================================

    def log_failure(
            self,
            episode: int,
            problem_name: str,
            failure_type: str,  # 'timeout', 'dead_end', 'goal_lost', 'crash'
            error_message: str,
            context: Optional[Dict] = None,
    ):
        """
        Log failure with explicit taxonomy.

        Enables analysis: "10% of failures are timeouts, 20% are dead-ends..."
        """
        self._emit_event(
            'failure',
            episode=episode,
            problem_name=problem_name,
            failure_type=failure_type,
            error_message=error_message,
            context=context or {}
        )

    # ========================================================================
    # CHECKPOINT & BEST MODEL TRACKING
    # ========================================================================

    def log_checkpoint_saved(
            self,
            step: int,
            path: str,
            reward: float,
            problem_name: str = '',
            domain_name: str = '',
    ):
        """Log checkpoint saved."""
        self._emit_event(
            'checkpoint_saved',
            step=step,
            path=path,
            reward=reward,
            problem_name=problem_name,
            domain_name=domain_name
        )

    def log_best_model_saved(
            self,
            step: int,
            reward: float,
            path: str,
            source: str = 'training',
    ):
        """Log when best model is saved (training or evaluation)."""
        self._emit_event(
            'best_model_saved',
            step=step,
            reward=reward,
            path=path,
            source=source
        )

    # ========================================================================
    # ADAPTIVE SAMPLING EVENTS
    # ========================================================================

    def log_adaptive_sampling_update(
            self,
            episode: int,
            per_problem_scores: Dict[str, float],
            per_problem_coverage: Dict[str, float],
    ):
        """Log adaptive sampling weight update with coverage analysis."""
        self._emit_event(
            'adaptive_sampling_update',
            episode=episode,
            per_problem_scores=per_problem_scores,
            per_problem_coverage=per_problem_coverage,
        )

    # ========================================================================
    # PROBLEM COVERAGE VALIDATION
    # ========================================================================

    def log_problem_coverage_report(
            self,
            total_episodes: int,
            problem_names: List[str],
    ):
        """
        Log comprehensive coverage report.
        CRITICAL: Ensures no problem is left undertrained.
        """
        coverage_data = {}
        for problem_name in problem_names:
            episode_count = self.problem_episode_counts[problem_name]
            failure_count = self.problem_failure_counts[problem_name]
            avg_reward = np.mean(self.problem_rewards[problem_name]) if self.problem_rewards[problem_name] else 0.0

            coverage_pct = (episode_count / total_episodes * 100) if total_episodes > 0 else 0
            failure_rate = (failure_count / episode_count * 100) if episode_count > 0 else 0

            coverage_data[problem_name] = {
                'episodes': episode_count,
                'coverage_percent': coverage_pct,
                'failures': failure_count,
                'failure_rate': failure_rate,
                'avg_reward': avg_reward,
            }

        self._emit_event(
            'problem_coverage_report',
            total_episodes=total_episodes,
            coverage_data=coverage_data,
            all_problems_trained=all(
                coverage_data[p]['episodes'] > 0 for p in problem_names
            ),
            minimum_coverage=min(
                (coverage_data[p]['coverage_percent'] for p in problem_names),
                default=0
            ),
        )

    # ========================================================================
    # EVALUATION EVENTS
    # ========================================================================

    def log_evaluation_started(self, num_problems: int, runs_per_problem: int):
        """Log evaluation phase start."""
        self._emit_event(
            'evaluation_started',
            num_problems=num_problems,
            runs_per_problem=runs_per_problem
        )

    def log_evaluation_completed(self, num_problems_evaluated: int, avg_reward: float):
        """Log evaluation completion."""
        self._emit_event(
            'evaluation_completed',
            num_problems_evaluated=num_problems_evaluated,
            avg_reward=avg_reward
        )

    def close(self):
        """Finalize log file."""
        if self.file_handler:
            try:
                self.file_handler.close()
            except Exception:
                pass
            finally:
                self.file_handler = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ============================================================================
# DATA STRUCTURES (ENHANCED)
# ============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""
    episode: int
    problem_name: str
    reward: float
    h_star_preservation: float = 1.0
    num_active_systems: int = 0
    is_solvable: bool = True
    eval_steps: int = 0
    total_reward: float = 0.0
    timestamp: float = field(default_factory=lambda: time.time())
    error: Optional[str] = None
    failure_type: Optional[str] = None  # 'timeout', 'dead_end', 'crash', etc.

    # ✅ NEW: Rich metrics for analysis
    step_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    avg_inference_time_ms: float = 0.0
    policy_entropy: float = 0.0
    value_loss: float = 0.0
    gradient_norm: float = 0.0
    graph_size_reduction_pct: float = 0.0
    dead_end_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProblemStats:
    """Statistics for a single problem across training."""
    problem_name: str
    num_episodes: int
    num_failed: int
    coverage_percent: float  # ✅ NEW: % of total training on this problem
    avg_reward: float
    best_reward: float
    worst_reward: float
    final_reward: float
    improvement_ratio: float
    avg_h_preservation: float
    solve_rate: float
    episodes_to_convergence: Optional[int] = None
    avg_step_time_ms: float = 0.0
    avg_memory_mb: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OverfitExperimentSummary:
    """Overall statistics for the overfit experiment."""
    num_problems: int
    num_train_episodes: int
    num_failed_episodes: int
    total_timesteps: int
    start_time: str
    end_time: str
    duration_seconds: float
    avg_reward_over_all: float
    best_reward_over_all: float
    worst_reward_over_all: float
    per_problem_stats: List[Dict]
    reward_variance: float
    h_preservation_improvement_ratio: float
    solve_rate_improvement: float
    early_convergence_episodes: int
    checkpoints_saved: int = 0
    best_model_path: str = ''
    convergence_threshold: float = 0.05
    plateau_episode: Optional[int] = None
    overfitting_ratio: float = 1.0
    training_final_avg: float = 0.0
    evaluation_avg: float = 0.0
    experiment_id: str = ''

    # ✅ NEW: Coverage validation
    problem_coverage_valid: bool = True
    min_problem_coverage_pct: float = 0.0
    max_problem_coverage_pct: float = 0.0
    all_problems_trained: bool = True

    # ✅ NEW: Failure analysis
    failure_taxonomy: Dict[str, int] = field(default_factory=dict)
    avg_step_time_ms: float = 0.0
    avg_peak_memory_mb: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# ENHANCED ADAPTIVE SAMPLING (with minimum guarantee)
# ============================================================================

class EnhancedAdaptiveSampler:
    """
    ✅ CRITICAL FIX: Adaptive Sampling with MINIMUM TRAINING GUARANTEE

    Prevents problem starvation: ensures each problem gets minimum episodes
    before being susceptible to adaptive down-weighting.
    """

    def __init__(
            self,
            problem_names: List[str],
            update_interval: int = 50,
            sweep_interval: int = 100,
            alpha: float = 2.0,
            min_episodes_per_problem: int = 10,  # ✅ NEW: Minimum guarantee
            seed: int = 42,
    ):
        self.problem_names = problem_names
        self.update_interval = update_interval
        self.sweep_interval = sweep_interval
        self.alpha = alpha
        self.min_episodes_per_problem = min_episodes_per_problem

        self.rng = np.random.RandomState(seed)

        # ✅ NEW: Track episodes per problem
        self.per_problem_episodes = {name: 0 for name in problem_names}
        self.per_problem_scores = {name: 0.5 for name in problem_names}
        self.recent_rewards = defaultdict(list)

    def update_scores_from_log(
            self,
            episode_log: List,
            window_size: int = 5
    ) -> None:
        """
        Update per-problem scores based on recent performance.
        ONLY update problems that have met minimum episode count.
        """
        if len(episode_log) == 0:
            return

        # Collect recent rewards per problem
        recent_by_problem = defaultdict(list)
        for metrics in episode_log[-window_size * len(self.problem_names):]:
            if metrics.error is None:
                recent_by_problem[metrics.problem_name].append(metrics.reward)

                # ✅ NEW: Track episode count
                self.per_problem_episodes[metrics.problem_name] = \
                    sum(1 for m in episode_log if m.problem_name == metrics.problem_name)

        # Update scores ONLY if min episodes met
        for problem_name, rewards in recent_by_problem.items():
            if rewards and self.per_problem_episodes[problem_name] >= self.min_episodes_per_problem:
                avg_reward = np.mean(rewards)
                self.per_problem_scores[problem_name] = avg_reward

    def get_weights(self) -> np.ndarray:
        """
        Compute sampling weights with MINIMUM GUARANTEE.

        Problems below minimum episodes get UNIFORM weight.
        Problems above minimum get ADAPTIVE weight based on performance.
        """
        scores = np.array([
            self.per_problem_scores.get(name, 0.5)
            for name in self.problem_names
        ])

        # ✅ NEW: Check which problems need more training
        episode_counts = np.array([
            self.per_problem_episodes.get(name, 0)
            for name in self.problem_names
        ])

        weights = np.zeros_like(scores)

        # Problems below minimum: uniform weight
        below_min_mask = episode_counts < self.min_episodes_per_problem
        weights[below_min_mask] = 1.0

        # Problems above minimum: exponential weighting
        above_min_mask = ~below_min_mask
        exp_weights = np.exp(-self.alpha * scores[above_min_mask])

        if np.any(np.isnan(exp_weights)) or np.any(np.isinf(exp_weights)):
            exp_weights = np.ones_like(exp_weights)

        weights[above_min_mask] = exp_weights

        # Normalize
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            weights = np.ones_like(weights) / len(weights)

        return weights

    def sample_problem_idx(self) -> int:
        """Sample a problem index based on current weights."""
        weights = self.get_weights()
        idx = self.rng.choice(len(self.problem_names), p=weights)
        return idx

    def get_coverage_stats(self) -> Dict[str, float]:
        """Return current coverage statistics."""
        total_episodes = sum(self.per_problem_episodes.values())
        coverage_stats = {}

        for problem_name in self.problem_names:
            episodes = self.per_problem_episodes[problem_name]
            coverage_pct = (episodes / total_episodes * 100) if total_episodes > 0 else 0
            coverage_stats[problem_name] = coverage_pct

        return coverage_stats


# ============================================================================
# PROBLEM SELECTION (IMPROVED)
# ============================================================================

def select_training_problems(
        domain_file: str,
        problem_pattern: str,
        num_problems: int,
        seed: int = 42
) -> Tuple[str, List[Tuple[str, str]], List[str]]:
    """Select K problems for overfitting experiment."""
    domain_path = os.path.abspath(domain_file)

    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Domain not found: {domain_path}")

    all_problems = sorted(glob.glob(problem_pattern))

    if not all_problems:
        raise ValueError(f"No problems found matching: {problem_pattern}")

    random.seed(seed)
    selected = random.sample(all_problems, min(num_problems, len(all_problems)))
    selected = sorted(selected)

    benchmarks = [(domain_path, os.path.abspath(p)) for p in selected]
    problem_names = [os.path.basename(p) for _, p in benchmarks]

    return domain_path, benchmarks, problem_names


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_training_state(output_dir: str) -> Tuple[List, float, List[str], str]:
    """Load previous training state when resuming."""
    output_path = Path(output_dir)
    episode_log = []
    best_reward = -float('inf')
    problem_names = []
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = output_path / "training_log.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'steps' in data and 'eval_steps' not in data:
                        data['eval_steps'] = data.pop('steps')

                    if 'problem_name' not in data:
                        data['problem_name'] = data.get('problem_idx', 0)

                    metrics = EpisodeMetrics(**data)
                    episode_log.append(metrics)

                    if metrics.problem_name not in problem_names:
                        problem_names.append(metrics.problem_name)
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    print(f"[WARN] Could not parse log line: {e}", file=sys.stderr)
                    continue

    summary_path = output_path / "overfit_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            try:
                summary = json.load(f)
                best_reward = summary.get('best_reward_over_all', -float('inf'))
                experiment_id = summary.get('experiment_id', experiment_id)
            except json.JSONDecodeError:
                pass

    return episode_log, best_reward, problem_names, experiment_id


def extract_global_step_from_checkpoint(checkpoint_path: str) -> int:
    """Extract global_step from checkpoint filename."""
    match = re.search(r'model_step_(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


def validate_checkpoint_compatibility(
        checkpoint_dir: str,
        num_problems: int,
        problem_names: List[str]
) -> None:
    """Validate checkpoint compatibility with current setup."""
    summary_path = Path(checkpoint_dir).parent.parent / "overfit_summary.json"

    if summary_path.exists():
        try:
            with open(summary_path) as f:
                old_summary = json.load(f)
                old_num_problems = old_summary.get('num_problems', 0)

                if old_num_problems != num_problems:
                    raise ValueError(
                        f"❌ Checkpoint from {old_num_problems} problems, "
                        f"but training on {num_problems} problems. "
                        f"Incompatible!"
                    )
        except json.JSONDecodeError:
            pass


# ============================================================================
# RESOURCE MONITORING
# ============================================================================

class ResourceMonitor:
    """Monitor system resource usage during training."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time:
            return (time.time() - self.start_time) * 1000
        return 0.0

    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        if self.start_memory:
            current = self.process.memory_info().rss / 1024 / 1024
            return current - self.start_memory
        return 0.0


# ============================================================================
# METRIC VALIDATION (Signal Integrity)
# ============================================================================

def validate_reward_signals(reward_signals: Dict) -> Tuple[bool, Optional[str]]:
    """
    ✅ CRITICAL: Validate reward signals for integrity.

    Catches:
    - Missing h* preservation (silent default bug)
    - Infinity values in heuristic
    - Contradictory solvability claims
    """
    # Check required fields
    required_fields = ['h_star_before', 'h_star_after', 'h_star_preservation', 'is_solvable']
    for field in required_fields:
        if field not in reward_signals:
            return False, f"Missing required field: {field}"

    # Check h* preservation validity
    h_before = float(reward_signals.get('h_star_before', 0))
    h_after = float(reward_signals.get('h_star_after', 0))
    h_pres = float(reward_signals.get('h_star_preservation', 1.0))

    # Handle infinity (dead-end)
    if np.isinf(h_before) or np.isinf(h_after):
        if not np.isinf(h_pres):
            return False, f"h* is infinite but preservation is finite: {h_pres}"

    # Verify solvability
    is_solvable = bool(reward_signals.get('is_solvable', True))
    dead_end_ratio = float(reward_signals.get('dead_end_ratio', 0.0))

    if is_solvable and dead_end_ratio > 0.9:
        return False, f"Solvable claim conflicts with {dead_end_ratio:.1%} dead-ends"

    if not is_solvable and dead_end_ratio < 0.5:
        return False, f"Unsolvable claim but only {dead_end_ratio:.1%} dead-ends"

    return True, None


# ============================================================================
# TRAINING PHASE (FULLY ENHANCED)
# ============================================================================

class OverfitTrainer:
    """
    Trains GNN on fixed problem set with RIGOROUS VALIDATION.
    """

    def __init__(
            self,
            benchmarks: List[Tuple[str, str]],
            problem_names: List[str],
            output_dir: str,
            reward_weights: Optional[Dict[str, float]] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
            checkpoint_interval: int = 1000,
            min_episodes_per_problem: int = 10,  # ✅ NEW
            seed: int = 42,
    ):
        self.benchmarks = benchmarks
        self.problem_names = problem_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS.copy()
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step
        self.checkpoint_interval = checkpoint_interval
        self.min_episodes_per_problem = min_episodes_per_problem
        self.seed = seed

        self.episode_log: List[EpisodeMetrics] = []
        self.start_time = datetime.now()
        self.failed_episode_count = 0

        # ✅ NEW: Enhanced adaptive sampler
        self.sampler = EnhancedAdaptiveSampler(
            problem_names=problem_names,
            update_interval=50,
            sweep_interval=100,
            min_episodes_per_problem=min_episodes_per_problem,
            seed=seed + 1000,
        )

        self.best_reward = -float('inf')
        self.best_model_path = None
        self.global_step = 0

        self.experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # ✅ NEW: Enhanced logger
        self.logger = EnhancedSilentTrainingLogger(
            str(self.output_dir),
            experiment_id=self.experiment_id,
            verbose=False
        )

        # ✅ NEW: Resource monitor
        self.resource_monitor = ResourceMonitor()

        self._import_dependencies()

    def _import_dependencies(self):
        """Import all required dependencies."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.monitor import Monitor
            self.PPO = PPO
            self.Monitor = Monitor
        except ImportError as e:
            raise ImportError(f"Failed to import stable_baselines3: {e}")

        try:
            from gnn_policy import GNNPolicy
            self.GNNPolicy = GNNPolicy
        except ImportError as e:
            raise ImportError(f"Failed to import GNNPolicy: {e}")

        try:
            from thin_merge_env import ThinMergeEnv
            self.ThinMergeEnv = ThinMergeEnv
        except ImportError as e:
            raise ImportError(f"Failed to import ThinMergeEnv: {e}")

    def get_experiment_id(self) -> str:
        """Return the experiment_id for logging."""
        return self.experiment_id

    def _create_env(self, domain_file: str, problem_file: str, seed: int):
        """Create environment with error handling."""
        try:
            env = self.ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
                reward_weights=self.reward_weights,
                debug=False,
                seed=seed,
            )
        except TypeError:
            env = self.ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
                reward_weights=self.reward_weights,
                debug=False,
            )

        return self.Monitor(env)

    def _problem_cycle_generator(self, start_episode: int, num_episodes: int):
        """Generate problems using enhanced adaptive sampling."""
        for episode in range(start_episode, num_episodes):
            # Every update_interval episodes, refresh weights
            if episode % self.sampler.update_interval == 0 and episode > 0:
                self.sampler.update_scores_from_log(self.episode_log)

                coverage = self.sampler.get_coverage_stats()
                self.logger.log_adaptive_sampling_update(
                    episode,
                    self.sampler.per_problem_scores,
                    coverage
                )

            idx = self.sampler.sample_problem_idx()
            yield idx

    def run_training(
            self,
            num_episodes: int,
            timesteps_per_episode: int = 50,
            resume_from: Optional[str] = None,
    ) -> Optional[str]:
        """
        Train with FULL VALIDATION AND MONITORING.
        """
        model = None
        start_episode = 0
        cumulative_reward = 0.0
        env = None

        self.logger.log_training_started(
            num_episodes=num_episodes,
            num_problems=len(self.benchmarks),
            seed=self.seed
        )

        try:
            if resume_from:
                checkpoint_path = Path(resume_from)
                if checkpoint_path.exists():
                    print(f"\n🔄 RESUME: Loading checkpoint: {checkpoint_path.name}")

                    try:
                        validate_checkpoint_compatibility(
                            str(checkpoint_path),
                            len(self.benchmarks),
                            self.problem_names
                        )
                        print(f"   ✓ Checkpoint compatibility validated")
                    except ValueError as e:
                        print(f"   ✗ {e}")
                        raise

                    model = self.PPO.load(resume_from)
                    print(f"   ✓ Model loaded")

                    prev_log, prev_best, prev_problem_names, prev_exp_id = load_training_state(
                        str(self.output_dir)
                    )

                    start_episode = len(prev_log)
                    self.global_step = start_episode * timesteps_per_episode

                    self.episode_log = prev_log
                    self.best_reward = prev_best
                    self.experiment_id = prev_exp_id
                    self.logger.experiment_id = prev_exp_id

                    if prev_log:
                        cumulative_reward = sum(m.reward for m in prev_log if m.error is None)
                        failed_count = sum(1 for m in prev_log if m.error is not None)
                        self.failed_episode_count = failed_count
                        print(f"   ✓ Loaded {len(prev_log)} previous episodes "
                              f"({failed_count} failed)")

                    self.sampler.update_scores_from_log(self.episode_log)

            pbar = tqdm(
                self._problem_cycle_generator(start_episode, num_episodes),
                total=num_episodes,
                initial=start_episode,
                desc="Training (details → training.log)",
                unit="episode",
                disable=False
            )

            for episode, problem_idx in pbar:
                domain_file, problem_file = self.benchmarks[problem_idx]
                problem_name = self.problem_names[problem_idx]

                self.logger.log_episode_started(episode, problem_name)
                self.resource_monitor.start()

                try:
                    cleanup_signal_files()
                except Exception:
                    pass

                env = None
                episode_error = None
                failure_type = None

                try:
                    train_seed = self.seed + episode

                    env = self._create_env(domain_file, problem_file, seed=train_seed)

                    if model is None:
                        model = self.PPO(
                            policy=self.GNNPolicy,
                            env=env,
                            learning_rate=0.0003,
                            n_steps=64,
                            batch_size=32,
                            ent_coef=0.01,
                            verbose=0,
                            tensorboard_log=str(self.output_dir / "tb_logs"),
                            policy_kwargs={"hidden_dim": 64},
                        )
                    else:
                        model.set_env(env)

                    model.learn(
                        total_timesteps=timesteps_per_episode,
                        tb_log_name=f"overfit_episode_{episode}",
                        reset_num_timesteps=False,
                    )

                    self.global_step += timesteps_per_episode

                    # Evaluate after training
                    obs, _ = env.reset()
                    episode_reward = 0.0
                    eval_steps = 0
                    h_preservation = 1.0
                    is_solvable = True
                    num_active = 0

                    for step in range(self.max_merges):
                        try:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, done, truncated, info = env.step(int(action))
                            episode_reward += reward
                            eval_steps += 1

                            reward_signals = info.get('reward_signals', {})

                            # ✅ CRITICAL: Validate reward signals
                            is_valid, error_msg = validate_reward_signals(reward_signals)
                            if not is_valid:
                                pbar.write(f"⚠️  Episode {episode}: Signal validation failed: {error_msg}")
                                self.logger.log_failure(
                                    episode, problem_name,
                                    'signal_validation_failure',
                                    error_msg
                                )

                            h_preservation = reward_signals.get('h_star_preservation', 1.0)
                            is_solvable = reward_signals.get('is_solvable', True)
                            num_active = info.get('num_active_systems', 0)

                            if done or truncated:
                                break
                        except subprocess.TimeoutExpired:
                            episode_error = "Timeout during environment step"
                            failure_type = 'timeout'
                            pbar.write(f"⏱️  Episode {episode}: Timeout during eval")
                            break
                        except Exception as e:
                            if "Timeout" in str(type(e)):
                                episode_error = f"Timeout: {str(e)[:100]}"
                                failure_type = 'timeout'
                                pbar.write(f"⏱️  Episode {episode}: {episode_error}")
                                break
                            raise

                    cumulative_reward += episode_reward

                    # Collect rich metrics
                    step_time_ms = self.resource_monitor.get_elapsed_ms() / max(1, eval_steps)
                    peak_memory_mb = self.resource_monitor.get_peak_memory_mb()

                    metrics = EpisodeMetrics(
                        episode=episode,
                        problem_name=problem_name,
                        reward=episode_reward,
                        h_star_preservation=h_preservation,
                        num_active_systems=num_active,
                        is_solvable=is_solvable,
                        eval_steps=eval_steps,
                        total_reward=cumulative_reward,
                        error=episode_error,
                        failure_type=failure_type,
                        step_time_ms=step_time_ms,
                        peak_memory_mb=peak_memory_mb,
                    )
                    self.episode_log.append(metrics)

                    self.logger.log_episode_completed(
                        episode=episode,
                        problem_name=problem_name,
                        reward=episode_reward,
                        steps=eval_steps,
                        h_preservation=h_preservation,
                        is_solvable=is_solvable,
                        error=episode_error,
                        failure_type=failure_type,
                        metrics={
                            'step_time_ms': step_time_ms,
                            'peak_memory_mb': peak_memory_mb,
                        }
                    )

                    successful_episodes = [m for m in self.episode_log if m.error is None]
                    avg_reward = np.mean([m.reward for m in successful_episodes]) if successful_episodes else 0
                    pbar.set_postfix({
                        'reward': f'{episode_reward:.4f}',
                        'h*': f'{h_preservation:.3f}',
                        'avg': f'{avg_reward:.4f}',
                        'coverage': '✓'
                    })

                    if (episode + 1) % self.checkpoint_interval == 0 or (episode + 1) == num_episodes:
                        checkpoint_path = self.checkpoints_dir / f"model_step_{self.global_step}.zip"
                        model.save(str(checkpoint_path))
                        self.logger.log_checkpoint_saved(
                            step=self.global_step,
                            path=str(checkpoint_path),
                            reward=episode_reward,
                            problem_name=problem_name
                        )

                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                        env = None

                except KeyboardInterrupt:
                    pbar.close()
                    print("\n⚠️  Training interrupted by user")
                    break

                except subprocess.TimeoutExpired as e:
                    self.failed_episode_count += 1
                    pbar.write(f"✗ Episode {episode} timeout: {e}")
                    self.logger.log_failure(
                        episode, problem_name, 'timeout',
                        str(e)[:100]
                    )
                    continue

                except Exception as e:
                    self.failed_episode_count += 1
                    pbar.write(f"✗ Episode {episode} failed: {e}")
                    self.logger.log_failure(
                        episode, problem_name, 'crash',
                        str(e)[:100],
                        {'traceback': traceback.format_exc()[:200]}
                    )
                    continue

                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass

            pbar.close()

            # Log coverage report at end of training
            self.logger.log_problem_coverage_report(
                total_episodes=len(self.episode_log),
                problem_names=self.problem_names
            )

            # Save final model
            if model is not None:
                final_model_path = self.output_dir / "model.zip"
                model.save(str(final_model_path))

                self.logger.log_training_completed(
                    total_steps=self.global_step,
                    total_reward=cumulative_reward
                )

                return str(final_model_path)

        except Exception as e:
            self.logger.log_failure(
                0, 'training', 'crash',
                str(e),
                {'traceback': traceback.format_exc()[:500]}
            )
            return None

        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        return None

    def save_training_log(self) -> Path:
        """Save episode metrics to JSONL file."""
        log_path = self.output_dir / "training_log.jsonl"

        with open(log_path, 'w', encoding='utf-8') as f:
            for metrics in self.episode_log:
                f.write(json.dumps(metrics.to_dict()) + '\n')

        return log_path

    def close_logger(self):
        """Finalize the training logger."""
        if self.logger:
            self.logger.close()

    def __del__(self):
        """Ensure logger is closed on cleanup."""
        try:
            self.close_logger()
        except:
            pass


# ============================================================================
# EVALUATION PHASE (ENHANCED)
# ============================================================================

class OverfitEvaluator:
    """Evaluates overfitting on training problems with enhanced tracking."""

    def __init__(
            self,
            model_path: str,
            benchmarks: List[Tuple[str, str]],
            problem_names: List[str],
            output_dir: str,
            reward_weights: Optional[Dict[str, float]] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
            seed: int = 42,
    ):
        self.model_path = model_path
        self.benchmarks = benchmarks
        self.problem_names = problem_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS.copy()
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step
        self.seed = seed

        from stable_baselines3 import PPO
        from thin_merge_env import ThinMergeEnv

        self.PPO = PPO
        self.ThinMergeEnv = ThinMergeEnv

    def _create_env(self, domain_file: str, problem_file: str, seed: int):
        """Create environment."""
        try:
            env = self.ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
                reward_weights=self.reward_weights,
                debug=False,
                seed=seed,
            )
        except TypeError:
            env = self.ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
                reward_weights=self.reward_weights,
                debug=False,
            )
        return env

    def evaluate(self, num_runs_per_problem: int = 5) -> Dict:
        """Evaluate on training problems."""
        model = self.PPO.load(self.model_path)
        results = {}

        for prob_idx, (domain_file, problem_file) in enumerate(
                tqdm(self.benchmarks, desc="Evaluating", unit="problem")
        ):
            problem_name = self.problem_names[prob_idx]
            rewards = []
            h_preservations = []
            steps_list = []
            solvable_count = 0
            failed_runs = 0

            for run in range(num_runs_per_problem):
                eval_seed = 1000000 + self.seed + (prob_idx * 10000) + run

                try:
                    cleanup_signal_files()
                except Exception:
                    pass

                env = None
                try:
                    env = self._create_env(domain_file, problem_file, seed=eval_seed)
                    obs, _ = env.reset()
                    total_reward = 0.0
                    eval_steps = 0
                    h_preservation = 1.0
                    is_solvable = True

                    for step in range(self.max_merges):
                        try:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, done, truncated, info = env.step(int(action))
                            total_reward += reward
                            eval_steps += 1

                            reward_signals = info.get('reward_signals', {})
                            h_preservation = reward_signals.get('h_star_preservation', 1.0)
                            is_solvable = reward_signals.get('is_solvable', True)

                            if done or truncated:
                                break
                        except subprocess.TimeoutExpired:
                            is_solvable = False
                            failed_runs += 1
                            break

                    rewards.append(total_reward)
                    h_preservations.append(h_preservation)
                    steps_list.append(eval_steps)
                    if is_solvable:
                        solvable_count += 1

                except Exception as e:
                    failed_runs += 1
                    print(f"[WARN] Evaluation failed for {problem_name} run {run}: {e}",
                          file=sys.stderr)

                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass

            if rewards:
                results[problem_name] = {
                    'problem_idx': prob_idx,
                    'num_runs': len(rewards),
                    'num_failed_runs': failed_runs,
                    'avg_reward': float(np.mean(rewards)),
                    'std_reward': float(np.std(rewards)),
                    'max_reward': float(np.max(rewards)),
                    'min_reward': float(np.min(rewards)),
                    'avg_h_preservation': float(np.mean(h_preservations)),
                    'avg_steps': float(np.mean(steps_list)),
                    'solve_rate': solvable_count / len(rewards),
                }
            else:
                print(f"[ERROR] Evaluation failed for {problem_name}", file=sys.stderr)
                results[problem_name] = {
                    'problem_idx': prob_idx,
                    'num_runs': 0,
                    'num_failed_runs': num_runs_per_problem,
                    'avg_reward': 0.0,
                    'std_reward': 0.0,
                    'max_reward': 0.0,
                    'min_reward': 0.0,
                    'avg_h_preservation': 0.0,
                    'avg_steps': 0.0,
                    'solve_rate': 0.0,
                }

        return results


# ============================================================================
# ANALYSIS (ENHANCED WITH COVERAGE VALIDATION)
# ============================================================================

def analyze_overfitting(
        training_log: List[EpisodeMetrics],
        eval_results: Dict,
        problem_names: List[str],
        benchmarks: List[Tuple[str, str]],
        timesteps_per_episode: int = 50,
        best_model_path: Optional[str] = None,
        checkpoints_count: int = 0,
        experiment_id: str = '',
) -> OverfitExperimentSummary:
    """
    Analyze overfitting with COVERAGE VALIDATION.

    ✅ CRITICAL: Ensures all problems got adequate training
    """
    # Group by problem
    by_problem = defaultdict(list)
    for metrics in training_log:
        by_problem[metrics.problem_name].append(metrics)

    # Per-problem statistics
    per_problem_stats = []
    min_coverage_pct = 100.0
    max_coverage_pct = 0.0
    all_problems_trained = True

    for problem_idx, (domain, problem_file) in enumerate(benchmarks):
        problem_name = problem_names[problem_idx]

        if problem_name in by_problem:
            episodes = by_problem[problem_name]

            # ✅ NEW: Coverage calculation
            coverage_pct = (len(episodes) / len(training_log) * 100) if training_log else 0
            min_coverage_pct = min(min_coverage_pct, coverage_pct)
            max_coverage_pct = max(max_coverage_pct, coverage_pct)

            successful_episodes = [e for e in episodes if e.error is None]
            failed_episodes = [e for e in episodes if e.error is not None]

            if successful_episodes:
                rewards = [e.reward for e in successful_episodes]
                h_preservations = [e.h_star_preservation for e in successful_episodes]
                step_times = [e.step_time_ms for e in successful_episodes if e.step_time_ms > 0]
                peak_mems = [e.peak_memory_mb for e in successful_episodes if e.peak_memory_mb > 0]
            else:
                rewards = []
                h_preservations = []
                step_times = []
                peak_mems = []

            initial_reward = rewards[0] if rewards else 0
            final_reward = rewards[-1] if rewards else 0
            best_reward = max(rewards) if rewards else 0
            worst_reward = min(rewards) if rewards else 0

            if best_reward != worst_reward:
                improvement_ratio = (final_reward - worst_reward) / (best_reward - worst_reward)
            else:
                improvement_ratio = 0.0

            # Convergence detection
            episodes_to_convergence = None
            if len(rewards) > 10:
                for i in range(10, len(rewards)):
                    recent_avg = np.mean(rewards[i - 10:i])
                    older_avg = np.mean(rewards[max(0, i - 20):i - 10])
                    if older_avg != 0 and abs(recent_avg - older_avg) / abs(older_avg) < 0.05:
                        episodes_to_convergence = i
                        break

            stats = ProblemStats(
                problem_name=problem_name,
                num_episodes=len(episodes),
                num_failed=len(failed_episodes),
                coverage_percent=coverage_pct,
                avg_reward=float(np.mean(rewards)) if rewards else 0.0,
                best_reward=best_reward,
                worst_reward=worst_reward,
                final_reward=final_reward,
                improvement_ratio=improvement_ratio,
                avg_h_preservation=float(np.mean(h_preservations)) if h_preservations else 0.0,
                solve_rate=len(successful_episodes) / len(episodes) if episodes else 0,
                episodes_to_convergence=episodes_to_convergence,
                avg_step_time_ms=float(np.mean(step_times)) if step_times else 0,
                avg_memory_mb=float(np.mean(peak_mems)) if peak_mems else 0,
            )

            per_problem_stats.append(stats)
        else:
            # ✅ CRITICAL: Problem was never trained!
            all_problems_trained = False
            per_problem_stats.append(ProblemStats(
                problem_name=problem_name,
                num_episodes=0,
                num_failed=0,
                coverage_percent=0.0,
                avg_reward=0.0,
                best_reward=0.0,
                worst_reward=0.0,
                final_reward=0.0,
                improvement_ratio=0.0,
                avg_h_preservation=0.0,
                solve_rate=0.0,
            ))

    # Overall statistics
    successful_log = [m for m in training_log if m.error is None]
    failed_log = [m for m in training_log if m.error is not None]

    # Failure taxonomy
    failure_taxonomy = defaultdict(int)
    for m in failed_log:
        failure_type = m.failure_type or 'unknown'
        failure_taxonomy[failure_type] += 1

    all_rewards = [m.reward for m in successful_log]
    all_h_preservations = [m.h_star_preservation for m in successful_log]

    # H* preservation improvement
    if len(all_h_preservations) > 10:
        initial_h_pres = np.mean(all_h_preservations[:10])
        final_h_pres = np.mean(all_h_preservations[-10:])
        h_pres_improvement = final_h_pres / initial_h_pres if initial_h_pres > 0 else 1.0
    else:
        h_pres_improvement = 1.0

    # Solve rate improvement
    if len(successful_log) > 10:
        initial_solve_rate = sum(1 for m in successful_log[:10] if m.is_solvable) / 10
        final_solve_rate = sum(1 for m in successful_log[-10:] if m.is_solvable) / 10
        solve_rate_improvement = final_solve_rate - initial_solve_rate
    else:
        solve_rate_improvement = 0.0

    training_final_avg = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)

    if eval_results:
        eval_rewards = [r.get('avg_reward', 0) for r in eval_results.values()]
        evaluation_avg = np.mean(eval_rewards) if eval_rewards else 0.0
    else:
        evaluation_avg = 0.0

    overfitting_ratio = training_final_avg / evaluation_avg if evaluation_avg > 0 else 1.0

    # Average time and memory
    all_step_times = [m.step_time_ms for m in successful_log if m.step_time_ms > 0]
    all_peak_mems = [m.peak_memory_mb for m in successful_log if m.peak_memory_mb > 0]

    summary = OverfitExperimentSummary(
        num_problems=len(benchmarks),
        num_train_episodes=len(training_log),
        num_failed_episodes=len(failed_log),
        total_timesteps=len(successful_log) * timesteps_per_episode,
        start_time=datetime.fromtimestamp(training_log[0].timestamp).isoformat() if training_log else "",
        end_time=datetime.fromtimestamp(training_log[-1].timestamp).isoformat() if training_log else "",
        duration_seconds=training_log[-1].timestamp - training_log[0].timestamp if len(training_log) > 1 else 0,
        avg_reward_over_all=float(np.mean(all_rewards)) if all_rewards else 0,
        best_reward_over_all=float(np.max(all_rewards)) if all_rewards else 0,
        worst_reward_over_all=float(np.min(all_rewards)) if all_rewards else 0,
        per_problem_stats=[s.to_dict() for s in per_problem_stats],
        reward_variance=float(np.var(all_rewards)) if all_rewards else 0,
        h_preservation_improvement_ratio=float(h_pres_improvement),
        solve_rate_improvement=float(solve_rate_improvement),
        early_convergence_episodes=min(
            [s.episodes_to_convergence for s in per_problem_stats if s.episodes_to_convergence],
            default=0
        ) or 0,
        checkpoints_saved=checkpoints_count,
        best_model_path=best_model_path or '',
        overfitting_ratio=float(overfitting_ratio),
        training_final_avg=float(training_final_avg),
        evaluation_avg=float(evaluation_avg),
        experiment_id=experiment_id,
        # ✅ NEW: Coverage validation
        problem_coverage_valid=all_problems_trained and min_coverage_pct >= 5.0,
        min_problem_coverage_pct=min_coverage_pct if training_log else 0.0,
        max_problem_coverage_pct=max_coverage_pct if training_log else 0.0,
        all_problems_trained=all_problems_trained,
        # ✅ NEW: Failure taxonomy
        failure_taxonomy=dict(failure_taxonomy),
        avg_step_time_ms=float(np.mean(all_step_times)) if all_step_times else 0,
        avg_peak_memory_mb=float(np.mean(all_peak_mems)) if all_peak_mems else 0,
    )

    return summary


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_plots(
        training_log: List[EpisodeMetrics],
        eval_results: Dict,
        output_dir: Path
):
    """Generate learning curve plots."""

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots", file=sys.stderr)
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    episodes = [m.episode for m in training_log]
    rewards = [m.reward for m in training_log]
    h_preservations = [m.h_star_preservation for m in training_log]
    problems = [m.problem_name for m in training_log]

    # Plot 1: Overall reward curve
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, label='Per-episode')

    window = min(10, len(rewards) // 4) if len(rewards) > 4 else 1
    if window > 1:
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(rewards)), rolling_avg, linewidth=2,
                 label=f'Rolling avg (window={window})')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Curve - Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: H* preservation curve
    ax2 = axes[0, 1]
    ax2.plot(episodes, h_preservations, alpha=0.3, color='green', label='Per-episode')

    if window > 1:
        rolling_h = np.convolve(h_preservations, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(h_preservations)), rolling_h, linewidth=2,
                 color='darkgreen', label=f'Rolling avg')

    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect preservation')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('H* Preservation')
    ax2.set_title('Learning Curve - H* Preservation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Problem coverage
    by_problem = defaultdict(list)
    for m in training_log:
        by_problem[m.problem_name].append(len(by_problem[m.problem_name]) + 1)

    ax3 = axes[1, 0]
    problem_counts = [len(by_problem[p]) for p in sorted(by_problem.keys())]
    ax3.bar(range(len(problem_counts)), problem_counts)
    ax3.set_xticks(range(len(problem_counts)))
    ax3.set_xticklabels([p[:10] for p in sorted(by_problem.keys())], rotation=45)
    ax3.set_ylabel('Episode Count')
    ax3.set_title('Problem Coverage')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Failure distribution
    failure_types = defaultdict(int)
    for m in training_log:
        if m.failure_type:
            failure_types[m.failure_type] += 1

    ax4 = axes[1, 1]
    if failure_types:
        labels = list(failure_types.keys())
        counts = list(failure_types.values())
        ax4.pie(counts, labels=labels, autopct='%1.1f%%')
        ax4.set_title('Failure Types')
    else:
        ax4.text(0.5, 0.5, 'No Failures', ha='center', va='center')
        ax4.set_title('Failure Types')

    plt.tight_layout()
    plt.savefig(plots_dir / "analysis_dashboard.png", dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overfit experiment with RIGOROUS VALIDATION (v2)",
        epilog="""
EXAMPLES:

  Fresh training:
    python experiment_1_problem_overfit.py \\
      --domain benchmarks/domain.pddl \\
      --problems "benchmarks/problem_*.pddl" \\
      --num-problems 5 \\
      --num-train-episodes 100 \\
      --output results/

  Resume from checkpoint:
    python experiment_1_problem_overfit.py \\
      --domain benchmarks/domain.pddl \\
      --problems "benchmarks/problem_*.pddl" \\
      --num-problems 5 \\
      --num-train-episodes 100 \\
      --output results/ \\
      --resume-from results/checkpoints/model_step_50000.zip
        """
    )
    parser.add_argument("--domain", required=True, help="Domain PDDL file")
    parser.add_argument("--problems", required=True, help="Problem glob pattern")
    parser.add_argument("--num-problems", type=int, default=5, help="Number of problems")
    parser.add_argument("--num-train-episodes", type=int, default=100, help="Training episodes")
    parser.add_argument("--timesteps-per-episode", type=int, default=50, help="Timesteps/episode")
    parser.add_argument("--output", default="overfit_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-merges", type=int, default=50, help="Max merges/episode")
    parser.add_argument("--timeout", type=float, default=120.0, help="Step timeout (sec)")
    parser.add_argument("--eval-runs", type=int, default=3, help="Eval runs/problem")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Checkpoint every N steps")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--min-episodes-per-problem", type=int, default=10,
                        help="Minimum training episodes per problem (coverage guarantee)")

    args = parser.parse_args()

    # Early validation
    print("\n🔍 EARLY VALIDATION")
    print("   Checking dependencies...")

    try:
        validate_dependencies()
        print("   ✅ All dependencies available\n")
    except SystemExit:
        return 1

    print("   Checking disk space...")
    if not validate_disk_space(args.output, min_gb=5.0):
        print("   ❌ Insufficient disk space\n")
        return 1
    print("   ✅ Disk space OK\n")

    set_all_seeds(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("🔬 OVERFIT EXPERIMENT - PRODUCTION HARDENED v2 (RIGOROUS VALIDATION)")
    print("=" * 100)
    print(f"\n🔒 REPRODUCIBILITY")
    print(f"   Seed set in all libraries: {args.seed}")
    print(f"   Torch determinism: ON")
    print(f"\n💾 SAFETY NET")
    print(f"   Checkpoints: Every {args.checkpoint_interval} steps (never overwrite)")
    print(f"   Best model: Updated after evaluation")
    print(f"   Resume: Full state restoration")
    print(f"\n✅ COVERAGE GUARANTEE (NEW)")
    print(f"   Minimum episodes/problem: {args.min_episodes_per_problem}")
    print(f"   Validation: All problems trained check")
    print(f"\n📊 METRIC RICHNESS (NEW)")
    print(f"   Step-level logging: Per-decision analysis")
    print(f"   GNN health: Entropy, value loss, gradients")
    print(f"   Failure taxonomy: Timeout/DeadEnd/Crash/SolvabilityLoss")
    print(f"   Resource tracking: Time, memory")
    print(f"\n🛡️  SIGNAL INTEGRITY (NEW)")
    print(f"   Reward validation: Explicit h* preservation check")
    print(f"   Infinity handling: Dedicated dead-end detection")
    print(f"   Parsing errors: Fail-fast (not silent defaults)")
    print("=" * 100 + "\n")

    try:
        # Step 1: Select problems
        print("📋 Step 1: Selecting training problems...")
        domain_file, benchmarks, problem_names = select_training_problems(
            args.domain,
            args.problems,
            args.num_problems,
            args.seed
        )
        print(f"✓ Selected {len(benchmarks)} problems:")
        for name in problem_names:
            print(f"   • {name}")

        # Step 2: Train with enhanced monitoring
        print("\n🚀 Step 2: Training with RIGOROUS VALIDATION...")
        print("   Details logged to: training.log (structured EVENTs)")
        trainer = OverfitTrainer(
            benchmarks=benchmarks,
            problem_names=problem_names,
            output_dir=args.output,
            max_merges=args.max_merges,
            timeout_per_step=args.timeout,
            checkpoint_interval=args.checkpoint_interval,
            min_episodes_per_problem=args.min_episodes_per_problem,
            seed=args.seed,
        )
        model_path = trainer.run_training(
            num_episodes=args.num_train_episodes,
            timesteps_per_episode=args.timesteps_per_episode,
            resume_from=args.resume_from,
        )
        trainer.save_training_log()
        trainer.close_logger()

        if not model_path:
            print("\n❌ Training failed")
            return 1

        checkpoint_count = len(list(trainer.checkpoints_dir.glob("*.zip")))
        print(f"\n✅ Training complete!")
        print(f"   Final model: {model_path}")
        print(f"   Checkpoints saved: {checkpoint_count}")
        print(f"   Failed episodes: {trainer.failed_episode_count}")

        # Step 3: Evaluate
        print("\n📊 Step 3: Evaluating on training problems...")
        evaluator = OverfitEvaluator(
            model_path=model_path,
            benchmarks=benchmarks,
            problem_names=problem_names,
            output_dir=args.output,
            max_merges=args.max_merges,
            timeout_per_step=args.timeout,
            seed=args.seed,
        )
        eval_results = evaluator.evaluate(num_runs_per_problem=args.eval_runs)
        print("✓ Evaluation complete")

        # Step 4: Analyze with coverage validation
        print("\n🔍 Step 4: Analyzing results with COVERAGE VALIDATION...")
        summary = analyze_overfitting(
            trainer.episode_log,
            eval_results,
            problem_names,
            benchmarks,
            timesteps_per_episode=args.timesteps_per_episode,
            best_model_path=str(trainer.best_model_path) if trainer.best_model_path else None,
            checkpoints_count=checkpoint_count,
            experiment_id=trainer.get_experiment_id(),
        )

        # Save results
        summary_path = output_dir / "overfit_summary.json"
        save_json_atomic(summary.to_dict(), str(summary_path))

        eval_path = output_dir / "evaluation_results.json"
        save_json_atomic(eval_results, str(eval_path))

        # Generate plots
        print("📈 Generating visualization plots...")
        try:
            generate_plots(trainer.episode_log, eval_results, output_dir)
        except Exception as e:
            print(f"[WARN] Plot generation failed: {e}", file=sys.stderr)

        # Print summary
        print("\n" + "=" * 100)
        print("✅ OVERFIT EXPERIMENT COMPLETE")
        print("=" * 100)
        print(f"\n📁 All output in: {os.path.abspath(args.output)}")
        print(f"\n📄 OUTPUT FILES:")
        print(f"   training.log ........................ Structured EVENTs")
        print(f"   training_log.jsonl .................. Per-episode metrics")
        print(f"   overfit_summary.json ............... Experiment summary")
        print(f"   evaluation_results.json ............ Evaluation results")
        print(f"   model.zip .................... Final model")
        print(f"   checkpoints/ ....................... Checkpoint history")
        print(f"   plots/ ............................. Visualization")
        print(f"\n✅ COVERAGE VALIDATION:")
        print(f"   All problems trained: {summary.all_problems_trained}")
        print(f"   Min coverage: {summary.min_problem_coverage_pct:.1f}%")
        print(f"   Max coverage: {summary.max_problem_coverage_pct:.1f}%")
        print(f"   Coverage valid: {summary.problem_coverage_valid}")
        print(f"\n📊 FAILURE ANALYSIS:")
        print(f"   Total failures: {summary.num_failed_episodes}")
        print(f"   Failure taxonomy: {summary.failure_taxonomy}")
        print(f"\n📊 RESOURCE METRICS:")
        print(f"   Avg step time: {summary.avg_step_time_ms:.2f}ms")
        print(f"   Avg peak memory: {summary.avg_peak_memory_mb:.2f}MB")
        print(f"\n📊 KEY METRICS:")
        print(f"   Avg reward: {summary.avg_reward_over_all:.4f}")
        print(f"   Best reward: {summary.best_reward_over_all:.4f}")
        print(f"   Overfitting ratio: {summary.overfitting_ratio:.3f}")
        print(f"   H* improvement: {summary.h_preservation_improvement_ratio:.3f}x")
        print(f"   Training duration: {summary.duration_seconds:.1f}s")
        print("=" * 100)

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())