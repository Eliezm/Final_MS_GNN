#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHARED EXPERIMENT UTILITIES - ENHANCED WITH THIN CLIENT INTEGRATION
====================================================================
Common code used across all experiments + evaluation framework integration.

Provides:
  - Checkpoint/resume functionality
  - Standardized logging
  - Common training/evaluation patterns
  - Result persistence
  - Experiment-to-evaluation conversion
  - Multi-experiment result aggregation
  - ThinMergeEnv integration with enhanced features
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict

from src.environments.thin_merge_env import ThinMergeEnv
from src.models.gnn_policy import GNNPolicy
from src.utils.common_utils import (
    PROJECT_ROOT, DOWNWARD_DIR, FD_OUTPUT_DIR, GNN_OUTPUT_DIR,
    ThinClientConfig, ensure_directories, cleanup_signal_files
)


import logging

logger = logging.getLogger(__name__)

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure directories exist
os.makedirs("downward/gnn_output", exist_ok=True)
os.makedirs("downward/fd_output", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("tb_logs", exist_ok=True)


# ============================================================================
# LOGGING & FORMATTING
# ============================================================================

def setup_logging(experiment_name: str, output_dir: str) -> logging.Logger:
    """Setup consistent logging for experiment."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(output_dir, f"{experiment_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8'),
        ],
        force=True
    )

    exp_logger = logging.getLogger(experiment_name)
    return exp_logger


def print_section(title: str, exp_logger: logging.Logger, width: int = 90):
    """Print formatted section header."""
    exp_logger.info("\n" + "=" * width)
    exp_logger.info(f"// {title.upper()}")
    exp_logger.info("=" * width + "\n")


def print_subsection(title: str, exp_logger: logging.Logger):
    """Print formatted subsection header."""
    exp_logger.info("\n" + "-" * 80)
    exp_logger.info(f">>> {title}")
    exp_logger.info("-" * 80 + "\n")


# ============================================================================
# CHECKPOINT & RESUME
# ============================================================================

class ExperimentCheckpoint:
    """Simple checkpoint manager for robust experiment recovery."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "experiment_checkpoint.json"

    def save(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state."""
        state['checkpoint_time'] = datetime.now().isoformat()

        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if it exists."""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def clear(self) -> None:
        """Clear checkpoint (call after successful completion)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


# ============================================================================
# THIN CLIENT REWARD WEIGHTS
# ============================================================================

DEFAULT_REWARD_WEIGHTS = {
    'w_h_preservation': 0.40,
    'w_shrinkability': 0.25,
    'w_state_control': 0.20,
    'w_solvability': 0.15,
}

REWARD_FUNCTION_CONFIG = {
    'type': 'learning_focused',  # 'learning_focused' or 'enhanced'
    'learning_focused': {
        'debug': False,
        'total_episodes': 1500,
        'use_curriculum': True,  # Enable episode-aware thresholds
    },
    'enhanced': {
        'debug': False,
    }
}


# ============================================================================
# TRAINING HELPERS (UPDATED FOR THIN CLIENT)
# ============================================================================

def train_gnn_model(
        benchmarks: List[Tuple[str, str]],
        total_timesteps: int = 5000,
        timesteps_per_problem: int = 500,
        model_output_path: str = "models/gnn_model.zip",
        exp_logger: Optional[logging.Logger] = None,
        tb_log_name: str = "experiment_training",
        reward_weights: Optional[Dict[str, float]] = None,
        max_merges: int = 128,  # ✅ CHANGED from 50
        timeout_per_step: float = 120.0,
        reward_function_config: Optional[Dict] = None,  # ✅ NEW
) -> Optional[str]:
    """
    Train GNN model on benchmarks using ThinMergeEnv.

    Returns:
        Path to trained model, or None if failed
    """
    if exp_logger is None:
        exp_logger = logging.getLogger("train_gnn")

    if reward_function_config is None:
        reward_function_config = REWARD_FUNCTION_CONFIG  # ✅ NEW

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        # from gnn_policy import GNNPolicy
        from src.environments.thin_merge_env import ThinMergeEnv

        exp_logger.info(f"Training on {len(benchmarks)} problem(s)")
        exp_logger.info(f"Total timesteps: {total_timesteps}")
        exp_logger.info(f"Timesteps per problem: {timesteps_per_problem}\n")

        os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)

        if reward_weights is None:
            reward_weights = DEFAULT_REWARD_WEIGHTS.copy()

        model = None
        total_steps = 0

        for step, (domain_file, problem_file) in enumerate(benchmarks):
            problem_name = os.path.basename(problem_file)
            exp_logger.info(f"  [{step + 1}/{len(benchmarks)}] {problem_name}")

            try:
                env = ThinMergeEnv(
                    domain_file=os.path.abspath(domain_file),
                    problem_file=os.path.abspath(problem_file),
                    max_merges=max_merges,
                    timeout_per_step=timeout_per_step,
                    reward_weights=reward_weights,
                    debug=False,
                )
                env = Monitor(env)

                if model is None:
                    exp_logger.info("    Creating new PPO model...")
                    model = PPO(
                        policy=GNNPolicy,
                        env=env,
                        learning_rate=0.0003,
                        n_steps=64,
                        batch_size=32,
                        ent_coef=0.01,
                        verbose=0,
                        tensorboard_log="tb_logs/",
                        policy_kwargs={"hidden_dim": 64},
                    )
                else:
                    model.set_env(env)

                exp_logger.info(f"    Training for {timesteps_per_problem} timesteps...")
                model.learn(
                    total_timesteps=timesteps_per_problem,
                    tb_log_name=f"{tb_log_name}_{step}",
                    reset_num_timesteps=False,
                )

                total_steps += timesteps_per_problem
                exp_logger.info(f"    ✓ Training complete (total: {total_steps} steps)")

                env.close()

                if total_steps >= total_timesteps:
                    break

            except Exception as e:
                exp_logger.error(f"    ⚠️ Problem failed: {e}")
                exp_logger.error(traceback.format_exc())
                continue

        if model is None:
            exp_logger.error("Training failed - no model created")
            return None

        model.save(model_output_path)
        exp_logger.info(f"\n✅ Model saved: {model_output_path}")

        return model_output_path

    except Exception as e:
        exp_logger.error(f"Training failed: {e}")
        exp_logger.error(traceback.format_exc())
        return None


def evaluate_model_on_problems(
        model_path: str,
        benchmarks: List[Tuple[str, str]],
        exp_logger: Optional[logging.Logger] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        max_merges: int = 50,
        timeout_per_step: float = 120.0,
) -> Dict[str, Any]:
    """
    Quick evaluation of trained model on test problems using ThinMergeEnv.

    Returns:
        Dict with evaluation results
    """
    if exp_logger is None:
        exp_logger = logging.getLogger("evaluate")

    try:
        from stable_baselines3 import PPO
        from src.environments.thin_merge_env import ThinMergeEnv

        model = PPO.load(model_path)
        exp_logger.info(f"Model loaded: {model_path}")

        if reward_weights is None:
            reward_weights = DEFAULT_REWARD_WEIGHTS.copy()

        results = {
            'total_problems': len(benchmarks),
            'solved_count': 0,
            'avg_reward': 0.0,
            'avg_time': 0.0,
            'details': []
        }

        rewards = []
        times = []

        for i, (domain_file, problem_file) in enumerate(benchmarks):
            problem_name = os.path.basename(problem_file)
            exp_logger.info(f"  [{i + 1}/{len(benchmarks)}] {problem_name}")

            try:
                env = ThinMergeEnv(
                    domain_file=os.path.abspath(domain_file),
                    problem_file=os.path.abspath(problem_file),
                    max_merges=max_merges,
                    timeout_per_step=timeout_per_step,
                    reward_weights=reward_weights,
                    debug=False,
                )

                start_time = time.time()
                obs, _ = env.reset()
                episode_reward = 0.0
                steps = 0

                for step in range(max_merges):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))
                    episode_reward += reward
                    steps += 1

                    if done or truncated:
                        break

                elapsed = time.time() - start_time

                rewards.append(episode_reward)
                times.append(elapsed)
                results['solved_count'] += 1

                exp_logger.info(f"    ✓ Reward: {episode_reward:.4f}, Time: {elapsed:.2f}s, Steps: {steps}")

                # Extract additional info from final state
                reward_signals = info.get('reward_signals', {})

                results['details'].append({
                    'problem': problem_name,
                    'reward': float(episode_reward),
                    'time': float(elapsed),
                    'steps': steps,
                    'solved': True,
                    'h_star_preservation': reward_signals.get('h_star_preservation', 1.0),
                    'is_solvable': reward_signals.get('is_solvable', True),
                    'num_active_systems': info.get('num_active_systems', 1),
                })

                env.close()

            except Exception as e:
                exp_logger.warning(f"    ⚠️ Failed: {e}")
                results['details'].append({
                    'problem': problem_name,
                    'solved': False,
                    'error': str(e)[:100]
                })

        if rewards:
            results['avg_reward'] = float(np.mean(rewards))
            results['avg_time'] = float(np.mean(times))
            results['solve_rate'] = (results['solved_count'] / len(benchmarks)) * 100

        return results

    except Exception as e:
        exp_logger.error(f"Evaluation failed: {e}")
        exp_logger.error(traceback.format_exc())
        return {}


# ============================================================================
# DATASET UTILITIES
# ============================================================================

def load_and_split_problems(
        domain_file: str,
        problem_pattern: str,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        exp_logger: Optional[logging.Logger] = None
) -> Tuple[List[str], List[str]]:
    """Load problems and split into train/test."""
    if exp_logger is None:
        exp_logger = logging.getLogger("dataset")

    import glob
    import random

    if not os.path.exists(domain_file):
        raise FileNotFoundError(f"Domain not found: {domain_file}")

    all_problems = sorted(glob.glob(problem_pattern))

    if not all_problems:
        raise ValueError(f"No problems found matching: {problem_pattern}")

    if len(all_problems) < 2:
        raise ValueError("Need at least 2 problems for train/test split")

    random.seed(random_seed)
    problems_shuffled = all_problems.copy()
    random.shuffle(problems_shuffled)

    split_idx = int(len(problems_shuffled) * train_ratio)
    train_problems = sorted(problems_shuffled[:split_idx])
    test_problems = sorted(problems_shuffled[split_idx:])

    exp_logger.info(f"Loaded {len(all_problems)} problems")
    exp_logger.info(f"Train/test split: {len(train_problems)} train, {len(test_problems)} test")

    return train_problems, test_problems


def load_benchmarks_by_difficulty(
        benchmarks_dir: str = "benchmarks",
        exp_logger: Optional[logging.Logger] = None
) -> Dict[str, List[Tuple[str, str]]]:
    """Load benchmarks organized by difficulty."""
    if exp_logger is None:
        exp_logger = logging.getLogger("dataset")

    import glob

    benchmarks_dir = os.path.abspath(benchmarks_dir)

    if not os.path.isdir(benchmarks_dir):
        exp_logger.warning(f"Benchmarks dir not found: {benchmarks_dir}")
        return {}

    all_benchmarks = {}

    for difficulty in ["small", "medium", "large"]:
        difficulty_dir = os.path.join(benchmarks_dir, difficulty)

        domain_file = os.path.join(difficulty_dir, "domain.pddl")
        if not os.path.exists(domain_file):
            exp_logger.warning(f"Domain not found for {difficulty}")
            all_benchmarks[difficulty] = []
            continue

        problems = sorted(glob.glob(os.path.join(difficulty_dir, "problem_*.pddl")))

        if not problems:
            exp_logger.warning(f"No problems found for {difficulty}")
            all_benchmarks[difficulty] = []
            continue

        benchmarks_list = [
            (os.path.abspath(domain_file), os.path.abspath(prob))
            for prob in problems
        ]

        all_benchmarks[difficulty] = benchmarks_list
        exp_logger.info(f"  {difficulty:<10}: {len(benchmarks_list)} problems")

    return all_benchmarks


def load_and_validate_benchmarks(
        benchmark_dir: str = "benchmarks",
        timeout_per_problem: int = 480,
        exp_logger: Optional[logging.Logger] = None
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load and validate benchmarks from directory structure.

    Expected structure:
        benchmarks/
        ├── blocksworld/
        │   ├── small/
        │   │   ├── domain.pddl
        │   │   ├── problem_small_00.pddl
        │   │   └── ...
        │   ├── medium/
        │   └── large/
        ├── logistics/
        └── parking/

    OR simple structure:
        benchmarks/
        ├── small/
        │   ├── domain.pddl
        │   └── problem_*.pddl
        ├── medium/
        └── large/

    Returns:
        Dict mapping key -> list of (domain_file, problem_file) tuples
    """
    if exp_logger is None:
        exp_logger = logging.getLogger("benchmark_loader")

    import glob

    exp_logger.info(f"Loading benchmarks from: {benchmark_dir}")

    benchmarks = {}

    # Check for simple structure first (benchmarks/small/, etc.)
    simple_sizes = ["small", "medium", "large"]
    simple_structure = any(
        os.path.isdir(os.path.join(benchmark_dir, size))
        for size in simple_sizes
    )

    if simple_structure:
        # Simple structure: benchmarks/small/domain.pddl
        for size_name in simple_sizes:
            size_dir = os.path.join(benchmark_dir, size_name)
            if not os.path.isdir(size_dir):
                continue

            domain_file = os.path.join(size_dir, "domain.pddl")
            if not os.path.exists(domain_file):
                exp_logger.warning(f"Domain file not found: {domain_file}")
                continue

            problems = sorted(glob.glob(os.path.join(size_dir, "problem_*.pddl")))

            if not problems:
                exp_logger.warning(f"No problems found in: {size_dir}")
                continue

            key = size_name
            benchmarks[key] = [
                (os.path.abspath(domain_file), os.path.abspath(p))
                for p in problems
            ]

            exp_logger.info(f"  {key}: {len(benchmarks[key])} problems")
    else:
        # Complex structure: benchmarks/domain/size/
        for domain_dir in glob.glob(os.path.join(benchmark_dir, "*")):
            if not os.path.isdir(domain_dir):
                continue

            domain_name = os.path.basename(domain_dir)

            for size_dir in glob.glob(os.path.join(domain_dir, "*")):
                if not os.path.isdir(size_dir):
                    continue

                size_name = os.path.basename(size_dir)

                domain_file = os.path.join(size_dir, "domain.pddl")
                if not os.path.exists(domain_file):
                    exp_logger.warning(f"Domain file not found: {domain_file}")
                    continue

                problems = sorted(glob.glob(os.path.join(size_dir, "problem_*.pddl")))

                if not problems:
                    exp_logger.warning(f"No problems found in: {size_dir}")
                    continue

                key = f"{domain_name}_{size_name}"
                benchmarks[key] = [
                    (os.path.abspath(domain_file), os.path.abspath(p))
                    for p in problems
                ]

                exp_logger.info(f"  {key}: {len(benchmarks[key])} problems")

    exp_logger.info(f"\n✅ Loaded {sum(len(v) for v in benchmarks.values())} total problems")

    return benchmarks


def filter_benchmarks_by_size(
        all_benchmarks: Dict[str, List[Tuple[str, str]]],
        sizes: List[str]
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Filter benchmarks to only include specified sizes.

    Args:
        all_benchmarks: Dict from load_and_validate_benchmarks()
        sizes: List of sizes like ["small", "medium", "large"]

    Returns:
        Filtered benchmarks dict with only matching keys
    """
    filtered = {}
    for key, benchmark_list in all_benchmarks.items():
        for size in sizes:
            if key == size or key.endswith(f"_{size}"):
                filtered[key] = benchmark_list
                break

    logger.info(f"Filtered benchmarks by size {sizes}: {len(filtered)} combinations")
    return filtered


def get_benchmarks_for_sizes(
        all_benchmarks: Dict[str, List[Tuple[str, str]]],
        sizes: List[str],
        max_problems_per_combination: int = None
) -> List[Tuple[str, str]]:
    """
    Get flattened list of benchmarks for specified sizes.

    Args:
        all_benchmarks: Dict from load_and_validate_benchmarks()
        sizes: List of sizes like ["small", "medium"]
        max_problems_per_combination: Max problems per domain-size combo

    Returns:
        Flattened list of (domain_file, problem_file) tuples
    """
    benchmarks_list = []

    for key, benchmark_items in all_benchmarks.items():
        for size in sizes:
            if key == size or key.endswith(f"_{size}"):
                if max_problems_per_combination:
                    benchmarks_list.extend(benchmark_items[:max_problems_per_combination])
                else:
                    benchmarks_list.extend(benchmark_items)
                break

    logger.info(f"Collected {len(benchmarks_list)} benchmarks for sizes {sizes}")
    return benchmarks_list


# ============================================================================
# RESULT PERSISTENCE
# ============================================================================

def save_results_to_json(
        results: Dict[str, Any],
        output_path: str,
        exp_logger: Optional[logging.Logger] = None
) -> None:
    """Save results to JSON file."""
    if exp_logger is None:
        exp_logger = logging.getLogger("results")

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        exp_logger.info(f"✓ Results saved: {output_path}")

    except Exception as e:
        exp_logger.error(f"Failed to save results: {e}")


def save_results_to_txt(
        results: Dict[str, Any],
        output_path: str,
        experiment_name: str,
        exp_logger: Optional[logging.Logger] = None
) -> None:
    """Save human-readable results to text file with proper encoding."""
    if exp_logger is None:
        exp_logger = logging.getLogger("results")

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # ✅ FIX: Use UTF-8 encoding explicitly
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write(f"{experiment_name.upper()} - RESULTS SUMMARY\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

            for key, value in results.items():
                if key.startswith('_'):
                    continue

                if isinstance(value, dict):
                    f.write(f"\n{key.upper()}:\n")
                    f.write("-" * 50 + "\n")
                    for k, v in value.items():
                        # ✅ FIX: Sanitize unicode in values
                        v_str = str(v).encode('utf-8', errors='replace').decode('utf-8')
                        f.write(f"  {k:<40} {v_str}\n")
                elif isinstance(value, list):
                    f.write(f"\n{key.upper()}:\n")
                    f.write("-" * 50 + "\n")
                    for idx, item in enumerate(value[:5], 1):
                        item_str = str(item).encode('utf-8', errors='replace').decode('utf-8')
                        f.write(f"  [{idx}] {item_str}\n")
                    if len(value) > 5:
                        f.write(f"  ... and {len(value) - 5} more\n")
                else:
                    v_str = str(value).encode('utf-8', errors='replace').decode('utf-8')
                    f.write(f"{key:<40} {v_str}\n")

            f.write("\n" + "=" * 90 + "\n")

        exp_logger.info(f"✓ Results saved: {output_path}")

    except Exception as e:
        exp_logger.error(f"Failed to save results: {e}")


# ============================================================================
# EXPERIMENT-TO-EVALUATION INTEGRATION
# ============================================================================

@dataclass
class ExperimentResultsMetrics:
    """Unified metrics format for evaluation."""
    experiment_type: str
    problem_name: str
    difficulty: str
    solved: bool
    solve_time: float
    reward: float
    merge_episodes: int
    num_problems_total: int
    num_problems_solved: int
    avg_reward: float
    avg_time: float
    solve_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_experiment_results_from_directory(
        experiment_dir: str,
        experiment_type: str,
        exp_logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Load results from a single experiment directory.

    Expected structure:
        experiment_dir/
        ├── results.json      # Contains evaluation data
        └── results.txt       # Human-readable summary
    """
    if exp_logger is None:
        exp_logger = logging.getLogger("experiment_loader")

    # Try multiple common result file names
    result_files = ["results.json", "overfit_summary.json", "experiment_results.json"]
    experiment_results = None

    for result_file in result_files:
        results_path = os.path.join(experiment_dir, result_file)
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    experiment_results = json.load(f)
                exp_logger.info(f"Loaded experiment results from {results_path}")
                break
            except Exception as e:
                exp_logger.warning(f"Failed to load {results_path}: {e}")

    if experiment_results is None:
        exp_logger.warning(f"No results file found in: {experiment_dir}")
        return []

    parsed_results = []

    # Parse evaluation details
    if 'evaluation' in experiment_results and 'details' in experiment_results['evaluation']:
        for detail in experiment_results['evaluation']['details']:
            parsed_results.append({
                'experiment_type': experiment_type,
                'problem_name': detail.get('problem', ''),
                'solved': detail.get('solved', False),
                'time': detail.get('time', 0),
                'reward': detail.get('reward', 0),
                'difficulty': _extract_difficulty_from_problem_name(detail.get('problem', '')),
            })

    elif 'details' in experiment_results:
        for detail in experiment_results['details']:
            parsed_results.append({
                'experiment_type': experiment_type,
                'problem_name': detail.get('problem', ''),
                'solved': detail.get('solved', False),
                'time': detail.get('time', 0),
                'reward': detail.get('reward', 0),
                'difficulty': _extract_difficulty_from_problem_name(detail.get('problem', '')),
            })

    elif 'per_problem_stats' in experiment_results:
        for stats in experiment_results['per_problem_stats']:
            parsed_results.append({
                'experiment_type': experiment_type,
                'problem_name': stats.get('problem_name', ''),
                'solved': stats.get('solve_rate', 0) > 0,
                'time': 0,
                'reward': stats.get('avg_reward', 0),
                'difficulty': _extract_difficulty_from_problem_name(stats.get('problem_name', '')),
            })

    exp_logger.info(f"Parsed {len(parsed_results)} problem results")

    return parsed_results


def aggregate_experiment_results(
        experiment_directories: Dict[str, str],
        exp_logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Aggregate results from multiple experiments.

    Args:
        experiment_directories: Dict mapping experiment_type -> directory_path
                               E.g., {
                                   'overfit': 'overfit_experiment_results',
                                   'problem_gen': 'problem_generalization_results',
                                   'scale_gen': 'scale_generalization_results',
                                   'curriculum': 'curriculum_learning_results'
                               }

    Returns:
        Aggregated results dictionary with statistics
    """
    if exp_logger is None:
        exp_logger = logging.getLogger("experiment_aggregator")

    exp_logger.info("Aggregating experiment results...")

    all_results = []
    experiment_summaries = {}

    for exp_type, exp_dir in experiment_directories.items():
        if not os.path.exists(exp_dir):
            exp_logger.warning(f"Experiment directory not found: {exp_dir}")
            continue

        exp_logger.info(f"\nLoading {exp_type} from {exp_dir}...")

        results = load_experiment_results_from_directory(
            exp_dir,
            exp_type,
            exp_logger
        )

        if results:
            all_results.extend(results)

            solved_count = sum(1 for r in results if r['solved'])
            total_count = len(results)

            solved_times = [r['time'] for r in results if r['solved'] and r['time'] > 0]
            avg_time = np.mean(solved_times) if solved_times else 0

            all_rewards = [r['reward'] for r in results]
            avg_reward = np.mean(all_rewards) if all_rewards else 0

            experiment_summaries[exp_type] = {
                'total_problems': total_count,
                'solved': solved_count,
                'solve_rate': (solved_count / total_count * 100) if total_count > 0 else 0,
                'avg_time': float(avg_time) if not np.isnan(avg_time) else 0,
                'avg_reward': float(avg_reward) if not np.isnan(avg_reward) else 0,
            }

            exp_logger.info(f"  ✓ {exp_type}: {solved_count}/{total_count} solved "
                            f"({experiment_summaries[exp_type]['solve_rate']:.1f}%)")

    exp_logger.info(f"\n✅ Total results aggregated: {len(all_results)} problems")

    return {
        'all_results': all_results,
        'experiment_summaries': experiment_summaries,
        'timestamp': datetime.now().isoformat()
    }


def _extract_difficulty_from_problem_name(problem_name: str) -> str:
    """Extract difficulty level from problem name."""
    problem_lower = problem_name.lower()
    if 'small' in problem_lower:
        return 'small'
    elif 'medium' in problem_lower:
        return 'medium'
    elif 'large' in problem_lower or 'hard' in problem_lower:
        return 'large'
    else:
        return 'unknown'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    for directory in [
        "downward/gnn_output",
        "downward/fd_output",
        "logs",
        "models",
        "benchmarks",
        "tb_logs"
    ]:
        os.makedirs(directory, exist_ok=True)


def get_timestamp_str() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Format duration nicely."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def cleanup_signal_files() -> int:
    """Remove all signal files from communication directories."""
    from pathlib import Path
    import glob

    deleted = 0
    for directory in ["downward/fd_output", "downward/gnn_output"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for pattern in ["*.json", "*.tmp"]:
            for filepath in glob.glob(str(dir_path / pattern)):
                try:
                    os.remove(filepath)
                    deleted += 1
                except:
                    pass
    return deleted