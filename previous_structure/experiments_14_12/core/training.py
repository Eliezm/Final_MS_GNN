#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAINING MODULE - Extracted from experiment_1_problem_overfit.py
Handles GNN training with full validation and monitoring.
"""

import sys
import json
import random
import re
import traceback
import subprocess
import warnings
import uuid
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
from datetime import datetime
import time

import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from shared_experiment_utils import DEFAULT_REWARD_WEIGHTS, cleanup_signal_files
from experiments.core.logging import EnhancedSilentTrainingLogger, EpisodeMetrics, MergeDecisionTrace

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# REPRODUCIBILITY
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
# FILE I/O
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
# ADAPTIVE SAMPLING
# ============================================================================

class EnhancedAdaptiveSampler:
    """Adaptive Sampling with MINIMUM TRAINING GUARANTEE."""

    def __init__(
            self,
            problem_names: List[str],
            update_interval: int = 50,
            sweep_interval: int = 100,
            alpha: float = 2.0,
            min_episodes_per_problem: int = 10,
            seed: int = 42,
    ):
        self.problem_names = problem_names
        self.update_interval = update_interval
        self.sweep_interval = sweep_interval
        self.alpha = alpha
        self.min_episodes_per_problem = min_episodes_per_problem

        self.rng = np.random.RandomState(seed)

        self.per_problem_episodes = {name: 0 for name in problem_names}
        self.per_problem_scores = {name: 0.5 for name in problem_names}
        self.recent_rewards = defaultdict(list)

    def update_scores_from_log(
            self,
            episode_log: List[EpisodeMetrics],
            window_size: int = 5
    ) -> None:
        """Update per-problem scores based on recent performance."""
        if len(episode_log) == 0:
            return

        recent_by_problem = defaultdict(list)
        for metrics in episode_log[-window_size * len(self.problem_names):]:
            if metrics.error is None:
                recent_by_problem[metrics.problem_name].append(metrics.reward)
                self.per_problem_episodes[metrics.problem_name] = \
                    sum(1 for m in episode_log if m.problem_name == metrics.problem_name)

        for problem_name, rewards in recent_by_problem.items():
            if rewards and self.per_problem_episodes[problem_name] >= self.min_episodes_per_problem:
                avg_reward = np.mean(rewards)
                self.per_problem_scores[problem_name] = avg_reward

    def get_weights(self) -> np.ndarray:
        """Compute sampling weights with MINIMUM GUARANTEE."""
        scores = np.array([
            self.per_problem_scores.get(name, 0.5)
            for name in self.problem_names
        ])

        episode_counts = np.array([
            self.per_problem_episodes.get(name, 0)
            for name in self.problem_names
        ])

        weights = np.zeros_like(scores)

        below_min_mask = episode_counts < self.min_episodes_per_problem
        weights[below_min_mask] = 1.0

        above_min_mask = ~below_min_mask
        exp_weights = np.exp(-self.alpha * scores[above_min_mask])

        if np.any(np.isnan(exp_weights)) or np.any(np.isinf(exp_weights)):
            exp_weights = np.ones_like(exp_weights)

        weights[above_min_mask] = exp_weights

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
# REWARD SIGNAL VALIDATION
# ============================================================================

def validate_reward_signals(reward_signals: Dict) -> Tuple[bool, Optional[str]]:
    """Validate reward signals for integrity."""
    required_fields = ['h_star_before', 'h_star_after', 'h_star_preservation', 'is_solvable']
    for field in required_fields:
        if field not in reward_signals:
            return False, f"Missing required field: {field}"

    h_before = float(reward_signals.get('h_star_before', 0))
    h_after = float(reward_signals.get('h_star_after', 0))
    h_pres = float(reward_signals.get('h_star_preservation', 1.0))

    if np.isinf(h_before) or np.isinf(h_after):
        if not np.isinf(h_pres):
            return False, f"h* is infinite but preservation is finite: {h_pres}"

    is_solvable = bool(reward_signals.get('is_solvable', True))
    dead_end_ratio = float(reward_signals.get('dead_end_ratio', 0.0))

    if is_solvable and dead_end_ratio > 0.9:
        return False, f"Solvable claim conflicts with {dead_end_ratio:.1%} dead-ends"

    if not is_solvable and dead_end_ratio < 0.5:
        return False, f"Unsolvable claim but only {dead_end_ratio:.1%} dead-ends"

    return True, None


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
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    summary_path = output_path / "experiment_summary.json"
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


# ============================================================================
# TRAINER CLASS
# ============================================================================

class GNNTrainer:
    """
    Trains GNN on problem set with RIGOROUS VALIDATION.
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
            min_episodes_per_problem: int = 10,
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

        self.logger = EnhancedSilentTrainingLogger(
            str(self.output_dir),
            experiment_id=self.experiment_id,
            verbose=False
        )

        self.resource_monitor = ResourceMonitor()
        self.episode_reward_signals = {}

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
            from src.models.gnn_policy import GNNPolicy
            self.GNNPolicy = GNNPolicy
        except ImportError as e:
            raise ImportError(f"Failed to import GNNPolicy: {e}")

        try:
            from src.environments.thin_merge_env import ThinMergeEnv
            self.ThinMergeEnv = ThinMergeEnv
        except ImportError as e:
            raise ImportError(f"Failed to import ThinMergeEnv: {e}")

    def get_experiment_id(self) -> str:
        """Return the experiment_id for logging."""
        return self.experiment_id

    def _create_env(self, domain_file: str, problem_file: str, seed: int):
        """Create environment with error handling."""
        downward_path = PROJECT_ROOT / "downward"

        try:
            env = self.ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
                reward_weights=self.reward_weights,
                debug=False,
                seed=seed,
                downward_dir=str(downward_path)
            )
        except TypeError:
            env = self.ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
                reward_weights=self.reward_weights,
                debug=False,
                downward_dir=str(downward_path)
            )

        return self.Monitor(env)

    def _problem_cycle_generator(self, start_episode: int, num_episodes: int):
        """Generate problems using enhanced adaptive sampling."""
        for episode in range(start_episode, num_episodes):
            if episode % self.sampler.update_interval == 0 and episode > 0:
                self.sampler.update_scores_from_log(self.episode_log)

                coverage = self.sampler.get_coverage_stats()
                self.logger.log_adaptive_sampling_update(
                    episode,
                    self.sampler.per_problem_scores,
                    coverage
                )

            idx = self.sampler.sample_problem_idx()
            yield episode, idx

    def run_training(
            self,
            num_episodes: int,
            timesteps_per_episode: int = 50,
            resume_from: Optional[str] = None,
    ) -> Optional[str]:
        """Train with FULL VALIDATION AND MONITORING."""
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
                        print(f"   ✓ Loaded {len(prev_log)} previous episodes ({failed_count} failed)")

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

                component_decisions_this_episode = []

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
                        tb_log_name=f"episode_{episode}",
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

                    component_rewards = []
                    component_h_pres = []
                    component_trans = []
                    component_opp = []
                    component_label = []
                    component_bonus = []
                    h_star_ratios = []
                    transition_growths = []
                    opp_scores = []
                    label_scores = []
                    reachability_ratios = []
                    dead_end_penalties = []
                    solvability_penalties = []
                    reward_signals_per_step = []

                    for step in range(self.max_merges):
                        try:
                            action, _states = model.predict(obs, deterministic=True)

                            gnn_logits = None
                            gnn_action_prob = 0.0

                            try:
                                from stable_baselines3.common.policies import ActorCriticPolicy
                                policy = model.policy
                                obs_tensor = policy.obs_to_tensor(obs)[0]

                                with torch.no_grad():
                                    dist = policy.get_distribution(obs_tensor)
                                    if hasattr(dist, 'logits'):
                                        gnn_logits = dist.logits.cpu().numpy()[0]
                                        gnn_action_prob = float(
                                            torch.softmax(dist.logits, dim=-1)[0, int(action)].item()
                                        )
                                    elif hasattr(dist, 'probs'):
                                        gnn_action_prob = float(dist.probs[0, int(action)].item())
                            except Exception:
                                pass

                            obs, reward, done, truncated, info = env.step(int(action))
                            episode_reward += reward
                            eval_steps += 1

                            reward_signals = info.get('reward_signals', {})
                            merge_pair = info.get('merge_pair', (0, 0))

                            is_valid, error_msg = validate_reward_signals(reward_signals)
                            if not is_valid:
                                pbar.write(f"⚠️  Episode {episode}: Signal validation failed: {error_msg}")
                                self.logger.log_failure(episode, problem_name, 'signal_validation_failure', error_msg)

                            h_pres = reward_signals.get('h_star_preservation', 1.0)
                            trans_growth = reward_signals.get('growth_ratio', 1.0)
                            opp_score = reward_signals.get('opp_score', 0.5)
                            label_comb = reward_signals.get('label_combinability_score', 0.5)

                            is_good = (h_pres > 0.8) and (trans_growth < 2.0)
                            is_bad = (h_pres < 0.7) or (trans_growth > 5.0)

                            if is_bad:
                                quality_category = 'bad'
                            elif h_pres < 0.8:
                                quality_category = 'poor'
                            elif trans_growth > 3.0:
                                quality_category = 'moderate'
                            elif is_good:
                                quality_category = 'excellent'
                            else:
                                quality_category = 'good'

                            decision_trace = MergeDecisionTrace(
                                step=step,
                                episode=episode,
                                problem_name=problem_name,
                                selected_merge_pair=merge_pair,
                                gnn_action_index=int(action),
                                gnn_logits=gnn_logits if gnn_logits is not None else np.array([]),
                                gnn_action_probability=gnn_action_prob,
                                node_features_used={},
                                edge_features_used=obs.get('edge_features', np.array([])) if isinstance(obs,
                                                                                                        dict) else np.array(
                                    []),
                                reward_signals=reward_signals,
                                immediate_reward=float(reward),
                                h_preservation=float(h_pres),
                                transition_growth=float(trans_growth),
                                opp_score=float(opp_score),
                                label_combinability=float(label_comb),
                                is_good_merge=is_good,
                                is_bad_merge=is_bad,
                                merge_quality_category=quality_category,
                            )

                            self.logger.log_merge_decision(
                                episode=episode,
                                problem_name=problem_name,
                                step=step,
                                decision_trace=decision_trace,
                            )

                            component_decisions_this_episode.append(decision_trace)

                            from src.rewards.reward_function_enhanced import create_enhanced_reward_function
                            reward_func = create_enhanced_reward_function(debug=False)

                            raw_obs = {
                                'reward_signals': reward_signals,
                                'edge_features': None,
                            }

                            component_breakdown = reward_func.compute_reward_with_breakdown(raw_obs)

                            component_rewards.append(component_breakdown['final_reward'])
                            component_h_pres.append(component_breakdown['components']['h_preservation'])
                            component_trans.append(component_breakdown['components']['transition_control'])
                            component_opp.append(component_breakdown['components']['operator_projection'])
                            component_label.append(component_breakdown['components']['label_combinability'])
                            component_bonus.append(component_breakdown['components']['bonus_signals'])

                            h_star_ratios.append(component_breakdown['component_details']['h_star_preservation'])
                            transition_growths.append(
                                component_breakdown['component_details']['transition_growth_ratio'])
                            opp_scores.append(component_breakdown['component_details']['opp_score'])
                            label_scores.append(component_breakdown['component_details']['label_combinability'])
                            reachability_ratios.append(component_breakdown['component_details']['reachability_ratio'])
                            dead_end_penalties.append(
                                component_breakdown['catastrophic_penalties'].get('dead_end_penalty', 0.0))
                            solvability_penalties.append(
                                component_breakdown['catastrophic_penalties'].get('solvability_loss', 0.0))

                            reward_signals_per_step.append({
                                'step': step,
                                'reward': reward,
                                'opp_score': component_breakdown['component_details']['opp_score'],
                                'label_combinability': component_breakdown['component_details']['label_combinability'],
                                'h_star_preservation': component_breakdown['component_details']['h_star_preservation'],
                                'transition_growth': component_breakdown['component_details'][
                                    'transition_growth_ratio'],
                                'reachability_ratio': component_breakdown['component_details']['reachability_ratio'],
                                'dead_end_ratio': reward_signals.get('dead_end_ratio', 0.0),
                                'is_solvable': reward_signals.get('is_solvable', True),
                                'gnn_action_probability': gnn_action_prob,
                                'merge_quality_category': quality_category,
                            })

                            self.logger.log_reward_component_breakdown(
                                episode=episode,
                                problem_name=problem_name,
                                step=step,
                                component_breakdown=component_breakdown,
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

                    step_time_ms = self.resource_monitor.get_elapsed_ms() / max(1, eval_steps)
                    peak_memory_mb = self.resource_monitor.get_peak_memory_mb()

                    component_summary = {
                        'avg_h_preservation': float(np.mean(component_h_pres)) if component_h_pres else 0.0,
                        'avg_transition_control': float(np.mean(component_trans)) if component_trans else 0.0,
                        'avg_operator_projection': float(np.mean(component_opp)) if component_opp else 0.0,
                        'avg_label_combinability': float(np.mean(component_label)) if component_label else 0.0,
                        'avg_bonus_signals': float(np.mean(component_bonus)) if component_bonus else 0.0,
                        'avg_h_star_ratio': float(np.mean(h_star_ratios)) if h_star_ratios else 1.0,
                        'avg_transition_growth': float(np.mean(transition_growths)) if transition_growths else 1.0,
                        'avg_opp_score': float(np.mean(opp_scores)) if opp_scores else 0.5,
                        'avg_label_score': float(np.mean(label_scores)) if label_scores else 0.5,
                        'min_reachability': float(np.min(reachability_ratios)) if reachability_ratios else 1.0,
                        'max_dead_end_penalty': float(np.max(dead_end_penalties)) if dead_end_penalties else 0.0,
                        'max_solvability_penalty': float(
                            np.max(solvability_penalties)) if solvability_penalties else 0.0,
                    }

                    decision_traces_dicts = []
                    for decision_trace in component_decisions_this_episode:
                        decision_traces_dicts.append(decision_trace.to_dict())

                    if not hasattr(self, 'episode_reward_signals'):
                        self.episode_reward_signals = {}
                    self.episode_reward_signals[episode] = {
                        'problem_name': problem_name,
                        'episode_reward': episode_reward,
                        'reward_signals_per_step': reward_signals_per_step,
                        'component_summary': component_summary,
                    }

                    quality_to_score = {
                        'excellent': 1.0,
                        'good': 0.75,
                        'moderate': 0.5,
                        'poor': 0.25,
                        'bad': 0.0,
                    }
                    merge_quality_scores = [
                        quality_to_score.get(d.get('merge_quality_category', 'moderate'), 0.5)
                        for d in decision_traces_dicts
                    ]

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
                        component_h_preservation=component_summary['avg_h_preservation'],
                        component_transition_control=component_summary['avg_transition_control'],
                        component_operator_projection=component_summary['avg_operator_projection'],
                        component_label_combinability=component_summary['avg_label_combinability'],
                        component_bonus_signals=component_summary['avg_bonus_signals'],
                        h_star_ratio=component_summary['avg_h_star_ratio'],
                        transition_growth_ratio=component_summary['avg_transition_growth'],
                        opp_score=component_summary['avg_opp_score'],
                        label_combinability_score=component_summary['avg_label_score'],
                        reachability_ratio=component_summary['min_reachability'],
                        penalty_dead_end=component_summary['max_dead_end_penalty'],
                        penalty_solvability_loss=component_summary['max_solvability_penalty'],
                        merge_decisions_per_step=decision_traces_dicts,
                        merge_quality_scores=merge_quality_scores,
                        gnn_action_probabilities=[float(d.get('gnn_action_probability', 0.5)) for d in
                                                  decision_traces_dicts],
                        selected_actions=[int(d.get('gnn_action_index', 0)) for d in decision_traces_dicts],
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
                        },
                        component_breakdown=component_summary,
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

            self.logger.log_problem_coverage_report(
                total_episodes=len(self.episode_log),
                problem_names=self.problem_names
            )

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