#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOGGING MODULE - Enhanced for Analysis & Plotting
==================================================
Comprehensive logging with proper serialization and analysis support.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime
import time
import numpy as np


def _serialize_value(val: Any) -> Any:
    """Recursively serialize values for JSON compatibility."""
    if val is None:
        return None
    elif isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.bool_,)):
        return bool(val)
    elif isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    elif isinstance(val, (int, float, str, bool)):
        return val
    else:
        return str(val)


@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode - ENHANCED for analysis."""

    # Core identifiers
    episode: int
    problem_name: str
    timestamp: float = field(default_factory=lambda: time.time())

    # Primary metrics
    reward: float = 0.0
    h_star_preservation: float = 1.0
    is_solvable: bool = True

    # Episode info
    num_active_systems: int = 0
    eval_steps: int = 0
    total_reward: float = 0.0

    # Error tracking
    error: Optional[str] = None
    failure_type: Optional[str] = None

    # Performance metrics
    step_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    avg_inference_time_ms: float = 0.0

    # PPO training metrics
    policy_entropy: float = 0.0
    value_loss: float = 0.0
    gradient_norm: float = 0.0

    # Abstraction quality metrics
    graph_size_reduction_pct: float = 0.0
    dead_end_ratio: float = 0.0

    # Component rewards (5 components from reward function)
    component_h_preservation: float = 0.0
    component_transition_control: float = 0.0
    component_operator_projection: float = 0.0
    component_label_combinability: float = 0.0
    component_bonus_signals: float = 0.0

    # Detailed signal values
    h_star_ratio: float = 1.0
    transition_growth_ratio: float = 1.0
    transition_density: float = 0.0
    opp_score: float = 0.5
    label_combinability_score: float = 0.5
    causal_proximity: float = 0.0
    landmark_preservation: float = 0.5
    reachability_ratio: float = 1.0

    # Penalties applied
    penalty_solvability_loss: float = 0.0
    penalty_dead_end: float = 0.0

    # Per-step decision data (for detailed analysis)
    merge_decisions_per_step: List[Dict[str, Any]] = field(default_factory=list)
    merge_quality_scores: List[float] = field(default_factory=list)
    gnn_action_probabilities: List[float] = field(default_factory=list)
    selected_actions: List[int] = field(default_factory=list)

    # Additional metadata
    domain_name: str = ""
    problem_size: str = ""
    training_phase: str = "standard"  # or "curriculum_phase_1", etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict with proper numpy handling."""
        result = {}
        for key, value in asdict(self).items():
            result[key] = _serialize_value(value)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodeMetrics':
        """Create from dict with validation and defaults."""
        # Handle missing fields gracefully
        field_defaults = {
            'episode': 0,
            'problem_name': 'unknown',
            'timestamp': time.time(),
            'reward': 0.0,
            'h_star_preservation': 1.0,
            'is_solvable': True,
            'num_active_systems': 0,
            'eval_steps': 0,
            'total_reward': 0.0,
            'error': None,
            'failure_type': None,
            'step_time_ms': 0.0,
            'peak_memory_mb': 0.0,
            'avg_inference_time_ms': 0.0,
            'policy_entropy': 0.0,
            'value_loss': 0.0,
            'gradient_norm': 0.0,
            'graph_size_reduction_pct': 0.0,
            'dead_end_ratio': 0.0,
            'component_h_preservation': 0.0,
            'component_transition_control': 0.0,
            'component_operator_projection': 0.0,
            'component_label_combinability': 0.0,
            'component_bonus_signals': 0.0,
            'h_star_ratio': 1.0,
            'transition_growth_ratio': 1.0,
            'transition_density': 0.0,
            'opp_score': 0.5,
            'label_combinability_score': 0.5,
            'causal_proximity': 0.0,
            'landmark_preservation': 0.5,
            'reachability_ratio': 1.0,
            'penalty_solvability_loss': 0.0,
            'penalty_dead_end': 0.0,
            'merge_decisions_per_step': [],
            'merge_quality_scores': [],
            'gnn_action_probabilities': [],
            'selected_actions': [],
            'domain_name': '',
            'problem_size': '',
            'training_phase': 'standard',
        }

        # Merge with data, using defaults for missing
        merged = {**field_defaults, **data}

        # Handle legacy field names
        if 'steps' in data and 'eval_steps' not in data:
            merged['eval_steps'] = data['steps']

        # Filter to only valid fields
        valid_fields = set(field_defaults.keys())
        filtered = {k: v for k, v in merged.items() if k in valid_fields}

        return cls(**filtered)

    def get_summary_metrics(self) -> Dict[str, float]:
        """Get key metrics for summary tables."""
        return {
            'reward': self.reward,
            'h_star_preservation': self.h_star_preservation,
            'is_solvable': float(self.is_solvable),
            'eval_steps': float(self.eval_steps),
            'component_h': self.component_h_preservation,
            'component_trans': self.component_transition_control,
            'transition_growth': self.transition_growth_ratio,
            'opp_score': self.opp_score,
            'label_comb': self.label_combinability_score,
        }


@dataclass
class MergeDecisionTrace:
    """Trace a single GNN merge decision - ENHANCED."""

    # Identifiers
    step: int
    episode: int
    problem_name: str
    timestamp: float = field(default_factory=time.time)

    # Merge selection
    selected_merge_pair: Tuple[int, int] = (0, 0)
    gnn_action_index: int = 0
    num_candidates: int = 0

    # GNN outputs
    gnn_logits: np.ndarray = field(default_factory=lambda: np.array([]))
    gnn_action_probability: float = 0.0
    gnn_entropy: float = 0.0

    # Features used
    node_features_used: Dict[str, List[float]] = field(default_factory=dict)
    edge_features_used: np.ndarray = field(default_factory=lambda: np.array([]))

    # Reward signals from this step
    reward_signals: Dict[str, Any] = field(default_factory=dict)
    immediate_reward: float = 0.0

    # Quality metrics
    h_preservation: float = 1.0
    transition_growth: float = 1.0
    opp_score: float = 0.5
    label_combinability: float = 0.5

    # Merge quality assessment
    is_good_merge: bool = False
    is_bad_merge: bool = False
    merge_quality_category: str = "neutral"  # excellent, good, neutral, poor, bad
    merge_quality_score: float = 0.5  # 0-1 scale

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'step': int(self.step),
            'episode': int(self.episode),
            'problem_name': str(self.problem_name),
            'timestamp': float(self.timestamp),
            'selected_merge_pair': list(self.selected_merge_pair),
            'gnn_action_index': int(self.gnn_action_index),
            'num_candidates': int(self.num_candidates),
            'gnn_logits': self.gnn_logits.tolist() if isinstance(self.gnn_logits, np.ndarray) else list(
                self.gnn_logits),
            'gnn_action_probability': float(self.gnn_action_probability),
            'gnn_entropy': float(self.gnn_entropy),
            'node_features_used': _serialize_value(self.node_features_used),
            'edge_features_used': self.edge_features_used.tolist() if isinstance(self.edge_features_used,
                                                                                 np.ndarray) else list(
                self.edge_features_used),
            'reward_signals': _serialize_value(self.reward_signals),
            'immediate_reward': float(self.immediate_reward),
            'h_preservation': float(self.h_preservation),
            'transition_growth': float(self.transition_growth),
            'opp_score': float(self.opp_score),
            'label_combinability': float(self.label_combinability),
            'is_good_merge': bool(self.is_good_merge),
            'is_bad_merge': bool(self.is_bad_merge),
            'merge_quality_category': str(self.merge_quality_category),
            'merge_quality_score': float(self.merge_quality_score),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MergeDecisionTrace':
        """Create from dict with validation."""
        return cls(
            step=data.get('step', 0),
            episode=data.get('episode', 0),
            problem_name=data.get('problem_name', ''),
            timestamp=data.get('timestamp', time.time()),
            selected_merge_pair=tuple(data.get('selected_merge_pair', (0, 0))),
            gnn_action_index=data.get('gnn_action_index', 0),
            num_candidates=data.get('num_candidates', 0),
            gnn_logits=np.array(data.get('gnn_logits', [])),
            gnn_action_probability=data.get('gnn_action_probability', 0.0),
            gnn_entropy=data.get('gnn_entropy', 0.0),
            node_features_used=data.get('node_features_used', {}),
            edge_features_used=np.array(data.get('edge_features_used', [])),
            reward_signals=data.get('reward_signals', {}),
            immediate_reward=data.get('immediate_reward', 0.0),
            h_preservation=data.get('h_preservation', 1.0),
            transition_growth=data.get('transition_growth', 1.0),
            opp_score=data.get('opp_score', 0.5),
            label_combinability=data.get('label_combinability', 0.5),
            is_good_merge=data.get('is_good_merge', False),
            is_bad_merge=data.get('is_bad_merge', False),
            merge_quality_category=data.get('merge_quality_category', 'neutral'),
            merge_quality_score=data.get('merge_quality_score', 0.5),
        )


@dataclass
class TrainingSummaryStats:
    """Aggregated statistics for a training run - for summary reports."""

    total_episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0

    # Reward statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    median_reward: float = 0.0

    # H* preservation statistics
    mean_h_preservation: float = 1.0
    std_h_preservation: float = 0.0
    min_h_preservation: float = 1.0

    # Component statistics (means)
    mean_component_h: float = 0.0
    mean_component_trans: float = 0.0
    mean_component_opp: float = 0.0
    mean_component_label: float = 0.0
    mean_component_bonus: float = 0.0

    # Per-problem breakdown
    per_problem_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Learning progress (window averages)
    reward_over_time: List[float] = field(default_factory=list)
    h_preservation_over_time: List[float] = field(default_factory=list)

    # Timing
    total_training_time_sec: float = 0.0
    avg_episode_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_episode_log(cls, episode_log: List[EpisodeMetrics], window_size: int = 50) -> 'TrainingSummaryStats':
        """Compute summary statistics from episode log."""
        if not episode_log:
            return cls()

        successful = [m for m in episode_log if m.error is None]

        rewards = [m.reward for m in successful]
        h_preservations = [m.h_star_preservation for m in successful]

        # Per-problem statistics
        per_problem = defaultdict(list)
        for m in successful:
            per_problem[m.problem_name].append(m)

        per_problem_stats = {}
        for problem_name, metrics_list in per_problem.items():
            rewards_p = [m.reward for m in metrics_list]
            h_pres_p = [m.h_star_preservation for m in metrics_list]
            per_problem_stats[problem_name] = {
                'count': len(metrics_list),
                'mean_reward': float(np.mean(rewards_p)) if rewards_p else 0.0,
                'std_reward': float(np.std(rewards_p)) if rewards_p else 0.0,
                'mean_h_preservation': float(np.mean(h_pres_p)) if h_pres_p else 1.0,
            }

        # Rolling averages for learning curves
        reward_over_time = []
        h_over_time = []
        for i in range(0, len(successful), window_size):
            window = successful[i:i + window_size]
            if window:
                reward_over_time.append(float(np.mean([m.reward for m in window])))
                h_over_time.append(float(np.mean([m.h_star_preservation for m in window])))

        # Component means
        comp_h = [m.component_h_preservation for m in successful]
        comp_trans = [m.component_transition_control for m in successful]
        comp_opp = [m.component_operator_projection for m in successful]
        comp_label = [m.component_label_combinability for m in successful]
        comp_bonus = [m.component_bonus_signals for m in successful]

        return cls(
            total_episodes=len(episode_log),
            successful_episodes=len(successful),
            failed_episodes=len(episode_log) - len(successful),
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            min_reward=float(np.min(rewards)) if rewards else 0.0,
            max_reward=float(np.max(rewards)) if rewards else 0.0,
            median_reward=float(np.median(rewards)) if rewards else 0.0,
            mean_h_preservation=float(np.mean(h_preservations)) if h_preservations else 1.0,
            std_h_preservation=float(np.std(h_preservations)) if h_preservations else 0.0,
            min_h_preservation=float(np.min(h_preservations)) if h_preservations else 1.0,
            mean_component_h=float(np.mean(comp_h)) if comp_h else 0.0,
            mean_component_trans=float(np.mean(comp_trans)) if comp_trans else 0.0,
            mean_component_opp=float(np.mean(comp_opp)) if comp_opp else 0.0,
            mean_component_label=float(np.mean(comp_label)) if comp_label else 0.0,
            mean_component_bonus=float(np.mean(comp_bonus)) if comp_bonus else 0.0,
            per_problem_stats=per_problem_stats,
            reward_over_time=reward_over_time,
            h_preservation_over_time=h_over_time,
        )


class EnhancedSilentTrainingLogger:
    """Enhanced logging system for training analysis and plotting."""

    def __init__(self, log_dir: str, experiment_id: str, verbose: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.verbose = verbose
        self.start_time = time.time()

        # File handles
        self.log_file = self.log_dir / "training.log"
        self.events_file = self.log_dir / "training_events.jsonl"
        self.file_handler = None
        self.events_handler = None

        self._open_files()

        # In-memory tracking for aggregation
        self.problem_episode_counts = defaultdict(int)
        self.problem_failure_counts = defaultdict(int)
        self.problem_rewards = defaultdict(list)
        self.problem_h_preservations = defaultdict(list)

        # Rolling window for real-time stats
        self.recent_rewards = []
        self.recent_h_preservations = []
        self.window_size = 50

        # Event counter for debugging
        self.event_count = 0

    def _open_files(self):
        """Open log files for writing."""
        # Main log file (human readable)
        mode = 'a' if self.log_file.exists() else 'w'
        self.file_handler = open(self.log_file, mode, encoding='utf-8', buffering=1)

        # Events file (JSONL for analysis)
        mode = 'a' if self.events_file.exists() else 'w'
        self.events_handler = open(self.events_file, mode, encoding='utf-8', buffering=1)

        # Write header if new file
        if mode == 'w':
            self.file_handler.write(f"# Training Log - {self.experiment_id}\n")
            self.file_handler.write(f"# Started: {datetime.now().isoformat()}\n")
            self.file_handler.write("=" * 100 + "\n\n")

    def _emit_event(self, event_type: str, **kwargs) -> None:
        """Emit structured event to both log files."""
        self.event_count += 1
        timestamp = datetime.now().isoformat()

        event_data = {
            'event_id': self.event_count,
            'event_type': event_type,
            'timestamp': timestamp,
            'experiment_id': self.experiment_id,
            'elapsed_sec': time.time() - self.start_time,
            **{k: _serialize_value(v) for k, v in kwargs.items()}
        }

        try:
            # JSONL for analysis
            if self.events_handler and not self.events_handler.closed:
                event_json = json.dumps(event_data, ensure_ascii=False)
                self.events_handler.write(event_json + "\n")
                self.events_handler.flush()

            # Human-readable log
            if self.file_handler and not self.file_handler.closed:
                self._write_human_readable(event_type, event_data)

        except Exception as e:
            print(f"[LOGGER ERROR] Failed to emit event: {e}", file=sys.stderr)

    def _write_human_readable(self, event_type: str, data: Dict) -> None:
        """Write human-readable version to log file."""
        timestamp = data.get('timestamp', '')[:19]  # Truncate microseconds

        if event_type == 'training_started':
            self.file_handler.write(f"\n[{timestamp}] 🚀 TRAINING STARTED\n")
            self.file_handler.write(f"  Episodes: {data.get('num_episodes')}\n")
            self.file_handler.write(f"  Problems: {data.get('num_problems')}\n")
            self.file_handler.write(f"  Seed: {data.get('seed')}\n")

        elif event_type == 'episode_completed':
            episode = data.get('episode', 0)
            problem = data.get('problem_name', '')
            reward = data.get('reward', 0)
            h_pres = data.get('h_preservation', 1.0)
            error = data.get('error')

            if error:
                self.file_handler.write(f"[{timestamp}] ❌ Episode {episode} ({problem}): FAILED - {error}\n")
            else:
                status = "✅" if reward > 0.2 else "⚠️" if reward > 0 else "🔴"
                self.file_handler.write(
                    f"[{timestamp}] {status} Episode {episode} ({problem}): "
                    f"R={reward:+.4f} h*={h_pres:.4f}\n"
                )

        elif event_type == 'checkpoint_saved':
            self.file_handler.write(f"[{timestamp}] 💾 Checkpoint: {data.get('path')}\n")

        elif event_type == 'training_completed':
            self.file_handler.write(f"\n[{timestamp}] 🏁 TRAINING COMPLETED\n")
            self.file_handler.write(f"  Total steps: {data.get('total_steps')}\n")
            self.file_handler.write(f"  Total reward: {data.get('total_reward'):.4f}\n")

        elif event_type == 'reward_component_breakdown':
            # Only write detailed breakdowns in verbose mode
            if self.verbose:
                self.file_handler.write(f"[{timestamp}] 📊 Reward breakdown for episode {data.get('episode')}\n")

        self.file_handler.flush()

    def log_training_started(self, num_episodes: int, num_problems: int, seed: int):
        """Log training start event."""
        self._emit_event(
            'training_started',
            num_episodes=num_episodes,
            num_problems=num_problems,
            seed=seed
        )

    def log_training_completed(self, total_steps: int, total_reward: float):
        """Log training completion event."""
        self._emit_event(
            'training_completed',
            total_steps=total_steps,
            total_reward=total_reward,
            total_time_sec=time.time() - self.start_time,
            final_stats={
                'total_episodes': sum(self.problem_episode_counts.values()),
                'total_failures': sum(self.problem_failure_counts.values()),
                'avg_reward': float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0,
            }
        )

    def log_episode_started(self, episode: int, problem_name: str):
        """Log episode start event."""
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
            failure_type: Optional[str] = None,
            metrics: Optional[Dict] = None,
            component_breakdown: Optional[Dict] = None,
    ):
        """Log episode completion with full metrics."""
        # Update tracking
        self.problem_episode_counts[problem_name] += 1
        if error:
            self.problem_failure_counts[problem_name] += 1
        else:
            self.problem_rewards[problem_name].append(reward)
            self.problem_h_preservations[problem_name].append(h_preservation)

            # Update rolling windows
            self.recent_rewards.append(reward)
            self.recent_h_preservations.append(h_preservation)
            if len(self.recent_rewards) > self.window_size:
                self.recent_rewards.pop(0)
            if len(self.recent_h_preservations) > self.window_size:
                self.recent_h_preservations.pop(0)

        # Compute rolling stats
        rolling_reward_avg = float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
        rolling_h_avg = float(np.mean(self.recent_h_preservations)) if self.recent_h_preservations else 1.0

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
            metrics=metrics or {},
            component_breakdown=component_breakdown or {},
            rolling_reward_avg=rolling_reward_avg,
            rolling_h_avg=rolling_h_avg,
            total_episodes_so_far=sum(self.problem_episode_counts.values()),
        )

    def log_reward_component_breakdown(
            self,
            episode: int,
            problem_name: str,
            step: int,
            component_breakdown: Dict[str, Any],
    ):
        """Log detailed reward component breakdown."""
        self._emit_event(
            'reward_component_breakdown',
            episode=episode,
            problem_name=problem_name,
            step=step,
            components=component_breakdown.get('components', {}),
            component_details=component_breakdown.get('component_details', {}),
            catastrophic_penalties=component_breakdown.get('catastrophic_penalties', {}),
            signal_validity=component_breakdown.get('signal_validity', {}),
            final_reward=component_breakdown.get('final_reward', 0.0),
        )

    def log_merge_decision(
            self,
            episode: int,
            problem_name: str,
            step: int,
            decision_trace: MergeDecisionTrace,
    ):
        """Log individual merge decision."""
        self._emit_event(
            'merge_decision',
            episode=episode,
            problem_name=problem_name,
            step=step,
            decision=decision_trace.to_dict(),
        )

    def log_adaptive_sampling_update(
            self,
            episode: int,
            per_problem_scores: Dict[str, float],
            per_problem_coverage: Dict[str, float],
    ):
        """Log adaptive sampling state."""
        self._emit_event(
            'adaptive_sampling_update',
            episode=episode,
            per_problem_scores=per_problem_scores,
            per_problem_coverage=per_problem_coverage,
        )

    def log_checkpoint_saved(
            self,
            step: int,
            path: str,
            reward: float,
            problem_name: str = '',
            domain_name: str = '',
    ):
        """Log checkpoint save event."""
        self._emit_event(
            'checkpoint_saved',
            step=step,
            path=path,
            reward=reward,
            problem_name=problem_name,
            domain_name=domain_name,
            current_stats={
                'recent_avg_reward': float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0,
                'recent_avg_h': float(np.mean(self.recent_h_preservations)) if self.recent_h_preservations else 1.0,
            }
        )

    def log_failure(
            self,
            episode: int,
            problem_name: str,
            failure_type: str,
            error_message: str,
            context: Optional[Dict] = None,
    ):
        """Log failure event."""
        self._emit_event(
            'failure',
            episode=episode,
            problem_name=problem_name,
            failure_type=failure_type,
            error_message=error_message,
            context=context or {}
        )

    def log_problem_coverage_report(
            self,
            total_episodes: int,
            problem_names: List[str],
    ):
        """Log problem coverage statistics."""
        coverage_data = {}
        for problem_name in problem_names:
            episode_count = self.problem_episode_counts[problem_name]
            failure_count = self.problem_failure_counts[problem_name]
            rewards = self.problem_rewards[problem_name]
            h_preservations = self.problem_h_preservations[problem_name]

            coverage_pct = (episode_count / total_episodes * 100) if total_episodes > 0 else 0
            failure_rate = (failure_count / episode_count * 100) if episode_count > 0 else 0

            coverage_data[problem_name] = {
                'episodes': episode_count,
                'coverage_percent': round(coverage_pct, 2),
                'failures': failure_count,
                'failure_rate': round(failure_rate, 2),
                'avg_reward': round(float(np.mean(rewards)), 4) if rewards else 0.0,
                'std_reward': round(float(np.std(rewards)), 4) if rewards else 0.0,
                'avg_h_preservation': round(float(np.mean(h_preservations)), 4) if h_preservations else 1.0,
            }

        self._emit_event(
            'problem_coverage_report',
            total_episodes=total_episodes,
            coverage_data=coverage_data,
            all_problems_trained=all(coverage_data[p]['episodes'] > 0 for p in problem_names),
            minimum_coverage=min((coverage_data[p]['coverage_percent'] for p in problem_names), default=0),
            maximum_coverage=max((coverage_data[p]['coverage_percent'] for p in problem_names), default=0),
        )

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get current summary statistics."""
        all_rewards = []
        all_h_pres = []
        for problem_name in self.problem_rewards:
            all_rewards.extend(self.problem_rewards[problem_name])
            all_h_pres.extend(self.problem_h_preservations[problem_name])

        return {
            'total_episodes': sum(self.problem_episode_counts.values()),
            'total_failures': sum(self.problem_failure_counts.values()),
            'num_problems': len(self.problem_episode_counts),
            'avg_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'std_reward': float(np.std(all_rewards)) if all_rewards else 0.0,
            'avg_h_preservation': float(np.mean(all_h_pres)) if all_h_pres else 1.0,
            'recent_avg_reward': float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0,
            'recent_avg_h': float(np.mean(self.recent_h_preservations)) if self.recent_h_preservations else 1.0,
        }

    def close(self):
        """Finalize log files."""
        # Write final summary
        if self.file_handler and not self.file_handler.closed:
            self.file_handler.write("\n" + "=" * 100 + "\n")
            self.file_handler.write("FINAL SUMMARY\n")
            self.file_handler.write("=" * 100 + "\n")

            stats = self.get_summary_stats()
            for key, value in stats.items():
                self.file_handler.write(f"  {key}: {value}\n")

            self.file_handler.write("=" * 100 + "\n")

            try:
                self.file_handler.close()
            except:
                pass
            self.file_handler = None

        if self.events_handler and not self.events_handler.closed:
            try:
                self.events_handler.close()
            except:
                pass
            self.events_handler = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ============================================================================
# UTILITY FUNCTIONS FOR LOADING AND ANALYZING LOGS
# ============================================================================

def load_training_log(log_path: Union[str, Path]) -> List[EpisodeMetrics]:
    """
    Load training log from JSONL file.

    Args:
        log_path: Path to training_log.jsonl

    Returns:
        List of EpisodeMetrics objects
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    metrics_list = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Handle missing episode field
                if 'episode' not in data:
                    data['episode'] = line_num

                metrics = EpisodeMetrics.from_dict(data)
                metrics_list.append(metrics)

            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue

    return metrics_list


def load_training_events(events_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load training events from JSONL file.

    Args:
        events_path: Path to training_events.jsonl

    Returns:
        List of event dicts
    """
    events_path = Path(events_path)
    if not events_path.exists():
        return []

    events = []

    with open(events_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                continue

    return events


def get_episode_completed_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter events to only episode_completed events."""
    return [e for e in events if e.get('event_type') == 'episode_completed']


def get_reward_breakdown_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter events to only reward_component_breakdown events."""
    return [e for e in events if e.get('event_type') == 'reward_component_breakdown']


def compute_summary_from_log(episode_log: List[EpisodeMetrics]) -> TrainingSummaryStats:
    """Compute summary statistics from episode log."""
    return TrainingSummaryStats.from_episode_log(episode_log)