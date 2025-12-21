#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOGGING MODULE - Enhanced logging with temporal resolution
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime
import time
import numpy as np


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
    failure_type: Optional[str] = None

    step_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    avg_inference_time_ms: float = 0.0
    policy_entropy: float = 0.0
    value_loss: float = 0.0
    gradient_norm: float = 0.0
    graph_size_reduction_pct: float = 0.0
    dead_end_ratio: float = 0.0

    component_h_preservation: float = 0.0
    component_transition_control: float = 0.0
    component_operator_projection: float = 0.0
    component_label_combinability: float = 0.0
    component_bonus_signals: float = 0.0

    h_star_ratio: float = 1.0
    transition_growth_ratio: float = 1.0
    transition_density: float = 0.0
    opp_score: float = 0.5
    label_combinability_score: float = 0.5
    causal_proximity: float = 0.0
    landmark_preservation: float = 0.5
    reachability_ratio: float = 1.0

    penalty_solvability_loss: float = 0.0
    penalty_dead_end: float = 0.0

    merge_decisions_per_step: List[Dict[str, Any]] = field(default_factory=list)
    merge_quality_scores: List[float] = field(default_factory=list)
    gnn_action_probabilities: List[float] = field(default_factory=list)
    selected_actions: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['merge_decisions_per_step'] = [
            {k: (list(v) if isinstance(v, np.ndarray) else v)
             for k, v in decision.items()}
            for decision in self.merge_decisions_per_step
        ]
        return d


@dataclass
class MergeDecisionTrace:
    """Trace a single GNN merge decision."""
    step: int
    episode: int
    problem_name: str

    selected_merge_pair: Tuple[int, int]
    gnn_action_index: int

    gnn_logits: np.ndarray
    gnn_action_probability: float

    node_features_used: Dict[str, List[float]]
    edge_features_used: np.ndarray

    reward_signals: Dict[str, Any]
    immediate_reward: float

    h_preservation: float
    transition_growth: float
    opp_score: float
    label_combinability: float

    is_good_merge: bool
    is_bad_merge: bool
    merge_quality_category: str

    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'episode': self.episode,
            'problem_name': self.problem_name,
            'selected_merge_pair': self.selected_merge_pair,
            'gnn_action_index': self.gnn_action_index,
            'gnn_logits': self.gnn_logits.tolist() if isinstance(self.gnn_logits, np.ndarray) else self.gnn_logits,
            'gnn_action_probability': float(self.gnn_action_probability),
            'node_features_used': self.node_features_used,
            'edge_features_used': self.edge_features_used.tolist() if isinstance(self.edge_features_used,
                                                                                 np.ndarray) else self.edge_features_used,
            'reward_signals': self.reward_signals,
            'immediate_reward': float(self.immediate_reward),
            'h_preservation': float(self.h_preservation),
            'transition_growth': float(self.transition_growth),
            'opp_score': float(self.opp_score),
            'label_combinability': float(self.label_combinability),
            'is_good_merge': bool(self.is_good_merge),
            'is_bad_merge': bool(self.is_bad_merge),
            'merge_quality_category': self.merge_quality_category,
        }


class EnhancedSilentTrainingLogger:
    """Enhanced logging system with temporal resolution & metric richness."""

    def __init__(self, log_dir: str, experiment_id: str, verbose: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.verbose = verbose

        self.log_file = self.log_dir / "training.log"
        self.file_handler = None
        self._open_file()

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

    def log_training_started(self, num_episodes: int, num_problems: int, seed: int):
        self._emit_event(
            'training_started',
            num_episodes=num_episodes,
            num_problems=num_problems,
            seed=seed
        )

    def log_training_completed(self, total_steps: int, total_reward: float):
        self._emit_event(
            'training_completed',
            total_steps=total_steps,
            total_reward=total_reward
        )

    def log_episode_started(self, episode: int, problem_name: str):
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
        )

        self.problem_episode_counts[problem_name] += 1
        if error:
            self.problem_failure_counts[problem_name] += 1
        else:
            self.problem_rewards[problem_name].append(reward)

    def log_reward_component_breakdown(
            self,
            episode: int,
            problem_name: str,
            step: int,
            component_breakdown: Dict[str, Any],
    ):
        self._emit_event(
            'reward_component_breakdown',
            episode=episode,
            problem_name=problem_name,
            step=step,
            components=component_breakdown.get('components', {}),
            component_details=component_breakdown.get('component_details', {}),
            catastrophic_penalties=component_breakdown.get('catastrophic_penalties', {}),
            signal_validity=component_breakdown.get('signal_validity', {}),
        )

    def log_merge_decision(
            self,
            episode: int,
            problem_name: str,
            step: int,
            decision_trace: MergeDecisionTrace,
    ):
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
        self._emit_event(
            'checkpoint_saved',
            step=step,
            path=path,
            reward=reward,
            problem_name=problem_name,
            domain_name=domain_name
        )

    def log_failure(
            self,
            episode: int,
            problem_name: str,
            failure_type: str,
            error_message: str,
            context: Optional[Dict] = None,
    ):
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