#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THIN MERGE ENVIRONMENT - Updated for Enhanced Features
========================================================
Now handles 15-dim node features and 10-dim edge features from C++.
Uses improved reward function with h* preservation focus.
"""

import os
import sys
import json
import time
import glob
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.rewards.reward_function_enhanced import EnhancedRewardFunction
from src.rewards.reward_function_learning_focused import LearningFocusedRewardFunctionComplete  # ✅ NEW
from src.utils.common_utils import ThinClientConfig


logger = logging.getLogger(__name__)


class ThinMergeEnv(gym.Env):
    """
    Thin Client Environment with Enhanced Features.

    Now uses:
    - 15-dimensional node features (was 7)
    - 10-dimensional edge features from C++ (was derived)
    - h* preservation focused reward
    """

    metadata = {"render_modes": []}

    # Updated dimensions
    NODE_FEATURE_DIM = 9  # Expanded from 7
    EDGE_FEATURE_DIM = 11  # New: from C++
    MAX_NODES = 100
    MAX_EDGES = 1000

    # New reward weights (h* focused)
    DEFAULT_REWARD_WEIGHTS = {
        'w_h_preservation': 0.40,  # Primary signal!
        'w_shrinkability': 0.25,
        'w_state_control': 0.20,
        'w_solvability': 0.15,
    }

    def __init__(
            self,
            domain_file: str,
            problem_file: str,
            downward_dir: Optional[str] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
            reward_weights: Optional[Dict[str, float]] = None,
            debug: bool = False,
    ):
        # ✅ FIX 4: Ensure CPU-only
        import torch
        torch.cuda.is_available = lambda: False

        super().__init__()

        self.domain_file = os.path.abspath(domain_file)
        self.problem_file = os.path.abspath(problem_file)

        if not os.path.exists(self.domain_file):
            raise FileNotFoundError(f"Domain file not found: {self.domain_file}")
        if not os.path.exists(self.problem_file):
            raise FileNotFoundError(f"Problem file not found: {self.problem_file}")

        if downward_dir is None:
            script_dir = Path(__file__).parent.absolute()
            self.downward_dir = script_dir / "downward"
        else:
            self.downward_dir = Path(downward_dir).absolute()

        if not self.downward_dir.exists():
            raise FileNotFoundError(f"Fast Downward directory not found: {self.downward_dir}")

        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step
        self.debug = debug
        # self.reward_weights = reward_weights or self.DEFAULT_REWARD_WEIGHTS.copy()

        self.reward_weights = reward_weights or self.DEFAULT_REWARD_WEIGHTS.copy()

        # ✅ NEW: Initialize reward function (with curriculum support)
        self._reward_function_type = "learning_focused"  # or "enhanced"
        self._episode_number = 0
        self._total_episodes = 1500  # Default: can be overridden
        self._reward_function = None
        self._init_reward_function()

        self.fd_output_dir = self.downward_dir / "fd_output"
        self.gnn_output_dir = self.downward_dir / "gnn_output"
        self._setup_directories()

        # Updated observation space
        self.observation_space = spaces.Dict({
            "x": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.MAX_NODES, self.NODE_FEATURE_DIM),
                dtype=np.float32
            ),
            "edge_index": spaces.Box(
                low=0, high=self.MAX_NODES,
                shape=(2, self.MAX_EDGES),
                dtype=np.int64
            ),
            "edge_features": spaces.Box(  # NEW!
                low=-np.inf, high=np.inf,
                shape=(self.MAX_EDGES, self.EDGE_FEATURE_DIM),
                dtype=np.float32
            ),
            "num_nodes": spaces.Box(low=0, high=self.MAX_NODES, shape=(), dtype=np.int32),
            "num_edges": spaces.Box(low=0, high=self.MAX_EDGES, shape=(), dtype=np.int32),
        })

        self.action_space = spaces.Discrete(self.MAX_EDGES)

        self.current_iteration = -1
        self.process: Optional[subprocess.Popen] = None
        self.fd_log_file = None
        self._cached_edge_index: Optional[np.ndarray] = None
        self._last_observation: Optional[Dict] = None

        logger.info(f"[THIN_ENV] Initialized with {self.NODE_FEATURE_DIM} node features, "
                    f"{self.EDGE_FEATURE_DIM} edge features")

    def _init_reward_function(self):
        """Initialize reward function based on configured type."""
        if self._reward_function_type == "learning_focused":
            from src.rewards.reward_function_learning_focused import (
                create_learning_focused_reward_function
            )
            self._reward_function = create_learning_focused_reward_function(
                debug=self.debug,
                episode=self._episode_number,
                total_episodes=self._total_episodes
            )
        else:
            from src.rewards.reward_function_enhanced import (
                create_enhanced_reward_function
            )
            self._reward_function = create_enhanced_reward_function(
                debug=self.debug
            )

        logger.info(f"[THIN_ENV] Using reward function: {self._reward_function_type}")

    def set_episode_number(self, episode: int, total_episodes: int = 1500) -> None:
        """Set current episode for curriculum-aware reward functions."""
        self._episode_number = episode
        self._total_episodes = total_episodes

        # Reinitialize reward function with new episode info
        if self._reward_function_type == "learning_focused":
            self._init_reward_function()

    def _setup_directories(self):
        self.fd_output_dir.mkdir(parents=True, exist_ok=True)
        self.gnn_output_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_files(self):
        patterns = [
            str(self.fd_output_dir / "*.json"),
            str(self.gnn_output_dir / "*.json"),
            str(self.fd_output_dir / "*.tmp"),
            str(self.gnn_output_dir / "*.tmp"),
        ]
        deleted = 0
        for pattern in patterns:
            for filepath in glob.glob(pattern):
                try:
                    os.remove(filepath)
                    deleted += 1
                except:
                    pass
        logger.debug(f"[THIN_ENV] Cleaned up {deleted} files")

    def _terminate_process(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except:
                pass
        self.process = None

        if self.fd_log_file:
            try:
                self.fd_log_file.close()
            except:
                pass
            self.fd_log_file = None

    def _launch_fd_process(self):
        """Launch Fast Downward with GNN merge strategy."""
        logger.info("[THIN_ENV] Launching Fast Downward...")

        translator_script = self.downward_dir / "builds" / "release" / "bin" / "translate" / "translate.py"
        if not translator_script.exists():
            raise FileNotFoundError(f"Translator not found: {translator_script}")

        translate_cmd = [
            sys.executable,
            str(translator_script),
            self.domain_file,
            self.problem_file,
            "--sas-file", "output.sas"
        ]

        translate_result = subprocess.run(
            translate_cmd,
            cwd=str(self.downward_dir),
            capture_output=True,
            timeout=60,
            text=True
        )

        if translate_result.returncode != 0:
            raise RuntimeError(f"Translator failed: {translate_result.stderr}")

        import platform

        # Detect executable name based on OS
        if platform.system() == "Windows":
            downward_exe = self.downward_dir / "builds" / "release" / "bin" / "downward.exe"
        else:
            # Linux/macOS
            downward_exe = self.downward_dir / "builds" / "release" / "bin" / "downward"

        if not downward_exe.exists():
            raise FileNotFoundError(
                f"Downward executable not found: {downward_exe}\n"
                f"Expected location: {self.downward_dir / 'builds' / 'release' / 'bin'}\n"
                f"OS: {platform.system()}"
            )

        search_config = (
            "astar(merge_and_shrink("
            "merge_strategy=merge_gnn(),"
            "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
            "label_reduction=exact(before_shrinking=true,before_merging=false),"
            "max_states=200000,"
            "threshold_before_merge=1))"
        )

        downward_cmd = [str(downward_exe), "--search", search_config]

        log_path = self.fd_output_dir / "downward.log"
        self.fd_log_file = open(log_path, "w", buffering=1)

        self.process = subprocess.Popen(
            downward_cmd,
            cwd=str(self.downward_dir),
            stdin=subprocess.PIPE,
            stdout=self.fd_log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        sas_path = self.downward_dir / "output.sas"
        with open(sas_path, "r") as f:
            sas_content = f.read()

        try:
            self.process.stdin.write(sas_content)
            self.process.stdin.close()
        except BrokenPipeError:
            logger.error("[THIN_ENV] Failed to write SAS file to process")
            raise

        logger.info(f"[THIN_ENV] FD process started (PID: {self.process.pid})")

    def _wait_for_observation(self, iteration: int, timeout: Optional[float] = None) -> Dict[str, Any]:
        if timeout is None:
            timeout = self.timeout_per_step

        obs_path = self.fd_output_dir / f"observation_{iteration}.json"
        start_time = time.time()
        last_error = None

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time

            if elapsed > 2.0 and self.process and self.process.poll() is not None:
                return_code = self.process.returncode
                if obs_path.exists():
                    try:
                        time.sleep(0.05)
                        with open(obs_path, 'r') as f:
                            content = f.read()
                        if content.strip():
                            return json.loads(content)
                    except:
                        pass
                raise RuntimeError(f"FD process died with code {return_code}")

            if obs_path.exists():
                try:
                    time.sleep(0.01)
                    with open(obs_path, 'r') as f:
                        content = f.read()
                    if content.strip():
                        data = json.loads(content)
                        return data
                except json.JSONDecodeError as e:
                    last_error = f"JSON decode: {e}"
                except PermissionError as e:
                    last_error = f"Permission: {e}"
                except IOError as e:
                    last_error = f"IO error: {e}"

            time.sleep(0.05)

        raise TimeoutError(f"Timeout waiting for observation_{iteration}.json: {last_error}")

    def _send_merge_decision(self, iteration: int, merge_pair: Tuple[int, int]) -> None:
        import platform

        decision = {
            "iteration": iteration,
            "merge_pair": list(merge_pair),
            "timestamp": time.time()
        }

        decision_path = self.gnn_output_dir / f"merge_{iteration}.json"
        temp_path = decision_path.with_suffix('.tmp')

        with open(temp_path, 'w') as f:
            json.dump(decision, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        max_rename_attempts = 5
        for attempt in range(max_rename_attempts):
            try:
                os.replace(temp_path, decision_path)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_rename_attempts - 1:
                    time.sleep(0.05)
                else:
                    raise RuntimeError(f"Failed to write merge decision: {e}")

        if platform.system() == 'Windows':
            time.sleep(0.02)

    def _observation_to_tensors(self, raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert C++ observation JSON to numpy tensors - UPDATED for edge features."""

        # Node features (15-dim)
        x_raw = raw_obs.get('x', [])
        num_nodes = len(x_raw)

        x = np.zeros((self.MAX_NODES, self.NODE_FEATURE_DIM), dtype=np.float32)
        n = min(num_nodes, self.MAX_NODES)

        for i in range(n):
            if i < len(x_raw):
                features = x_raw[i]
                for j in range(min(len(features), self.NODE_FEATURE_DIM)):
                    x[i, j] = float(features[j])

        # Edge index
        edge_index_raw = raw_obs.get('edge_index', [[], []])
        num_edges = len(edge_index_raw[0]) if len(edge_index_raw) == 2 else 0

        edge_index = np.zeros((2, self.MAX_EDGES), dtype=np.int64)
        ne = min(num_edges, self.MAX_EDGES)

        if ne > 0 and len(edge_index_raw) == 2:
            edge_index[0, :ne] = np.array(edge_index_raw[0][:ne], dtype=np.int64)
            edge_index[1, :ne] = np.array(edge_index_raw[1][:ne], dtype=np.int64)

        self._cached_edge_index = edge_index_raw

        # Edge features (10-dim) - NEW from C++!
        edge_features_raw = raw_obs.get('edge_features', None)
        edge_features = np.zeros((self.MAX_EDGES, self.EDGE_FEATURE_DIM), dtype=np.float32)

        if edge_features_raw is not None and ne > 0:
            for i, ef in enumerate(edge_features_raw[:ne]):
                for j, val in enumerate(ef[:self.EDGE_FEATURE_DIM]):
                    edge_features[i, j] = float(val)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_features": edge_features,  # NEW!
            "num_nodes": np.int32(n),
            "num_edges": np.int32(ne),
        }

    # def _compute_reward(self, raw_obs: Dict[str, Any]) -> float:
    #     """
    #     Compute scalar reward with h* preservation focus.
    #
    #     Weighted combination:
    #     - w_h_preservation = 0.40 (PRIMARY!)
    #     - w_shrinkability = 0.25
    #     - w_state_control = 0.20
    #     - w_solvability = 0.15
    #     """
    #     signals = raw_obs.get('reward_signals', {})
    #
    #     # ========================================================================
    #     # EXTRACT SIGNALS
    #     # ========================================================================
    #
    #     # H* preservation (PRIMARY SIGNAL!)
    #     h_star_before = float(signals.get('h_star_before', 0))
    #     h_star_after = float(signals.get('h_star_after', 0))
    #     h_star_preservation = float(signals.get('h_star_preservation', 1.0))
    #
    #     # Shrinkability
    #     theoretical_product = int(signals.get('theoretical_product_size', 1))
    #     actual_size = int(signals.get('merged_size', 1))
    #     shrinkability = float(signals.get('shrinkability', 0.0))
    #
    #     # State control
    #     state_control_score = float(signals.get('state_control_score', 0.5))
    #     state_explosion_penalty = float(signals.get('state_explosion_penalty', 0.0))
    #
    #     # Solvability
    #     is_solvable = bool(signals.get('is_solvable', True))
    #
    #     # Dead-end ratio (new!)
    #     dead_end_ratio = float(signals.get('dead_end_ratio', 0.0))
    #
    #     # ========================================================================
    #     # LOG SIGNALS FOR DEBUGGING
    #     # ========================================================================
    #
    #     if self.debug:
    #         logger.debug(f"[REWARD] h* before={h_star_before}, after={h_star_after}, "
    #                      f"preservation={h_star_preservation:.3f}")
    #         logger.debug(f"[REWARD] shrinkability={shrinkability:.3f}, "
    #                      f"state_control={state_control_score:.3f}")
    #         logger.debug(f"[REWARD] solvable={is_solvable}, dead_ends={dead_end_ratio:.3f}")
    #
    #     # ========================================================================
    #     # COMPUTE WEIGHTED REWARD
    #     # ========================================================================
    #
    #     w_h = self.reward_weights.get('w_h_preservation', 0.40)
    #     w_shrink = self.reward_weights.get('w_shrinkability', 0.25)
    #     w_state = self.reward_weights.get('w_state_control', 0.20)
    #     w_solv = self.reward_weights.get('w_solvability', 0.15)
    #
    #     # h* preservation: 1.0 = preserved, > 1.0 = improved, < 1.0 = degraded
    #     h_component = min(1.0, h_star_preservation)
    #
    #     # Shrinkability: [-1, 1] -> [0, 1]
    #     shrink_component = max(0.0, shrinkability + 0.5)
    #
    #     # State control
    #     state_component = state_control_score
    #
    #     # Solvability
    #     solv_component = 1.0 if is_solvable else 0.0
    #
    #     # Weighted sum
    #     weighted_sum = (
    #             w_h * h_component +
    #             w_shrink * shrink_component +
    #             w_state * state_component +
    #             w_solv * solv_component
    #     )
    #
    #     reward = weighted_sum
    #
    #     # ========================================================================
    #     # BONUSES AND PENALTIES
    #     # ========================================================================
    #
    #     # Heavy penalty for losing solvability
    #     if not is_solvable:
    #         reward -= 1.0
    #
    #     # Penalty for high dead-end ratio
    #     if dead_end_ratio > 0.3:
    #         reward -= 0.1 * dead_end_ratio
    #
    #     # Bonus for h* improvement
    #     if h_star_preservation > 1.0:
    #         reward += 0.1 * (h_star_preservation - 1.0)
    #
    #     # Small step reward
    #     reward += 0.02
    #
    #     # Scale to reasonable RL range
    #     reward = (reward - 0.5) * 2.0
    #
    #     # Clamp
    #     reward = max(-5.0, min(2.0, float(reward)))
    #
    #     logger.debug(f"[REWARD] Final: weighted_sum={weighted_sum:.3f}, reward={reward:.3f}")
    #
    #     return reward

    # def _compute_reward(self, raw_obs: Dict[str, Any]) -> float:
    #     """
    #     Compute reward with clear positive/negative signals.
    #
    #     Design:
    #     - Range: [-2.0, +2.0] centered around 0
    #     - Positive rewards for good merges (h* preserved, controlled growth)
    #     - Negative rewards for bad merges (h* degradation, explosion, dead-ends)
    #     - Severe penalties for catastrophic failures (unsolvable)
    #
    #     Components:
    #     1. H* Preservation (±0.5): Primary signal for heuristic quality
    #     2. State Explosion (±0.4): Penalize uncontrolled growth
    #     3. Shrinkability (±0.15): Reward good shrinking
    #     4. Solvability (-1.0): Severe penalty for losing solvability
    #     5. Dead-End Ratio (±0.3): Penalize creating dead-ends
    #     6. Reachability (±0.2): Reward maintaining reachability
    #     7. F-Value Stability (±0.1): Reward stable heuristics
    #     8. Progress (+0.02): Small positive for each step
    #     """
    #     signals = raw_obs.get('reward_signals', {})
    #
    #     # ========================================================================
    #     # EXTRACT ALL SIGNALS FROM C++
    #     # ========================================================================
    #
    #     # Primary: h* preservation (most important!)
    #     h_star_before = float(signals.get('h_star_before', 0))
    #     h_star_after = float(signals.get('h_star_after', 0))
    #     h_star_preservation = float(signals.get('h_star_preservation', 1.0))
    #
    #     # State management
    #     states_before = int(signals.get('states_before', 1))
    #     states_after = int(signals.get('states_after', 1))
    #     state_explosion_penalty = float(signals.get('state_explosion_penalty', 0.0))
    #     shrinkability = float(signals.get('shrinkability', 0.0))
    #
    #     # Quality metrics
    #     is_solvable = bool(signals.get('is_solvable', True))
    #     dead_end_ratio = float(signals.get('dead_end_ratio', 0.0))
    #     reachability_ratio = float(signals.get('reachability_ratio', 1.0))
    #
    #     # F-value stability
    #     f_value_stability = float(signals.get('f_value_stability', 1.0))
    #     f_preservation_score = float(signals.get('f_preservation_score', 1.0))
    #
    #     # Transition metrics
    #     transition_density = float(signals.get('transition_density', 1.0))
    #     total_dead_ends = int(signals.get('total_dead_ends', 0))
    #
    #     # ========================================================================
    #     # REWARD COMPONENT 1: H* PRESERVATION (±0.5)
    #     # This is the PRIMARY signal - h* is the whole point!
    #     # ========================================================================
    #
    #     h_reward = 0.0
    #     if h_star_preservation >= 1.0:
    #         # GOOD: h* preserved or improved
    #         # Base reward + bonus for improvement
    #         improvement = min(1.0, h_star_preservation - 1.0)
    #         h_reward = 0.3 + 0.2 * improvement  # Range: [0.3, 0.5]
    #
    #         if self.debug:
    #             logger.debug(f"[REWARD] h* improved/preserved: +{h_reward:.3f}")
    #     else:
    #         # BAD: h* degraded - penalty proportional to loss
    #         degradation = 1.0 - h_star_preservation
    #         h_reward = -0.5 * degradation  # Range: [-0.5, 0]
    #
    #         # Extra penalty for severe degradation (>20% loss)
    #         if degradation > 0.2:
    #             h_reward -= 0.2 * (degradation - 0.2)  # Additional penalty
    #
    #         if self.debug:
    #             logger.debug(f"[REWARD] h* DEGRADED by {degradation:.1%}: {h_reward:.3f}")
    #
    #     # ========================================================================
    #     # REWARD COMPONENT 2: STATE EXPLOSION CONTROL (±0.4)
    #     # Penalize merges that cause excessive state growth
    #     # ========================================================================
    #
    #     explosion_reward = 0.0
    #     if states_before > 0:
    #         growth_ratio = states_after / max(1, states_before)
    #
    #         if growth_ratio > 10:
    #             # VERY BAD: Severe explosion (>10x growth)
    #             explosion_reward = -0.4  # Maximum penalty
    #             if self.debug:
    #                 logger.debug(f"[REWARD] SEVERE explosion ({growth_ratio:.1f}x): {explosion_reward:.3f}")
    #
    #         elif growth_ratio > 5:
    #             # BAD: Significant explosion (5-10x growth)
    #             explosion_reward = -0.2 - 0.2 * (growth_ratio - 5) / 5
    #             if self.debug:
    #                 logger.debug(f"[REWARD] Significant explosion ({growth_ratio:.1f}x): {explosion_reward:.3f}")
    #
    #         elif growth_ratio > 2:
    #             # MODERATE: Some growth (2-5x)
    #             explosion_reward = -0.05 - 0.15 * (growth_ratio - 2) / 3
    #             if self.debug:
    #                 logger.debug(f"[REWARD] Moderate growth ({growth_ratio:.1f}x): {explosion_reward:.3f}")
    #
    #         elif growth_ratio > 1:
    #             # MILD: Slight growth (1-2x)
    #             explosion_reward = -0.02 * (growth_ratio - 1)
    #
    #         else:
    #             # GOOD: Shrinking or no growth
    #             shrink_factor = 1.0 - growth_ratio
    #             explosion_reward = 0.1 * shrink_factor  # Bonus for shrinking
    #             if self.debug:
    #                 logger.debug(f"[REWARD] Good shrinking ({growth_ratio:.2f}x): +{explosion_reward:.3f}")
    #
    #     # ========================================================================
    #     # REWARD COMPONENT 3: SHRINKABILITY (±0.15)
    #     # Reward merges that allow effective shrinking
    #     # shrinkability: 1.0 = perfect, 0 = neutral, -1.0 = expansion
    #     # ========================================================================
    #
    #     shrink_reward = 0.0
    #     if shrinkability > 0:
    #         # GOOD: Effective shrinking possible
    #         shrink_reward = 0.15 * shrinkability
    #     else:
    #         # BAD: Poor shrinkability
    #         shrink_reward = 0.1 * shrinkability  # Penalty (shrinkability is negative)
    #
    #     # ========================================================================
    #     # REWARD COMPONENT 4: SOLVABILITY (-1.0 penalty)
    #     # Catastrophic failure - losing ability to find solutions
    #     # ========================================================================
    #
    #     solvability_reward = 0.0
    #     if not is_solvable:
    #         # CATASTROPHIC: Lost solvability - large penalty
    #         solvability_reward = -1.0
    #         if self.debug:
    #             logger.debug(f"[REWARD] CATASTROPHIC: Lost solvability: {solvability_reward:.3f}")
    #     else:
    #         # Small bonus for maintaining solvability
    #         solvability_reward = 0.05
    #
    #     # ========================================================================
    #     # REWARD COMPONENT 5: DEAD-END RATIO (±0.3)
    #     # Penalize creating dead-ends (states with no path to goal)
    #     # ========================================================================
    #
    #     dead_end_reward = 0.0
    #     if dead_end_ratio > 0.5:
    #         # VERY BAD: >50% dead-ends
    #         dead_end_reward = -0.3 * (dead_end_ratio - 0.5) / 0.5 - 0.1
    #         if self.debug:
    #             logger.debug(f"[REWARD] HIGH dead-ends ({dead_end_ratio:.1%}): {dead_end_reward:.3f}")
    #
    #     elif dead_end_ratio > 0.2:
    #         # BAD: 20-50% dead-ends
    #         dead_end_reward = -0.1 * (dead_end_ratio - 0.2) / 0.3
    #
    #     elif dead_end_ratio > 0.1:
    #         # MODERATE: 10-20% dead-ends - small penalty
    #         dead_end_reward = -0.02
    #
    #     else:
    #         # GOOD: <10% dead-ends
    #         dead_end_reward = 0.02  # Small bonus
    #
    #     # ========================================================================
    #     # REWARD COMPONENT 6: REACHABILITY (±0.2)
    #     # Reward maintaining high fraction of reachable states
    #     # ========================================================================
    #
    #     reach_reward = 0.0
    #     if reachability_ratio < 0.3:
    #         # VERY BAD: <30% reachable
    #         reach_reward = -0.2
    #         if self.debug:
    #             logger.debug(f"[REWARD] LOW reachability ({reachability_ratio:.1%}): {reach_reward:.3f}")
    #
    #     elif reachability_ratio < 0.5:
    #         # BAD: 30-50% reachable
    #         reach_reward = -0.15 * (0.5 - reachability_ratio) / 0.2
    #
    #     elif reachability_ratio < 0.7:
    #         # MODERATE: 50-70% reachable
    #         reach_reward = -0.05
    #
    #     elif reachability_ratio > 0.9:
    #         # GOOD: >90% reachable
    #         reach_reward = 0.05
    #
    #     # ========================================================================
    #     # REWARD COMPONENT 7: F-VALUE STABILITY (±0.1)
    #     # Reward stable f-values (indicates good abstraction quality)
    #     # ========================================================================
    #
    #     stability_avg = (f_value_stability + f_preservation_score) / 2.0
    #     stability_reward = 0.1 * (stability_avg - 0.5)  # Centered at 0.5
    #
    #     # ========================================================================
    #     # REWARD COMPONENT 8: PROGRESS REWARD (+0.02)
    #     # Small positive reward for each step to encourage exploration
    #     # ========================================================================
    #
    #     progress_reward = 0.02
    #
    #     # ========================================================================
    #     # COMBINE ALL COMPONENTS
    #     # ========================================================================
    #
    #     reward = (
    #             h_reward +  # ±0.5 (primary signal)
    #             explosion_reward +  # ±0.4
    #             shrink_reward +  # ±0.15
    #             solvability_reward +  # -1.0 or +0.05
    #             dead_end_reward +  # ±0.3
    #             reach_reward +  # ±0.2
    #             stability_reward +  # ±0.1
    #             progress_reward  # +0.02
    #     )
    #
    #     # ========================================================================
    #     # FINAL CLAMPING AND LOGGING
    #     # ========================================================================
    #
    #     # Clamp to [-2.0, +2.0]
    #     reward = max(-2.0, min(2.0, float(reward)))
    #
    #     if self.debug:
    #         logger.debug(
    #             f"[REWARD] Components: h*={h_reward:.3f}, explosion={explosion_reward:.3f}, "
    #             f"shrink={shrink_reward:.3f}, solv={solvability_reward:.3f}, "
    #             f"dead={dead_end_reward:.3f}, reach={reach_reward:.3f}, "
    #             f"stab={stability_reward:.3f}, prog={progress_reward:.3f}"
    #         )
    #         logger.debug(f"[REWARD] FINAL: {reward:.4f}")
    #
    #     return reward

    # def _compute_reward(self, raw_obs: Dict[str, Any]) -> float:
    #     """
    #     ENHANCED REWARD FUNCTION - Theory-Informed
    #
    #     Based on:
    #     1. Nissim et al. (2011) - Perfect Heuristics & Bisimulation
    #     2. Helmert et al. (2014) - M&S Implementation
    #     3. Katz & Hoffmann (2013) - M&S Lower Bounds
    #
    #     Component Weights:
    #     - H* Preservation (50%)      [Greedy Bisimulation]
    #     - Transition Control (20%)   [Avoid Explosion]
    #     - Operator Projection (15%)  [Enable Compression]
    #     - Label Combinability (10%)  [Label Reduction]
    #     - Bonuses (5%)              [Architecture Signals]
    #     """
    #     # from reward_function_enhanced import EnhancedRewardFunction
    #
    #     if not hasattr(self, '_enhanced_reward_fn'):
    #         self._enhanced_reward_fn = EnhancedRewardFunction(debug=self.debug)
    #
    #     return self._enhanced_reward_fn.compute_reward(raw_obs)

    def _compute_reward(self, raw_obs: Dict[str, Any]) -> float:
        """
        ✅ FIXED: Compute reward and ensure Python float return
        """
        if self._reward_function is None:
            self._init_reward_function()

        # Get reward (might be numpy)
        reward = self._reward_function.compute_reward(raw_obs)

        # ✅ CRITICAL FIX: Convert to Python float explicitly
        reward_float = float(reward)

        # Validate it's actually a Python float
        if not isinstance(reward_float, float):
            raise TypeError(f"Reward must be Python float, got {type(reward_float)}")

        return reward_float

    def _action_to_merge_pair(self, action: int) -> Tuple[int, int]:
        if self._cached_edge_index is None or len(self._cached_edge_index) != 2:
            return (0, 1)

        src_list, tgt_list = self._cached_edge_index

        if len(src_list) == 0:
            return (0, 1)

        num_edges = len(src_list)
        action = max(0, min(action, num_edges - 1))

        src = int(src_list[action])
        tgt = int(tgt_list[action])

        if src > tgt:
            src, tgt = tgt, src

        return (src, tgt)

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:

        super().reset(seed=seed)

        # ✅ FIX: Actually use the seed if provided
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        logger.info(f"[THIN_ENV] === RESET: {os.path.basename(self.problem_file)} ===")

        self._terminate_process()
        self._cleanup_files()

        self.current_iteration = -1
        self._cached_edge_index = None
        self._last_observation = None

        self._launch_fd_process()

        raw_obs = self._wait_for_observation(iteration=-1)

        obs = self._observation_to_tensors(raw_obs)
        self._last_observation = obs

        info = {
            "iteration": -1,
            "num_active_systems": raw_obs.get('num_active_systems', 0),
            "reward_signals": raw_obs.get('reward_signals', {}),
        }

        logger.info(f"[THIN_ENV] Reset complete: {obs['num_nodes']} nodes, {obs['num_edges']} edges")

        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        ✅ COMPLETELY FIXED v4: Aggressive type validation and conversion
        """
        self.current_iteration += 1
        iteration = self.current_iteration

        # ✅ FIX 1: Validate and convert action to Python int immediately
        try:
            action_python_int = int(action)
            if isinstance(action_python_int, (np.integer, np.ndarray)):
                action_python_int = int(
                    action_python_int.item() if hasattr(action_python_int, 'item') else action_python_int)
            assert isinstance(action_python_int, int) and not isinstance(action_python_int, bool)
        except Exception as e:
            logger.error(f"[THIN_ENV] Invalid action type: {type(action)} - {e}")
            # Fallback: use action 0
            action_python_int = 0

        merge_pair = self._action_to_merge_pair(action_python_int)
        logger.info(f"[THIN_ENV] Step {iteration}: action={action_python_int} -> merge_pair={merge_pair}")

        self._send_merge_decision(iteration, merge_pair)

        try:
            raw_obs = self._wait_for_observation(iteration)
        except TimeoutError as e:
            logger.error(f"[THIN_ENV] Timeout: {e}")
            return (
                self._last_observation if self._last_observation else self._create_dummy_observation(),
                -1.0,  # ✅ EXPLICIT Python float
                True,  # ✅ EXPLICIT Python bool
                False,  # ✅ EXPLICIT Python bool
                {"error": str(e), "error_type": "timeout"}
            )
        except RuntimeError as e:
            logger.error(f"[THIN_ENV] Step failed: {e}")
            return (
                self._last_observation if self._last_observation else self._create_dummy_observation(),
                -1.0,  # ✅ EXPLICIT Python float
                True,  # ✅ EXPLICIT Python bool
                False,  # ✅ EXPLICIT Python bool
                {"error": str(e), "error_type": "runtime_error"}
            )

        # ✅ FIX 2: Convert observation tensors properly
        obs = self._observation_to_tensors(raw_obs)
        self._last_observation = obs

        # ✅ FIX 3: AGGRESSIVE reward computation with triple validation
        reward = self._compute_reward_with_validation(raw_obs)

        # ✅ FIX 4: Extract and validate status flags
        num_active = int(raw_obs.get('num_active_systems', 1))
        if isinstance(num_active, (np.integer, np.ndarray)):
            num_active = int(num_active.item() if hasattr(num_active, 'item') else num_active)

        is_done = bool(raw_obs.get('is_terminal', False))
        if isinstance(is_done, (np.bool_, np.ndarray)):
            is_done = bool(is_done.item() if hasattr(is_done, 'item') else is_done)

        # ✅ FIX 5: Explicit bool conversions
        terminated = bool((num_active <= 1) or is_done)
        truncated = bool(iteration >= self.max_merges - 1)

        # Validate types before return
        assert isinstance(reward, float) and not isinstance(reward, (bool, np.bool_)), \
            f"reward must be Python float, got {type(reward)}: {reward}"
        assert isinstance(terminated, bool) and not isinstance(terminated, np.bool_), \
            f"terminated must be Python bool, got {type(terminated)}"
        assert isinstance(truncated, bool) and not isinstance(truncated, np.bool_), \
            f"truncated must be Python bool, got {type(truncated)}"

        info = {
            "iteration": int(iteration),
            "merge_pair": tuple(int(x) for x in merge_pair),
            "num_active_systems": num_active,
            "reward_signals": raw_obs.get('reward_signals', {}),
            "solved": bool(raw_obs.get('solved', False)),
            "plan_cost": int(raw_obs.get('plan_cost', 0)),
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward_with_validation(self, raw_obs: Dict[str, Any]) -> float:
        """
        ✅ AGGRESSIVE reward computation with triple validation.

        Ensures:
        1. Reward function returns value
        2. Value is converted to Python float
        3. Float is validated as proper Python type
        4. Value is within valid range
        """
        if self._reward_function is None:
            self._init_reward_function()

        try:
            # STEP 1: Call reward function
            reward_raw = self._reward_function.compute_reward(raw_obs)

            # STEP 2: Convert to Python float
            # Handle numpy scalars, arrays, tensors
            if isinstance(reward_raw, np.ndarray):
                if reward_raw.shape == ():  # 0-d array
                    reward_float = float(reward_raw.item())
                else:
                    raise TypeError(f"Reward array shape {reward_raw.shape}, expected scalar")

            elif isinstance(reward_raw, (np.floating, np.integer)):
                # Numpy scalar type
                reward_float = float(reward_raw.item())

            elif isinstance(reward_raw, float):
                # Already Python float
                reward_float = reward_raw

            else:
                # Try generic conversion
                reward_float = float(reward_raw)

            # STEP 3: Validate it's actually Python float
            if not isinstance(reward_float, float):
                raise TypeError(f"After conversion, reward is {type(reward_float)}, not float")

            # STEP 4: Check for NaN/Inf
            if np.isnan(reward_float):
                logger.warning("[THIN_ENV] Reward is NaN, setting to -1.0")
                reward_float = -1.0
            elif np.isinf(reward_float):
                logger.warning(f"[THIN_ENV] Reward is Inf, clamping to ±2.0")
                reward_float = 2.0 if reward_float > 0 else -2.0

            # STEP 5: Clamp to valid range
            reward_float = max(-2.0, min(2.0, reward_float))

            # Final validation
            assert isinstance(reward_float, float) and not isinstance(reward_float, bool), \
                f"Final reward validation failed: {type(reward_float)}"

            return reward_float

        except Exception as e:
            logger.error(f"[THIN_ENV] Reward computation failed: {e}")
            logger.error(f"  Raw reward: {reward_raw if 'reward_raw' in locals() else 'NOT SET'}")
            logger.error(f"  Raw type: {type(reward_raw) if 'reward_raw' in locals() else 'N/A'}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            # Return safe default
            return -1.0

    def _create_dummy_observation(self) -> Dict[str, Any]:
        """Create a dummy observation when environment fails."""
        return {
            "x": np.zeros((self.MAX_NODES, self.NODE_FEATURE_DIM), dtype=np.float32),
            "edge_index": np.zeros((2, self.MAX_EDGES), dtype=np.int64),
            "edge_features": np.zeros((self.MAX_EDGES, self.EDGE_FEATURE_DIM), dtype=np.float32),
            "num_nodes": np.int32(1),
            "num_edges": np.int32(0),
        }

    def close(self):
        self._terminate_process()
        logger.info("[THIN_ENV] Environment closed")


def make_thin_env(domain_file: str, problem_file: str, **kwargs) -> ThinMergeEnv:
    return ThinMergeEnv(domain_file=domain_file, problem_file=problem_file, **kwargs)