#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOCUSED REWARD FUNCTION - Clean Per-Step Rewards for M&S Learning
===================================================================

DESIGN PRINCIPLES:
1. Per-step reward in range [-1.0, +1.0] for RL stability
2. Episode reward = sum of step rewards (interpretable: ~+3 to +7 for good episodes)
3. Clear positive/negative signals for gradient-based learning
4. Uses C++ signals directly from merge_and_shrink_signals.cc
5. Component names match analysis/visualization expectations

REWARD COMPONENTS (matching logging.EpisodeMetrics and analysis modules):
- component_h_preservation (40%)     - Primary: heuristic quality
- component_transition_control (25%) - Secondary: explosion avoidance
- component_operator_projection (15%) - Tertiary: compression potential
- component_label_combinability (10%) - Quaternary: label reduction
- component_bonus_signals (10%)       - Safety & architectural signals

C++ SIGNALS USED (from merge_and_shrink_signals.cc):
- h_star_preservation: h* after / h* before (1.0 = preserved, <1 = degraded)
- growth_ratio: states after / states before (<1 = shrunk, >1 = grew)
- opp_score: operator projection potential [0, 1]
- label_combinability_score: label reduction potential [0, 1]
- is_solvable: boolean (solvability maintained)
- dead_end_ratio: fraction of dead-ends [0, 1]
- reachability_ratio: fraction of reachable states [0, 1]

EXPECTED EPISODE REWARDS:
- Excellent episode (7 good merges): +5.0 to +7.0
- Good episode (7 ok merges): +2.0 to +5.0
- Poor episode (mixed merges): 0.0 to +2.0
- Bad episode (several bad merges): -3.0 to 0.0
- Catastrophic episode (unsolvable): -7.0 to -3.0

LEARNING INTERPRETATION:
- Increasing episode rewards = GNN learning to choose good merges
- Decreasing rewards = GNN making worse choices (shouldn't happen with proper exploration)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FocusedRewardFunction:
    """
    Clean, focused reward function for M&S merge learning.

    Per-step reward in [-1.0, +1.0].
    Episode reward = sum of step rewards.

    Components (exactly matching EpisodeMetrics and analysis code):
    - h_preservation (40%): Primary h* quality signal
    - transition_control (25%): Explosion avoidance
    - operator_projection (15%): Compression potential
    - label_combinability (10%): Label reduction potential
    - bonus_signals (10%): Safety and reachability signals

    Each component returns value in [-1.0, +1.0].
    Final reward is weighted sum, clamped to [-1.0, +1.0].
    """

    # Component weights (must sum to 1.0)
    W_H_PRESERVATION = 0.40
    W_TRANSITION_CONTROL = 0.25
    W_OPERATOR_PROJECTION = 0.15
    W_LABEL_COMBINABILITY = 0.10
    W_BONUS_SIGNALS = 0.10

    # Thresholds from M&S literature (Helmert et al. 2014, Nissim et al. 2011)
    H_STAR_PERFECT = 1.0  # Perfect h* preservation
    H_STAR_GOOD = 0.95  # 5% loss is acceptable
    H_STAR_MODERATE = 0.85  # 15% loss is concerning
    H_STAR_BAD = 0.70  # 30% loss is bad

    GROWTH_EXCELLENT = 1.0  # Shrinking or stable
    GROWTH_GOOD = 2.0  # 2x growth is tolerable
    GROWTH_MODERATE = 3.0  # 3x growth is concerning
    GROWTH_BAD = 5.0  # 5x growth is explosion

    OPP_EXCELLENT = 0.70  # Excellent projection potential
    OPP_GOOD = 0.50  # Good projection potential
    OPP_MODERATE = 0.30  # Moderate potential

    LABEL_EXCELLENT = 0.60  # Excellent combinability
    LABEL_GOOD = 0.40  # Good combinability
    LABEL_MODERATE = 0.20  # Moderate combinability

    DEAD_END_WARN = 0.20  # 20% dead-ends is warning
    DEAD_END_BAD = 0.40  # 40% dead-ends is bad

    REACHABILITY_GOOD = 0.80  # 80% reachable is good
    REACHABILITY_WARN = 0.50  # 50% reachable is warning

    def __init__(self, debug: bool = False):
        """
        Initialize focused reward function.

        Args:
            debug: If True, log detailed reward computation
        """
        self.debug = debug
        self._step_count = 0
        self._episode_reward_sum = 0.0

    def reset(self):
        """Reset per-episode state (call at episode start)."""
        self._step_count = 0
        self._episode_reward_sum = 0.0

    def compute_reward(self, raw_obs: Dict[str, Any]) -> float:
        """
        Compute per-step reward from C++ signals.

        This is the main reward function called by ThinMergeEnv.step().

        Args:
            raw_obs: Observation dict with 'reward_signals' from C++

        Returns:
            reward: Python float in [-1.0, +1.0]
        """
        signals = raw_obs.get('reward_signals', {})

        # ====================================================================
        # STEP 1: Extract and validate C++ signals
        # ====================================================================
        h_preservation = self._extract_signal(signals, 'h_star_preservation', 1.0, 0.0, 2.0)
        growth_ratio = self._extract_signal(signals, 'growth_ratio', 1.0, 0.01, 100.0)
        opp_score = self._extract_signal(signals, 'opp_score', 0.5, 0.0, 1.0)
        label_score = self._extract_signal(signals, 'label_combinability_score', 0.5, 0.0, 1.0)
        is_solvable = bool(signals.get('is_solvable', True))
        dead_end_ratio = self._extract_signal(signals, 'dead_end_ratio', 0.0, 0.0, 1.0)
        reachability = self._extract_signal(signals, 'reachability_ratio', 1.0, 0.0, 1.0)

        # ====================================================================
        # STEP 2: Compute each component reward (each in [-1.0, +1.0])
        # ====================================================================
        h_reward = self._compute_h_preservation_reward(h_preservation)
        trans_reward = self._compute_transition_control_reward(growth_ratio)
        opp_reward = self._compute_operator_projection_reward(opp_score)
        label_reward = self._compute_label_combinability_reward(label_score)
        bonus_reward = self._compute_bonus_signals_reward(is_solvable, dead_end_ratio, reachability)

        # ====================================================================
        # STEP 3: Weighted combination
        # ====================================================================
        final_reward = (
                self.W_H_PRESERVATION * h_reward +
                self.W_TRANSITION_CONTROL * trans_reward +
                self.W_OPERATOR_PROJECTION * opp_reward +
                self.W_LABEL_COMBINABILITY * label_reward +
                self.W_BONUS_SIGNALS * bonus_reward
        )

        # ====================================================================
        # STEP 4: Clamp and convert to Python float
        # ====================================================================
        final_reward = max(-1.0, min(1.0, float(final_reward)))

        # Update tracking
        self._step_count += 1
        self._episode_reward_sum += final_reward

        if self.debug:
            logger.debug(
                f"[REWARD] Step {self._step_count}: "
                f"h={h_reward:.3f} trans={trans_reward:.3f} "
                f"opp={opp_reward:.3f} label={label_reward:.3f} "
                f"bonus={bonus_reward:.3f} → {final_reward:.4f} "
                f"(episode_sum={self._episode_reward_sum:.3f})"
            )

        return final_reward

    def _compute_h_preservation_reward(self, h_preservation: float) -> float:
        """
        H* preservation reward component (PRIMARY SIGNAL).

        h_preservation = h* after merge / h* before merge
        - >1.0: h* improved (rare but excellent)
        - =1.0: h* perfectly preserved
        - 0.95-1.0: slight loss (acceptable)
        - 0.85-0.95: moderate loss (concerning)
        - 0.70-0.85: significant loss (bad)
        - <0.70: severe loss (very bad)

        Returns value in [-1.0, +1.0]
        """
        if h_preservation >= self.H_STAR_PERFECT:
            # Perfect or improved
            improvement = min(1.0, (h_preservation - 1.0) * 5.0)  # Scale improvement
            return 0.6 + 0.4 * improvement  # [0.6, 1.0]

        elif h_preservation >= self.H_STAR_GOOD:
            # Good: 95-100% preserved
            ratio = (h_preservation - self.H_STAR_GOOD) / (self.H_STAR_PERFECT - self.H_STAR_GOOD)
            return 0.3 + 0.3 * ratio  # [0.3, 0.6]

        elif h_preservation >= self.H_STAR_MODERATE:
            # Moderate: 85-95% preserved
            ratio = (h_preservation - self.H_STAR_MODERATE) / (self.H_STAR_GOOD - self.H_STAR_MODERATE)
            return 0.0 + 0.3 * ratio  # [0.0, 0.3]

        elif h_preservation >= self.H_STAR_BAD:
            # Concerning: 70-85% preserved
            ratio = (h_preservation - self.H_STAR_BAD) / (self.H_STAR_MODERATE - self.H_STAR_BAD)
            return -0.5 + 0.5 * ratio  # [-0.5, 0.0]

        else:
            # Severe: <70% preserved
            ratio = max(0.0, h_preservation / self.H_STAR_BAD)
            return -1.0 + 0.5 * ratio  # [-1.0, -0.5]

    def _compute_transition_control_reward(self, growth_ratio: float) -> float:
        """
        Transition/state explosion control reward.

        growth_ratio = states after / states before
        - <1.0: shrinking (excellent)
        - 1.0: stable (good)
        - 1.0-2.0: mild growth (acceptable)
        - 2.0-3.0: moderate growth (concerning)
        - 3.0-5.0: significant growth (bad)
        - >5.0: explosion (very bad)

        Returns value in [-1.0, +1.0]
        """
        if growth_ratio <= self.GROWTH_EXCELLENT:
            # Shrinking or stable - excellent!
            shrink_factor = 1.0 - growth_ratio
            return 0.5 + 0.5 * min(1.0, shrink_factor * 2.0)  # [0.5, 1.0]

        elif growth_ratio <= self.GROWTH_GOOD:
            # Mild growth (1.0-2.0)
            ratio = (growth_ratio - self.GROWTH_EXCELLENT) / (self.GROWTH_GOOD - self.GROWTH_EXCELLENT)
            return 0.3 - 0.2 * ratio  # [0.1, 0.3]

        elif growth_ratio <= self.GROWTH_MODERATE:
            # Moderate growth (2.0-3.0)
            ratio = (growth_ratio - self.GROWTH_GOOD) / (self.GROWTH_MODERATE - self.GROWTH_GOOD)
            return 0.0 - 0.3 * ratio  # [-0.3, 0.0]

        elif growth_ratio <= self.GROWTH_BAD:
            # Significant growth (3.0-5.0)
            ratio = (growth_ratio - self.GROWTH_MODERATE) / (self.GROWTH_BAD - self.GROWTH_MODERATE)
            return -0.5 - 0.3 * ratio  # [-0.8, -0.5]

        else:
            # Explosion (>5.0)
            return -1.0

    def _compute_operator_projection_reward(self, opp_score: float) -> float:
        """
        Operator projection potential reward.

        opp_score in [0, 1]: fraction of operators that can be projected
        High OPP = merge enables good compression

        Returns value in [-1.0, +1.0]
        """
        if opp_score >= self.OPP_EXCELLENT:
            # Excellent OPP
            ratio = (opp_score - self.OPP_EXCELLENT) / (1.0 - self.OPP_EXCELLENT)
            return 0.5 + 0.5 * ratio  # [0.5, 1.0]

        elif opp_score >= self.OPP_GOOD:
            # Good OPP
            ratio = (opp_score - self.OPP_GOOD) / (self.OPP_EXCELLENT - self.OPP_GOOD)
            return 0.2 + 0.3 * ratio  # [0.2, 0.5]

        elif opp_score >= self.OPP_MODERATE:
            # Moderate OPP
            ratio = (opp_score - self.OPP_MODERATE) / (self.OPP_GOOD - self.OPP_MODERATE)
            return 0.0 + 0.2 * ratio  # [0.0, 0.2]

        else:
            # Low OPP - small penalty
            ratio = opp_score / self.OPP_MODERATE
            return -0.3 + 0.3 * ratio  # [-0.3, 0.0]

    def _compute_label_combinability_reward(self, label_score: float) -> float:
        """
        Label combinability reward.

        label_score in [0, 1]: fraction of labels that can combine
        High = labels will collapse post-merge = smaller abstraction

        Returns value in [-1.0, +1.0]
        """
        if label_score >= self.LABEL_EXCELLENT:
            # Excellent combinability
            ratio = (label_score - self.LABEL_EXCELLENT) / (1.0 - self.LABEL_EXCELLENT)
            return 0.5 + 0.5 * ratio  # [0.5, 1.0]

        elif label_score >= self.LABEL_GOOD:
            # Good combinability
            ratio = (label_score - self.LABEL_GOOD) / (self.LABEL_EXCELLENT - self.LABEL_GOOD)
            return 0.2 + 0.3 * ratio  # [0.2, 0.5]

        elif label_score >= self.LABEL_MODERATE:
            # Moderate combinability
            ratio = (label_score - self.LABEL_MODERATE) / (self.LABEL_GOOD - self.LABEL_MODERATE)
            return 0.0 + 0.2 * ratio  # [0.0, 0.2]

        else:
            # Low combinability - small penalty
            ratio = label_score / self.LABEL_MODERATE
            return -0.3 + 0.3 * ratio  # [-0.3, 0.0]

    def _compute_bonus_signals_reward(
            self,
            is_solvable: bool,
            dead_end_ratio: float,
            reachability: float
    ) -> float:
        """
        Bonus signals reward (safety and architectural).

        Combines:
        - Solvability (must maintain)
        - Dead-end ratio (lower is better)
        - Reachability (higher is better)

        Returns value in [-1.0, +1.0]
        """
        # Catastrophic: lost solvability
        if not is_solvable:
            return -1.0

        # Dead-end penalty
        if dead_end_ratio >= self.DEAD_END_BAD:
            dead_penalty = -0.5
        elif dead_end_ratio >= self.DEAD_END_WARN:
            ratio = (dead_end_ratio - self.DEAD_END_WARN) / (self.DEAD_END_BAD - self.DEAD_END_WARN)
            dead_penalty = -0.2 - 0.3 * ratio  # [-0.5, -0.2]
        else:
            dead_penalty = 0.0

        # Reachability reward
        if reachability >= self.REACHABILITY_GOOD:
            reach_reward = 0.3
        elif reachability >= self.REACHABILITY_WARN:
            ratio = (reachability - self.REACHABILITY_WARN) / (self.REACHABILITY_GOOD - self.REACHABILITY_WARN)
            reach_reward = 0.0 + 0.3 * ratio  # [0.0, 0.3]
        else:
            ratio = reachability / self.REACHABILITY_WARN
            reach_reward = -0.3 + 0.3 * ratio  # [-0.3, 0.0]

        # Solvability bonus
        solv_bonus = 0.2

        # Combine
        total = solv_bonus + dead_penalty + reach_reward
        return max(-1.0, min(1.0, total))

    def _extract_signal(
            self,
            signals: Dict[str, Any],
            key: str,
            default: float,
            min_val: float,
            max_val: float
    ) -> float:
        """
        Safely extract and validate a signal from C++ output.

        Handles:
        - Missing keys
        - None values
        - numpy types
        - NaN/Inf values
        - Out-of-range values
        """
        value = signals.get(key, default)

        # Handle None
        if value is None:
            return default

        # Convert to Python float
        try:
            if isinstance(value, np.ndarray):
                if value.shape == ():
                    f = float(value.item())
                elif value.size > 0:
                    f = float(value.flat[0])
                else:
                    return default
            elif isinstance(value, (np.floating, np.integer)):
                f = float(value.item())
            elif isinstance(value, bool):
                f = 1.0 if value else 0.0
            else:
                f = float(value)
        except (TypeError, ValueError):
            return default

        # Handle NaN/Inf
        if np.isnan(f) or np.isinf(f):
            if self.debug:
                logger.warning(f"[REWARD] Signal '{key}' has NaN/Inf value, using default {default}")
            return default

        # Clamp to valid range
        if f < min_val or f > max_val:
            if self.debug:
                logger.warning(f"[REWARD] Signal '{key}' value {f} out of range [{min_val}, {max_val}], clamping")
            f = max(min_val, min(max_val, f))

        return f

    def compute_reward_with_breakdown(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute reward AND return detailed component breakdown.

        This is called by training.py for logging episode metrics.
        The component names MUST match EpisodeMetrics and analysis modules.

        Returns:
            Dict with:
            - final_reward: float in [-1.0, +1.0]
            - components: Dict with exact names for EpisodeMetrics
            - component_details: Raw signal values for debugging
            - interpretation: Human-readable quality assessment
        """
        signals = raw_obs.get('reward_signals', {})

        # Extract signals
        h_preservation = self._extract_signal(signals, 'h_star_preservation', 1.0, 0.0, 2.0)
        growth_ratio = self._extract_signal(signals, 'growth_ratio', 1.0, 0.01, 100.0)
        opp_score = self._extract_signal(signals, 'opp_score', 0.5, 0.0, 1.0)
        label_score = self._extract_signal(signals, 'label_combinability_score', 0.5, 0.0, 1.0)
        is_solvable = bool(signals.get('is_solvable', True))
        dead_end_ratio = self._extract_signal(signals, 'dead_end_ratio', 0.0, 0.0, 1.0)
        reachability = self._extract_signal(signals, 'reachability_ratio', 1.0, 0.0, 1.0)

        # Compute components
        h_reward = self._compute_h_preservation_reward(h_preservation)
        trans_reward = self._compute_transition_control_reward(growth_ratio)
        opp_reward = self._compute_operator_projection_reward(opp_score)
        label_reward = self._compute_label_combinability_reward(label_score)
        bonus_reward = self._compute_bonus_signals_reward(is_solvable, dead_end_ratio, reachability)

        # Final weighted reward
        final_reward = (
                self.W_H_PRESERVATION * h_reward +
                self.W_TRANSITION_CONTROL * trans_reward +
                self.W_OPERATOR_PROJECTION * opp_reward +
                self.W_LABEL_COMBINABILITY * label_reward +
                self.W_BONUS_SIGNALS * bonus_reward
        )
        final_reward = max(-1.0, min(1.0, float(final_reward)))

        # Interpretation
        if final_reward >= 0.6:
            interpretation = "EXCELLENT merge - h* preserved, controlled growth"
        elif final_reward >= 0.3:
            interpretation = "GOOD merge - acceptable quality"
        elif final_reward >= 0.0:
            interpretation = "MODERATE merge - some concerns"
        elif final_reward >= -0.3:
            interpretation = "POOR merge - degraded quality"
        else:
            interpretation = "BAD merge - significant problems"

        return {
            'final_reward': final_reward,

            # Component names EXACTLY matching EpisodeMetrics fields
            'components': {
                'h_preservation': float(h_reward),
                'transition_control': float(trans_reward),
                'operator_projection': float(opp_reward),
                'label_combinability': float(label_reward),
                'bonus_signals': float(bonus_reward),
            },

            # Raw signal values for debugging
            'component_details': {
                'h_star_preservation': float(h_preservation),
                'transition_growth_ratio': float(growth_ratio),
                'opp_score': float(opp_score),
                'label_combinability': float(label_score),
                'is_solvable': is_solvable,
                'dead_end_ratio': float(dead_end_ratio),
                'reachability_ratio': float(reachability),
            },

            # Weights for reference
            'weights': {
                'h_preservation': self.W_H_PRESERVATION,
                'transition_control': self.W_TRANSITION_CONTROL,
                'operator_projection': self.W_OPERATOR_PROJECTION,
                'label_combinability': self.W_LABEL_COMBINABILITY,
                'bonus_signals': self.W_BONUS_SIGNALS,
            },

            # For catastrophic penalty tracking
            'catastrophic_penalties': {
                'solvability_loss': -1.0 if not is_solvable else 0.0,
                'dead_end_penalty': -0.5 if dead_end_ratio > self.DEAD_END_BAD else 0.0,
            },

            # Signal validity check
            'signal_validity': {
                'is_solvable': is_solvable,
                'dead_end_ratio': float(dead_end_ratio),
                'signals_present': len(signals) > 0,
            },

            'interpretation': interpretation,
        }

    def update_episode(self, episode: int, total_episodes: int) -> None:
        """
        Update episode info (for curriculum learning compatibility).

        This is a no-op for the focused reward function since we don't
        use episode-aware thresholds. Kept for interface compatibility.
        """
        pass


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_focused_reward_function(debug: bool = False) -> FocusedRewardFunction:
    """
    Factory function for focused reward function.

    Usage:
        reward_fn = create_focused_reward_function(debug=False)
        reward = reward_fn.compute_reward(obs)
        breakdown = reward_fn.compute_reward_with_breakdown(obs)
    """
    return FocusedRewardFunction(debug=debug)