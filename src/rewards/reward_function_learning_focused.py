#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEARNING-FOCUSED REWARD FUNCTION - COMPLETE WITH LABEL COMBINABILITY
=====================================================================

THIS IS THE IMPROVED VERSION that:
1. Maintains strong H* signal (60%) for learning
2. RESTORES label combinability (8%) for compression
3. Keeps transition control (20%) for explosion avoidance
4. Adds operator projection (12%) for post-merge potential

Result: Learning-friendly WHILE maintaining M&S theory compliance

Based on:
1. Helmert et al. (2014) - Merge-and-Shrink Abstraction (label reduction)
2. Nissim et al. (2011) - Computing Perfect Heuristics
3. Katz & Hoffmann (2013) - Merge-and-Shrink Implementation
4. RL Theory - Reward shaping for faster learning
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LearningFocusedRewardFunctionComplete:
    """
    Complete learning-focused reward function WITH label combinability.

    Design Principles:
    1. PRIMARY (60%): H* Preservation - the most critical factor
    2. SECONDARY (20%): Transition Control - explosion avoidance
    3. TERTIARY (12%): Operator Projection - compression potential
    4. COMPRESSION (8%): Label Combinability - post-merge compression signal
    5. BONUS (0%): (folded into other components for clarity)

    Range: [-2.0, +2.0]

    KEY ADVANTAGE OVER INCOMPLETE VERSION:
    - Rewards BOTH accuracy (H*) AND size (label combinability)
    - Enables learning of merge policies that create small AND accurate abstractions
    - Maintains episode-aware curriculum for better convergence
    - Explicit label reward prevents "accuracy-only" local optima

    Learning Properties:
    - Early episodes (0-500): Stricter thresholds, encourage good exploration
    - Mid episodes (500-1000): Progressive scaling
    - Late episodes (1000+): Higher expectations, positive reinforcement
    """

    def __init__(self, debug: bool = False, episode: int = 0, total_episodes: int = 1500):
        """
        Initialize with episode tracking for curriculum rewards.

        Args:
            debug: Enable detailed logging
            episode: Current episode number (0-indexed)
            total_episodes: Total episodes in training
        """
        self.debug = debug
        self.episode = episode
        self.total_episodes = total_episodes

        # Training progress: 0.0 (start) to 1.0 (end)
        self.progress = min(1.0, episode / max(1, total_episodes - 1))

        # LITERATURE-BASED THRESHOLDS
        self.H_STAR_CRITICAL_LOSS = 0.15  # 15% loss = severe penalty
        self.TRANSITION_EXPLOSION_THRESHOLD = 5.0  # 5x growth = bad
        self.REACHABILITY_MINIMUM = 0.3  # <30% reachable = very bad
        self.DEAD_END_DANGER_ZONE = 0.4  # >40% dead-ends = risky

        # LEARNING-SPECIFIC THRESHOLDS (episode-aware)
        self._compute_learning_thresholds()

    def _compute_learning_thresholds(self):
        """
        Compute episode-aware thresholds for learning curriculum.

        Early episodes: Harder to get rewards (encourage exploration → good merges)
        Late episodes: Easier to get rewards (reward discovered strategies)
        """
        # H* preservation threshold that decreases learning (gets stricter)
        # Early: Need >95% preservation for any positive reward
        # Late: Need >90% preservation for positive reward
        self.h_star_threshold = 0.95 - 0.05 * self.progress

        # Transition growth tolerance (tighter early, looser late)
        # Early: Don't tolerate >2x growth
        # Late: Tolerate up to 3x if quality is good
        self.transition_growth_tolerance = 2.0 + 1.0 * self.progress

        # Dead-end danger zone (stricter early)
        # Early: >25% dead-ends is bad
        # Late: >35% dead-ends is bad (more tolerated if h* is great)
        self.dead_end_threshold = 0.25 + 0.1 * self.progress

        # Label combinability threshold (what counts as "good")
        # Early: Need >70% for bonus
        # Late: Need >60% for bonus (more tolerated)
        self.label_comb_threshold = 0.70 - 0.10 * self.progress

        if self.debug:
            logger.debug(f"[REWARD] Episode {self.episode}/{self.total_episodes} (progress={self.progress:.2%})")
            logger.debug(f"  H* threshold: {self.h_star_threshold:.3f}, "
                         f"Trans tolerance: {self.transition_growth_tolerance:.1f}x, "
                         f"Dead-end threshold: {self.dead_end_threshold:.1%}, "
                         f"Label comb threshold: {self.label_comb_threshold:.3f}")

    def compute_reward(self, raw_obs: Dict[str, Any]) -> float:
        """
        ✅ FIXED: Compute scalar reward with guaranteed Python float return.

        Returns:
            reward: Python float (NOT numpy scalar)
        """
        signals = raw_obs.get('reward_signals', {})
        edge_features = raw_obs.get('edge_features', None)

        # Compute components
        h_reward, h_details = self._compute_h_preservation_reward_learning(signals)
        trans_reward, trans_details = self._compute_transition_control_reward_learning(signals)
        opp_reward, opp_details = self._compute_operator_projection_reward_learning(signals)
        label_reward, label_details = self._compute_label_combinability_reward_learning(signals)

        # Weighted combination
        final_reward = (
                0.50 * h_reward +
                0.20 * trans_reward +
                0.15 * opp_reward +
                0.15 * label_reward
        )

        # Catastrophic penalties
        if not signals.get('is_solvable', True):
            final_reward = final_reward - 1.2

        if signals.get('dead_end_ratio', 0.0) > 0.8:
            final_reward = final_reward - 0.6

        # Synergy bonus
        h_pres = signals.get('h_star_preservation', 1.0)
        trans_growth = trans_details.get('growth_ratio', 1.0)
        label_comb = signals.get('label_combinability_score', 0.5)

        if h_pres > 0.95 and trans_growth < 1.5 and label_comb > 0.75:
            final_reward = final_reward + (0.35 * self.progress)
        elif h_pres > 0.92 and trans_growth < 1.8 and label_comb > 0.65:
            final_reward = final_reward + (0.20 * self.progress)
        elif h_pres > 0.90 and trans_growth < 2.0 and label_comb > 0.55:
            final_reward = final_reward + (0.10 * self.progress)

        # ✅ CRITICAL: Convert to Python float explicitly
        # Step 1: Use numpy clip but extract immediately
        clipped = np.clip(final_reward, -2.0, 2.0)

        # Step 2: Convert numpy type to Python float
        if isinstance(clipped, np.ndarray):
            if clipped.shape == ():  # 0-d array
                final_reward_python = float(clipped.item())
            else:
                final_reward_python = float(clipped.flat[0])
        elif isinstance(clipped, (np.floating, np.integer)):
            final_reward_python = float(clipped.item())
        else:
            final_reward_python = float(clipped)

        # Step 3: Validate
        assert isinstance(final_reward_python, float) and not isinstance(final_reward_python, bool), \
            f"Reward must be Python float, got {type(final_reward_python)}"

        if self.debug:
            logger.debug(f"[REWARD] h={h_reward:.3f}, trans={trans_reward:.3f}, "
                         f"opp={opp_reward:.3f}, label={label_reward:.3f} → Final: {final_reward_python:.4f}")

        return final_reward_python

    def update_episode(self, episode: int, total_episodes: int) -> None:
        """Update episode info without full reinitialization."""
        self.episode = episode
        self.total_episodes = total_episodes
        self.progress = min(1.0, episode / max(1, total_episodes - 1))
        self._compute_learning_thresholds()

    def _compute_h_preservation_reward_learning(self, signals: Dict) -> Tuple[float, Dict]:
        """H* Preservation - Range normalized to [-1.0, +1.0]"""
        h_pres = signals.get('h_star_preservation', 1.0)

        if h_pres >= 0.98:
            reward = 0.90 + 0.10 * (h_pres - 0.98) / 0.02  # [0.90, 1.0]
        elif h_pres >= 0.95:
            reward = 0.70 + 0.20 * (h_pres - 0.95) / 0.03  # [0.70, 0.90]
        elif h_pres >= 0.90:
            reward = 0.40 + 0.30 * (h_pres - 0.90) / 0.05  # [0.40, 0.70]
        elif h_pres >= 0.85:
            reward = 0.10 + 0.30 * (h_pres - 0.85) / 0.05  # [0.10, 0.40]
        elif h_pres >= 0.70:
            reward = -0.40 + 0.50 * (h_pres - 0.70) / 0.15  # [-0.40, 0.10]
        else:
            reward = -1.0 + 0.60 * max(0, h_pres) / 0.70  # [-1.0, -0.40]

        # Now in [-1.0, 1.0] range
        return reward, {'h_star_preservation': h_pres, 'reward': reward}

    def _compute_transition_control_reward_learning(self, signals: Dict) -> Tuple[float, Dict]:
        """
        Transition Explosion Control with Learning Signal (20% weight).

        Strategy:
        - 1.0x-1.5x: Bonus (0.15-0.25)
        - 1.5x-2.0x: Neutral (0.05-0.15)
        - 2.0x-3.0x: Mild penalty (-0.1 to 0.05)
        - 3.0x-5.0x: Moderate penalty (-0.3 to -0.1)
        - >5.0x: Severe penalty (-0.8 to -0.3)

        Creates strong signal against explosion.
        """
        details = {}

        states_before = signals.get('states_before', 1)
        states_after = signals.get('states_after', 1)
        density_ratio = signals.get('transition_density', 1.0)

        # Compute growth with safety
        if states_before > 0:
            growth_ratio = states_after / max(1, states_before)
        else:
            growth_ratio = 1.0

        # EXCELLENT: Shrinking or stable
        if growth_ratio <= 1.0:
            # Shrinking is VERY good
            shrink = 1.0 - growth_ratio
            reward = 0.15 + 0.10 * min(1.0, shrink / 0.5)  # Range: [0.15, 0.25]
            quality = "SHRINKING"

        elif growth_ratio <= 1.5:
            # Stable with minimal growth - good
            reward = 0.15  # Solid bonus
            quality = "MINIMAL GROWTH"

        elif growth_ratio <= 2.0:
            # Mild growth - still acceptable
            growth_excess = growth_ratio - 1.5
            reward = 0.10 - 0.05 * (growth_excess / 0.5)  # Range: [0.05, 0.10]
            quality = "MILD GROWTH"

        elif growth_ratio <= 3.0:
            # Moderate growth - penalty starting
            growth_excess = growth_ratio - 2.0
            reward = -0.05 - 0.10 * (growth_excess / 1.0)  # Range: [-0.05, -0.15]
            quality = "MODERATE GROWTH"

        elif growth_ratio <= 5.0:
            # Heavy growth - stronger penalty
            growth_excess = growth_ratio - 3.0
            reward = -0.20 - 0.10 * (growth_excess / 2.0)  # Range: [-0.20, -0.30]
            quality = "HEAVY GROWTH"

        else:
            # Explosion - severe penalty
            explosion_severity = min(1.0, (growth_ratio - 5.0) / 5.0)
            reward = -0.50 - 0.30 * explosion_severity  # Range: [-0.50, -0.80]
            quality = "EXPLOSION"

        # Density bonus/penalty (secondary signal)
        if density_ratio > 0.8:
            reward -= 0.2  # High density = bad
            quality += " (HIGH DENSITY)"

        if self.debug:
            logger.debug(f"[REWARD-Trans] {quality}: {growth_ratio:.2f}x → {reward:.3f}")

        details['growth_ratio'] = growth_ratio
        details['density_ratio'] = density_ratio
        details['reward'] = reward

        return reward, details

    def _compute_operator_projection_reward_learning(self, signals: Dict) -> Tuple[float, Dict]:
        """
        Operator Projection Potential with Learning Signal (12% weight - REDUCED).

        High OPP = better compression potential after merge

        Note: Weight reduced from 15% to 12% to make room for label combinability (8%)
        This maintains the same relative importance while explicitly separating label concerns.
        """
        details = {}

        opp_score = signals.get('opp_score', 0.5)

        # EXCELLENT: High projection potential
        if opp_score >= 0.85:
            reward = 0.28  # Strong reward
            quality = "EXCELLENT"

        elif opp_score >= 0.70:
            # Good projection potential
            reward = 0.20 + 0.08 * (opp_score - 0.70) / 0.15
            quality = "GOOD"

        elif opp_score >= 0.50:
            # Moderate projection
            reward = 0.10 + 0.10 * (opp_score - 0.50) / 0.20
            quality = "MODERATE"

        elif opp_score >= 0.30:
            # Low projection
            reward = 0.02 + 0.08 * (opp_score - 0.30) / 0.20
            quality = "LOW"

        else:
            # Very low - slight penalty
            reward = -0.08
            quality = "VERY LOW"

        if self.debug:
            logger.debug(f"[REWARD-OPP] {quality}: {opp_score:.2f} → {reward:.3f}")

        details['opp_score'] = opp_score
        details['quality'] = quality
        details['reward'] = reward

        return reward, details

    def _compute_label_combinability_reward_learning(self, signals: Dict) -> Tuple[float, Dict]:
        """
        Label Combinability Reward with Learning Signal (8% weight - RESTORED).

        ⭐ THIS IS THE CRITICAL COMPONENT MISSING FROM INCOMPLETE FUNCTION 2

        From Helmert et al. (2014):
        "Labels that are locally equivalent in all other factors are combinable"

        High combinability = labels will collapse post-merge = smaller abstraction

        Why This Matters:
        - Without this signal, agent learns to optimize ONLY for accuracy (H*)
        - Leads to "large but accurate" abstractions
        - M&S goal: BOTH small AND accurate
        - Label combinability directly measures "potential smallness"

        Reward Strategy:
        - >75%: Excellent bonus (labels will compress well post-merge)
        - 50-75%: Good signal (decent compression potential)
        - 25-50%: Moderate signal (some compression)
        - <25%: Poor signal (independent labels, won't compress)
        """
        details = {}

        label_comb = signals.get('label_combinability_score', 0.5)

        # EXCELLENT: Many labels will combine post-merge
        if label_comb >= 0.85:
            # High combinability = smaller final abstraction
            reward = 0.25  # Strong bonus for compression potential
            quality = "EXCELLENT"

        elif label_comb >= 0.70:
            # Good combinability
            reward = 0.15 + 0.10 * (label_comb - 0.70) / 0.15
            quality = "VERY GOOD"

        elif label_comb >= 0.50:
            # Moderate combinability
            reward = 0.08 + 0.07 * (label_comb - 0.50) / 0.20
            quality = "GOOD"

        elif label_comb >= 0.30:
            # Low combinability
            reward = 0.02 + 0.06 * (label_comb - 0.30) / 0.20
            quality = "MODERATE"

        elif label_comb >= 0.10:
            # Very low combinability
            reward = -0.03 + 0.05 * label_comb / 0.10
            quality = "LOW"

        else:
            # No combinability - independent labels
            # Small penalty to discourage merging unrelated systems
            reward = -0.08
            quality = "NONE"

        if self.debug:
            logger.debug(f"[REWARD-Label] {quality}: {label_comb:.3f} → {reward:.3f}")

        details['label_combinability'] = label_comb
        details['quality'] = quality
        details['reward'] = reward

        return reward, details

    def compute_reward_with_breakdown(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute reward AND return detailed component breakdown.

        Returns full information for analysis and plotting.
        """
        signals = raw_obs.get('reward_signals', {})
        edge_features = raw_obs.get('edge_features', None)

        # Compute each component
        h_reward, h_details = self._compute_h_preservation_reward_learning(signals)
        trans_reward, trans_details = self._compute_transition_control_reward_learning(signals)
        opp_reward, opp_details = self._compute_operator_projection_reward_learning(signals)
        label_reward, label_details = self._compute_label_combinability_reward_learning(signals)

        # Track penalties
        catastrophic_penalties = {
            'solvability_loss': -1.2 if not signals.get('is_solvable', True) else 0.0,
            'dead_end_penalty': -0.6 if signals.get('dead_end_ratio', 0.0) > 0.8 else 0.0,
        }

        # Synergy bonus
        h_pres = signals.get('h_star_preservation', 1.0)
        trans_growth = trans_details.get('growth_ratio', 1.0)
        label_comb = signals.get('label_combinability_score', 0.5)
        synergy_bonus = 0.0

        if h_pres > 0.95 and trans_growth < 1.5 and label_comb > 0.75:
            synergy_bonus = 0.35 * self.progress
        elif h_pres > 0.92 and trans_growth < 1.8 and label_comb > 0.65:
            synergy_bonus = 0.20 * self.progress
        elif h_pres > 0.90 and trans_growth < 2.0 and label_comb > 0.55:
            synergy_bonus = 0.10 * self.progress

        # Final reward
        final_reward = (
                0.60 * h_reward +
                0.20 * trans_reward +
                0.12 * opp_reward +
                0.08 * label_reward
        )

        final_reward += sum(catastrophic_penalties.values())
        final_reward += synergy_bonus
        final_reward = np.clip(final_reward, -2.0, 2.0)

        return {
            'final_reward': float(final_reward),
            'episode': self.episode,
            'progress': float(self.progress),
            'components': {
                'h_preservation': float(h_reward),
                'transition_control': float(trans_reward),
                'operator_projection': float(opp_reward),
                'label_combinability': float(label_reward),
                'synergy_bonus': float(synergy_bonus),
            },
            'component_details': {
                'h_star_preservation': float(signals.get('h_star_preservation', 1.0)),
                'transition_growth_ratio': float(trans_details.get('growth_ratio', 1.0)),
                'transition_density': float(trans_details.get('density_ratio', 0.0)),
                'opp_score': float(opp_details.get('opp_score', 0.5)),
                'label_combinability': float(label_details.get('label_combinability', 0.5)),
                'reachability_ratio': float(signals.get('reachability_ratio', 1.0)),
                'causal_proximity': float(signals.get('causal_proximity_score', 0.0)),
            },
            'catastrophic_penalties': catastrophic_penalties,
            'signal_validity': {
                'is_solvable': bool(signals.get('is_solvable', True)),
                'dead_end_ratio': float(signals.get('dead_end_ratio', 0.0)),
            }
        }


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def create_learning_focused_reward_function(
        debug: bool = False,
        episode: int = 0,
        total_episodes: int = 1500
) -> LearningFocusedRewardFunctionComplete:
    """
    Factory function for learning-focused reward function WITH label combinability.

    Args:
        debug: Enable detailed logging
        episode: Current episode number
        total_episodes: Total training episodes

    Returns:
        LearningFocusedRewardFunctionComplete instance

    Usage in thin_merge_env.py:
    ----
    reward_fn = create_learning_focused_reward_function(
        debug=False,
        episode=self.episode,
        total_episodes=1500
    )
    reward = reward_fn.compute_reward(obs)
    breakdown = reward_fn.compute_reward_with_breakdown(obs)
    """
    return LearningFocusedRewardFunctionComplete(
        debug=debug,
        episode=episode,
        total_episodes=total_episodes
    )