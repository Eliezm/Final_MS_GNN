#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED REWARD FUNCTION - Theory-Informed Design
==================================================

Based on:
1. Helmert et al. (2014) - Merge-and-Shrink Abstraction
2. Nissim et al. (2011) - Computing Perfect Heuristics
3. Katz & Hoffmann (2013) - Merge-and-Shrink Implementation

Core Principles:
- Bisimulation (especially Greedy Bisimulation) is the gold standard
- H* preservation is PRIMARY (w=0.50)
- Transition count matters more than state count
- Label reduction potential is critical
- Operator projection enables compression
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedRewardFunction:
    """
    Theory-informed reward function balancing Small & Accurate abstractions.

    Reward Components (in order of importance from literature):
    1. H* Preservation (50%)         - Greedy bisimulation
    2. Transition Control (20%)      - Avoid explosion
    3. Operator Projection (15%)     - Post-merge compression potential
    4. Label Combinability (10%)     - Label reduction potential
    5. Bonus Signals (5%)            - Architecture-specific insights

    Range: [-2.0, +2.0]
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

        # Thresholds from literature
        self.H_STAR_CRITICAL_LOSS = 0.15  # 15% h* loss = severe penalty
        self.TRANSITION_EXPLOSION_THRESHOLD = 5.0  # 5x growth = bad
        self.REACHABILITY_MINIMUM = 0.3  # <30% reachable = very bad
        self.DEAD_END_DANGER_ZONE = 0.4  # >40% dead-ends = risky

    def compute_reward(self, raw_obs: Dict[str, Any]) -> float:
        """
        ✅ FIXED: Compute scalar reward with guaranteed Python float return.

        Returns:
            reward: Python float (NOT numpy scalar)
        """
        signals = raw_obs.get('reward_signals', {})
        edge_features = raw_obs.get('edge_features', None)

        h_reward, h_details = self._compute_h_preservation_reward(signals)
        trans_reward, trans_details = self._compute_transition_control_reward(signals)
        opp_reward, opp_details = self._compute_operator_projection_reward(signals)
        label_reward, label_details = self._compute_label_combinability_reward(signals)
        bonus_reward, bonus_details = self._compute_bonus_signals(signals, edge_features)

        final_reward = (
                0.50 * h_reward +
                0.20 * trans_reward +
                0.15 * opp_reward +
                0.10 * label_reward +
                0.05 * bonus_reward
        )

        if not signals.get('is_solvable', True):
            final_reward = final_reward - 1.0

        if signals.get('dead_end_ratio', 0.0) > 0.7:
            final_reward = final_reward - 0.5

        # ✅ CRITICAL: Convert to Python float explicitly
        clipped = np.clip(final_reward, -2.0, 2.0)

        if isinstance(clipped, np.ndarray):
            if clipped.shape == ():
                final_reward_python = float(clipped.item())
            else:
                final_reward_python = float(clipped.flat[0])
        elif isinstance(clipped, (np.floating, np.integer)):
            final_reward_python = float(clipped.item())
        else:
            final_reward_python = float(clipped)

        assert isinstance(final_reward_python, float) and not isinstance(final_reward_python, bool), \
            f"Reward must be Python float, got {type(final_reward_python)}"

        return final_reward_python

    def _compute_h_preservation_reward(self, signals: Dict) -> Tuple[float, Dict]:
        """
        H* Preservation Reward Component (50% weight)

        From Nissim et al. (2011):
        - Gold standard: h* = h^* (bisimulation)
        - Practical: Greedy bisimulation (h* on optimal paths)

        This is the PRIMARY signal for heuristic quality.
        """
        details = {}

        h_star_before = signals.get('h_star_before', 0)
        h_star_after = signals.get('h_star_after', 0)
        h_star_preservation = signals.get('h_star_preservation', 1.0)

        # h_star_preservation = after / before (ratio)
        # >1.0 = improved, =1.0 = preserved, <1.0 = degraded

        if h_star_preservation >= 1.0:
            # GOOD: H* preserved or improved
            # Reward = improvement bonus
            improvement = min(1.0, h_star_preservation - 1.0)
            reward = 0.4 + 0.3 * improvement  # Range: [0.4, 0.7]

            if self.debug:
                logger.debug(f"[REWARD-H*] Preserved/improved: {h_star_preservation:.3f} → +{reward:.3f}")

        elif h_star_preservation >= (1.0 - self.H_STAR_CRITICAL_LOSS):
            # MODERATE: Small h* degradation (<15%)
            # Tolerable if other signals are good
            degradation = 1.0 - h_star_preservation
            penalty = 0.3 * (degradation / self.H_STAR_CRITICAL_LOSS)
            reward = 0.2 - penalty  # Range: [0.05, 0.2]

            if self.debug:
                logger.debug(f"[REWARD-H*] Minor degradation {degradation:.1%}: {reward:.3f}")

        else:
            # BAD: Severe h* degradation (>15%)
            # Large penalty - heuristic quality compromised
            degradation = 1.0 - h_star_preservation
            penalty = 0.5 * min(1.0, degradation / 0.5)  # Scale by severity
            reward = -penalty  # Range: [-0.5, 0]

            if self.debug:
                logger.debug(f"[REWARD-H*] SEVERE degradation {degradation:.1%}: {reward:.3f}")

        details['h_star_preservation'] = h_star_preservation
        details['reward'] = reward

        return reward, details

    def _compute_transition_control_reward(self, signals: Dict) -> Tuple[float, Dict]:
        """
        Transition Explosion Control (20% weight)

        From papers: "Transitions are the real killer"
        - Penalize merges causing explosive transition growth
        - Reward merges that compress or stabilize transitions
        - Use prediction from C++ for explosion detection
        """
        details = {}

        states_before = signals.get('states_before', 1)
        states_after = signals.get('states_after', 1)

        # Transition density prediction
        density_ratio = signals.get('transition_density', 1.0)

        # Compute growth factor
        if states_before > 0:
            growth_ratio = states_after / max(1, states_before)
        else:
            growth_ratio = 1.0

        # Explosion severity
        if growth_ratio > 10.0 or density_ratio > 0.9:
            # SEVERE EXPLOSION
            reward = -0.8
            if self.debug:
                logger.debug(f"[REWARD-Trans] SEVERE explosion: growth={growth_ratio:.1f}x, "
                             f"density={density_ratio:.2f}")

        elif growth_ratio > 5.0 or density_ratio > 0.7:
            # MODERATE EXPLOSION
            reward = -0.3 - 0.2 * min(1.0, (growth_ratio - 5.0) / 5.0)
            if self.debug:
                logger.debug(f"[REWARD-Trans] Moderate explosion: growth={growth_ratio:.1f}x")

        elif growth_ratio > 2.0:
            # MILD GROWTH
            reward = -0.1 * (growth_ratio - 1.0)
            if self.debug:
                logger.debug(f"[REWARD-Trans] Mild growth: {growth_ratio:.1f}x")

        elif growth_ratio >= 1.0:
            # STABLE
            reward = 0.05

        else:
            # SHRINKING - bonus!
            shrink = 1.0 - growth_ratio
            reward = 0.15 * shrink  # Range: [0, 0.15]
            if self.debug:
                logger.debug(f"[REWARD-Trans] Good shrinking: {growth_ratio:.2f}x")

        details['growth_ratio'] = growth_ratio
        details['density_ratio'] = density_ratio
        details['reward'] = reward

        return reward, details

    def _compute_operator_projection_reward(self, signals: Dict) -> Tuple[float, Dict]:
        """
        Operator Projection Potential (15% weight)

        From Nissim et al. (2011) - Section on Label Projection:
        "Maximal conservative label reduction: project operators to merged vars"

        High OPP = many operators become internal-only = post-merge compression
        This is crucial for keeping merged systems "small"
        """
        details = {}

        # Extract OPP score from enhanced signals
        merge_quality = signals.get('merge_quality_score', 0.5)
        opp_score = signals.get('opp_score', 0.5)  # 0-1

        # Use OPP directly if available, otherwise estimate from quality
        if 'opp_score' in signals:
            opp_score = signals['opp_score']
        else:
            # Fallback: estimate from shrinkability
            opp_score = max(0.0, signals.get('shrinkability', 0.0) + 0.5)

        # Convert to reward: high OPP = high reward
        if opp_score > 0.7:
            # EXCELLENT: Many projectable operators
            reward = 0.2 + 0.1 * (opp_score - 0.7) / 0.3  # Range: [0.2, 0.3]
            if self.debug:
                logger.debug(f"[REWARD-OPP] Excellent projection potential: {opp_score:.2f}")

        elif opp_score > 0.4:
            # GOOD: Reasonable projection potential
            reward = 0.1 + 0.1 * (opp_score - 0.4) / 0.3
            if self.debug:
                logger.debug(f"[REWARD-OPP] Good projection potential: {opp_score:.2f}")

        elif opp_score > 0.2:
            # MODERATE: Some projection possible
            reward = 0.02 + 0.08 * (opp_score - 0.2) / 0.2

        else:
            # POOR: Few projectable operators
            # Merge might not benefit from compression
            reward = -0.1
            if self.debug:
                logger.debug(f"[REWARD-OPP] Poor projection potential: {opp_score:.2f}")

        details['opp_score'] = opp_score
        details['reward'] = reward

        return reward, details

    def _compute_label_combinability_reward(self, signals: Dict) -> Tuple[float, Dict]:
        """
        Label Combinability Reward (10% weight)

        From Helmert et al. (2014):
        "Labels that are locally equivalent in all other factors are combinable"

        High combinability = labels will collapse post-merge = compression
        """
        details = {}

        # Extract label combinability score
        label_comb = signals.get('label_combinability_score', 0.5)

        # Convert to reward
        if label_comb > 0.6:
            # EXCELLENT: Many labels will combine
            reward = 0.15 * label_comb  # Range: [0.09, 0.15]
            if self.debug:
                logger.debug(f"[REWARD-Label] Excellent combinability: {label_comb:.2f}")

        elif label_comb > 0.3:
            # GOOD: Reasonable combinability
            reward = 0.08 * label_comb

        elif label_comb > 0.0:
            # MODERATE
            reward = 0.03 * label_comb

        else:
            # POOR: No label combinability
            reward = -0.05
            if self.debug:
                logger.debug(f"[REWARD-Label] No combinability: merging independent systems")

        details['label_combinability'] = label_comb
        details['reward'] = reward

        return reward, details

    def _compute_bonus_signals(self, signals: Dict,
                               edge_features: Any) -> Tuple[float, Dict]:
        """
        Bonus Signals (5% weight)

        Architecture-specific insights that don't fit main categories.
        """
        details = {}
        reward = 0.0

        # Bonus 1: Causal graph proximity
        causal_proximity = signals.get('causal_proximity_score', 0.0)
        if causal_proximity > 0.8:
            reward += 0.02  # Small bonus for merging causally adjacent vars

        # Bonus 2: Greedy bisimulation compliance
        gb_error = signals.get('gb_error', 0.0)
        if gb_error < 0.1:
            reward += 0.02  # Good h-value compatibility

        # Bonus 3: Landmark preservation
        landmark_score = signals.get('landmark_preservation', 0.5)
        if landmark_score > 0.8:
            reward += 0.01

        # Bonus 4: Stability/F-value preservation
        f_stability = signals.get('f_value_stability', 0.5)
        if f_stability > 0.8:
            reward += 0.01

        # Penalty: Reachability collapse
        reachability = signals.get('reachability_ratio', 1.0)
        if reachability < 0.3:
            reward -= 0.05
            if self.debug:
                logger.debug(f"[REWARD-Bonus] Low reachability: {reachability:.1%}")

        details['causal_proximity'] = causal_proximity
        details['gb_error'] = gb_error
        details['landmark_score'] = landmark_score
        details['f_stability'] = f_stability
        details['reachability'] = reachability
        details['reward'] = reward

        return reward, details

    def compute_reward_with_breakdown(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute reward AND return detailed component breakdown.

        Returns:
            {
                'final_reward': float,
                'components': {
                    'h_preservation': float,
                    'transition_control': float,
                    'operator_projection': float,
                    'label_combinability': float,
                    'bonus_signals': float,
                },
                'component_details': {
                    'h_star_preservation': float,
                    'transition_growth_ratio': float,
                    'opp_score': float,
                    'label_combinability': float,
                    'catastrophic_penalties': {
                        'solvability_loss': float,
                        'dead_end_penalty': float,
                    }
                }
            }
        """
        signals = raw_obs.get('reward_signals', {})
        edge_features = raw_obs.get('edge_features', None)

        # Compute each component
        h_reward, h_details = self._compute_h_preservation_reward(signals)
        trans_reward, trans_details = self._compute_transition_control_reward(signals)
        opp_reward, opp_details = self._compute_operator_projection_reward(signals)
        label_reward, label_details = self._compute_label_combinability_reward(signals)
        bonus_reward, bonus_details = self._compute_bonus_signals(signals, edge_features)

        # Track penalties
        catastrophic_penalties = {
            'solvability_loss': -1.0 if not signals.get('is_solvable', True) else 0.0,
            'dead_end_penalty': -0.5 if signals.get('dead_end_ratio', 0.0) > 0.7 else 0.0,
        }

        # Final reward
        final_reward = (
                0.50 * h_reward +
                0.20 * trans_reward +
                0.15 * opp_reward +
                0.10 * label_reward +
                0.05 * bonus_reward
        )

        final_reward += sum(catastrophic_penalties.values())
        final_reward = np.clip(final_reward, -2.0, 2.0)

        return {
            'final_reward': float(final_reward),
            'components': {
                'h_preservation': float(h_reward),
                'transition_control': float(trans_reward),
                'operator_projection': float(opp_reward),
                'label_combinability': float(label_reward),
                'bonus_signals': float(bonus_reward),
            },
            'component_details': {
                'h_star_preservation': float(signals.get('h_star_preservation', 1.0)),
                'transition_growth_ratio': float(trans_details.get('growth_ratio', 1.0)),
                'transition_density': float(trans_details.get('density_ratio', 0.0)),
                'opp_score': float(opp_details.get('opp_score', 0.5)),
                'label_combinability': float(label_details.get('label_combinability', 0.5)),
                'causal_proximity': float(bonus_details.get('causal_proximity', 0.0)),
                'landmark_preservation': float(bonus_details.get('landmark_score', 0.5)),
                'reachability_ratio': float(bonus_details.get('reachability', 1.0)),
            },
            'catastrophic_penalties': catastrophic_penalties,
            'signal_validity': {
                'is_solvable': bool(signals.get('is_solvable', True)),
                'dead_end_ratio': float(signals.get('dead_end_ratio', 0.0)),
            }
        }


# ============================================================================
# INTEGRATION: Replace in thin_merge_env.py
# ============================================================================

def create_enhanced_reward_function(debug: bool = False) -> EnhancedRewardFunction:
    """Factory function for reward function."""
    return EnhancedRewardFunction(debug=debug)