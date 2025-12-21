#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEATURE ANALYSIS - Input feature importance & correlation
==========================================================
Analyzes which GNN input features matter most.

Computes:
- Correlation between features and GNN decision confidence
- Correlation between features and reward
- Feature importance ranking
"""

from typing import Dict, List, Any
import numpy as np
import warnings

from experiments.core.logging import EpisodeMetrics


def analyze_feature_reward_correlation(
        episode_reward_signals: Dict,
        output_dir=None,
) -> Dict[str, Any]:
    """
    Analyze correlation between input features and reward.

    Returns:
        {
            'feature_correlations': {feature_name: {correlation, p_value, significant}},
            'num_episodes': int,
            'reward_stats': {...},
        }
    """

    try:
        import scipy.stats as stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        return {}

    if not episode_reward_signals:
        return {}

    # ====================================================================
    # EXTRACT FEATURE VALUES
    # ====================================================================

    all_episodes = []
    all_rewards = []
    all_opp_scores = []
    all_label_scores = []
    all_h_star_ratios = []
    all_transition_growth = []
    all_reachability = []
    all_dead_ends = []

    for episode, data in sorted(episode_reward_signals.items()):
        episode_reward = data['episode_reward']
        summary = data['component_summary']

        all_episodes.append(episode)
        all_rewards.append(episode_reward)
        all_opp_scores.append(summary['avg_opp_score'])
        all_label_scores.append(summary['avg_label_combinability'])
        all_h_star_ratios.append(summary['avg_h_star_ratio'])
        all_transition_growth.append(summary['avg_transition_growth'])
        all_reachability.append(summary['min_reachability'])
        all_dead_ends.append(summary['max_dead_end_penalty'])

    if len(all_episodes) < 3:
        return {}

    # ====================================================================
    # COMPUTE CORRELATIONS
    # ====================================================================

    correlations = {}
    features_to_test = {
        'opp_score': all_opp_scores,
        'label_combinability': all_label_scores,
        'h_star_preservation': all_h_star_ratios,
        'transition_control': [1.0 / max(0.1, g) for g in all_transition_growth],
        'reachability_ratio': all_reachability,
        'dead_end_avoidance': [1.0 - d for d in all_dead_ends],
    }

    for feature_name, feature_values in features_to_test.items():
        if len(feature_values) < 3:
            continue

        feature_array = np.array(feature_values)
        reward_array = np.array(all_rewards)

        # Skip invalid data
        if np.any(~np.isfinite(feature_array)) or np.any(~np.isfinite(reward_array)):
            continue

        if np.std(feature_array) < 1e-6 or np.std(reward_array) < 1e-6:
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                corr, p_val = stats.pearsonr(feature_array, reward_array)

            if np.isnan(corr) or np.isinf(corr):
                continue

            correlations[feature_name] = {
                'correlation': float(corr),
                'p_value': float(p_val),
                'significant': p_val < 0.05,
            }
        except Exception:
            continue

    return {
        'feature_correlations': correlations,
        'num_episodes': len(all_episodes),
        'reward_stats': {
            'mean': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'std': float(np.std(all_rewards)) if all_rewards else 0.0,
            'min': float(np.min(all_rewards)) if all_rewards else 0.0,
            'max': float(np.max(all_rewards)) if all_rewards else 0.0,
        }
    }


def analyze_feature_importance_from_decisions(
        training_log: List[EpisodeMetrics],
        output_dir=None,
) -> Dict[str, Any]:
    """
    Estimate feature importance from decision traces.

    Uses correlation with GNN action probability to infer importance.

    Returns:
        {
            'feature_importance': {feature_name: {importance, correlation, significant}},
            'num_decisions_analyzed': int,
        }
    """

    try:
        import scipy.stats as stats
    except ImportError:
        return {}

    if not training_log:
        return {}

    # ====================================================================
    # EXTRACT DECISION DATA
    # ====================================================================

    all_h_pres_values = []
    all_trans_growth_values = []
    all_opp_scores = []
    all_label_scores = []
    all_reachability = []
    all_gnn_probs = []

    for metrics in training_log:
        if metrics.error is None and metrics.merge_decisions_per_step:
            for decision_dict in metrics.merge_decisions_per_step:
                h_pres = decision_dict.get('h_preservation', 1.0)
                trans_growth = decision_dict.get('transition_growth', 1.0)
                opp = decision_dict.get('opp_score', 0.5)
                label_comb = decision_dict.get('label_combinability', 0.5)
                reachability = decision_dict.get('reachability_ratio', 1.0)
                gnn_prob = decision_dict.get('gnn_action_probability', 0.5)

                all_h_pres_values.append(h_pres)
                all_trans_growth_values.append(trans_growth)
                all_opp_scores.append(opp)
                all_label_scores.append(label_comb)
                all_reachability.append(reachability)
                all_gnn_probs.append(gnn_prob)

    if not all_gnn_probs or len(all_gnn_probs) < 3:
        return {}

    # ====================================================================
    # COMPUTE IMPORTANCE VIA CORRELATION
    # ====================================================================

    feature_importance = {}
    features_to_analyze = {
        'H* Preservation': all_h_pres_values,
        'Transition Growth (inverse)': [1.0 / max(0.1, x) for x in all_trans_growth_values],
        'OPP Score': all_opp_scores,
        'Label Combinability': all_label_scores,
        'Reachability Ratio': all_reachability,
    }

    for feature_name, feature_values in features_to_analyze.items():
        if len(feature_values) < 3:
            continue

        feature_array = np.array(feature_values)
        gnn_array = np.array(all_gnn_probs)

        # Skip constant arrays
        if np.std(feature_array) < 1e-6 or np.std(gnn_array) < 1e-6:
            continue

        # Skip invalid data
        if np.any(~np.isfinite(feature_array)) or np.any(~np.isfinite(gnn_array)):
            continue

        try:
            corr, p_val = stats.spearmanr(feature_array, gnn_array)

            if np.isnan(corr) or np.isinf(corr):
                continue

            importance = abs(corr)
            feature_importance[feature_name] = {
                'importance': float(importance),
                'correlation': float(corr),
                'p_value': float(p_val),
                'significant': float(p_val) < 0.05,
            }
        except Exception:
            continue

    return {
        'feature_importance': feature_importance,
        'num_decisions_analyzed': len(all_gnn_probs),
    }