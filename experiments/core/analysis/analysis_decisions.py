#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DECISION ANALYSIS - Merge strategy quality
==========================================
Analyzes whether the GNN learned to make good merge decisions.

Answers:
- Does GNN distinguish good merges from bad?
- Does merge order respect causal structure?
- Can GNN predict transition explosions?
"""

from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

from experiments.core.logging import EpisodeMetrics


def analyze_causal_alignment(
        training_log: List[EpisodeMetrics],
        output_dir=None,
) -> Dict[str, Any]:
    """
    Analyze whether GNN merge order respects causal graph structure.

    Good strategy: early merges preserve quality, later merges control size.
    """

    if not training_log:
        return {}

    # ====================================================================
    # COLLECT MERGE SEQUENCE
    # ====================================================================

    merge_order = []
    for metrics in training_log:
        if metrics.error is None and metrics.merge_decisions_per_step:
            for i, decision_dict in enumerate(metrics.merge_decisions_per_step):
                merge_pair = decision_dict.get('selected_merge_pair', (0, 0))
                is_good = decision_dict.get('is_good_merge', False)
                h_pres = decision_dict.get('h_preservation', 1.0)

                merge_order.append({
                    'step': i,
                    'pair': merge_pair,
                    'h_preservation': h_pres,
                    'is_good': is_good,
                })

    if not merge_order:
        return {}

    # ====================================================================
    # PHASE ANALYSIS
    # ====================================================================

    early_merges = merge_order[:len(merge_order) // 3]
    late_merges = merge_order[-len(merge_order) // 3:]

    early_h_pres = np.mean([m['h_preservation'] for m in early_merges]) if early_merges else 1.0
    late_h_pres = np.mean([m['h_preservation'] for m in late_merges]) if late_merges else 1.0

    early_good_rate = sum(1 for m in early_merges if m['is_good']) / max(1, len(early_merges))
    late_good_rate = sum(1 for m in late_merges if m['is_good']) / max(1, len(late_merges))

    # ====================================================================
    # PAIR DISTANCE ANALYSIS
    # ====================================================================

    pair_distances = [abs(m['pair'][1] - m['pair'][0]) for m in merge_order]
    avg_pair_distance = np.mean(pair_distances) if pair_distances else 0

    # ====================================================================
    # ALIGNMENT SCORE
    # ====================================================================

    strategy_alignment_score = late_h_pres / max(0.01, early_h_pres)

    return {
        'early_h_preservation': float(early_h_pres),
        'late_h_preservation': float(late_h_pres),
        'early_good_rate': float(early_good_rate),
        'late_good_rate': float(late_good_rate),
        'avg_pair_distance': float(avg_pair_distance),
        'strategy_alignment_score': float(strategy_alignment_score),
        'learning_trend': 'improving' if strategy_alignment_score > 1.0 else 'declining',
        'num_merges_analyzed': len(merge_order),
    }


def analyze_transition_explosion_risk(
        training_log: List[EpisodeMetrics],
        output_dir=None,
) -> Dict[str, Any]:
    """
    Analyze whether GNN learned to predict transition explosions.

    Good GNN: low confidence when transition growth would be high.
    """

    if not training_log:
        return {}

    all_trans_growth = []
    all_gnn_probs = []

    for metrics in training_log:
        if metrics.error is None and metrics.merge_decisions_per_step:
            for decision_dict in metrics.merge_decisions_per_step:
                trans_growth = decision_dict.get('transition_growth', 1.0)
                gnn_prob = decision_dict.get('gnn_action_probability', 0.5)

                all_trans_growth.append(trans_growth)
                all_gnn_probs.append(gnn_prob)

    if not all_trans_growth:
        return {}

    # ====================================================================
    # EXPLOSION DETECTION ACCURACY
    # ====================================================================

    explosion_threshold = 5.0
    is_explosion = [x > explosion_threshold for x in all_trans_growth]

    # GNN should give low confidence for explosions
    correct_explosions = sum(
        1 for (exp, prob) in zip(is_explosion, all_gnn_probs)
        if (exp and prob < 0.3) or (not exp and prob >= 0.3)
    )
    explosion_prediction_accuracy = correct_explosions / max(1, len(is_explosion))

    # ====================================================================
    # CONFIDENCE SEPARATION
    # ====================================================================

    explosion_probs = [p for e, p in zip(is_explosion, all_gnn_probs) if e]
    safe_probs = [p for e, p in zip(is_explosion, all_gnn_probs) if not e]

    avg_conf_explosion = np.mean(explosion_probs) if explosion_probs else 0.5
    avg_conf_safe = np.mean(safe_probs) if safe_probs else 0.5

    separation = abs(avg_conf_safe - avg_conf_explosion)

    return {
        'explosion_prediction_accuracy': float(explosion_prediction_accuracy),
        'num_explosions_detected': sum(is_explosion),
        'num_safe_merges': len(is_explosion) - sum(is_explosion),
        'avg_gnn_confidence_for_explosions': float(avg_conf_explosion),
        'avg_gnn_confidence_for_safe': float(avg_conf_safe),
        'confidence_separation': float(separation),
        'gnn_learned_to_avoid_explosions': separation > 0.2,
        'total_transitions_analyzed': len(all_trans_growth),
    }


def analyze_gnn_decision_quality(
        decision_traces_by_episode: Dict[int, List[Dict]],
        output_dir=None,
) -> Dict[str, Any]:
    """
    Analyze whether GNN discriminates good vs bad merges.

    High quality: correctly selects good merges, avoids bad merges.
    """

    if not decision_traces_by_episode:
        return {}

    all_decisions = []
    for episode_id, traces in decision_traces_by_episode.items():
        all_decisions.extend(traces)

    if not all_decisions:
        return {}

    # ====================================================================
    # ACCURACY METRICS
    # ====================================================================

    correct_good = 0
    correct_bad = 0
    incorrect_good = 0
    incorrect_bad = 0

    confidence_by_category = {}

    for decision in all_decisions:
        gnn_prob = decision.get('gnn_action_probability', 0.5)

        # Classify decision
        if decision.get('is_good_merge'):
            if gnn_prob > 0.5:
                correct_good += 1
            else:
                incorrect_bad += 1
        elif decision.get('is_bad_merge'):
            if gnn_prob < 0.3:
                correct_bad += 1
            else:
                incorrect_good += 1

        # Confidence by category
        category = decision.get('merge_quality_category', 'unknown')
        if category not in confidence_by_category:
            confidence_by_category[category] = []
        confidence_by_category[category].append(gnn_prob)

    # ====================================================================
    # COMPUTE STATISTICS
    # ====================================================================

    total_decisions = len(all_decisions)
    correct_decisions = correct_good + correct_bad
    gnn_accuracy = correct_decisions / max(1, total_decisions)

    # Entropy
    all_probs = [d.get('gnn_action_probability', 0.5) for d in all_decisions]
    entropy = -np.mean([
        p * np.log(p + 1e-8) + (1 - p) * np.log(1 - p + 1e-8)
        for p in all_probs
    ])

    confidence_stats = {}
    for category, probs in confidence_by_category.items():
        if probs:
            confidence_stats[category] = {
                'mean_confidence': float(np.mean(probs)),
                'std_confidence': float(np.std(probs)),
                'count': len(probs),
            }

    return {
        'gnn_accuracy': float(gnn_accuracy),
        'confusion_matrix': {
            'correct_good_merges': correct_good,
            'incorrect_bad_merges': incorrect_bad,
            'correct_bad_merges': correct_bad,
            'incorrect_good_merges': incorrect_good,
        },
        'confidence_by_category': confidence_stats,
        'decision_entropy': float(entropy),
        'total_decisions_analyzed': total_decisions,
    }