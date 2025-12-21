#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 9-10: TRANSITION SYSTEM CONTROL
=====================================
Analyze abstraction size control and transition explosion risk.

Shows:
- Label combinability impact on merge quality
- Transition explosion prediction accuracy
- GNN confidence distributions
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

from experiments.core.logging import EpisodeMetrics
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, format_plot_labels,
    create_plots_directory, save_plot_safely,
)


def plot_label_reduction_impact(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create label combinability impact plots.

    Shows relationship between label combinability and merge quality.

    Args:
        training_log: Episode metrics
        output_dir: Output directory

    Returns:
        Path to saved plot or None
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not training_log:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ====================================================================
    # EXTRACT DATA
    # ====================================================================

    label_scores = []
    opp_scores = []
    rewards = []
    episodes = []

    for metrics in training_log:
        if metrics.error is None:
            episodes.append(metrics.episode)
            label_scores.append(metrics.label_combinability_score)
            opp_scores.append(metrics.opp_score)
            rewards.append(metrics.reward)

    if not label_scores:
        return None

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Label Combinability Impact on Merge Quality\n'
                 '(Helmert et al. 2014 - Local Equivalence)',
                 fontsize=14, fontweight='bold')

    # ====================================================================
    # PANEL 1: SCATTER WITH TREND
    # ====================================================================

    scatter = ax1.scatter(label_scores, rewards, alpha=0.5, s=50,
                          c=range(len(label_scores)), cmap='viridis',
                          edgecolor='black', linewidth=0.5)

    # Trend line
    if len(label_scores) > 2:
        z = np.polyfit(label_scores, rewards, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(label_scores), max(label_scores), 100)
        ax1.plot(x_line, p(x_line), "r--", linewidth=2.5, label='Trend')

    format_plot_labels(ax1, 'Label Combinability Score', 'Episode Reward',
                       'Label Combinability vs Reward')
    ax1.legend(fontsize=9)

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Episode Number', rotation=270, labelpad=20)

    # ====================================================================
    # PANEL 2: COMPONENT EVOLUTION
    # ====================================================================

    ax2.plot(episodes, label_scores, 'o-', linewidth=2, label='Label Score',
             alpha=0.7, markersize=4, color='blue')

    ax2_twin1 = ax2.twinx()
    ax2_twin1.plot(episodes, opp_scores, 's-', linewidth=2, label='OPP Score',
                   color='orange', alpha=0.7, markersize=4)

    ax2_twin2 = ax2_twin1.twinx()
    ax2_twin2.spines['right'].set_position(('outward', 60))
    ax2_twin2.plot(episodes, rewards, '^-', linewidth=2, label='Reward',
                   color='green', alpha=0.7, markersize=4)

    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Label Combinability', fontweight='bold', color='blue')
    ax2_twin1.set_ylabel('OPP Score', fontweight='bold', color='orange')
    ax2_twin2.set_ylabel('Reward', fontweight='bold', color='green')

    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin1.tick_params(axis='y', labelcolor='orange')
    ax2_twin2.tick_params(axis='y', labelcolor='green')

    ax2.set_title('Label Combinability + OPP Score Evolution', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin1.get_legend_handles_labels()
    lines3, labels3 = ax2_twin2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
               loc='upper left', fontsize=9)

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "08_label_reduction_impact.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_transition_explosion(
        training_log: List[EpisodeMetrics],
        explosion_analysis: Dict[str, Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create transition explosion risk analysis plots.

    Panel 1: Scatter - transition growth vs GNN confidence
    Panel 2: Explosion prediction accuracy
    Panel 3: GNN confidence distribution
    Panel 4: Summary statistics

    Args:
        training_log: Episode metrics
        explosion_analysis: From analyze_transition_explosion_risk()
        output_dir: Output directory

    Returns:
        Path to saved plot or None
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not training_log:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ====================================================================
    # EXTRACT DATA
    # ====================================================================

    all_trans_growth = []
    all_gnn_probs = []
    explosion_threshold = 5.0

    for metrics in training_log:
        if metrics.error is None and metrics.merge_decisions_per_step:
            for decision_dict in metrics.merge_decisions_per_step:
                trans_growth = decision_dict.get('transition_growth', 1.0)
                gnn_prob = decision_dict.get('gnn_action_probability', 0.5)

                all_trans_growth.append(trans_growth)
                all_gnn_probs.append(gnn_prob)

    if not all_trans_growth:
        return None

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Transition Explosion Risk Analysis',
                 fontsize=14, fontweight='bold')

    # ====================================================================
    # PANEL 1: SCATTER
    # ====================================================================

    is_explosion = [x > explosion_threshold for x in all_trans_growth]
    colors_scatter = ['#e74c3c' if x else '#27ae60' for x in is_explosion]

    ax1.scatter(all_gnn_probs, all_trans_growth, c=colors_scatter, alpha=0.5,
                s=50, edgecolors='black', linewidth=0.5)

    ax1.axhline(y=explosion_threshold, color='red', linestyle='--', linewidth=2,
                label='Explosion threshold (5x growth)')
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.3,
                label='Neutral confidence (50%)')

    format_plot_labels(ax1, 'GNN Action Probability', 'Transition Growth Ratio',
                       'Transition Growth vs GNN Confidence')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0.5, min(max(all_trans_growth) * 1.1, 20)])
    ax1.legend(fontsize=9, loc='upper right')

    # ====================================================================
    # PANEL 2: ACCURACY
    # ====================================================================

    accuracy = explosion_analysis.get('explosion_prediction_accuracy', 0.0)
    baseline = 0.5

    metrics_names = ['GNN Prediction', 'Random Baseline']
    metrics_values = [accuracy, baseline]
    colors_metrics = ['#27ae60' if accuracy > 0.7 else '#f39c12' if accuracy > 0.6 else '#e74c3c',
                      '#95a5a6']

    bars = ax2.bar(metrics_names, metrics_values, color=colors_metrics, alpha=0.7,
                   edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels
    for bar, val in zip(bars, metrics_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val,
                 f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (80%)')

    format_plot_labels(ax2, '', 'Prediction Accuracy',
                       'Can GNN Predict Transition Explosions?')
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=9)

    # ====================================================================
    # PANEL 3: CONFIDENCE DISTRIBUTION
    # ====================================================================

    explosion_probs = [p for e, p in zip(is_explosion, all_gnn_probs) if e]
    safe_probs = [p for e, p in zip(is_explosion, all_gnn_probs) if not e]

    if explosion_probs and safe_probs:
        ax3.hist(safe_probs, bins=20, alpha=0.6, label='Non-explosion merges',
                 color='#27ae60', edgecolor='black')
        ax3.hist(explosion_probs, bins=20, alpha=0.6, label='Explosion merges',
                 color='#e74c3c', edgecolor='black')

        ax3.axvline(x=np.mean(safe_probs), color='darkgreen', linestyle='--',
                    linewidth=2, label=f'Mean (safe): {np.mean(safe_probs):.2f}')
        ax3.axvline(x=np.mean(explosion_probs), color='darkred', linestyle='--',
                    linewidth=2, label=f'Mean (explosion): {np.mean(explosion_probs):.2f}')

        format_plot_labels(ax3, 'GNN Action Probability', 'Frequency',
                           'GNN Confidence Distribution by Merge Safety')
        ax3.legend(fontsize=9)

    # ====================================================================
    # PANEL 4: SUMMARY
    # ====================================================================

    ax4.axis('off')

    summary_text = f"""TRANSITION EXPLOSION RISK SUMMARY

Detection Capability:
  • Prediction Accuracy: {explosion_analysis.get('explosion_prediction_accuracy', 0):.1%}
  • Confidence Separation: {explosion_analysis.get('confidence_separation', 0):.3f}
  • GNN Learned to Avoid: {explosion_analysis.get('gnn_learned_to_avoid_explosions', False)}

Merge Classification:
  • Explosions Detected (>5x): {explosion_analysis.get('num_explosions_detected', 0)}
  • Safe Merges: {explosion_analysis.get('num_safe_merges', 0)}
  • Total Analyzed: {explosion_analysis.get('total_transitions_analyzed', 0)}

GNN Confidence Analysis:
  • Avg conf for explosion merges: {explosion_analysis.get('avg_gnn_confidence_for_explosions', 0):.3f}
  • Avg conf for safe merges: {explosion_analysis.get('avg_gnn_confidence_for_safe', 0):.3f}
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "09_transition_explosion.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None