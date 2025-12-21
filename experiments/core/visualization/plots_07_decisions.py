#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 11-12: MERGE DECISION ANALYSIS
===================================
Analyze GNN merge strategy and decision quality.

✅ FIXED: Handle empty decision_quality_analysis
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from collections import defaultdict
import logging

from experiments.core.logging import EpisodeMetrics
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, format_plot_labels,
    create_plots_directory, save_plot_safely,
)

logger = logging.getLogger(__name__)


def plot_causal_alignment(
        training_log: List[EpisodeMetrics],
        causal_analysis: Dict[str, Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create causal alignment analysis plots.

    Shows whether GNN merge order respects causal structure.

    Args:
        training_log: Episode metrics
        causal_analysis: From analyze_causal_alignment()
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
    # EXTRACT MERGE SEQUENCE DATA
    # ====================================================================

    merge_h_pres = []
    merge_quality = []
    merge_steps = []
    step_counter = 0

    for metrics in training_log:
        if metrics.error is None and metrics.merge_decisions_per_step:
            for decision_dict in metrics.merge_decisions_per_step:
                h_pres = decision_dict.get('h_preservation', 1.0)
                is_good = decision_dict.get('is_good_merge', False)

                merge_h_pres.append(h_pres)
                merge_quality.append(1.0 if is_good else 0.0)
                merge_steps.append(step_counter)
                step_counter += 1

    if not merge_steps:
        return None

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Causal Alignment & Merge Order Strategy',
                 fontsize=14, fontweight='bold')

    # ====================================================================
    # PANEL 1: H* PRESERVATION OVER MERGE SEQUENCE
    # ====================================================================

    ax1.scatter(merge_steps, merge_h_pres, alpha=0.3, s=20, label='Per-merge')

    window = min(20, len(merge_h_pres) // 4)
    if window > 1 and len(merge_h_pres) > window:
        rolling_avg = np.convolve(merge_h_pres, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(merge_h_pres)), rolling_avg,
                 linewidth=2.5, color='darkblue', label='Rolling avg')

    ax1.axhline(y=0.95, color='green', linestyle='--', linewidth=2,
                alpha=0.5, label='Target (0.95)')

    format_plot_labels(ax1, 'Merge Sequence Step', 'H* Preservation',
                       'H* Preservation Over Merge Sequence')
    ax1.set_ylim([0.7, 1.1])
    ax1.legend(fontsize=9)

    # ====================================================================
    # PANEL 2: MERGE QUALITY BY PHASE
    # ====================================================================

    n_merges = len(merge_quality)
    if n_merges > 0:
        early_quality = np.mean(merge_quality[:n_merges // 3]) if n_merges > 0 else 0
        mid_quality = np.mean(merge_quality[n_merges // 3:2 * n_merges // 3]) if n_merges > 0 else 0
        late_quality = np.mean(merge_quality[-n_merges // 3:]) if n_merges > 0 else 0

        phases = ['Early\n(0-33%)', 'Mid\n(33-67%)', 'Late\n(67-100%)']
        qualities = [early_quality, mid_quality, late_quality]
        colors = ['#e74c3c' if q < 0.5 else '#f39c12' if q < 0.7 else '#27ae60'
                  for q in qualities]

        bars = ax2.bar(range(len(phases)), qualities, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, qual in zip(bars, qualities):
            ax2.text(bar.get_x() + bar.get_width() / 2, qual,
                     f'{qual:.1%}', ha='center', va='bottom', fontweight='bold')

        ax2.set_xticks(range(len(phases)))
        ax2.set_xticklabels(phases)

        format_plot_labels(ax2, 'Training Phase', '% Good Merges',
                           'Merge Quality by Training Phase')
        ax2.set_ylim([0, 1.1])

    # ====================================================================
    # PANEL 3: VARIABLE PAIR DISTANCE DISTRIBUTION
    # ====================================================================

    pair_distances = []
    for metrics in training_log:
        if metrics.error is None and metrics.merge_decisions_per_step:
            for decision_dict in metrics.merge_decisions_per_step:
                pair = decision_dict.get('selected_merge_pair', (0, 1))
                distance = abs(pair[1] - pair[0])
                pair_distances.append(distance)

    if pair_distances:
        ax3.hist(pair_distances, bins=20, color='steelblue', alpha=0.7,
                 edgecolor='black', linewidth=1.5)
        ax3.axvline(x=np.mean(pair_distances), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(pair_distances):.1f}')

        format_plot_labels(ax3, 'Variable Distance in Merge Pair', 'Frequency',
                           'Distribution of Variable Distances in Merges')
        ax3.legend(fontsize=9)

    # ====================================================================
    # PANEL 4: SUMMARY
    # ====================================================================

    ax4.axis('off')

    alignment_data = causal_analysis or {}

    summary_text = f"""MERGE ORDER STRATEGY ANALYSIS

Early Phase (0-33% of merges):
  * H* Preservation: {alignment_data.get('early_h_preservation', 0):.3f}
  * Good Merge Rate: {alignment_data.get('early_good_rate', 0):.1%}

Late Phase (67-100% of merges):
  * H* Preservation: {alignment_data.get('late_h_preservation', 0):.3f}
  * Good Merge Rate: {alignment_data.get('late_good_rate', 0):.1%}

Strategy Alignment Score: {alignment_data.get('strategy_alignment_score', 0):.3f}
  (>1.0 = Improving Strategy)

Total Merges Analyzed: {alignment_data.get('num_merges_analyzed', 0)}
Learning Trend: {alignment_data.get('learning_trend', 'unknown')}
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "10_causal_alignment.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_gnn_decision_quality(
        decision_quality_analysis: Dict[str, Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create GNN decision quality analysis plots.

    ✅ FIXED: Handle empty or None decision_quality_analysis
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ✅ FIX: Handle empty decision_quality_analysis
    if not decision_quality_analysis:
        logger.warning("No decision quality analysis data - generating placeholder plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'GNN Decision Quality Analysis\n\n'
                          'No decision quality data available.\n\n'
                          'This occurs when:\n'
                          '1. Training log has no merge decisions\n'
                          '2. Decision analysis was not computed\n'
                          '3. All episodes failed\n\n'
                          'Try running with more episodes.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "11_gnn_decision_quality.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GNN Decision Quality Analysis',
                 fontsize=14, fontweight='bold')

    # ====================================================================
    # PANEL 1: CONFUSION MATRIX
    # ====================================================================

    confusion = decision_quality_analysis.get('confusion_matrix', {})

    correct_good = confusion.get('correct_good_merges', 0)
    incorrect_bad = confusion.get('incorrect_bad_merges', 0)
    correct_bad = confusion.get('correct_bad_merges', 0)
    incorrect_good = confusion.get('incorrect_good_merges', 0)

    matrix_data = np.array([
        [correct_good, incorrect_bad],
        [incorrect_good, correct_bad]
    ])

    im = ax1.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Good Merge\n(True Positive)', 'Bad Merge\n(True Negative)'],
                        fontweight='bold', fontsize=9)
    ax1.set_yticklabels(['GNN Selected\n(High Prob)', 'GNN Avoided\n(Low Prob)'],
                        fontweight='bold', fontsize=9)

    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, int(matrix_data[i, j]),
                            ha="center", va="center", color="black",
                            fontweight='bold', fontsize=14)

    ax1.set_title('Confusion Matrix: GNN Decisions vs Merge Quality',
                  fontweight='bold', fontsize=11)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Count', rotation=270, labelpad=20)

    # ====================================================================
    # PANEL 2: ACCURACY METRICS
    # ====================================================================

    accuracy = decision_quality_analysis.get('gnn_accuracy', 0.0)
    acc_good = decision_quality_analysis.get('gnn_accuracy_good_merges', accuracy)
    acc_bad = decision_quality_analysis.get('gnn_accuracy_bad_merges', accuracy)

    metrics = ['Overall\nAccuracy', 'Good Merge\nRecall', 'Bad Merge\nRecall']
    values = [accuracy, acc_good, acc_bad]
    colors = ['#27ae60' if v > 0.8 else '#f39c12' if v > 0.6 else '#e74c3c'
              for v in values]

    bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black',
                   linewidth=1.5, width=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5,
                linewidth=2, label='Target (80%)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5,
                linewidth=2, label='Baseline (50%)')

    format_plot_labels(ax2, '', 'Accuracy',
                       'Decision Accuracy by Merge Category')
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=9)

    # ====================================================================
    # PANEL 3: CONFIDENCE DISTRIBUTION
    # ====================================================================

    confidence_by_cat = decision_quality_analysis.get('confidence_by_category', {})

    categories = sorted(confidence_by_cat.keys())
    if categories:
        means = [confidence_by_cat[cat]['mean_confidence'] for cat in categories]
        stds = [confidence_by_cat[cat]['std_confidence'] for cat in categories]

        x_pos = np.arange(len(categories))
        colors_cats = ['#e74c3c', '#f39c12', '#f1c40f', '#a4d65e', '#27ae60'][:len(categories)]

        ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                color=colors_cats, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([c.replace('_', ' ').title() for c in categories],
                            rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('GNN Action Probability', fontweight='bold')
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='Neutral (50%)')
        ax3.set_title('GNN Confidence by Merge Quality Category', fontweight='bold')
        ax3.set_ylim([0, 1.0])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No confidence data by category',
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.axis('off')

    # ====================================================================
    # PANEL 4: SUMMARY
    # ====================================================================

    ax4.axis('off')

    summary_text = f"""GNN DECISION QUALITY SUMMARY

Overall Accuracy: {accuracy:.1%}
  * Correctly selected good merges: {acc_good:.1%}
  * Correctly avoided bad merges: {acc_bad:.1%}

Decisions Analyzed: {decision_quality_analysis.get('total_decisions_analyzed', 0)}

Decision Entropy: {decision_quality_analysis.get('decision_entropy', 0):.3f}
  (Lower = more confident, Higher = more uncertain)
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "11_gnn_decision_quality.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None