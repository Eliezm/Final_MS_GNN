#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 1: LEARNING CURVES - FIXED
================================
Visualize training progress from a learning dynamics perspective.

FIXES:
✅ Proper shape handling for rolling averages
✅ Handles variable episode counts
✅ No array indexing mismatches
"""

from pathlib import Path
from typing import List, Optional
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt

from experiments.core.logging import EpisodeMetrics
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, safe_mean_and_rolling, format_plot_labels,
    add_target_line, create_plots_directory, save_plot_safely,
)


def plot_learning_curves(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create 4-panel learning curves plot (FIXED v2).

    Panel 1: Reward trajectory (per-episode + rolling avg)
    Panel 2: H* preservation over time
    Panel 3: Problem coverage distribution
    Panel 4: Failure type breakdown

    Args:
        training_log: List of episode metrics
        output_dir: Output directory

    Returns:
        Path to saved plot or None if failed
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

    episodes = list(range(len(training_log)))
    rewards = np.array([m.reward for m in training_log])
    h_preservations = np.array([m.h_star_preservation for m in training_log])

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ====================================================================
    # PANEL 1: REWARD CURVE (FIXED)
    # ====================================================================

    ax1 = axes[0, 0]

    ax1.plot(episodes, rewards, alpha=0.3, label='Per-episode', linewidth=1, color='blue')

    # ✅ FIX: Properly handle rolling average with correct indices
    window = min(10, max(2, len(rewards) // 4)) if len(rewards) > 10 else 1

    if window > 1 and len(rewards) > window:
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        # ✅ FIX: Align rolling average indices correctly
        rolling_episodes = list(range(window - 1, len(rewards)))
        ax1.plot(rolling_episodes, rolling_avg, linewidth=2.5, label=f'Rolling avg (w={window})',
                 color='darkblue')

    format_plot_labels(ax1, 'Episode', 'Reward', 'Learning Curve - Reward')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ====================================================================
    # PANEL 2: H* PRESERVATION (FIXED)
    # ====================================================================

    ax2 = axes[0, 1]

    ax2.plot(episodes, h_preservations, alpha=0.3, color='green',
             label='Per-episode', linewidth=1)

    # ✅ FIX: Same rolling average fix for H* preservation
    if window > 1 and len(h_preservations) > window:
        rolling_h = np.convolve(h_preservations, np.ones(window) / window, mode='valid')
        rolling_episodes = list(range(window - 1, len(h_preservations)))
        ax2.plot(rolling_episodes, rolling_h, linewidth=2.5, color='darkgreen',
                 label=f'Rolling avg (w={window})')

    add_target_line(ax2, 1.0, 'Perfect preservation', color='red')
    add_target_line(ax2, 0.95, 'Target (>0.95)', color='orange')

    format_plot_labels(ax2, 'Episode', 'H* Preservation',
                       'Learning Curve - H* Preservation')
    ax2.set_ylim([0.7, 1.05])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ====================================================================
    # PANEL 3: PROBLEM COVERAGE
    # ====================================================================

    ax3 = axes[1, 0]

    by_problem = defaultdict(int)
    for m in training_log:
        by_problem[m.problem_name] += 1

    problem_names = sorted(by_problem.keys())
    problem_counts = [by_problem[p] for p in problem_names]

    ax3.bar(range(len(problem_counts)), problem_counts, color='steelblue',
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(problem_names)))
    ax3.set_xticklabels([p[:10] for p in problem_names], rotation=45, ha='right', fontsize=9)

    format_plot_labels(ax3, 'Problem', 'Episode Count', 'Problem Coverage Distribution')
    ax3.grid(True, alpha=0.3, axis='y')

    # ====================================================================
    # PANEL 4: FAILURE DISTRIBUTION
    # ====================================================================

    ax4 = axes[1, 1]

    failure_types = defaultdict(int)
    for m in training_log:
        if m.failure_type:
            failure_types[m.failure_type] += 1

    if failure_types:
        labels = list(failure_types.keys())
        counts = list(failure_types.values())
        colors = ['#e74c3c', '#f39c12', '#e67e22', '#c0392b'][:len(counts)]

        ax4.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors,
                startangle=90)
        ax4.set_title('Failure Type Distribution', fontweight='bold', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No Failures Recorded ✓',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax4.transAxes, color='green')
        ax4.axis('off')

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "01_learning_curves.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_step_reward_progression(
        training_log: List['EpisodeMetrics'],
        output_dir: str,
) -> Optional[Path]:
    """
    Plot 33: Step reward progression analysis.

    Shows:
    - Average reward by step position (early/mid/late)
    - Reward variance across episodes by position
    - Component contribution by step position
    """
    from experiments.core.analysis.analysis_step_rewards import analyze_step_rewards_from_training_log

    output_path = Path(output_dir) / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    analysis = analyze_step_rewards_from_training_log(training_log)

    if "error" in analysis:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Phase-wise average rewards
    ax1 = axes[0, 0]
    pos = analysis.get("step_position_analysis", {})
    phases = ['Early', 'Mid', 'Late']
    avgs = [
        pos.get('early_step_avg_reward', 0),
        pos.get('mid_step_avg_reward', 0),
        pos.get('late_step_avg_reward', 0),
    ]
    stds = [
        pos.get('early_step_std_reward', 0),
        pos.get('mid_step_std_reward', 0),
        pos.get('late_step_std_reward', 0),
    ]

    colors = ['#3498db', '#f39c12', '#2ecc71']
    bars = ax1.bar(phases, avgs, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward by Merge Phase')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, avg in zip(bars, avgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{avg:+.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Position-wise statistics
    ax2 = axes[0, 1]
    step_stats = analysis.get("step_position_stats", {})
    positions = sorted([int(k.split('_')[1]) for k in step_stats.keys()])
    pos_avgs = [step_stats.get(f"position_{p}", {}).get('mean_reward', 0) for p in positions]
    pos_stds = [step_stats.get(f"position_{p}", {}).get('std_reward', 0) for p in positions]

    ax2.errorbar(positions, pos_avgs, yerr=pos_stds, marker='o', capsize=3,
                 color='#9b59b6', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Normalized Step Position (0=start, 10=end)')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Reward by Step Position')
    ax2.grid(alpha=0.3)

    # Plot 3: Component trends
    ax3 = axes[1, 0]
    comp = analysis.get("step_component_analysis", {})

    components = ['H* Pres.', 'Trans. Growth', 'OPP', 'Label Comb.']
    early_vals = [
        comp.get('early_h_preservation', 1.0),
        comp.get('early_transition_growth', 1.0) / 5.0,  # Normalize
        comp.get('early_opp_score', 0.5),
        comp.get('early_label_comb', 0.5),
    ]
    late_vals = [
        comp.get('late_h_preservation', 1.0),
        comp.get('late_transition_growth', 1.0) / 5.0,
        comp.get('late_opp_score', 0.5),
        comp.get('late_label_comb', 0.5),
    ]

    x = np.arange(len(components))
    width = 0.35

    ax3.bar(x - width / 2, early_vals, width, label='Early', color='#3498db', alpha=0.8)
    ax3.bar(x + width / 2, late_vals, width, label='Late', color='#2ecc71', alpha=0.8)
    ax3.set_ylabel('Normalized Value')
    ax3.set_title('Component Values: Early vs Late Steps')
    ax3.set_xticks(x)
    ax3.set_xticklabels(components)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Quality distribution
    ax4 = axes[1, 1]
    quality_dist = analysis.get("quality_distribution", {})

    categories = ['excellent', 'good', 'neutral', 'poor', 'bad']
    early_counts = [quality_dist.get('early', {}).get(c, 0) for c in categories]
    late_counts = [quality_dist.get('late', {}).get(c, 0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    ax4.bar(x - width / 2, early_counts, width, label='Early', color='#3498db', alpha=0.8)
    ax4.bar(x + width / 2, late_counts, width, label='Late', color='#2ecc71', alpha=0.8)
    ax4.set_ylabel('Count')
    ax4.set_title('Merge Quality Distribution: Early vs Late')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Excellent', 'Good', 'Neutral', 'Poor', 'Bad'])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('Step-Wise Reward Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_path / "33_step_reward_progression.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Step reward plot saved: {plot_path}")
    return plot_path