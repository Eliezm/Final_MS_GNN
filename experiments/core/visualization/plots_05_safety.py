#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 8: SAFETY ANALYSIS
=======================
Analyze problem soundness: dead-end prevention and solvability.

Shows:
- Dead-end creation timeline
- Cumulative dead-end risk
- Solvability maintenance by phase
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

from experiments.core.logging import EpisodeMetrics
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, format_plot_labels,
    create_plots_directory, save_plot_safely,
)


def plot_dead_end_analysis(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """Create dead-end risk timeline plot (with legend fixes)."""
    plt = setup_matplotlib()
    if not plt:
        return None

    if not training_log:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ✅ EXTRACT DATA
    episodes = []
    dead_end_penalties = []
    is_solvable_flags = []
    cumulative_dead_ends = []
    cumulative_sum = 0

    for metrics in training_log:
        if metrics.error is None:
            episodes.append(metrics.episode)

            penalty = abs(metrics.penalty_dead_end)
            dead_end_penalties.append(penalty)
            is_solvable_flags.append(1.0 if metrics.is_solvable else 0.0)

            cumulative_sum += penalty
            cumulative_dead_ends.append(cumulative_sum)

    if not episodes:
        return None

    # ✅ CREATE FIGURE
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Dead-End Creation Risk Analysis',
                 fontsize=14, fontweight='bold')

    # PANEL 1: PER-EPISODE PENALTY
    colors_penalty = ['#e74c3c' if p > 0.3 else '#f39c12' if p > 0.1 else '#27ae60'
                      for p in dead_end_penalties]

    ax1.scatter(episodes, dead_end_penalties, alpha=0.4, s=30, c=colors_penalty,
                label='Per-episode penalty', edgecolor='black', linewidth=0.5)

    # Trend line
    window = min(10, len(dead_end_penalties) // 4)
    if window > 1 and len(dead_end_penalties) > window:
        rolling_avg = np.convolve(dead_end_penalties, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(dead_end_penalties)), rolling_avg,
                 linewidth=2.5, color='darkred', label='Trend (rolling avg)')

    format_plot_labels(ax1, 'Episode', 'Dead-End Penalty (0-0.5)',
                       'Per-Episode Dead-End Penalty')

    # ✅ FIX: Proper legend
    ax1.legend(fontsize=9, loc='best')

    # PANEL 2: CUMULATIVE DEAD-ENDS
    ax2.fill_between(episodes, cumulative_dead_ends, alpha=0.3, color='red',
                     label='Cumulative penalty')
    ax2.plot(episodes, cumulative_dead_ends, linewidth=2, color='darkred',
             marker='o', markersize=3, label='Cumulative trajectory')

    format_plot_labels(ax2, 'Episode', 'Cumulative Dead-End Penalty',
                       'Cumulative Dead-End Creation Risk')
    ax2.legend(fontsize=9, loc='best')

    # PANEL 3: SOLVABILITY BY PHASE
    n_episodes_solved = len(is_solvable_flags)
    solvable_per_phase = []
    phase_names = []

    for phase_idx in range(3):
        start = (phase_idx * n_episodes_solved) // 3
        end = ((phase_idx + 1) * n_episodes_solved) // 3

        if start < end:
            phase_values = is_solvable_flags[start:end]
            solve_rate = sum(phase_values) / len(phase_values) * 100
            solvable_per_phase.append(solve_rate)
            phase_names.append(['Early', 'Mid', 'Late'][phase_idx])

    if solvable_per_phase:
        colors_phase = ['#e74c3c' if r < 70 else '#f39c12' if r < 90 else '#27ae60'
                        for r in solvable_per_phase]

        bars = ax3.bar(range(len(phase_names)), solvable_per_phase, color=colors_phase,
                       alpha=0.7, edgecolor='black', linewidth=1.5, label='Solvable rate')

        # Add value labels
        for bar, val in zip(bars, solvable_per_phase):
            ax3.text(bar.get_x() + bar.get_width() / 2., val,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax3.axhline(y=95, color='green', linestyle='--', linewidth=2, label='Target: >95%')

        format_plot_labels(ax3, 'Training Phase', 'Solvable Episodes (%)',
                           'Solvability Maintenance by Phase')
        ax3.set_xticks(range(len(phase_names)))
        ax3.set_xticklabels(phase_names)
        ax3.set_ylim([0, 120])
        ax3.legend(fontsize=9, loc='best')

    # ✅ SAVE
    plt.tight_layout()
    plot_path = plots_dir / "07_dead_end_timeline.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None