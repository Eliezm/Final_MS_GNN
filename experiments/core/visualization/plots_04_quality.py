#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 6-7: HEURISTIC QUALITY ANALYSIS - FIXED
====================================
Analyze bisimulation preservation and H* protection.

Shows:
- H* preservation over time with targets
- Bisimulation preservation rates by phase
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

from experiments.core.logging import EpisodeMetrics
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, format_plot_labels, add_target_line,
    create_plots_directory, save_plot_safely,
)


def plot_bisimulation_preservation(
        training_log: List[EpisodeMetrics],
        bisim_analysis: Dict[str, Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create bisimulation preservation analysis plots.

    Panel 1: H* preservation timeline
    Panel 2: Preservation rates by training phase

    Args:
        training_log: Episode metrics
        bisim_analysis: From analyze_bisimulation_preservation()
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

    episodes = []
    h_pres_values = []

    for metrics in training_log:
        if metrics.error is None:
            episodes.append(metrics.episode)
            h_pres_values.append(metrics.h_star_preservation)

    if not episodes:
        return None

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Bisimulation Preservation Analysis (Greedy Algorithm Quality)',
                 fontsize=14, fontweight='bold')

    # ====================================================================
    # PANEL 1: H* PRESERVATION TIMELINE
    # ====================================================================

    ax1.scatter(episodes, h_pres_values, alpha=0.3, s=20, label='Per-episode', color='blue')

    # Rolling average
    window = min(10, len(h_pres_values) // 4)
    if window > 1 and len(h_pres_values) > window:
        rolling_avg = np.convolve(h_pres_values, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(h_pres_values)), rolling_avg, linewidth=2.5,
                 color='darkblue', label=f'Rolling avg (window={window})')

    # Target lines
    add_target_line(ax1, 1.0, 'Perfect preservation', color='green', linestyle='-')
    add_target_line(ax1, 0.95, 'Target (>0.95)', color='orange')

    format_plot_labels(ax1, 'Episode', 'H* Preservation Ratio',
                       'Bisimulation Preservation During Training')
    ax1.set_ylim([0.7, 1.1])
    ax1.legend(fontsize=9)

    # ====================================================================
    # PANEL 2: PHASE ANALYSIS - ✅ FIXED: Handle variable phase counts
    # ====================================================================

    n_episodes = len(h_pres_values)
    phase_names_full = ['Early', 'Mid', 'Late']
    phase_rates = []
    phase_names_active = []  # ✅ Track which phases have data

    for phase_idx in range(3):
        start = (phase_idx * n_episodes) // 3
        end = ((phase_idx + 1) * n_episodes) // 3

        # ✅ Only add phase if it has data
        if start < end:
            phase_values = h_pres_values[start:end]
            good_count = sum(1 for v in phase_values if v >= 0.95)
            rate = good_count / len(phase_values) * 100
            phase_rates.append(rate)
            phase_names_active.append(phase_names_full[phase_idx])

    # ✅ Create colors for active phases only
    colors_phase = ['#27ae60' if r >= 95 else '#f39c12' if r >= 80 else '#e74c3c'
                    for r in phase_rates]

    # ✅ Use active phases for plotting
    if phase_rates:  # Guard against empty data
        bars = ax2.bar(range(len(phase_names_active)), phase_rates,
                       color=colors_phase,
                       alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, phase_rates):
            ax2.text(bar.get_x() + bar.get_width() / 2., val,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

        add_target_line(ax2, 95, 'Target: 95%', color='green')

        ax2.set_ylabel('Episodes with H* Preservation > 0.95 (%)', fontweight='bold')
        ax2.set_xlabel('Training Phase', fontweight='bold')
        ax2.set_title('Bisimulation Preservation by Phase', fontweight='bold')

        # ✅ Set ticks to match active phases
        ax2.set_xticks(range(len(phase_names_active)))
        ax2.set_xticklabels(phase_names_active)
        ax2.set_ylim([0, 120])
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        # ✅ Handle case with no valid episodes
        ax2.text(0.5, 0.5, 'Insufficient data for phase analysis',
                 ha='center', va='center', fontsize=12,
                 transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "06_bisimulation_preservation.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None