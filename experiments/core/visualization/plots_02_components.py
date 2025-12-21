#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOTS 2-4: COMPONENT ANALYSIS
==============================
Analyze reward component contributions and stability.

Shows:
- Component trajectories over time
- Component stability metrics
- Merge quality heatmap
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import logging

from experiments.core.logging import EpisodeMetrics
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, format_plot_labels, create_plots_directory,
    save_plot_safely, COMPONENT_COLORS,
)

logger = logging.getLogger(__name__)


def plot_component_trajectories(
        training_log: List[EpisodeMetrics],
        component_analysis: Dict[str, Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create 5-panel component trajectory plot.

    Shows how each reward component evolves during training.

    Args:
        training_log: Episode metrics
        component_analysis: Results from analyze_component_trajectories()
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

    episodes = list(range(len(training_log)))

    # Define components
    components = [
        ('component_h_preservation', 'H* Preservation', '#2ecc71'),
        ('component_transition_control', 'Transition Control', '#3498db'),
        ('component_operator_projection', 'Operator Projection', '#e67e22'),
        ('component_label_combinability', 'Label Combinability', '#e74c3c'),
        ('component_bonus_signals', 'Bonus Signals', '#9b59b6'),
    ]

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Reward Component Evolution During Training',
                 fontsize=16, fontweight='bold')

    # ====================================================================
    # PLOT EACH COMPONENT
    # ====================================================================

    for idx, (attr_name, label, color) in enumerate(components):
        ax = axes[idx // 2, idx % 2]

        # Extract values
        values = [getattr(m, attr_name, 0.0) for m in training_log]

        # Plot per-episode
        ax.plot(episodes, values, alpha=0.4, color=color, label='Per-episode', linewidth=1)

        # Plot rolling average
        window = min(10, len(values) // 4) if len(values) > 4 else 1
        if window > 1 and len(values) > window:
            rolling_avg = np.convolve(values, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(values)), rolling_avg, linewidth=2.5,
                    color=color, label=f'Rolling avg (window={window})')

        # Baseline
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        format_plot_labels(ax, 'Episode', 'Component Reward', label)
        ax.legend(fontsize=9)

    # Remove extra subplot
    fig.delaxes(axes[2, 1])

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "02_component_trajectories.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_component_stability(
        component_analysis: Dict[str, Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create stability metrics bar chart.

    Shows stability (1 - variance) of each component.

    Args:
        component_analysis: Results from analyze_component_trajectories()
        output_dir: Output directory

    Returns:
        Path to saved plot or None
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ✅ FIX: Handle None or empty component_analysis
    if not component_analysis:
        logger.warning("No component analysis data - generating placeholder plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Component stability analysis requires\nmore training episodes.\n\n'
                          'Try running with more episodes or\nensure component metrics are being logged.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "03_component_stability.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    stability_metrics = component_analysis.get('stability_metrics', {})

    # ✅ FIX: Handle empty stability_metrics
    if not stability_metrics:
        logger.warning("No stability metrics available - generating placeholder plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No stability metrics available.\n\n'
                          'This occurs when:\n'
                          '1. Training log is too short\n'
                          '2. Component analysis was not computed\n'
                          '3. All components have zero variance',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "03_component_stability.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    # ====================================================================
    # PREPARE DATA
    # ====================================================================

    component_labels = [k.replace('component_', '').replace('_', ' ').title()
                        for k in stability_metrics.keys()]
    stability_values = list(stability_metrics.values())
    colors = COMPONENT_COLORS[:len(stability_values)]

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(component_labels, stability_values, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, stability_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    format_plot_labels(ax, 'Component', 'Stability Score (1 - variance)',
                       'Component Stability During Training\n(Higher = More Stable)')
    ax.set_ylim([0, 1.2])

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "03_component_stability.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_merge_quality_heatmap(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create merge quality heatmap over training.

    Shows stability and quality of merge decisions across time.

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

    n_episodes = len(training_log)
    n_components = 5

    # ====================================================================
    # BUILD HEATMAP DATA
    # ====================================================================

    heatmap_data = np.zeros((n_components, n_episodes))

    components = [
        'component_h_preservation',
        'component_transition_control',
        'component_operator_projection',
        'component_label_combinability',
        'component_bonus_signals',
    ]

    for i, component_name in enumerate(components):
        values = [getattr(m, component_name, 0.0) for m in training_log]

        # Normalize to [0, 1]
        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        range_val = max_val - min_val if max_val != min_val else 1

        normalized = [(v - min_val) / range_val for v in values]
        heatmap_data[i, :] = normalized

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')

    ax.set_yticks(range(n_components))
    ax.set_yticklabels([c.replace('component_', '').replace('_', ' ').title()
                        for c in components])

    format_plot_labels(ax, 'Episode', 'Component',
                       'Merge Quality Heatmap - Component Stability Over Time')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Component Quality', rotation=270, labelpad=20)

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "04_merge_quality_heatmap.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None