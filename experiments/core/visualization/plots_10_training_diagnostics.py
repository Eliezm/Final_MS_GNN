#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 18-22: TRAINING DIAGNOSTICS
================================
Visualizes metrics that indicate GNN learning health.

Research Questions:
- Is the policy becoming more confident (entropy decreasing)?
- Is the value function learning effectively?
- Are gradients healthy throughout training?
- Is GNN inference fast enough for practical use?
- Does GNN learn to compress graphs?
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import logging

from experiments.core.logging import EpisodeMetrics
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, format_plot_labels, create_plots_directory,
    save_plot_safely, add_target_line,
)

logger = logging.getLogger(__name__)


def plot_policy_entropy_evolution(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 18: Policy Entropy Evolution

    Research Question: Does GNN become more confident over training?

    What it shows:
    - Policy entropy decreasing = GNN becoming more decisive
    - High entropy = exploratory, uncertain decisions
    - Low entropy = confident, exploitative decisions

    Key Insight: Healthy learning shows decreasing entropy with some exploration
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not training_log:
        return None

    plots_dir = create_plots_directory(output_dir)

    # Extract entropy values (compute from action probabilities if not stored)
    episodes = []
    entropies = []

    for metrics in training_log:
        if metrics.error is None:
            episodes.append(metrics.episode)

            # Use stored entropy or compute from action probabilities
            if hasattr(metrics, 'policy_entropy') and metrics.policy_entropy > 0:
                entropies.append(metrics.policy_entropy)
            elif metrics.gnn_action_probabilities:
                # Compute entropy from action probabilities
                probs = np.array(metrics.gnn_action_probabilities)
                probs = probs[probs > 0]  # Filter zeros
                if len(probs) > 0:
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(entropy)
                else:
                    entropies.append(0.0)
            else:
                entropies.append(0.0)

    if not episodes or all(e == 0 for e in entropies):
        logger.warning("No entropy data available")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Policy Entropy Evolution\n'
                 'Research Q: Does GNN become more confident over training?',
                 fontsize=14, fontweight='bold')

    # Panel 1: Entropy over time
    ax1.scatter(episodes, entropies, alpha=0.3, s=20, color='purple', label='Per-episode')

    # Rolling average
    window = min(20, len(entropies) // 4)
    if window > 1 and len(entropies) > window:
        rolling_avg = np.convolve(entropies, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(entropies)), rolling_avg,
                 linewidth=2.5, color='darkviolet', label=f'Rolling avg (w={window})')

    format_plot_labels(ax1, 'Episode', 'Policy Entropy (nats)',
                       'Entropy Timeline - Lower = More Confident')
    ax1.legend(fontsize=9)

    # Panel 2: Phase comparison
    n = len(entropies)
    if n >= 9:
        phases = ['Early\n(0-33%)', 'Mid\n(33-67%)', 'Late\n(67-100%)']
        phase_entropies = [
            np.mean(entropies[:n // 3]),
            np.mean(entropies[n // 3:2 * n // 3]),
            np.mean(entropies[2 * n // 3:])
        ]

        colors = ['#e74c3c' if e > phase_entropies[0] else '#27ae60' for e in phase_entropies]
        bars = ax2.bar(phases, phase_entropies, color=colors, alpha=0.7, edgecolor='black')

        for bar, val in zip(bars, phase_entropies):
            ax2.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.3f}',
                     ha='center', va='bottom', fontweight='bold')

        # Add trend arrow
        if phase_entropies[-1] < phase_entropies[0]:
            ax2.annotate('', xy=(2, phase_entropies[-1]), xytext=(0, phase_entropies[0]),
                         arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax2.text(1, max(phase_entropies) * 1.1, '✓ Entropy Decreasing\n(Learning!)',
                     ha='center', fontsize=10, color='green', fontweight='bold')

        format_plot_labels(ax2, 'Training Phase', 'Mean Entropy',
                           'Entropy by Phase - Expect Decrease')

    plt.tight_layout()
    plot_path = plots_dir / "18_policy_entropy_evolution.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_value_loss_evolution(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 19: Value Loss Evolution

    Research Question: Is the value function learning effectively?

    What it shows:
    - Value loss decreasing = better state value predictions
    - Stable low loss = accurate critic
    - Spikes = challenging problems or distribution shift

    Key Insight: Decreasing value loss indicates effective critic learning
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not training_log:
        return None

    plots_dir = create_plots_directory(output_dir)

    episodes = []
    value_losses = []

    for metrics in training_log:
        if metrics.error is None and hasattr(metrics, 'value_loss'):
            if metrics.value_loss > 0:
                episodes.append(metrics.episode)
                value_losses.append(metrics.value_loss)

    if not episodes:
        logger.warning("No value loss data available")
        # Create placeholder
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Value loss tracking requires\nPPO internal metrics logging.\n\n'
                          'Consider adding callback to track:\n'
                          '- model.logger.name_to_value["train/value_loss"]',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "19_value_loss_evolution.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.semilogy(episodes, value_losses, alpha=0.4, label='Per-episode', color='blue')

    window = min(20, len(value_losses) // 4)
    if window > 1:
        rolling_avg = np.convolve(value_losses, np.ones(window) / window, mode='valid')
        ax.semilogy(range(window - 1, len(value_losses)), rolling_avg,
                    linewidth=2.5, color='darkblue', label=f'Rolling avg')

    format_plot_labels(ax, 'Episode', 'Value Loss (log scale)',
                       'Value Function Learning\nResearch Q: Is critic learning effectively?')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plot_path = plots_dir / "19_value_loss_evolution.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_gradient_health(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 20: Gradient Health Analysis

    Research Question: Are gradients healthy (not exploding/vanishing)?

    What it shows:
    - Gradient norm over time
    - Spikes = potential instability
    - Near-zero = vanishing gradients
    - Healthy range typically 0.1 - 10.0

    Key Insight: Stable gradients indicate robust learning
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not training_log:
        return None

    plots_dir = create_plots_directory(output_dir)

    episodes = []
    gradient_norms = []

    for metrics in training_log:
        if metrics.error is None and hasattr(metrics, 'gradient_norm'):
            if metrics.gradient_norm > 0:
                episodes.append(metrics.episode)
                gradient_norms.append(metrics.gradient_norm)

    if not episodes:
        logger.warning("No gradient norm data available")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Gradient tracking requires\nLearningVerifier integration.\n\n'
                          'Enable with:\n'
                          '- learning_verifier.check_learning(model, episode)',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "20_gradient_health.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Gradient Health Analysis\n'
                 'Research Q: Are gradients healthy throughout training?',
                 fontsize=14, fontweight='bold')

    # Panel 1: Gradient norm timeline
    ax1.semilogy(episodes, gradient_norms, alpha=0.4, color='orange', label='Per-episode')

    # Add healthy range bands
    ax1.axhspan(0.1, 10.0, alpha=0.2, color='green', label='Healthy range')
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Explosion threshold')
    ax1.axhline(y=0.001, color='blue', linestyle='--', alpha=0.5, label='Vanishing threshold')

    format_plot_labels(ax1, 'Episode', 'Gradient Norm (log scale)',
                       'Gradient Magnitude Over Training')
    ax1.legend(fontsize=9)

    # Panel 2: Histogram
    ax2.hist(gradient_norms, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.median(gradient_norms), color='red', linestyle='--',
                label=f'Median: {np.median(gradient_norms):.3f}')

    format_plot_labels(ax2, 'Gradient Norm', 'Frequency',
                       'Gradient Norm Distribution')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = plots_dir / "20_gradient_health.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_inference_performance(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 21: GNN Inference Performance

    Research Question: Is GNN fast enough for practical use?

    What it shows:
    - Step time distribution
    - Inference time vs problem complexity
    - Memory usage patterns

    Key Insight: Practical GNN must be fast (<100ms per step)
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not training_log:
        return None

    plots_dir = create_plots_directory(output_dir)

    step_times = []
    memory_usage = []
    problem_sizes = []

    for metrics in training_log:
        if metrics.error is None:
            if metrics.step_time_ms > 0:
                step_times.append(metrics.step_time_ms)
            if metrics.peak_memory_mb > 0:
                memory_usage.append(metrics.peak_memory_mb)
            if metrics.eval_steps > 0:
                problem_sizes.append(metrics.eval_steps)

    if not step_times:
        logger.warning("No inference time data available")
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GNN Inference Performance Analysis\n'
                 'Research Q: Is GNN fast enough for practical use?',
                 fontsize=14, fontweight='bold')

    # Panel 1: Step time histogram
    ax1.hist(step_times, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=100, color='red', linestyle='--', label='Target: <100ms')
    ax1.axvline(x=np.median(step_times), color='green', linestyle='--',
                label=f'Median: {np.median(step_times):.1f}ms')

    format_plot_labels(ax1, 'Step Time (ms)', 'Frequency',
                       'Step Time Distribution')
    ax1.legend(fontsize=9)

    # Panel 2: Step time over training
    ax2.scatter(range(len(step_times)), step_times, alpha=0.3, s=10, color='steelblue')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)

    format_plot_labels(ax2, 'Episode', 'Step Time (ms)',
                       'Step Time Evolution')

    # Panel 3: Memory usage
    if memory_usage:
        ax3.hist(memory_usage, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax3.axvline(x=np.median(memory_usage), color='red', linestyle='--',
                    label=f'Median: {np.median(memory_usage):.1f}MB')
        format_plot_labels(ax3, 'Peak Memory (MB)', 'Frequency',
                           'Memory Usage Distribution')
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No memory data', ha='center', va='center',
                 transform=ax3.transAxes)
        ax3.axis('off')

    # Panel 4: Summary statistics
    ax4.axis('off')
    summary_text = f"""INFERENCE PERFORMANCE SUMMARY

Step Time:
  • Mean: {np.mean(step_times):.2f} ms
  • Median: {np.median(step_times):.2f} ms
  • 95th percentile: {np.percentile(step_times, 95):.2f} ms
  • Max: {np.max(step_times):.2f} ms

Memory Usage:
  • Mean: {np.mean(memory_usage):.1f} MB (if available)
  • Max: {np.max(memory_usage):.1f} MB (if available)

Practical Assessment:
  • {'✓ FAST' if np.median(step_times) < 100 else '✗ SLOW'}: Median step time {'<' if np.median(step_times) < 100 else '>'} 100ms
  • Episodes analyzed: {len(step_times)}
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plot_path = plots_dir / "21_inference_performance.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_graph_compression(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 22: Graph Compression Analysis

    Research Question: Does GNN learn to compress abstractions?

    What it shows:
    - Graph size reduction over training
    - Compression ratio evolution
    - Correlation with h* preservation

    Key Insight: Good GNN balances compression with heuristic quality
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not training_log:
        return None

    plots_dir = create_plots_directory(output_dir)

    episodes = []
    compression_ratios = []
    h_preservations = []

    for metrics in training_log:
        if metrics.error is None:
            episodes.append(metrics.episode)

            # Use stored compression or estimate from transitions
            if hasattr(metrics, 'graph_size_reduction_pct') and metrics.graph_size_reduction_pct != 0:
                compression_ratios.append(metrics.graph_size_reduction_pct)
            else:
                # Estimate from transition growth ratio (inverse)
                if metrics.transition_growth_ratio > 0:
                    compression_ratios.append(1.0 / metrics.transition_growth_ratio)
                else:
                    compression_ratios.append(1.0)

            h_preservations.append(metrics.h_star_preservation)

    if not episodes:
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Graph Compression Analysis\n'
                 'Research Q: Does GNN learn to compress while preserving quality?',
                 fontsize=14, fontweight='bold')

    # Panel 1: Compression over time
    ax1.scatter(episodes, compression_ratios, alpha=0.3, s=20, c=h_preservations,
                cmap='RdYlGn', vmin=0.8, vmax=1.0)

    window = min(20, len(compression_ratios) // 4)
    if window > 1:
        rolling_avg = np.convolve(compression_ratios, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(compression_ratios)), rolling_avg,
                 linewidth=2.5, color='black', label='Trend')

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No compression')
    format_plot_labels(ax1, 'Episode', 'Compression Ratio',
                       'Compression Over Training (color=h* preservation)')
    ax1.legend(fontsize=9)

    # Panel 2: Compression vs H* preservation scatter
    ax2.scatter(compression_ratios, h_preservations, alpha=0.5, s=30,
                c=range(len(compression_ratios)), cmap='viridis')

    # Add ideal region
    ax2.axhspan(0.95, 1.05, alpha=0.2, color='green', label='Good h* (>0.95)')
    ax2.axvspan(0.5, 1.0, alpha=0.2, color='blue', label='Good compression')

    format_plot_labels(ax2, 'Compression Ratio', 'H* Preservation',
                       'Compression vs Quality Trade-off')
    ax2.legend(fontsize=9)

    # Panel 3: Phase comparison
    n = len(compression_ratios)
    if n >= 9:
        phases = ['Early', 'Mid', 'Late']
        phase_compression = [
            np.mean(compression_ratios[:n // 3]),
            np.mean(compression_ratios[n // 3:2 * n // 3]),
            np.mean(compression_ratios[2 * n // 3:])
        ]
        phase_h_pres = [
            np.mean(h_preservations[:n // 3]),
            np.mean(h_preservations[n // 3:2 * n // 3]),
            np.mean(h_preservations[2 * n // 3:])
        ]

        x = np.arange(len(phases))
        width = 0.35

        bars1 = ax3.bar(x - width / 2, phase_compression, width, label='Compression',
                        color='steelblue', alpha=0.7)
        bars2 = ax3.bar(x + width / 2, phase_h_pres, width, label='H* Preservation',
                        color='coral', alpha=0.7)

        ax3.set_xticks(x)
        ax3.set_xticklabels(phases)
        format_plot_labels(ax3, 'Training Phase', 'Value',
                           'Compression and Quality by Phase')
        ax3.legend(fontsize=9)

    # Panel 4: Summary
    ax4.axis('off')

    # Compute Pareto efficiency
    good_both = sum(1 for c, h in zip(compression_ratios, h_preservations)
                    if c < 1.0 and h > 0.95)

    summary_text = f"""COMPRESSION ANALYSIS SUMMARY

Compression Statistics:
  • Mean ratio: {np.mean(compression_ratios):.3f}
  • Best compression: {np.min(compression_ratios):.3f}
  • Compression achieved: {sum(1 for c in compression_ratios if c < 1.0)}/{len(compression_ratios)} episodes

Quality Preservation:
  • Mean h* preservation: {np.mean(h_preservations):.3f}
  • High h* (>0.95): {sum(1 for h in h_preservations if h > 0.95)}/{len(h_preservations)} episodes

Pareto Efficiency:
  • Both good (compress + preserve): {good_both}/{len(compression_ratios)} episodes
  • Efficiency rate: {good_both / len(compression_ratios) * 100:.1f}%

Key Insight: {'✓ Learning to balance' if good_both > len(compression_ratios) * 0.3 else '✗ Needs improvement'}
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plot_path = plots_dir / "22_graph_compression.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None