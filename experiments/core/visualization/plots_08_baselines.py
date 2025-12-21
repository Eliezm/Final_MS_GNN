#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 13: COMPREHENSIVE 3-WAY BASELINE COMPARISON
===============================================
GNN vs Random vs FD Baselines (FULLY IMPLEMENTED)

✅ FIXED: Emoji rendering issues
✅ FIXED: Axis limits for single data points
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, format_plot_labels, create_plots_directory,
    save_plot_safely, QUALITY_COLORS,
)

logger = logging.getLogger(__name__)


def plot_merge_quality_distribution(
        episode_reward_signals: Dict,
        output_dir: Path,
) -> Optional[Path]:
    """
    Create merge quality distribution visualization.

    Args:
        episode_reward_signals: Episode reward data dict
        output_dir: Output directory

    Returns:
        Path to saved plot or None
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not episode_reward_signals:
        return None

    plots_dir = create_plots_directory(output_dir)

    # Extract quality categories
    quality_counts = {'excellent': 0, 'good': 0, 'moderate': 0, 'poor': 0, 'bad': 0}

    for episode_data in episode_reward_signals.values():
        reward_signals_list = episode_data.get('reward_signals_per_step', [])
        for signal in reward_signals_list:
            quality_cat = signal.get('merge_quality_category', 'moderate')
            if quality_cat in quality_counts:
                quality_counts[quality_cat] += 1

    fig, ax = plt.subplots(figsize=(12, 6))

    categories = list(quality_counts.keys())
    counts = list(quality_counts.values())
    colors = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c', '#c0392b']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Number of Merges', fontweight='bold', fontsize=12)
    ax.set_xlabel('Merge Quality Category', fontweight='bold', fontsize=12)
    ax.set_title('Merge Quality Distribution\n(From GNN Training)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = plots_dir / "12_merge_quality_distribution.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_three_way_comparison(
        gnn_stats: Dict[str, Any],
        random_stats: Dict[str, Any],
        baseline_stats: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create comprehensive 4-panel comparison plot.

    GNN vs Random vs FD Baselines on:
    1. Solve Rate
    2. Time Distribution
    3. Node Expansions
    4. H* Preservation (GNN-specific)
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = create_plots_directory(output_dir)

    # Handle empty stats
    gnn_stats = gnn_stats or {}
    random_stats = random_stats or {}
    baseline_stats = baseline_stats or {}

    if not gnn_stats and not random_stats and not baseline_stats:
        logger.warning("No statistics available for three-way comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No comparison data available.\n\n'
                          'Run GNN, Random, and Baseline evaluations first.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "13_three_way_comparison.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Strategy Comparison: GNN vs Random vs FD Baselines',
                 fontsize=16, fontweight='bold')

    # Build strategy list
    strategies = []
    solve_rates = []
    times = []
    expansions = []
    h_preservations = []
    colors = []

    # GNN
    if gnn_stats:
        strategies.append('GNN (Learned)')
        solve_rates.append(gnn_stats.get('solve_rate_pct', 0))
        times.append(gnn_stats.get('mean_time_sec', 0))
        expansions.append(gnn_stats.get('mean_expansions', 0))
        h_preservations.append(gnn_stats.get('mean_h_preservation', 1.0))
        colors.append('#2ecc71')  # Green

    # Random
    if random_stats:
        strategies.append('Random Merge')
        solve_rates.append(random_stats.get('solve_rate_pct', 0))
        times.append(random_stats.get('mean_time_sec', 0))
        expansions.append(random_stats.get('mean_expansions', 0))
        h_preservations.append(random_stats.get('mean_h_preservation', 1.0))
        colors.append('#95a5a6')  # Gray

    # Baselines
    baseline_colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    for i, (name, stats) in enumerate(list(baseline_stats.items())[:5]):
        strategies.append(name[:25])
        solve_rates.append(stats.get('solve_rate_%', stats.get('solve_rate_pct', 0)))
        times.append(stats.get('avg_time_total_s', stats.get('mean_time_sec', 0)))
        expansions.append(stats.get('avg_expansions', stats.get('mean_expansions', 0)))
        h_preservations.append(1.0)  # FD baselines are optimal
        colors.append(baseline_colors[i % len(baseline_colors)])

    # Panel 1: Solve Rate
    bars1 = ax1.bar(range(len(strategies)), solve_rates, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=2)
    for bar, rate in zip(bars1, solve_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Solve Rate (%)', fontweight='bold')
    ax1.set_title('Solve Rate Comparison', fontweight='bold')
    ax1.set_ylim([0, 110])
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Time Comparison
    bars2 = ax2.bar(range(len(strategies)), times, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=2)
    for bar, t in zip(bars2, times):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{t:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Mean Time (seconds)', fontweight='bold')
    ax2.set_title('Solve Time Comparison (Lower is Better)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Expansions (log scale)
    valid_exp = [e if e > 0 else 1 for e in expansions]
    bars3 = ax3.bar(range(len(strategies)), valid_exp, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=2)
    for bar, e in zip(bars3, expansions):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2., height * 1.1,
                     f'{int(e):,}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Mean Expansions (log scale)', fontweight='bold')
    ax3.set_title('Node Expansions (Lower is Better)', fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y', which='both')

    # Panel 4: H* Preservation (GNN-specific metric)
    bars4 = ax4.barh(range(len(strategies)), h_preservations, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=2)
    for bar, h in zip(bars4, h_preservations):
        width = bar.get_width()
        ax4.text(width - 0.02, bar.get_y() + bar.get_height() / 2.,
                 f'{h:.3f}', ha='right', va='center', fontweight='bold', fontsize=9,
                 color='white' if width > 0.5 else 'black')
    ax4.axvline(x=0.95, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax4.set_yticks(range(len(strategies)))
    ax4.set_yticklabels(strategies, fontsize=9)
    ax4.set_xlabel('H* Preservation Ratio', fontweight='bold')
    ax4.set_title('Heuristic Quality (H* Preservation)', fontweight='bold')
    ax4.set_xlim([0.7, 1.05])
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plot_path = plots_dir / "13_three_way_comparison.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_per_problem_winners(
        detailed_results: List[Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create per-problem winner heatmap.

    ✅ FIXED: Handle single problem case (axis limits)
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ✅ FIX: Handle empty or None results
    if not detailed_results:
        logger.warning("No detailed results for per-problem winners plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No detailed results available.\n\n'
                          'Run evaluation first to generate\n'
                          'per-problem comparison data.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "14_per_problem_winners.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    # Organize results by problem and strategy
    by_problem = defaultdict(dict)
    for result in detailed_results:
        # Handle both dict and object results
        if isinstance(result, dict):
            problem = result.get('problem_name', 'unknown')
            strategy = result.get('planner_name', 'unknown')
            time_val = result.get('wall_clock_time', float('inf'))
            solved = result.get('solved', False)
        else:
            problem = getattr(result, 'problem_name', 'unknown')
            strategy = getattr(result, 'planner_name', 'unknown')
            time_val = getattr(result, 'wall_clock_time', float('inf'))
            solved = getattr(result, 'solved', False)

        if problem not in by_problem:
            by_problem[problem] = {}
        by_problem[problem][strategy] = {
            'time': time_val,
            'solved': solved
        }

    problems = sorted(by_problem.keys())[:30]  # Top 30 problems
    strategies = sorted(set(s for p in by_problem.values() for s in p.keys()))

    # ✅ FIX: Handle case with only one problem or strategy
    if len(problems) < 1 or len(strategies) < 1:
        logger.warning("Not enough data for per-problem winners plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Insufficient data for per-problem comparison.\n\n'
                          f'Problems: {len(problems)}, Strategies: {len(strategies)}\n'
                          'Need at least 1 problem and 1 strategy.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "14_per_problem_winners.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    # Create matrix: rows=problems, cols=strategies
    matrix = np.zeros((len(problems), len(strategies)))

    for i, problem in enumerate(problems):
        for j, strategy in enumerate(strategies):
            if strategy in by_problem[problem]:
                result = by_problem[problem][strategy]
                if result['solved']:
                    matrix[i, j] = result['time']
                else:
                    matrix[i, j] = 1000  # Unsolved = penalty

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(problems) * 0.25)))

    # Identify winners per problem
    winners = []
    for i in range(len(problems)):
        valid_cols = [j for j in range(len(strategies))
                      if 0 < matrix[i, j] < 1000]
        if valid_cols:
            winner_idx = valid_cols[np.argmin([matrix[i, j] for j in valid_cols])]
            winners.append(winner_idx)
        else:
            winners.append(-1)

    # Color by winner
    colors_matrix = np.zeros((len(problems), len(strategies)))
    strategy_colors = {
        'GNN': 0,
        'Random': 1,
    }
    color_idx = 2
    for strategy in strategies:
        if strategy not in strategy_colors:
            strategy_colors[strategy] = color_idx
            color_idx += 1

    for i, winner_idx in enumerate(winners):
        if winner_idx >= 0:
            colors_matrix[i, winner_idx] = 2  # Winner = bright
        else:
            colors_matrix[i, :] = 0.5  # All failed = gray

    # ✅ FIX: Handle single row/column case for imshow
    if len(problems) == 1 or len(strategies) == 1:
        # Pad the matrix to avoid singular transformation
        if len(problems) == 1:
            colors_matrix = np.vstack([colors_matrix, colors_matrix])
        if len(strategies) == 1:
            colors_matrix = np.hstack([colors_matrix, colors_matrix])

    im = ax.imshow(colors_matrix[:len(problems), :len(strategies)],
                   cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)

    # Add time values as text
    for i in range(len(problems)):
        for j in range(len(strategies)):
            time_val = matrix[i, j]
            if 0 < time_val < 1000:
                ax.text(j, i, f'{time_val:.1f}s',
                        ha='center', va='center', fontsize=8,
                        color='white' if colors_matrix[i, j] > 1 else 'black',
                        fontweight='bold')
            elif time_val >= 1000:
                ax.text(j, i, 'X',  # ✅ FIX: Use 'X' instead of emoji
                        ha='center', va='center', fontsize=10, color='red',
                        fontweight='bold')

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(problems)))
    ax.set_yticklabels([p[:30] for p in problems], fontsize=8)

    ax.set_title('Per-Problem Performance Heatmap\n(Green=Winner, Yellow=Close, Red=Unsolved)',
                 fontweight='bold', fontsize=13)

    plt.tight_layout()
    plot_path = plots_dir / "14_per_problem_winners.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_cumulative_solved(
        detailed_results: List[Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot cumulative problems solved over time.

    ✅ FIXED: Emoji characters and empty legend
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ✅ FIX: Handle empty results
    if not detailed_results:
        logger.warning("No results for cumulative solved plot")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, 'No evaluation results available.\n\n'
                          'Run GNN and Random evaluation first.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "15_cumulative_solved.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    # Sort by time and compute cumulative solved
    strategies = set()
    for r in detailed_results:
        if isinstance(r, dict):
            strategies.add(r.get('planner_name', 'unknown'))
        else:
            strategies.add(getattr(r, 'planner_name', 'unknown'))

    cumulative_data = {}
    for strategy in strategies:
        strategy_results = []
        for r in detailed_results:
            if isinstance(r, dict):
                if r.get('planner_name') == strategy:
                    strategy_results.append(r)
            else:
                if getattr(r, 'planner_name', None) == strategy:
                    strategy_results.append(r)

        # Sort by time
        strategy_results.sort(key=lambda x: x.get('wall_clock_time', float('inf'))
        if isinstance(x, dict) else getattr(x, 'wall_clock_time', float('inf')))

        times = []
        cumulative_solved = []
        total_solved = 0

        for result in strategy_results:
            solved = result.get('solved', False) if isinstance(result, dict) else getattr(result, 'solved', False)
            time_val = result.get('wall_clock_time', 0) if isinstance(result, dict) else getattr(result,
                                                                                                 'wall_clock_time', 0)

            if solved:
                total_solved += 1
                times.append(time_val)
                cumulative_solved.append(total_solved)

        cumulative_data[strategy] = (times, cumulative_solved)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    colors_strat = {'GNN': '#2ecc71', 'Random': '#95a5a6'}
    color_idx = 0
    color_list = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

    has_data = False
    for strategy in sorted(strategies):
        if strategy not in colors_strat:
            if color_idx < len(color_list):
                colors_strat[strategy] = color_list[color_idx]
                color_idx += 1
            else:
                colors_strat[strategy] = '#34495e'

        times, cumulative_solved = cumulative_data[strategy]

        if times and cumulative_solved:
            ax.plot(times, cumulative_solved, marker='o', linewidth=2.5,
                    label=strategy, color=colors_strat.get(strategy, '#34495e'),
                    markersize=4, alpha=0.8)
            has_data = True

    # ✅ FIX: Removed emoji from title
    format_plot_labels(ax, 'Cumulative Time (seconds)', 'Problems Solved',
                       '[15] Cumulative Solved vs Time (Convergence Analysis)')
    ax.grid(True, alpha=0.3)

    # ✅ FIX: Only add legend if there's data
    if has_data:
        ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    plot_path = plots_dir / "15_cumulative_solved.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None


def plot_speedup_analysis(
        gnn_results: List[Dict[str, Any]],
        baseline_results: List[Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot speedup of GNN vs baselines (Pareto-style).

    ✅ FIXED: Handle empty results
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ✅ FIX: Handle empty results
    if not gnn_results or not baseline_results:
        logger.warning("Insufficient results for speedup analysis")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, 'Speedup analysis requires both\nGNN and Baseline results.\n\n'
                          f'GNN results: {len(gnn_results) if gnn_results else 0}\n'
                          f'Baseline results: {len(baseline_results) if baseline_results else 0}',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "16_speedup_analysis.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    # Compute speedups
    speedups = []

    gnn_by_problem = {}
    for r in gnn_results:
        if isinstance(r, dict):
            gnn_by_problem[r.get('problem_name')] = r
        else:
            gnn_by_problem[getattr(r, 'problem_name', None)] = r

    for baseline_result in baseline_results:
        if isinstance(baseline_result, dict):
            problem = baseline_result.get('problem_name')
            baseline_time = baseline_result.get('wall_clock_time', float('inf'))
        else:
            problem = getattr(baseline_result, 'problem_name', None)
            baseline_time = getattr(baseline_result, 'wall_clock_time', float('inf'))

        if problem in gnn_by_problem:
            gnn_r = gnn_by_problem[problem]
            if isinstance(gnn_r, dict):
                gnn_time = gnn_r.get('wall_clock_time', float('inf'))
            else:
                gnn_time = getattr(gnn_r, 'wall_clock_time', float('inf'))

            if baseline_time > 0 and gnn_time > 0 and baseline_time < 1000 and gnn_time < 1000:
                speedup = baseline_time / gnn_time
                speedups.append(speedup)

    if not speedups:
        logger.warning("No valid speedup comparisons found")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, 'No valid speedup comparisons.\n\n'
                          'This occurs when no problems were\n'
                          'solved by both GNN and Baselines.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "16_speedup_analysis.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    speedups = sorted(speedups)
    cumulative_pct = np.arange(1, len(speedups) + 1) / len(speedups) * 100

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(speedups, cumulative_pct, marker='o', linewidth=2.5, markersize=5,
            color='#2ecc71', label='GNN vs Best Baseline')

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5,
               label='No speedup')
    ax.axvline(x=np.median(speedups), color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Median: {np.median(speedups):.2f}x')

    # ✅ FIX: Removed emoji from title
    format_plot_labels(ax, 'Speedup Factor (Baseline Time / GNN Time)',
                       '% of Problems', '[16] GNN Speedup Analysis vs Baselines')

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plot_path = plots_dir / "16_speedup_analysis.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None