#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 23-27: GENERALIZATION ANALYSIS
===================================
Analyzes how well GNN generalizes to unseen problems.

Research Questions:
- How much does performance drop on unseen problems?
- Does training on larger/smaller problems generalize?
- Is there correlation between problem complexity and performance?
- What is the seen vs unseen performance gap?
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import logging

from experiments.core.logging import EpisodeMetrics
from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, format_plot_labels, create_plots_directory,
    save_plot_safely, add_target_line, QUALITY_COLORS,
)

logger = logging.getLogger(__name__)


def plot_performance_by_problem_size(
        test_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 23: Performance by Problem Size

    Research Question: How does GNN performance scale with problem complexity?

    What it shows:
    - Solve rate across small/medium/large problems
    - Performance degradation pattern
    - Size-specific strengths/weaknesses

    Key Insight: Reveals generalization limits of the trained model
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not test_results:
        logger.warning("No test results for performance by size plot")
        return None

    plots_dir = create_plots_directory(output_dir)

    # Extract data by size
    sizes = ['small', 'medium', 'large']
    size_data = {size: {'gnn_solved': 0, 'gnn_total': 0, 'random_solved': 0, 'random_total': 0}
                 for size in sizes}

    for test_name, test_data in test_results.items():
        # Determine size from test name
        detected_size = None
        for size in sizes:
            if size in test_name.lower():
                detected_size = size
                break

        if detected_size is None:
            continue

        results = test_data.get('results', {})
        summary = results.get('summary', {})

        size_data[detected_size]['gnn_solved'] += summary.get('gnn_solved', 0)
        size_data[detected_size]['gnn_total'] += summary.get('gnn_total', 0)
        size_data[detected_size]['random_solved'] += summary.get('random_solved', 0)
        size_data[detected_size]['random_total'] += summary.get('random_total', 0)

    # Check if we have data
    if all(size_data[s]['gnn_total'] == 0 for s in sizes):
        logger.warning("No size-categorized data found")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance by Problem Size\n'
                 'Research Q: How does GNN performance scale with problem complexity?',
                 fontsize=14, fontweight='bold')

    # Panel 1: Solve rates by size
    x = np.arange(len(sizes))
    width = 0.35

    gnn_rates = []
    random_rates = []
    for size in sizes:
        gnn_total = max(1, size_data[size]['gnn_total'])
        random_total = max(1, size_data[size]['random_total'])
        gnn_rates.append(size_data[size]['gnn_solved'] / gnn_total * 100)
        random_rates.append(size_data[size]['random_solved'] / random_total * 100)

    bars1 = ax1.bar(x - width / 2, gnn_rates, width, label='GNN', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, random_rates, width, label='Random', color='#e74c3c', alpha=0.8)

    # Add value labels
    for bar, rate in zip(bars1, gnn_rates):
        if rate > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, rate in zip(bars2, random_rates):
        if rate > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in sizes])
    format_plot_labels(ax1, 'Problem Size', 'Solve Rate (%)',
                       'Solve Rate by Problem Size')
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 110])

    # Panel 2: GNN Improvement over Random by size
    improvements = [gnn_rates[i] - random_rates[i] for i in range(len(sizes))]
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]

    bars = ax2.bar(sizes, improvements, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    for bar, imp in zip(bars, improvements):
        va = 'bottom' if imp >= 0 else 'top'
        offset = 1 if imp >= 0 else -1
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                 f'{imp:+.1f}%', ha='center', va=va, fontweight='bold', fontsize=11)

    format_plot_labels(ax2, 'Problem Size', 'GNN Improvement (%)',
                       'GNN vs Random Improvement by Size')

    plt.tight_layout()
    plot_path = plots_dir / "23_performance_by_problem_size.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_seen_vs_unseen_gap(
        test_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 24: Seen vs Unseen Performance Gap

    Research Question: How much does performance drop on unseen problems?

    What it shows:
    - Performance on training problems (seen)
    - Performance on test problems (unseen)
    - Generalization gap magnitude

    Key Insight: Measures true generalization vs memorization
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not test_results:
        return None

    plots_dir = create_plots_directory(output_dir)

    # Separate seen vs unseen
    seen_results = []
    unseen_results = []

    for test_name, test_data in test_results.items():
        results = test_data.get('results', {})
        summary = results.get('summary', {})

        gnn_total = summary.get('gnn_total', 0)
        gnn_solved = summary.get('gnn_solved', 0)

        if gnn_total == 0:
            continue

        solve_rate = gnn_solved / gnn_total * 100

        # Check if this is a "seen" test (training problems) or "unseen"
        if 'seen' in test_name.lower() or 'train' in test_name.lower():
            seen_results.append({'name': test_name, 'rate': solve_rate})
        else:
            unseen_results.append({'name': test_name, 'rate': solve_rate})

    if not seen_results and not unseen_results:
        logger.warning("No seen/unseen categorization possible")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Seen vs Unseen Performance Gap\n'
                 'Research Q: How much does performance drop on unseen problems?',
                 fontsize=14, fontweight='bold')

    # Panel 1: Bar comparison
    categories = ['Seen\n(Training)', 'Unseen\n(Test)']
    seen_avg = np.mean([r['rate'] for r in seen_results]) if seen_results else 0
    unseen_avg = np.mean([r['rate'] for r in unseen_results]) if unseen_results else 0

    bars = ax1.bar(categories, [seen_avg, unseen_avg],
                   color=['#3498db', '#9b59b6'], alpha=0.8, edgecolor='black', linewidth=2)

    for bar, rate in zip(bars, [seen_avg, unseen_avg]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add gap annotation
    if seen_avg > 0:
        gap = seen_avg - unseen_avg
        gap_pct = gap / seen_avg * 100 if seen_avg > 0 else 0
        ax1.annotate('', xy=(1, unseen_avg), xytext=(0, seen_avg),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax1.text(0.5, (seen_avg + unseen_avg) / 2, f'Gap: {gap:.1f}%\n({gap_pct:.0f}% drop)',
                 ha='center', va='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    format_plot_labels(ax1, '', 'Solve Rate (%)',
                       'Average Performance: Seen vs Unseen')
    ax1.set_ylim([0, max(seen_avg, unseen_avg) * 1.2 + 5])

    # Panel 2: Detailed breakdown
    all_names = []
    all_rates = []
    all_colors = []

    for r in seen_results:
        all_names.append(r['name'][:20])
        all_rates.append(r['rate'])
        all_colors.append('#3498db')
    for r in unseen_results:
        all_names.append(r['name'][:20])
        all_rates.append(r['rate'])
        all_colors.append('#9b59b6')

    if all_names:
        y_pos = np.arange(len(all_names))
        ax2.barh(y_pos, all_rates, color=all_colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(all_names, fontsize=9)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#3498db', label='Seen'),
                           Patch(facecolor='#9b59b6', label='Unseen')]
        ax2.legend(handles=legend_elements, loc='lower right')

        format_plot_labels(ax2, 'Solve Rate (%)', '',
                           'Per-Test-Set Breakdown')

    plt.tight_layout()
    plot_path = plots_dir / "24_seen_vs_unseen_gap.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_training_size_effect(
        experiment_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 25: Training Size Effect

    Research Question: Does training on larger problems generalize better?

    What it shows:
    - Comparison of models trained on different sizes
    - Performance on each test size
    - Optimal training size identification

    Key Insight: Answers "should we train on medium or large?"
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not experiment_results:
        return None

    plots_dir = create_plots_directory(output_dir)

    # This plot requires results from multiple experiments
    # Structure: experiment_results[exp_name][test_size] = solve_rate

    fig, ax = plt.subplots(figsize=(14, 8))

    train_sizes = list(experiment_results.keys())
    if not train_sizes:
        return None

    test_sizes = ['small', 'medium', 'large']
    x = np.arange(len(test_sizes))
    width = 0.8 / len(train_sizes)

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    for i, (train_size, results) in enumerate(experiment_results.items()):
        rates = []
        for test_size in test_sizes:
            rate = results.get(test_size, 0)
            rates.append(rate)

        offset = (i - len(train_sizes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=f'Train: {train_size}',
                      color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in test_sizes])
    format_plot_labels(ax, 'Test Problem Size', 'Solve Rate (%)',
                       'Training Size Effect on Generalization\n'
                       'Research Q: Does training on larger problems generalize better?')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim([0, 110])

    plt.tight_layout()
    plot_path = plots_dir / "25_training_size_effect.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_complexity_correlation(
        test_results: Dict[str, Dict[str, Any]],
        detailed_results: List[Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 26: Performance vs Problem Complexity

    Research Question: Is there correlation between problem complexity and performance?

    What it shows:
    - Scatter plot of problem complexity metrics vs performance
    - Regression line showing trend
    - Complexity threshold identification

    Key Insight: Identifies complexity limits of the learned strategy
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not detailed_results:
        return None

    plots_dir = create_plots_directory(output_dir)

    # Extract complexity metrics
    complexities = []
    solve_times = []
    solved_flags = []
    problem_names = []

    for result in detailed_results:
        if hasattr(result, 'problem_name'):
            # Estimate complexity from problem name or other metrics
            problem_name = result.problem_name

            # Extract size indicator from name
            complexity_score = 0
            if 'small' in problem_name.lower():
                complexity_score = 1
            elif 'medium' in problem_name.lower():
                complexity_score = 2
            elif 'large' in problem_name.lower():
                complexity_score = 3
            else:
                # Try to extract number from name
                import re
                numbers = re.findall(r'\d+', problem_name)
                if numbers:
                    complexity_score = min(int(numbers[-1]) / 10, 5)

            if hasattr(result, 'nodes_expanded') and result.nodes_expanded > 0:
                complexity_score = np.log10(result.nodes_expanded + 1)

            complexities.append(complexity_score)
            solve_times.append(result.wall_clock_time if hasattr(result, 'wall_clock_time') else 0)
            solved_flags.append(result.solved if hasattr(result, 'solved') else False)
            problem_names.append(problem_name)

    if not complexities or len(complexities) < 3:
        logger.warning("Insufficient complexity data for correlation plot")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance vs Problem Complexity\n'
                 'Research Q: Is there correlation between complexity and performance?',
                 fontsize=14, fontweight='bold')

    # Panel 1: Complexity vs Solve Time
    colors = ['#27ae60' if s else '#e74c3c' for s in solved_flags]
    ax1.scatter(complexities, solve_times, c=colors, alpha=0.6, s=80, edgecolors='black')

    # Add trend line for solved problems
    solved_complexities = [c for c, s in zip(complexities, solved_flags) if s]
    solved_times = [t for t, s in zip(solve_times, solved_flags) if s]

    if len(solved_complexities) > 2:
        z = np.polyfit(solved_complexities, solved_times, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(solved_complexities), max(solved_complexities), 100)
        ax1.plot(x_line, p(x_line), 'b--', linewidth=2, label='Trend (solved)')

        # Compute correlation
        corr = np.corrcoef(solved_complexities, solved_times)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                 transform=ax1.transAxes, fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    format_plot_labels(ax1, 'Problem Complexity', 'Solve Time (s)',
                       'Complexity vs Solve Time (Green=Solved, Red=Unsolved)')
    ax1.legend(fontsize=9)

    # Panel 2: Solve rate by complexity band
    if len(set(complexities)) > 1:
        # Bin complexities
        complexity_bins = np.percentile(complexities, [0, 33, 67, 100])
        bin_labels = ['Low', 'Medium', 'High']
        bin_solve_rates = []

        for i in range(len(bin_labels)):
            low = complexity_bins[i]
            high = complexity_bins[i + 1]

            in_bin = [(c >= low and c <= high) for c in complexities]
            solved_in_bin = sum(1 for j, s in enumerate(solved_flags) if in_bin[j] and s)
            total_in_bin = sum(in_bin)

            rate = solved_in_bin / max(1, total_in_bin) * 100
            bin_solve_rates.append(rate)

        colors = ['#27ae60', '#f39c12', '#e74c3c']
        bars = ax2.bar(bin_labels, bin_solve_rates, color=colors, alpha=0.8, edgecolor='black')

        for bar, rate in zip(bars, bin_solve_rates):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        format_plot_labels(ax2, 'Complexity Level', 'Solve Rate (%)',
                           'Solve Rate by Complexity Band')
        ax2.set_ylim([0, 110])

    plt.tight_layout()
    plot_path = plots_dir / "26_complexity_correlation.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_generalization_heatmap(
        test_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 27: Generalization Heatmap

    Research Question: What is the full generalization matrix?

    What it shows:
    - Heatmap: Train size (rows) vs Test size (columns)
    - Cell values: Solve rate or improvement over random
    - Identifies best training strategy

    Key Insight: Complete view of generalization patterns
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not test_results:
        return None

    plots_dir = create_plots_directory(output_dir)

    # Build matrix: this requires data from multiple experiments
    # For single experiment, we show test performance matrix

    test_sizes = ['small', 'medium', 'large']
    test_types = ['seen', 'unseen']

    # Build matrix
    matrix_data = np.zeros((len(test_types), len(test_sizes)))

    for test_name, test_data in test_results.items():
        results = test_data.get('results', {})
        summary = results.get('summary', {})

        gnn_total = summary.get('gnn_total', 0)
        gnn_solved = summary.get('gnn_solved', 0)

        if gnn_total == 0:
            continue

        rate = gnn_solved / gnn_total * 100

        # Determine row (seen/unseen)
        row = 1  # Default to unseen
        if 'seen' in test_name.lower():
            row = 0

        # Determine column (size)
        col = -1
        for i, size in enumerate(test_sizes):
            if size in test_name.lower():
                col = i
                break

        if col >= 0:
            matrix_data[row, col] = rate

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(test_sizes)))
    ax.set_yticks(np.arange(len(test_types)))
    ax.set_xticklabels([s.capitalize() for s in test_sizes], fontweight='bold')
    ax.set_yticklabels([t.capitalize() for t in test_types], fontweight='bold')

    # Add text annotations
    for i in range(len(test_types)):
        for j in range(len(test_sizes)):
            value = matrix_data[i, j]
            color = 'white' if value > 50 else 'black'
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                    color=color, fontweight='bold', fontsize=12)

    plt.colorbar(im, ax=ax, label='Solve Rate (%)')

    ax.set_xlabel('Test Problem Size', fontweight='bold', fontsize=12)
    ax.set_ylabel('Test Type', fontweight='bold', fontsize=12)
    ax.set_title('Generalization Heatmap\n'
                 'Research Q: Full view of generalization patterns',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()
    plot_path = plots_dir / "27_generalization_heatmap.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None