#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUALIZATION MODULE - All plots from experiment_1_problem_overfit.py
Organized for modularity and clarity.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.core.logging import EpisodeMetrics


def setup_matplotlib():
    """Setup matplotlib with safe backend."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return None


# ============================================================================
# CORE PLOTTING FUNCTIONS
# ============================================================================

def create_learning_curves(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """Plot 1: Learning curves - reward, h* preservation, coverage, failures."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    episodes = [m.episode for m in training_log]
    rewards = [m.reward for m in training_log]
    h_preservations = [m.h_star_preservation for m in training_log]

    # Plot 1: Overall reward curve
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, label='Per-episode')

    window = min(10, len(rewards) // 4) if len(rewards) > 4 else 1
    if window > 1:
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(rewards)), rolling_avg, linewidth=2,
                 label=f'Rolling avg (window={window})')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Curve - Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: H* preservation curve
    ax2 = axes[0, 1]
    ax2.plot(episodes, h_preservations, alpha=0.3, color='green', label='Per-episode')

    if window > 1:
        rolling_h = np.convolve(h_preservations, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(h_preservations)), rolling_h, linewidth=2,
                 color='darkgreen', label=f'Rolling avg')

    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect preservation')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='Target (>0.95)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('H* Preservation')
    ax2.set_title('Learning Curve - H* Preservation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Problem coverage
    ax3 = axes[1, 0]
    by_problem = defaultdict(list)
    for m in training_log:
        by_problem[m.problem_name].append(len(by_problem[m.problem_name]) + 1)

    problem_counts = [len(by_problem[p]) for p in sorted(by_problem.keys())]
    ax3.bar(range(len(problem_counts)), problem_counts)
    ax3.set_xticks(range(len(problem_counts)))
    ax3.set_xticklabels([p[:10] for p in sorted(by_problem.keys())], rotation=45)
    ax3.set_ylabel('Episode Count')
    ax3.set_title('Problem Coverage')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Failure distribution
    ax4 = axes[1, 1]
    failure_types = defaultdict(int)
    for m in training_log:
        if m.failure_type:
            failure_types[m.failure_type] += 1

    if failure_types:
        labels = list(failure_types.keys())
        counts = list(failure_types.values())
        ax4.pie(counts, labels=labels, autopct='%1.1f%%')
        ax4.set_title('Failure Types')
    else:
        ax4.text(0.5, 0.5, 'No Failures', ha='center', va='center')
        ax4.set_title('Failure Types')

    plt.tight_layout()
    plot_path = plots_dir / "01_learning_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_component_tracking_plots(
        training_log: List[EpisodeMetrics],
        analysis_results: Dict,
        output_dir: Path,
) -> Optional[Path]:
    """Plot 2-5: Component evolution, stability, degradation, per-problem."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    episodes = list(range(len(training_log)))

    # Plot 1: Individual Component Trajectories
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Reward Component Evolution During Training', fontsize=16, fontweight='bold')

    component_info = [
        ('component_h_preservation', 'H* Preservation', 'green'),
        ('component_transition_control', 'Transition Control', 'blue'),
        ('component_operator_projection', 'Operator Projection', 'orange'),
        ('component_label_combinability', 'Label Combinability', 'red'),
        ('component_bonus_signals', 'Bonus Signals', 'purple'),
    ]

    for idx, (attr_name, label, color) in enumerate(component_info):
        ax = axes[idx // 2, idx % 2]

        values = [getattr(m, attr_name, 0.0) for m in training_log]

        ax.plot(episodes, values, alpha=0.5, color=color, label='Per-episode')

        window = min(10, len(values) // 4) if len(values) > 4 else 1
        if window > 1 and len(values) > window:
            rolling_avg = np.convolve(values, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(values)), rolling_avg, linewidth=2.5,
                    color=color, label=f'Rolling avg (window={window})')

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Component Reward')
        ax.set_title(label, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.delaxes(axes[2, 1])

    plt.tight_layout()
    plot_path = plots_dir / "02_component_trajectories.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Merge Quality Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    n_episodes = len(training_log)
    n_components = 5

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
        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        range_val = max_val - min_val if max_val != min_val else 1
        normalized = [(v - min_val) / range_val for v in values]
        heatmap_data[i, :] = normalized

    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_yticks(range(n_components))
    ax.set_yticklabels([c.replace('component_', '').replace('_', ' ').title()
                        for c in components])
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_title('Merge Quality Heatmap - Component Stability Over Time',
                 fontweight='bold', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Component Quality', rotation=270, labelpad=20)

    plt.tight_layout()
    plot_path = plots_dir / "03_merge_quality_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Stability Metrics
    fig, ax = plt.subplots(figsize=(12, 6))

    stability_metrics = analysis_results.get('stability_metrics', {})
    component_labels = [k.replace('component_', '').replace('_', ' ').title()
                        for k in stability_metrics.keys()]
    stability_values = list(stability_metrics.values())

    colors = ['green', 'blue', 'orange', 'red', 'purple']
    bars = ax.bar(component_labels, stability_values, color=colors, alpha=0.7, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Stability Score (1 - variance)', fontweight='bold')
    ax.set_title('Component Stability During Training\n(Higher = More Stable)',
                 fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = plots_dir / "04_component_stability.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_feature_analysis_plots(
        training_log: List[EpisodeMetrics],
        correlation_analysis: Dict,
        feature_importance_analysis: Dict,
        output_dir: Path,
) -> Optional[Path]:
    """Plot 6: Feature importance and correlation."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Feature Importance
    fig, ax = plt.subplots(figsize=(12, 6))

    feature_imp = feature_importance_analysis.get('feature_importance', {})
    if feature_imp:
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1]['importance'], reverse=True)
        feature_names = [f[0] for f in sorted_features]
        importances = [f[1]['importance'] for f in sorted_features]
        significances = [f[1]['significant'] for f in sorted_features]

        colors = ['green' if sig else 'orange' for sig in significances]
        bars = ax.barh(range(len(feature_names)), importances, color=colors, alpha=0.7, edgecolor='black')

        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax.text(imp, bar.get_y() + bar.get_height() / 2, f'{imp:.3f}',
                    va='center', ha='left', fontweight='bold')

        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Importance (|Correlation| with GNN Confidence)', fontweight='bold')
        ax.set_title('Feature Importance Ranking\n(Green = Significant at p<0.05)',
                     fontweight='bold')
        ax.set_xlim([0, max(importances) * 1.15])
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plot_path = plots_dir / "05_feature_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Feature-Reward Correlation
    fig, ax = plt.subplots(figsize=(12, 6))

    correlations = correlation_analysis.get('feature_correlations', {})
    if correlations:
        feature_names = []
        corr_values = []
        colors = []

        for feature_name, stats_dict in sorted(correlations.items()):
            corr = stats_dict['correlation']
            p_val = stats_dict['p_value']

            feature_names.append(feature_name.replace('_', ' ').title())
            corr_values.append(corr)

            colors.append('green' if p_val < 0.05 else 'gray')

        bars = ax.bar(range(len(feature_names)), corr_values, color=colors, alpha=0.7, edgecolor='black')

        for i, (bar, val) in enumerate(zip(bars, corr_values)):
            ax.text(bar.get_x() + bar.get_width() / 2., val,
                    f'{val:.3f}',
                    ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Theory threshold (r > 0.4)')
        ax.axhline(y=-0.4, color='red', linestyle='--', alpha=0.5)

        ax.set_ylabel('Pearson Correlation with Reward', fontweight='bold')
        ax.set_xlabel('Feature', fontweight='bold')
        ax.set_title('Feature-Reward Correlation\n(Green = Significant at p<0.05)',
                     fontweight='bold', fontsize=14)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylim([-1.0, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = plots_dir / "06_feature_correlation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_bisimulation_plots(
        training_log: List[EpisodeMetrics],
        bisim_analysis: Dict,
        output_dir: Path,
) -> Optional[Path]:
    """Plot 7: Bisimulation preservation and h-value protection."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    episodes = []
    h_pres_values = []

    for metrics in training_log:
        if metrics.error is None:
            episodes.append(metrics.episode)
            h_pres_values.append(metrics.h_star_preservation)

    if episodes:
        ax1.scatter(episodes, h_pres_values, alpha=0.3, s=20, label='Per-episode')

        window = min(10, len(h_pres_values) // 4)
        if window > 1 and len(h_pres_values) > window:
            rolling_avg = np.convolve(h_pres_values, np.ones(window) / window, mode='valid')
            ax1.plot(range(window - 1, len(h_pres_values)), rolling_avg, linewidth=2.5,
                     color='darkblue', label=f'Rolling avg (window={window})')

        ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Perfect preservation')
        ax1.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Target (>0.95)')
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('H* Preservation Ratio', fontweight='bold')
        ax1.set_title('Bisimulation Preservation During Training',
                      fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.7, 1.1])

    # Plot 2: Preservation rate by phase
    preservation_rates_per_phase = []
    phase_labels = []

    n_episodes = len(h_pres_values)
    for phase_idx in range(3):
        start = (phase_idx * n_episodes) // 3
        end = ((phase_idx + 1) * n_episodes) // 3

        if start < end:
            phase_values = h_pres_values[start:end]
            good_count = sum(1 for v in phase_values if v >= 0.95)
            rate = good_count / len(phase_values) * 100
            preservation_rates_per_phase.append(rate)
            phase_labels.append(['Early', 'Mid', 'Late'][phase_idx])

    colors_phase = ['red' if r < 50 else 'orange' if r < 80 else 'green'
                    for r in preservation_rates_per_phase]

    bars = ax2.bar(range(len(phase_labels)), preservation_rates_per_phase,
                   color=colors_phase, alpha=0.7, edgecolor='black')

    for bar, val in zip(bars, preservation_rates_per_phase):
        ax2.text(bar.get_x() + bar.get_width() / 2., val,
                 f'{val:.1f}%',
                 ha='center', va='bottom', fontweight='bold')

    ax2.axhline(y=95, color='green', linestyle='--', linewidth=2, label='Target: 95%')
    ax2.set_ylabel('Episodes with H* Preservation > 0.95 (%)', fontweight='bold')
    ax2.set_xlabel('Training Phase', fontweight='bold')
    ax2.set_title('Bisimulation Preservation by Phase', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(phase_labels)))
    ax2.set_xticklabels(phase_labels)
    ax2.set_ylim([0, 120])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = plots_dir / "07_bisimulation_preservation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_dead_end_analysis_plots(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """Plot 8: Dead-end creation timeline and avoidance learning."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Dead-End Creation Risk Analysis', fontweight='bold', fontsize=14)

    # Plot 1: Per-episode dead-end penalty
    colors_penalty = ['red' if p > 0.3 else 'orange' if p > 0.1 else 'green'
                      for p in dead_end_penalties]
    ax1.scatter(episodes, dead_end_penalties, alpha=0.4, s=30, c=colors_penalty)

    window = min(10, len(dead_end_penalties) // 4)
    if window > 1 and len(dead_end_penalties) > window:
        rolling_avg = np.convolve(dead_end_penalties, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(dead_end_penalties)), rolling_avg, linewidth=2.5,
                 color='darkred', label='Trend')

    ax1.set_ylabel('Dead-End Penalty (0-0.5)', fontweight='bold')
    ax1.set_title('Per-Episode Dead-End Penalty')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Cumulative dead-ends
    ax2.fill_between(episodes, cumulative_dead_ends, alpha=0.3, color='red', label='Cumulative penalty')
    ax2.plot(episodes, cumulative_dead_ends, linewidth=2, color='darkred', marker='o', markersize=3)

    ax2.set_ylabel('Cumulative Dead-End Penalty', fontweight='bold')
    ax2.set_title('Cumulative Dead-End Creation Risk')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Solvability maintenance
    solvable_per_phase = []
    phase_labels = []
    n_episodes = len(is_solvable_flags)

    for phase_idx in range(3):
        start = (phase_idx * n_episodes) // 3
        end = ((phase_idx + 1) * n_episodes) // 3
        if start < end:
            phase_values = is_solvable_flags[start:end]
            solve_rate = sum(phase_values) / len(phase_values) * 100
            solvable_per_phase.append(solve_rate)
            phase_labels.append(['Early', 'Mid', 'Late'][phase_idx])

    colors_phase = ['red' if r < 70 else 'orange' if r < 90 else 'green'
                    for r in solvable_per_phase]
    bars = ax3.bar(range(len(phase_labels)), solvable_per_phase, color=colors_phase, alpha=0.7)

    for bar, val in zip(bars, solvable_per_phase):
        ax3.text(bar.get_x() + bar.get_width() / 2., val,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax3.axhline(y=95, color='green', linestyle='--', linewidth=2, label='Target: >95% solvable')
    ax3.set_ylabel('Solvable Episodes (%)', fontweight='bold')
    ax3.set_xlabel('Training Phase', fontweight='bold')
    ax3.set_title('Solvability Maintenance by Phase')
    ax3.set_xticks(range(len(phase_labels)))
    ax3.set_xticklabels(phase_labels)
    ax3.set_ylim([0, 120])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = plots_dir / "08_dead_end_timeline.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_label_reduction_plots(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """Plot 9: Label combinability impact on merge quality."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Label Combinability Impact on Merge Quality\n(Helmert et al. 2014)',
                 fontweight='bold', fontsize=14)

    episodes = []
    label_scores = []
    component_label_rewards = []
    opp_scores = []
    merged_rewards = []

    for metrics in training_log:
        if metrics.error is None:
            episodes.append(metrics.episode)
            label_scores.append(metrics.label_combinability_score)
            component_label_rewards.append(metrics.component_label_combinability)
            opp_scores.append(metrics.opp_score)
            merged_rewards.append(metrics.reward)

    if label_scores:
        ax1.scatter(label_scores, merged_rewards, alpha=0.5, s=50, c=range(len(label_scores)),
                    cmap='viridis', edgecolor='black', linewidth=0.5)

        if len(label_scores) > 2:
            z = np.polyfit(label_scores, merged_rewards, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(label_scores), max(label_scores), 100)
            ax1.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend')

        ax1.set_xlabel('Label Combinability Score', fontweight='bold')
        ax1.set_ylabel('Episode Reward', fontweight='bold')
        ax1.set_title('Label Combinability vs Reward',
                      fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Plot 2: Component evolution
    ax2.plot(episodes, label_scores, 'o-', linewidth=2, label='Label Combinability Score',
             alpha=0.7, markersize=4)
    ax2_twin1 = ax2.twinx()
    ax2_twin1.plot(episodes, opp_scores, 's-', linewidth=2, label='OPP Score',
                   color='orange', alpha=0.7, markersize=4)
    ax2_twin2 = ax2_twin1.twinx()
    ax2_twin2.spines['right'].set_position(('outward', 60))
    ax2_twin2.plot(episodes, merged_rewards, '^-', linewidth=2, label='Reward',
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

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin1.get_legend_handles_labels()
    lines3, labels3 = ax2_twin2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.tight_layout()
    plot_path = plots_dir / "09_label_reduction_impact.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_causal_alignment_plots(
        training_log: List[EpisodeMetrics],
        causal_analysis: Dict,
        output_dir: Path,
) -> Optional[Path]:
    """Plot 10: Causal alignment and merge sequence strategy."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Causal Alignment & Merge Order Strategy',
                 fontweight='bold', fontsize=14)

    merge_qualities = []
    merge_h_pres = []
    merge_steps = []

    step_counter = 0
    for metrics in training_log:
        if metrics.error is None and metrics.merge_decisions_per_step:
            for decision_dict in metrics.merge_decisions_per_step:
                h_pres = decision_dict.get('h_preservation', 1.0)
                is_good = decision_dict.get('is_good_merge', False)

                merge_qualities.append(1.0 if is_good else 0.0)
                merge_h_pres.append(h_pres)
                merge_steps.append(step_counter)
                step_counter += 1

    # Plot 1: H* preservation over merge sequence
    if merge_steps and merge_h_pres:
        ax1.scatter(merge_steps, merge_h_pres, alpha=0.3, s=20, label='Per-merge')

        window = min(20, len(merge_h_pres) // 4)
        if window > 1 and len(merge_h_pres) > window:
            rolling_avg = np.convolve(merge_h_pres, np.ones(window) / window, mode='valid')
            ax1.plot(range(window - 1, len(merge_h_pres)), rolling_avg, linewidth=2.5,
                     color='darkblue', label=f'Rolling avg')

        ax1.axhline(y=0.95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (0.95)')
        ax1.set_xlabel('Merge Sequence Step', fontweight='bold')
        ax1.set_ylabel('H* Preservation', fontweight='bold')
        ax1.set_title('H* Preservation Over Merge Sequence', fontweight='bold')
        ax1.set_ylim([0.7, 1.1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Merge quality by phase
    if merge_qualities:
        n_merges = len(merge_qualities)
        early_quality = np.mean(merge_qualities[:n_merges // 3]) if n_merges > 0 else 0
        mid_quality = np.mean(merge_qualities[n_merges // 3:2 * n_merges // 3]) if n_merges > 0 else 0
        late_quality = np.mean(merge_qualities[-n_merges // 3:]) if n_merges > 0 else 0

        phases = ['Early (0-33%)', 'Mid (33-67%)', 'Late (67-100%)']
        qualities = [early_quality, mid_quality, late_quality]
        colors = ['red' if q < 0.5 else 'orange' if q < 0.7 else 'green' for q in qualities]

        bars = ax2.bar(range(len(phases)), qualities, color=colors, alpha=0.7, edgecolor='black')

        for bar, qual in zip(bars, qualities):
            ax2.text(bar.get_x() + bar.get_width() / 2, qual, f'{qual:.1%}',
                     ha='center', va='bottom', fontweight='bold')

        ax2.set_xticks(range(len(phases)))
        ax2.set_xticklabels(phases)
        ax2.set_ylabel('% Good Merges', fontweight='bold')
        ax2.set_title('Merge Quality by Training Phase', fontweight='bold')
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Strategy explanation
    ax3.axis('off')

    alignment_data = causal_analysis
    alignment_text = f"""
MERGE ORDER STRATEGY ANALYSIS

Early Phase (0-33% of merges):
  • H* Preservation: {alignment_data.get('early_h_preservation', 0):.3f}
  • Good Merge Rate: {alignment_data.get('early_good_rate', 0):.1%}

Late Phase (67-100% of merges):
  • H* Preservation: {alignment_data.get('late_h_preservation', 0):.3f}
  • Good Merge Rate: {alignment_data.get('late_good_rate', 0):.1%}

Strategy Alignment Score: {alignment_data.get('strategy_alignment_score', 0):.3f}
  (>1.0 = Improving = Good strategy)

Merge Pair Distance Analysis:
  • Avg variable distance in pairs: {alignment_data.get('avg_pair_distance', 0):.1f}

Total Merges Analyzed: {alignment_data.get('num_merges_analyzed', 0)}
Learning Trend: {alignment_data.get('learning_trend', 'unknown')}
"""

    ax3.text(0.05, 0.95, alignment_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Pair distance distribution
    if training_log:
        pair_distances = []
        for metrics in training_log:
            if metrics.error is None and metrics.merge_decisions_per_step:
                for decision_dict in metrics.merge_decisions_per_step:
                    pair = decision_dict.get('selected_merge_pair', (0, 1))
                    distance = abs(pair[1] - pair[0])
                    pair_distances.append(distance)

        if pair_distances:
            ax4.hist(pair_distances, bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax4.axvline(x=np.mean(pair_distances), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(pair_distances):.1f}')
            ax4.set_xlabel('Variable Distance in Merge Pair', fontweight='bold')
            ax4.set_ylabel('Frequency', fontweight='bold')
            ax4.set_title('Distribution of Variable Distances in Merges',
                          fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = plots_dir / "10_causal_alignment.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_transition_explosion_plots(
        training_log: List[EpisodeMetrics],
        explosion_analysis: Dict,
        output_dir: Path,
) -> Optional[Path]:
    """Plot 11: Transition explosion risk and GNN prediction."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Transition Explosion Risk Analysis',
                 fontweight='bold', fontsize=14)

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

    if all_trans_growth:
        # Plot 1: Scatter - transition growth vs GNN confidence
        colors_scatter = ['red' if x > explosion_threshold else 'green'
                          for x in all_trans_growth]

        ax1.scatter(all_gnn_probs, all_trans_growth, c=colors_scatter, alpha=0.5, s=50,
                    edgecolors='black', linewidth=0.5)

        ax1.axhline(y=explosion_threshold, color='red', linestyle='--', linewidth=2,
                    label='Explosion threshold (5x growth)')
        ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.3,
                    label='Neutral confidence (50%)')

        ax1.set_xlabel('GNN Action Probability', fontweight='bold')
        ax1.set_ylabel('Transition Growth Ratio', fontweight='bold')
        ax1.set_title('Transition Growth vs GNN Confidence', fontweight='bold')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0.5, min(max(all_trans_growth) * 1.1, 20)])
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy of explosion prediction
    accuracy = explosion_analysis.get('explosion_prediction_accuracy', 0.0)
    baseline = 0.5

    metrics_names = ['GNN Prediction', 'Random Baseline']
    metrics_values = [accuracy, baseline]
    colors_metrics = ['green' if accuracy > 0.7 else 'orange' if accuracy > 0.6 else 'red',
                      'gray']

    bars = ax2.bar(metrics_names, metrics_values, color=colors_metrics, alpha=0.7,
                   edgecolor='black', width=0.6)

    for bar, val in zip(bars, metrics_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.1%}',
                 ha='center', va='bottom', fontweight='bold')

    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
    ax2.set_ylabel('Prediction Accuracy', fontweight='bold')
    ax2.set_title('Can GNN Predict Transition Explosions?', fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: GNN confidence distribution
    explosion_mask = [x > explosion_threshold for x in all_trans_growth]
    explosion_probs = [p for e, p in zip(explosion_mask, all_gnn_probs) if e]
    safe_probs = [p for e, p in zip(explosion_mask, all_gnn_probs) if not e]

    if explosion_probs and safe_probs:
        ax3.hist(safe_probs, bins=20, alpha=0.6, label='Non-explosion merges',
                 color='green', edgecolor='black')
        ax3.hist(explosion_probs, bins=20, alpha=0.6, label='Explosion merges',
                 color='red', edgecolor='black')

        ax3.axvline(x=np.mean(safe_probs), color='darkgreen', linestyle='--', linewidth=2,
                    label=f'Mean (safe): {np.mean(safe_probs):.2f}')
        ax3.axvline(x=np.mean(explosion_probs), color='darkred', linestyle='--', linewidth=2,
                    label=f'Mean (explosion): {np.mean(explosion_probs):.2f}')

        ax3.set_xlabel('GNN Action Probability', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('GNN Confidence Distribution by Merge Safety', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary statistics
    ax4.axis('off')

    summary_text = f"""
TRANSITION EXPLOSION RISK SUMMARY

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

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plot_path = plots_dir / "11_transition_explosion.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_gnn_decision_quality_plots(
        decision_quality_analysis: Dict,
        output_dir: Path,
) -> Optional[Path]:
    """Plot 12: GNN decision quality - confusion matrix & confidence."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if not decision_quality_analysis:
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GNN Decision Quality Analysis',
                 fontweight='bold', fontsize=14)

    # Plot 1: Confusion Matrix
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
    ax1.set_xticklabels(['Good Merge\n(True Positive)', 'Bad Merge\n(True Negative)'], fontweight='bold')
    ax1.set_yticklabels(['GNN Selected\n(High Prob)', 'GNN Avoided\n(Low Prob)'], fontweight='bold')

    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, int(matrix_data[i, j]),
                            ha="center", va="center", color="black", fontweight='bold', fontsize=14)

    ax1.set_title('Confusion Matrix: GNN Decisions vs Merge Quality', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Count', rotation=270, labelpad=20)

    # Plot 2: Accuracy Metrics
    accuracy = decision_quality_analysis.get('gnn_accuracy', 0.0)
    acc_good = decision_quality_analysis.get('gnn_accuracy_good_merges', 0.0)
    acc_bad = decision_quality_analysis.get('gnn_accuracy_bad_merges', 0.0)

    metrics = ['Overall\nAccuracy', 'Good Merge\nRecall', 'Bad Merge\nRecall']
    values = [accuracy, acc_good, acc_bad]
    colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]

    bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', width=0.6)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target (80%)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Baseline (50%)')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.set_title('Decision Accuracy by Merge Category', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Confidence Distribution
    confidence_by_cat = decision_quality_analysis.get('confidence_by_category', {})

    categories = sorted(confidence_by_cat.keys())
    means = [confidence_by_cat[cat]['mean_confidence'] for cat in categories]
    stds = [confidence_by_cat[cat]['std_confidence'] for cat in categories]

    if categories:
        x_pos = np.arange(len(categories))
        ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                color=['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(categories)],
                edgecolor='black')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=45, ha='right')
        ax3.set_ylabel('GNN Action Probability', fontweight='bold')
        ax3.set_title('GNN Confidence by Merge Quality Category', fontweight='bold')
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='Neutral (50%)')
        ax3.set_ylim([0, 1.0])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary
    ax4.axis('off')

    summary_text = f"""
GNN DECISION QUALITY SUMMARY

Overall Accuracy: {accuracy:.1%}
  ✓ Correctly selected good merges: {acc_good:.1%}
  ✓ Correctly avoided bad merges: {acc_bad:.1%}

Decisions Analyzed: {decision_quality_analysis.get('total_decisions_analyzed', 0)}

Decision Entropy: {decision_quality_analysis.get('decision_entropy', 0):.3f}
  (Lower = more confident, Higher = more uncertain)
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = plots_dir / "12_gnn_decision_quality.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_merge_quality_distribution_plots(
        episode_reward_signals: Dict,
        output_dir: Path,
) -> Optional[Path]:
    """Plot 13: Merge pair quality distribution and Pareto frontier."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Merge Pair Quality Distribution',
                 fontweight='bold', fontsize=14)

    quality_categories = defaultdict(int)

    for episode_id, data in episode_reward_signals.items():
        steps_data = data.get('reward_signals_per_step', [])
        for step_data in steps_data:
            h_pres = step_data.get('h_star_preservation', 1.0)
            trans_growth = step_data.get('transition_growth', 1.0)

            if h_pres > 0.8 and trans_growth < 2.0:
                category = 'Excellent'
            elif h_pres > 0.8 and trans_growth < 3.0:
                category = 'Good'
            elif h_pres > 0.7 and trans_growth < 5.0:
                category = 'Moderate'
            elif h_pres > 0.5 and trans_growth < 10.0:
                category = 'Poor'
            else:
                category = 'Bad'

            quality_categories[category] += 1

    categories = ['Excellent', 'Good', 'Moderate', 'Poor', 'Bad']
    counts = [quality_categories.get(cat, 0) for cat in categories]
    colors = ['darkgreen', 'green', 'yellow', 'orange', 'red']

    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{count}\n({count / sum(counts) * 100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    ax1.set_ylabel('Number of Merges', fontweight='bold')
    ax1.set_title('Distribution of Selected Merge Qualities', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Pareto frontier
    all_h_pres = []
    all_trans_growth = []
    colors_scatter = []

    for episode_id, data in episode_reward_signals.items():
        steps_data = data.get('reward_signals_per_step', [])
        for step_data in steps_data:
            h_pres = step_data.get('h_star_preservation', 1.0)
            trans_growth = step_data.get('transition_growth', 1.0)
            all_h_pres.append(h_pres)
            all_trans_growth.append(trans_growth)

            if h_pres > 0.8 and trans_growth < 2.0:
                colors_scatter.append('darkgreen')
            elif h_pres > 0.7 and trans_growth < 5.0:
                colors_scatter.append('orange')
            else:
                colors_scatter.append('red')

    ax2.scatter(all_trans_growth, all_h_pres, c=colors_scatter, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)

    ax2.axvspan(0, 2.0, alpha=0.1, color='green', label='Target region')
    ax2.axhspan(0.8, 1.0, alpha=0.1, color='green')

    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axvline(x=2.0, color='green', linestyle='--', alpha=0.5, linewidth=2)

    ax2.set_xlabel('Transition Growth (lower = better)', fontweight='bold')
    ax2.set_ylabel('H* Preservation (higher = better)', fontweight='bold')
    ax2.set_title('Pareto Frontier: Accuracy vs Size Tradeoff', fontweight='bold')
    ax2.set_xlim([0.5, max(10, max(all_trans_growth) * 1.1)])
    ax2.set_ylim([0.4, 1.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = plots_dir / "13_merge_quality_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_baseline_comparison_plots(
        training_log: List[EpisodeMetrics],
        output_dir: Path,
) -> Optional[Path]:
    """Plot 14: Baseline comparison (GNN vs RL-G vs DFP vs Random)."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    successful_log = [m for m in training_log if m.error is None]
    if not successful_log:
        return None

    gnn_final_reward = np.mean([m.reward for m in successful_log[-50:]])
    gnn_h_preservation = np.mean([m.h_star_preservation for m in successful_log[-50:]])
    gnn_solve_rate = sum(1 for m in successful_log[-50:] if m.is_solvable) / len(successful_log[-50:])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GNN vs Baseline Comparison',
                 fontweight='bold', fontsize=14)

    baselines = {
        'GNN (Ours)': {
            'reward': gnn_final_reward,
            'h_preservation': gnn_h_preservation,
            'solve_rate': gnn_solve_rate,
            'color': 'blue',
        },
        'RL-G': {
            'reward': 0.0,
            'h_preservation': 0.0,
            'solve_rate': 0.0,
            'color': 'green',
        },
        'DFP': {
            'reward': 0.0,
            'h_preservation': 0.0,
            'solve_rate': 0.0,
            'color': 'orange',
        },
        'Random': {
            'reward': 0.0,
            'h_preservation': 0.0,
            'solve_rate': 0.0,
            'color': 'red',
        },
    }

    baseline_names = list(baselines.keys())
    rewards = [baselines[name]['reward'] for name in baseline_names]
    colors = [baselines[name]['color'] for name in baseline_names]

    bars = ax1.bar(baseline_names, rewards, color=colors, alpha=0.7, edgecolor='black')
    for bar, reward in zip(bars, rewards):
        if reward > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, reward, f'{reward:.3f}',
                     ha='center', va='bottom', fontweight='bold')

    ax1.set_ylabel('Average Reward', fontweight='bold')
    ax1.set_title('Final Reward Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    h_preservations = [baselines[name]['h_preservation'] for name in baseline_names]
    bars = ax2.bar(baseline_names, h_preservations, color=colors, alpha=0.7, edgecolor='black')
    for bar, h_pres in zip(bars, h_preservations):
        if h_pres > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h_pres, f'{h_pres:.3f}',
                     ha='center', va='bottom', fontweight='bold')

    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target (0.95)')
    ax2.set_ylabel('H* Preservation Ratio', fontweight='bold')
    ax2.set_title('H* Preservation Comparison', fontweight='bold')
    ax2.set_ylim([0.8, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    solve_rates = [baselines[name]['solve_rate'] * 100 for name in baseline_names]
    bars = ax3.bar(baseline_names, solve_rates, color=colors, alpha=0.7, edgecolor='black')
    for bar, rate in zip(bars, solve_rates):
        if rate > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, rate, f'{rate:.1f}%',
                     ha='center', va='bottom', fontweight='bold')

    ax3.axhline(y=95, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target (95%)')
    ax3.set_ylabel('Problem Solve Rate (%)', fontweight='bold')
    ax3.set_title('Solvability Comparison', fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    ax4.axis('off')

    summary_text = """
BASELINE COMPARISON RESULTS

STATUS: ⚠️  PLACEHOLDER
To complete this analysis:
1. Run RL-G baseline on same problems
2. Run DFP baseline on same problems
3. Run random baseline on same problems
4. Replace placeholder values in code
5. Re-generate this plot

GNN ADVANTAGE:
If GNN reward > RL-G reward: ✅ GNN learned better strategy
If GNN h_pres > DFP h_pres: ✅ GNN preserves heuristic quality
If GNN reward > Random reward: ✅ GNN learns non-trivial policy
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plot_path = plots_dir / "14_baseline_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def create_literature_alignment_report(
        checklist: Dict[str, bool],
        output_dir: Path,
) -> Optional[Path]:
    """Plot 15: Literature alignment checklist visualization."""
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    check_names = []
    check_passes = []
    categories = []

    for check_name, passes in sorted(checklist.items()):
        check_names.append(check_name.replace('_', ' ').title())
        check_passes.append(1.0 if passes else 0.0)

        if any(x in check_name for x in ['label_', 'transition_', 'irrelevance_']):
            categories.append('Helmert et al. 2014')
        elif any(x in check_name for x in ['opp_', 'h_preservation', 'equivalence']):
            categories.append('Nissim et al. 2011')
        elif any(x in check_name for x in ['node_', 'edge_', 'gnn_']):
            categories.append('GNN Architecture')
        else:
            categories.append('Validation')

    helmert_checks = [(n, p) for n, p, c in zip(check_names, check_passes, categories)
                      if c == 'Helmert et al. 2014']
    nissim_checks = [(n, p) for n, p, c in zip(check_names, check_passes, categories)
                     if c == 'Nissim et al. 2011']
    gnn_checks = [(n, p) for n, p, c in zip(check_names, check_passes, categories)
                  if c == 'GNN Architecture']
    val_checks = [(n, p) for n, p, c in zip(check_names, check_passes, categories)
                  if c == 'Validation']

    all_groups = [
        ('Helmert et al. 2014', helmert_checks),
        ('Nissim et al. 2011', nissim_checks),
        ('GNN Architecture', gnn_checks),
        ('Validation', val_checks),
    ]

    y_pos = 0
    colors_list = {'Helmert et al. 2014': '#1f77b4', 'Nissim et al. 2011': '#ff7f0e',
                   'GNN Architecture': '#2ca02c', 'Validation': '#d62728'}

    for group_name, checks in all_groups:
        if checks:
            ax.text(-0.1, y_pos + len(checks) / 2, group_name, fontweight='bold',
                    fontsize=11, ha='right', va='center',
                    bbox=dict(boxstyle='round', facecolor=colors_list[group_name], alpha=0.3))

            for check_name, passes in checks:
                color = 'green' if passes else 'red'
                marker = '✅' if passes else '❌'
                ax.barh(y_pos, passes, color=color, alpha=0.7, edgecolor='black')
                ax.text(-0.05, y_pos, marker, fontsize=12, ha='right', va='center', fontweight='bold')
                ax.text(0.5, y_pos, check_name, fontsize=10, va='center')
                y_pos += 1

            y_pos += 0.5

    ax.set_xlim([-0.3, 1.3])
    ax.set_ylim([-1, y_pos])
    ax.set_xlabel('Implementation Status', fontweight='bold')
    ax.set_title('Literature Alignment Checklist',
                 fontweight='bold', fontsize=14)
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Implemented', 'Implemented'])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plot_path = plots_dir / "15_literature_alignment.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


# ============================================================================
# ORCHESTRATION
# ============================================================================

def generate_all_plots(
        training_log: List[EpisodeMetrics],
        eval_results: Dict,
        output_dir: Path,
        component_analysis: Optional[Dict] = None,
        correlation_analysis: Optional[Dict] = None,
        feature_importance_analysis: Optional[Dict] = None,
        bisim_analysis: Optional[Dict] = None,
        causal_alignment_analysis: Optional[Dict] = None,
        explosion_analysis: Optional[Dict] = None,
        decision_quality_analysis: Optional[Dict] = None,
        episode_reward_signals: Optional[Dict] = None,
        literature_checklist: Optional[Dict] = None,
) -> Dict[str, Optional[Path]]:
    """Generate all visualization plots for an experiment."""
    output_path = Path(output_dir)

    results = {}

    # Core plots
    results['learning_curves'] = create_learning_curves(training_log, output_path)
    results['component_tracking'] = create_component_tracking_plots(
        training_log, component_analysis or {}, output_path
    )
    results['feature_analysis'] = create_feature_analysis_plots(
        training_log, correlation_analysis or {}, feature_importance_analysis or {}, output_path
    )
    results['bisimulation'] = create_bisimulation_plots(
        training_log, bisim_analysis or {}, output_path
    )
    results['dead_end_analysis'] = create_dead_end_analysis_plots(training_log, output_path)
    results['label_reduction'] = create_label_reduction_plots(training_log, output_path)
    results['causal_alignment'] = create_causal_alignment_plots(
        training_log, causal_alignment_analysis or {}, output_path
    )
    results['transition_explosion'] = create_transition_explosion_plots(
        training_log, explosion_analysis or {}, output_path
    )
    results['gnn_decision_quality'] = create_gnn_decision_quality_plots(
        decision_quality_analysis or {}, output_path
    )
    results['merge_quality_distribution'] = create_merge_quality_distribution_plots(
        episode_reward_signals or {}, output_path
    )
    results['baseline_comparison'] = create_baseline_comparison_plots(training_log, output_path)
    results['literature_alignment'] = create_literature_alignment_report(
        literature_checklist or {}, output_path
    )

    return results