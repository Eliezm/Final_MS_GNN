#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 28-32: CURRICULUM LEARNING ANALYSIS
========================================
Specialized plots for curriculum learning experiments.

Research Questions:
- Does curriculum learning help compared to direct training?
- How does knowledge transfer between phases?
- When do phase transitions help/hurt?
- Can learned strategies transfer across domains?
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


def plot_curriculum_phase_transitions(
        phase_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 28: Curriculum Phase Transitions

    Research Question: How does performance evolve across curriculum phases?

    What it shows:
    - Performance metrics per phase
    - Transition effects (performance dips/jumps at phase boundaries)
    - Cumulative learning curve

    Key Insight: Reveals whether curriculum structure helps learning
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not phase_results:
        logger.warning("No phase results for curriculum plot")
        return None

    plots_dir = create_plots_directory(output_dir)

    # Sort phases by name/number
    phase_names = sorted(phase_results.keys())
    if not phase_names:
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curriculum Learning Phase Analysis\n'
                 'Research Q: How does performance evolve across curriculum phases?',
                 fontsize=14, fontweight='bold')

    # Panel 1: Average reward per phase
    phase_rewards = []
    phase_h_pres = []
    phase_episodes = []

    for phase_name in phase_names:
        phase_data = phase_results[phase_name]
        summary = phase_data.get('summary', {})

        avg_reward = summary.get('avg_reward_over_all', 0)
        phase_rewards.append(avg_reward)

        # Extract h* preservation if available
        h_pres = summary.get('avg_h_preservation', 1.0)
        if h_pres == 0:
            # Try to get from per_problem_stats
            per_problem = summary.get('per_problem_stats', [])
            if per_problem:
                h_pres = np.mean([p.get('avg_h_preservation', 1.0) for p in per_problem])
        phase_h_pres.append(h_pres)

        num_episodes = phase_data.get('num_episodes', 0)
        phase_episodes.append(num_episodes)

    x = np.arange(len(phase_names))

    # Reward bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(phase_names)))
    bars = ax1.bar(x, phase_rewards, color=colors, alpha=0.8, edgecolor='black')

    for bar, reward in zip(bars, phase_rewards):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{reward:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace('phase_', 'P').replace('_', '\n')[:15] for p in phase_names], fontsize=9)
    format_plot_labels(ax1, 'Phase', 'Average Reward',
                       'Average Reward per Phase')

    # Panel 2: H* Preservation across phases
    bars = ax2.bar(x, phase_h_pres, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0.95, color='green', linestyle='--', label='Target: 0.95', linewidth=2)

    for bar, h in zip(bars, phase_h_pres):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace('phase_', 'P').replace('_', '\n')[:15] for p in phase_names], fontsize=9)
    format_plot_labels(ax2, 'Phase', 'H* Preservation',
                       'H* Preservation per Phase')
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 1.1])

    # Panel 3: Episode distribution
    ax3.bar(x, phase_episodes, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels([p.replace('phase_', 'P').replace('_', '\n')[:15] for p in phase_names], fontsize=9)
    format_plot_labels(ax3, 'Phase', 'Number of Episodes',
                       'Training Distribution Across Phases')

    # Panel 4: Cumulative learning curve
    cumulative_reward = np.cumsum([r * e for r, e in zip(phase_rewards, phase_episodes)])
    cumulative_episodes = np.cumsum(phase_episodes)

    ax4.plot(cumulative_episodes, cumulative_reward, 'b-o', linewidth=2, markersize=10)

    # Mark phase boundaries
    for i, (ep, rew) in enumerate(zip(cumulative_episodes, cumulative_reward)):
        ax4.axvline(x=ep, color='gray', linestyle='--', alpha=0.3)
        ax4.annotate(phase_names[i].replace('phase_', 'P')[:8],
                     xy=(ep, rew), xytext=(5, 10), textcoords='offset points',
                     fontsize=8, fontweight='bold')

    format_plot_labels(ax4, 'Cumulative Episodes', 'Cumulative Reward',
                       'Cumulative Learning Progress')

    plt.tight_layout()
    plot_path = plots_dir / "28_curriculum_phase_transitions.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_knowledge_transfer_analysis(
        phase_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 29: Knowledge Transfer Between Phases

    Research Question: Is knowledge from earlier phases helping later phases?

    What it shows:
    - Learning speed comparison (episodes to threshold per phase)
    - Initial performance at phase start (warm start effect)
    - Catastrophic forgetting detection

    Key Insight: Validates curriculum design effectiveness
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not phase_results:
        return None

    plots_dir = create_plots_directory(output_dir)

    phase_names = sorted(phase_results.keys())
    if len(phase_names) < 2:
        logger.warning("Need at least 2 phases for knowledge transfer analysis")
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Knowledge Transfer Analysis\n'
                 'Research Q: Is knowledge from earlier phases helping later phases?',
                 fontsize=14, fontweight='bold')

    # Extract episode logs from each phase
    phase_episode_logs = {}
    for phase_name in phase_names:
        phase_data = phase_results[phase_name]
        training_log = phase_data.get('training_log', [])

        if training_log:
            # Extract just rewards
            rewards = [ep.get('reward', ep.reward if hasattr(ep, 'reward') else 0)
                       for ep in training_log if isinstance(ep, dict) or hasattr(ep, 'reward')]
            phase_episode_logs[phase_name] = rewards

    # Panel 1: Initial performance at each phase start
    initial_performances = []
    for phase_name in phase_names:
        rewards = phase_episode_logs.get(phase_name, [])
        if len(rewards) >= 5:
            initial_perf = np.mean(rewards[:5])
        elif rewards:
            initial_perf = rewards[0]
        else:
            initial_perf = 0
        initial_performances.append(initial_perf)

    x = np.arange(len(phase_names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(phase_names)))

    bars = ax1.bar(x, initial_performances, color=colors, alpha=0.8, edgecolor='black')

    # Add improvement arrows
    for i in range(1, len(initial_performances)):
        if initial_performances[i] > initial_performances[i - 1]:
            ax1.annotate('', xy=(i, initial_performances[i]),
                         xytext=(i - 1, initial_performances[i - 1]),
                         arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace('phase_', 'P')[:12] for p in phase_names], fontsize=9)
    format_plot_labels(ax1, 'Phase', 'Initial Performance (first 5 episodes)',
                       'Warm Start Effect - Higher = Better Transfer')

    # Panel 2: Learning speed (episodes to reach threshold)
    threshold = 0.5  # Configurable threshold
    episodes_to_threshold = []

    for phase_name in phase_names:
        rewards = phase_episode_logs.get(phase_name, [])
        if rewards:
            # Find first episode where reward exceeds threshold
            reached = False
            for i, r in enumerate(rewards):
                if r >= threshold:
                    episodes_to_threshold.append(i + 1)
                    reached = True
                    break
            if not reached:
                episodes_to_threshold.append(len(rewards))  # Never reached
        else:
            episodes_to_threshold.append(0)

    bars = ax2.bar(x, episodes_to_threshold, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=np.mean(episodes_to_threshold), color='red', linestyle='--',
                label=f'Mean: {np.mean(episodes_to_threshold):.0f}')

    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace('phase_', 'P')[:12] for p in phase_names], fontsize=9)
    format_plot_labels(ax2, 'Phase', 'Episodes to Threshold',
                       f'Learning Speed (threshold={threshold}) - Lower = Faster')
    ax2.legend(fontsize=9)

    # Panel 3: All phase learning curves overlaid
    for i, phase_name in enumerate(phase_names):
        rewards = phase_episode_logs.get(phase_name, [])
        if rewards:
            # Normalize episode numbers to 0-1
            normalized_x = np.linspace(0, 1, len(rewards))
            ax3.plot(normalized_x, rewards, label=phase_name.replace('phase_', 'P')[:10],
                     color=colors[i], alpha=0.7, linewidth=2)

    format_plot_labels(ax3, 'Normalized Episode (0=start, 1=end)', 'Reward',
                       'Overlaid Learning Curves per Phase')
    ax3.legend(fontsize=8, loc='lower right')

    # Panel 4: Transfer efficiency matrix
    if len(phase_names) >= 2:
        # Simple transfer metric: improvement at start of phase N vs end of phase N-1
        transfer_scores = []
        for i in range(1, len(phase_names)):
            prev_phase = phase_names[i - 1]
            curr_phase = phase_names[i]

            prev_rewards = phase_episode_logs.get(prev_phase, [])
            curr_rewards = phase_episode_logs.get(curr_phase, [])

            if prev_rewards and curr_rewards:
                prev_final = np.mean(prev_rewards[-5:]) if len(prev_rewards) >= 5 else prev_rewards[-1]
                curr_initial = np.mean(curr_rewards[:5]) if len(curr_rewards) >= 5 else curr_rewards[0]

                # Transfer score: how much of previous learning is retained
                if prev_final != 0:
                    transfer = curr_initial / abs(prev_final)
                else:
                    transfer = 1.0
                transfer_scores.append(min(2.0, transfer))
            else:
                transfer_scores.append(1.0)

        transfer_labels = [f'{phase_names[i - 1][:6]}→{phase_names[i][:6]}'
                           for i in range(1, len(phase_names))]

        transfer_colors = ['#27ae60' if t >= 0.8 else '#f39c12' if t >= 0.5 else '#e74c3c'
                           for t in transfer_scores]

        bars = ax4.bar(range(len(transfer_scores)), transfer_scores,
                       color=transfer_colors, alpha=0.8, edgecolor='black')
        ax4.axhline(y=1.0, color='green', linestyle='--', label='Perfect transfer', linewidth=2)
        ax4.axhline(y=0.5, color='red', linestyle='--', label='50% retention', linewidth=2)

        ax4.set_xticks(range(len(transfer_labels)))
        ax4.set_xticklabels(transfer_labels, fontsize=9, rotation=45, ha='right')
        format_plot_labels(ax4, 'Phase Transition', 'Transfer Score (ratio)',
                           'Knowledge Transfer Efficiency')
        ax4.legend(fontsize=8)

    plt.tight_layout()
    plot_path = plots_dir / "29_knowledge_transfer_analysis.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_domain_transfer_results(
        same_domain_results: Dict[str, Dict[str, Any]],
        transfer_domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 30: Domain Transfer Analysis (Blocksworld → Logistics)

    Research Question: Can learned strategies transfer across domains?

    What it shows:
    - Performance on same domain (Blocksworld) vs different domain (Logistics)
    - Transfer gap by problem size
    - Structural similarity correlation

    Key Insight: Tests true domain-independent learning
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not same_domain_results and not transfer_domain_results:
        logger.warning("No domain transfer data available")
        return None

    plots_dir = create_plots_directory(output_dir)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Domain Transfer Analysis: Blocksworld → Logistics\n'
                 'Research Q: Can learned strategies transfer across domains?',
                 fontsize=14, fontweight='bold')

    # Extract solve rates
    sizes = ['small', 'medium', 'large']
    same_domain_rates = []
    transfer_domain_rates = []

    for size in sizes:
        # Same domain
        same_rate = 0
        for test_name, test_data in same_domain_results.items():
            if size in test_name.lower() and 'blocksworld' in test_name.lower():
                summary = test_data.get('results', {}).get('summary', {})
                total = summary.get('gnn_total', 0)
                solved = summary.get('gnn_solved', 0)
                if total > 0:
                    same_rate = solved / total * 100
                break
        same_domain_rates.append(same_rate)

        # Transfer domain
        transfer_rate = 0
        for test_name, test_data in transfer_domain_results.items():
            if size in test_name.lower() and 'logistics' in test_name.lower():
                summary = test_data.get('results', {}).get('summary', {})
                total = summary.get('gnn_total', 0)
                solved = summary.get('gnn_solved', 0)
                if total > 0:
                    transfer_rate = solved / total * 100
                break
        transfer_domain_rates.append(transfer_rate)

    # Panel 1: Side-by-side comparison
    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, same_domain_rates, width, label='Blocksworld (same domain)',
                    color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, transfer_domain_rates, width, label='Logistics (transfer)',
                    color='#9b59b6', alpha=0.8)

    for bar, rate in zip(bars1, same_domain_rates):
        if rate > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, rate in zip(bars2, transfer_domain_rates):
        if rate > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in sizes])
    format_plot_labels(ax1, 'Problem Size', 'Solve Rate (%)',
                       'Same Domain vs Transfer Domain Performance')
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 110])

    # Panel 2: Transfer gap
    transfer_gap = [same - transfer for same, transfer in zip(same_domain_rates, transfer_domain_rates)]
    colors = ['#27ae60' if gap < 10 else '#f39c12' if gap < 30 else '#e74c3c' for gap in transfer_gap]

    bars = ax2.bar(sizes, transfer_gap, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='black', linewidth=1)

    for bar, gap in zip(bars, transfer_gap):
        va = 'bottom' if gap >= 0 else 'top'
        offset = 1 if gap >= 0 else -1
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                 f'{gap:.1f}%', ha='center', va=va, fontweight='bold', fontsize=11)

    format_plot_labels(ax2, 'Problem Size', 'Transfer Gap (%)',
                       'Performance Drop on Transfer Domain\n(Lower = Better Transfer)')

    # Panel 3: Transfer ratio (what fraction of performance is retained)
    transfer_ratios = []
    for same, transfer in zip(same_domain_rates, transfer_domain_rates):
        if same > 0:
            ratio = transfer / same
        else:
            ratio = 1.0 if transfer > 0 else 0.0
        transfer_ratios.append(min(1.5, ratio))

    colors = ['#27ae60' if r >= 0.7 else '#f39c12' if r >= 0.3 else '#e74c3c' for r in transfer_ratios]
    bars = ax3.bar(sizes, transfer_ratios, color=colors, alpha=0.8, edgecolor='black')
    ax3.axhline(y=1.0, color='green', linestyle='--', label='Perfect transfer', linewidth=2)
    ax3.axhline(y=0.5, color='orange', linestyle='--', label='50% retention', linewidth=2)

    for bar, ratio in zip(bars, transfer_ratios):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    format_plot_labels(ax3, 'Problem Size', 'Transfer Ratio',
                       'Knowledge Transfer Ratio (1.0 = Perfect)')
    ax3.legend(fontsize=9)

    # Panel 4: Summary statistics
    ax4.axis('off')

    avg_same = np.mean(same_domain_rates) if same_domain_rates else 0
    avg_transfer = np.mean(transfer_domain_rates) if transfer_domain_rates else 0
    avg_gap = np.mean(transfer_gap) if transfer_gap else 0
    avg_ratio = np.mean(transfer_ratios) if transfer_ratios else 0

    summary_text = f"""
DOMAIN TRANSFER SUMMARY
{'=' * 50}

Training Domain: Blocksworld
Transfer Domain: Logistics

Performance Comparison:
  • Same domain avg:     {avg_same:.1f}%
  • Transfer domain avg: {avg_transfer:.1f}%
  • Average gap:         {avg_gap:.1f}%
  • Transfer ratio:      {avg_ratio:.2f}

Size-Specific Transfer:
  • Small:  {transfer_ratios[0]:.2f} ratio ({transfer_gap[0]:+.1f}% gap)
  • Medium: {transfer_ratios[1]:.2f} ratio ({transfer_gap[1]:+.1f}% gap)
  • Large:  {transfer_ratios[2]:.2f} ratio ({transfer_gap[2]:+.1f}% gap)

Key Insight:
  {'✓ Good transfer!' if avg_ratio >= 0.7 else '⚠ Limited transfer' if avg_ratio >= 0.3 else '✗ Poor transfer'}
  The GNN strategy {'generalizes well' if avg_ratio >= 0.7 else 'partially transfers' if avg_ratio >= 0.3 else 'does not transfer well'} 
  from {' Blocksworld'} to {'Logistics'}.
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plot_path = plots_dir / "30_domain_transfer_results.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_curriculum_vs_direct_training(
        curriculum_results: Dict[str, Any],
        direct_training_results: Dict[str, Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 31: Curriculum vs Direct Training Comparison

    Research Question: Does curriculum learning outperform direct training on hard problems?

    What it shows:
    - Head-to-head comparison on same test sets
    - Learning efficiency (episodes to same performance)
    - Final performance comparison

    Key Insight: Validates the curriculum learning hypothesis
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not curriculum_results and not direct_training_results:
        return None

    plots_dir = create_plots_directory(output_dir)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curriculum vs Direct Training Comparison\n'
                 'Research Q: Does curriculum learning outperform direct training?',
                 fontsize=14, fontweight='bold')

    # Extract test results from both approaches
    sizes = ['small', 'medium', 'large']

    curriculum_rates = []
    direct_rates = []

    for size in sizes:
        # Curriculum results
        curr_rate = 0
        for test_name, test_data in curriculum_results.get('test_results', {}).items():
            if size in test_name.lower():
                summary = test_data.get('results', {}).get('summary', {})
                total = summary.get('gnn_total', 0)
                solved = summary.get('gnn_solved', 0)
                if total > 0:
                    curr_rate = solved / total * 100
                break
        curriculum_rates.append(curr_rate)

        # Direct training results
        dir_rate = 0
        for test_name, test_data in direct_training_results.get('test_results', {}).items():
            if size in test_name.lower():
                summary = test_data.get('results', {}).get('summary', {})
                total = summary.get('gnn_total', 0)
                solved = summary.get('gnn_solved', 0)
                if total > 0:
                    dir_rate = solved / total * 100
                break
        direct_rates.append(dir_rate)

    # Panel 1: Performance comparison
    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, curriculum_rates, width, label='Curriculum',
                    color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, direct_rates, width, label='Direct Training',
                    color='#3498db', alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in sizes])
    format_plot_labels(ax1, 'Test Problem Size', 'Solve Rate (%)',
                       'Final Performance Comparison')
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 110])

    # Panel 2: Curriculum advantage
    advantages = [curr - direct for curr, direct in zip(curriculum_rates, direct_rates)]
    colors = ['#27ae60' if adv > 0 else '#e74c3c' for adv in advantages]

    bars = ax2.bar(sizes, advantages, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='black', linewidth=1)

    for bar, adv in zip(bars, advantages):
        va = 'bottom' if adv >= 0 else 'top'
        offset = 1 if adv >= 0 else -1
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                 f'{adv:+.1f}%', ha='center', va=va, fontweight='bold', fontsize=11)

    format_plot_labels(ax2, 'Problem Size', 'Curriculum Advantage (%)',
                       'Curriculum vs Direct: Positive = Curriculum Better')

    # Panel 3: Training efficiency (episodes used)
    curriculum_episodes = curriculum_results.get('total_episodes', 0)
    direct_episodes = direct_training_results.get('total_episodes', 0)

    if curriculum_episodes > 0 or direct_episodes > 0:
        bars = ax3.bar(['Curriculum', 'Direct'], [curriculum_episodes, direct_episodes],
                       color=['#2ecc71', '#3498db'], alpha=0.8, edgecolor='black')

        for bar in bars:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

        format_plot_labels(ax3, 'Training Approach', 'Total Episodes',
                           'Training Effort Comparison')

    # Panel 4: Summary
    ax4.axis('off')

    avg_curr = np.mean(curriculum_rates) if curriculum_rates else 0
    avg_direct = np.mean(direct_rates) if direct_rates else 0
    avg_advantage = np.mean(advantages) if advantages else 0

    winner = "Curriculum" if avg_advantage > 0 else "Direct Training" if avg_advantage < 0 else "Tie"

    summary_text = f"""
CURRICULUM VS DIRECT TRAINING SUMMARY
{'=' * 50}

Average Performance:
  • Curriculum:     {avg_curr:.1f}%
  • Direct:         {avg_direct:.1f}%
  • Advantage:      {avg_advantage:+.1f}%

Per-Size Breakdown:
  • Small:   Curr={curriculum_rates[0]:.1f}% vs Direct={direct_rates[0]:.1f}% ({advantages[0]:+.1f}%)
  • Medium:  Curr={curriculum_rates[1]:.1f}% vs Direct={direct_rates[1]:.1f}% ({advantages[1]:+.1f}%)
  • Large:   Curr={curriculum_rates[2]:.1f}% vs Direct={direct_rates[2]:.1f}% ({advantages[2]:+.1f}%)

Training Effort:
  • Curriculum: {curriculum_episodes} episodes
  • Direct:     {direct_episodes} episodes

Winner: {winner}

Key Insight:
  {'✓ Curriculum learning helps!' if avg_advantage > 5 else '~ Similar performance' if abs(avg_advantage) <= 5 else '✗ Direct training is better'}
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.tight_layout()
    plot_path = plots_dir / "31_curriculum_vs_direct_training.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None


def plot_phase_difficulty_progression(
        phase_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
) -> Optional[Path]:
    """
    Plot 32: Phase Difficulty Progression

    Research Question: How does problem difficulty change across curriculum phases?

    What it shows:
    - Average episode length per phase (proxy for difficulty)
    - Failure rates per phase
    - Time-to-solve trends

    Key Insight: Validates curriculum is actually increasing difficulty
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    if not phase_results:
        return None

    plots_dir = create_plots_directory(output_dir)

    phase_names = sorted(phase_results.keys())
    if not phase_names:
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase Difficulty Progression\n'
                 'Research Q: Is curriculum actually increasing difficulty?',
                 fontsize=14, fontweight='bold')

    # Extract difficulty metrics per phase
    avg_steps = []
    failure_rates = []
    avg_times = []

    for phase_name in phase_names:
        phase_data = phase_results[phase_name]
        summary = phase_data.get('summary', {})

        # Average steps per episode
        steps = summary.get('avg_steps_per_episode', 0)
        if steps == 0:
            # Try to compute from per_problem_stats
            per_problem = summary.get('per_problem_stats', [])
            if per_problem:
                steps = np.mean([p.get('avg_steps', 0) for p in per_problem])
        avg_steps.append(steps)

        # Failure rate
        num_failed = phase_data.get('num_failed', 0)
        num_episodes = phase_data.get('num_episodes', 1)
        failure_rates.append(num_failed / max(1, num_episodes) * 100)

        # Average time
        avg_time = summary.get('avg_step_time_ms', 0)
        avg_times.append(avg_time)

    x = np.arange(len(phase_names))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(phase_names)))

    # Panel 1: Average steps (difficulty proxy)
    bars = ax1.bar(x, avg_steps, color=colors, alpha=0.8, edgecolor='black')

    # Add trend line
    if len(avg_steps) > 1:
        z = np.polyfit(x, avg_steps, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
        ax1.legend(fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace('phase_', 'P')[:12] for p in phase_names], fontsize=9)
    format_plot_labels(ax1, 'Phase', 'Average Steps per Episode',
                       'Episode Length by Phase (Higher = Harder)')

    # Panel 2: Failure rates
    bars = ax2.bar(x, failure_rates, color=colors, alpha=0.8, edgecolor='black')

    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace('phase_', 'P')[:12] for p in phase_names], fontsize=9)
    format_plot_labels(ax2, 'Phase', 'Failure Rate (%)',
                       'Failure Rate by Phase')

    # Panel 3: Step time (computational difficulty)
    bars = ax3.bar(x, avg_times, color=colors, alpha=0.8, edgecolor='black')

    ax3.set_xticks(x)
    ax3.set_xticklabels([p.replace('phase_', 'P')[:12] for p in phase_names], fontsize=9)
    format_plot_labels(ax3, 'Phase', 'Average Step Time (ms)',
                       'Computational Cost by Phase')

    # Panel 4: Difficulty progression summary
    ax4.axis('off')

    # Compute difficulty slope
    steps_increasing = all(avg_steps[i] <= avg_steps[i + 1] for i in range(len(avg_steps) - 1))

    summary_text = f"""
DIFFICULTY PROGRESSION ANALYSIS
{'=' * 50}

Per-Phase Metrics:
"""
    for i, phase_name in enumerate(phase_names):
        summary_text += f"\n  {phase_name.replace('phase_', 'P')[:15]}:"
        summary_text += f"\n    • Steps: {avg_steps[i]:.1f}"
        summary_text += f"\n    • Failures: {failure_rates[i]:.1f}%"
        summary_text += f"\n    • Time: {avg_times[i]:.1f}ms"

    summary_text += f"""

Progression Analysis:
  • Steps increasing: {'✓ Yes' if steps_increasing else '✗ No'}
  • Avg difficulty ratio: {avg_steps[-1] / max(0.1, avg_steps[0]):.2f}x (last/first)
  • Failure rate change: {failure_rates[-1] - failure_rates[0]:+.1f}%

Key Insight:
  {'✓ Curriculum follows proper difficulty progression' if steps_increasing else '⚠ Difficulty progression may not be optimal'}
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plot_path = plots_dir / "32_phase_difficulty_progression.png"

    if save_plot_safely(fig, plot_path):
        return plot_path
    return None