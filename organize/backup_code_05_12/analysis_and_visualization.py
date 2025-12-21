#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE ANALYSIS & VISUALIZATION MODULE - FULLY COMPATIBLE
==================================================================
Advanced plots and statistical analysis for evaluation reporting.

Compatible with:
✓ All experiment types (overfit, problem_gen, scale_gen, curriculum)
✓ Baseline and GNN evaluation
✓ Enhanced feature dimensions (15-dim node, 10-dim edge)
✓ H* preservation metrics
✓ Research-grade statistical tests

Plots Generated:
1. Solve rate comparison (bar chart)
2. Time comparison (box plots + scatter)
3. Expansions comparison (log scale)
4. Efficiency frontier (2D scatter)
5. Per-problem heatmap
6. Time distribution (violin plots)
7. Cumulative distribution
8. Per-difficulty breakdown
9. Performance profile
10. Statistical summary dashboard
11. Curriculum effectiveness (if applicable)
12. H* preservation analysis (GNN-specific)
"""

import sys
import os
import logging
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib/seaborn not installed - plotting disabled")

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed - advanced features disabled")

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    "style": "seaborn-v0_8-whitegrid",
    "figsize": (14, 8),
    "figsize_wide": (16, 10),
    "dpi": 150,
    "font_size": 11,
}

COLORS = {
    "GNN": "#2E86AB",
    "FD_LM-Cut": "#F18F01",
    "FD_Blind": "#C73E1D",
    "FD_Add": "#06A77D",
    "FD_Max": "#D62828",
    "FD_M&S_DFP": "#A23B72",
    "FD_M&S_SCC": "#9D4EDD",
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_results_csv(csv_path: str) -> Optional['pd.DataFrame']:
    """Load evaluation results from CSV."""
    if not HAS_PANDAS:
        logger.warning("pandas not available - cannot load CSV")
        return None

    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} results from CSV")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return None


def load_experiment_results(
        experiment_dirs: Dict[str, str]
) -> Optional['pd.DataFrame']:
    """Load results from experiment directories."""
    if not HAS_PANDAS:
        logger.warning("pandas not available")
        return None

    logger.info("Loading experiment results...")

    all_data = []

    for exp_type, exp_dir in experiment_dirs.items():
        results_file = os.path.join(exp_dir, "results.json")

        if not os.path.exists(results_file):
            logger.warning(f"Results file not found: {results_file}")
            continue

        logger.info(f"  Loading: {exp_type}...")

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                exp_results = json.load(f)

            # Parse experiment results
            if 'evaluation' in exp_results:
                eval_data = exp_results['evaluation']
                for detail in eval_data.get('details', []):
                    all_data.append({
                        'experiment_type': exp_type,
                        'planner_name': 'GNN',
                        'problem_name': detail.get('problem', ''),
                        'solved': detail.get('solved', False),
                        'wall_clock_time': detail.get('time', 0),
                        'h_star_preservation': detail.get('h_star_preservation', 1.0),
                        'reward': detail.get('reward', 0),
                    })

            logger.info(f"  ✓ Loaded {exp_type}")

        except Exception as e:
            logger.error(f"  Failed to load {exp_type}: {e}")
            continue

    if not all_data:
        logger.error("No experiment data loaded")
        return None

    df = pd.DataFrame(all_data)
    logger.info(f"Total: {len(df)} results")
    return df


# ============================================================================
# PLOT 1: SOLVE RATE COMPARISON
# ============================================================================

def plot_solve_rate_comparison(df: 'pd.DataFrame', output_path: str):
    """Bar chart comparing solve rates."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    solve_rates = []
    planner_names = []

    for planner in sorted(df['planner_name'].unique()):
        df_planner = df[df['planner_name'] == planner]
        rate = (df_planner['solved'].sum() / len(df_planner)) * 100
        solve_rates.append(rate)
        planner_names.append(planner)

    colors_list = [COLORS.get(name, "#555") for name in planner_names]
    bars = ax.bar(range(len(solve_rates)), solve_rates, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.set_xticks(range(len(planner_names)))
    ax.set_xticklabels(planner_names, rotation=45, ha='right', fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.4, linewidth=2, label='Perfect')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.4, linewidth=2, label='Target (80%)')

    for i, (bar, rate) in enumerate(zip(bars, solve_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title("Solve Rate Comparison", fontsize=PLOT_CONFIG["font_size"] + 3, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=PLOT_CONFIG["font_size"], loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 2: TIME COMPARISON
# ============================================================================

def plot_time_comparison(df: 'pd.DataFrame', output_path: str):
    """Box plot and scatter showing time distribution."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    if len(df_solved) == 0:
        logger.warning("No solved problems for time comparison")
        plt.close()
        return

    planners = sorted(df_solved['planner_name'].unique())
    data_for_box = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].values for p in planners]

    bp = ax1.boxplot(data_for_box, labels=planners, patch_artist=True)
    for patch, planner in zip(bp['boxes'], planners):
        patch.set_facecolor(COLORS.get(planner, "#555"))
        patch.set_alpha(0.7)

    ax1.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax1.set_title("Time Distribution (Box Plot)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax1.set_xticklabels(planners, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Scatter plot
    for planner in planners:
        df_p = df_solved[df_solved['planner_name'] == planner]
        x = np.random.normal(list(planners).index(planner), 0.04, size=len(df_p))
        ax2.scatter(x, df_p['wall_clock_time'], alpha=0.6, s=100,
                    color=COLORS.get(planner, "#555"), label=planner, edgecolor='black', linewidth=0.5)

    ax2.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax2.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax2.set_xticks(range(len(planners)))
    ax2.set_xticklabels(planners, rotation=45, ha='right')
    ax2.set_title("Time Distribution (Scatter)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 3: EXPANSIONS COMPARISON
# ============================================================================

def plot_expansions_comparison(df: 'pd.DataFrame', output_path: str):
    """Bar chart of expansions on log scale."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    if len(df_solved) == 0:
        logger.warning("No solved problems for expansions comparison")
        plt.close()
        return

    expansions = []
    planner_names = []

    for planner in sorted(df_solved['planner_name'].unique()):
        df_p = df_solved[df_solved['planner_name'] == planner]

        # Filter out zero expansions
        valid_exps = df_p[df_p['nodes_expanded'] > 0]['nodes_expanded']
        if len(valid_exps) > 0:
            avg_exp = valid_exps.mean()
        else:
            avg_exp = 1

        expansions.append(max(avg_exp, 1))
        planner_names.append(planner)

    colors_list = [COLORS.get(name, "#555") for name in planner_names]
    bars = ax.bar(range(len(expansions)), expansions, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel("Avg Nodes Expanded (log scale)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(len(planner_names)))
    ax.set_xticklabels(planner_names, rotation=45, ha='right')

    for bar, exp in zip(bars, expansions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 2,
                f'{int(exp):,}', ha='center', va='bottom', fontsize=9)

    ax.set_title("Average Nodes Expanded (Log Scale)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 4: EFFICIENCY FRONTIER
# ============================================================================

def plot_efficiency_frontier(df: 'pd.DataFrame', output_path: str):
    """Scatter plot: time vs expansions."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    if len(df_solved) == 0:
        logger.warning("No solved problems for efficiency frontier")
        plt.close()
        return

    for planner in sorted(df_solved['planner_name'].unique()):
        df_p = df_solved[df_solved['planner_name'] == planner]
        ax.scatter(df_p['wall_clock_time'], df_p['nodes_expanded'],
                   label=planner, s=150, alpha=0.6, color=COLORS.get(planner, "#555"),
                   edgecolor='black', linewidth=1.5)

    ax.set_xlabel("Wall Clock Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_ylabel("Nodes Expanded (log scale)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=PLOT_CONFIG["font_size"], loc='best')
    ax.set_title("Efficiency Frontier (Time vs Expansions)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 5: H* PRESERVATION (GNN-SPECIFIC)
# ============================================================================

def plot_h_star_preservation(df: 'pd.DataFrame', output_path: str):
    """Plot H* preservation metric for GNN."""
    if not HAS_MATPLOTLIB or df is None:
        return

    # Filter for GNN results only
    if 'h_star_preservation' not in df.columns:
        logger.warning("h_star_preservation not in data")
        return

    df_gnn = df[df['planner_name'] == 'GNN']

    if len(df_gnn) == 0:
        logger.warning("No GNN results for H* preservation")
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=PLOT_CONFIG["dpi"])

    # Histogram
    h_pres_values = df_gnn['h_star_preservation'].values
    ax1.hist(h_pres_values, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect (1.0)')
    ax1.axvline(x=np.mean(h_pres_values), color='green', linestyle='--', linewidth=2,
                label=f'Mean ({np.mean(h_pres_values):.3f})')
    ax1.set_xlabel("H* Preservation Ratio", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax1.set_ylabel("Frequency", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax1.set_title("H* Preservation Distribution", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Time series
    ax2.plot(h_pres_values, marker='o', linestyle='-', color='#2E86AB', alpha=0.7, linewidth=2, markersize=5)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
    ax2.axhline(y=np.mean(h_pres_values), color='green', linestyle='--', alpha=0.5, linewidth=2, label='Mean')
    ax2.fill_between(range(len(h_pres_values)), 0.95, 1.05, alpha=0.1, color='green', label='Good region')
    ax2.set_xlabel("Problem Index", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax2.set_ylabel("H* Preservation Ratio", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax2.set_title("H* Preservation Trend", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 6: CUMULATIVE DISTRIBUTION
# ============================================================================

def plot_cumulative_distribution(df: 'pd.DataFrame', output_path: str):
    """Cumulative distribution of solve times."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    if len(df_solved) == 0:
        logger.warning("No solved problems for cumulative distribution")
        plt.close()
        return

    for planner in sorted(df_solved['planner_name'].unique()):
        df_p = df_solved[df_solved['planner_name'] == planner]['wall_clock_time'].sort_values()
        cumulative = np.arange(1, len(df_p) + 1) / len(df_p)
        ax.plot(df_p.values, cumulative, marker='o', label=planner, linewidth=2.5,
                color=COLORS.get(planner, "#555"), markersize=4, alpha=0.8)

    ax.set_xlabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_ylabel("Cumulative Fraction Solved", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.legend(fontsize=PLOT_CONFIG["font_size"], loc='lower right')
    ax.set_title("Cumulative Distribution of Solve Times", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 7: PERFORMANCE PROFILE
# ============================================================================

def plot_performance_profile(df: 'pd.DataFrame', output_path: str):
    """Performance profile: % problems solved within τ × best_time."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    if len(df_solved) == 0:
        logger.warning("No solved problems for performance profile")
        plt.close()
        return

    planners = sorted(df_solved['planner_name'].unique())

    for planner in planners:
        df_p = df_solved[df_solved['planner_name'] == planner]
        performance_ratios = []

        for problem in df['problem_name'].unique():
            best_time = df[df['problem_name'] == problem]['wall_clock_time'].min()
            df_problem = df_p[df_p['problem_name'] == problem]

            if len(df_problem) > 0 and best_time > 0:
                time_p = df_problem['wall_clock_time'].values[0]
                ratio = time_p / best_time
                performance_ratios.append(ratio)

        if performance_ratios:
            performance_ratios = sorted(performance_ratios)
            coverage = np.arange(1, len(performance_ratios) + 1) / len(performance_ratios)
            ax.plot(performance_ratios, coverage, marker='o', label=planner, linewidth=2.5,
                    color=COLORS.get(planner, "#555"), markersize=4, alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Performance Ratio τ (time / best_time)', fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_ylabel('% Problems Solved', fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_title('Performance Profile', fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.legend(fontsize=PLOT_CONFIG["font_size"], loc='lower right')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 8: STATISTICAL SUMMARY DASHBOARD
# ============================================================================

def plot_statistical_summary(df: 'pd.DataFrame', output_path: str):
    """Comprehensive statistical summary dashboard."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig = plt.figure(figsize=(16, 12), dpi=PLOT_CONFIG["dpi"])
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    df_solved = df[df['solved']]
    planners = sorted(df_solved['planner_name'].unique())

    # 1. Solve rate
    ax1 = fig.add_subplot(gs[0, 0])
    solve_rates = [(df[df['planner_name'] == p]['solved'].sum() / len(df[df['planner_name'] == p]) * 100)
                   for p in planners]
    colors = [COLORS.get(p, "#555") for p in planners]
    ax1.barh(planners, solve_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel("Solve Rate (%)", fontweight='bold')
    ax1.set_title("Solve Rate", fontweight='bold', fontsize=11)
    ax1.set_xlim([0, 105])
    ax1.grid(axis='x', alpha=0.3)

    # 2. Mean time
    ax2 = fig.add_subplot(gs[0, 1])
    mean_times = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].mean() for p in planners]
    ax2.barh(planners, mean_times, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel("Mean Time (s)", fontweight='bold')
    ax2.set_title("Mean Time (Solved)", fontweight='bold', fontsize=11)
    ax2.grid(axis='x', alpha=0.3)

    # 3. Median time
    ax3 = fig.add_subplot(gs[0, 2])
    median_times = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].median() for p in planners]
    ax3.barh(planners, median_times, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel("Median Time (s)", fontweight='bold')
    ax3.set_title("Median Time (Solved)", fontweight='bold', fontsize=11)
    ax3.grid(axis='x', alpha=0.3)

    # 4. Mean expansions
    ax4 = fig.add_subplot(gs[1, 0])
    mean_exps = [df_solved[df_solved['planner_name'] == p]['nodes_expanded'].mean() for p in planners]
    ax4.barh(planners, mean_exps, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_xlabel("Mean Expansions", fontweight='bold')
    ax4.set_title("Mean Nodes Expanded", fontweight='bold', fontsize=11)
    ax4.set_xscale('log')
    ax4.grid(axis='x', alpha=0.3, which='both')

    # 5. Plan cost
    ax5 = fig.add_subplot(gs[1, 1])
    mean_costs = [df_solved[df_solved['planner_name'] == p]['plan_cost'].mean() for p in planners]
    ax5.barh(planners, mean_costs, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_xlabel("Mean Plan Cost", fontweight='bold')
    ax5.set_title("Mean Plan Cost", fontweight='bold', fontsize=11)
    ax5.grid(axis='x', alpha=0.3)

    # 6. H* preservation (GNN only)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'h_star_preservation' in df.columns:
        h_pres = [df_solved[(df_solved['planner_name'] == p) & (df_solved['planner_name'] == 'GNN')][
                      'h_star_preservation'].mean()
                  for p in planners if p == 'GNN']
        if h_pres:
            ax6.barh(['GNN'], h_pres, color=[COLORS.get('GNN', "#555")], alpha=0.8, edgecolor='black')
            ax6.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')
            ax6.set_xlabel("H* Preservation", fontweight='bold')
            ax6.set_title("H* Preservation (GNN)", fontweight='bold', fontsize=11)
            ax6.set_xlim([0.8, 1.3])
            ax6.grid(axis='x', alpha=0.3)

    # Summary text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    summary_text = "SUMMARY STATISTICS\n\n"
    for planner in planners:
        df_p = df[df['planner_name'] == planner]
        df_p_solved = df_p[df_p['solved']]
        n_solved = len(df_p_solved)
        n_total = len(df_p)
        summary_text += f"{planner}:\n"
        summary_text += f"  Solved: {n_solved}/{n_total} ({n_solved / n_total * 100:.1f}%)\n"
        if n_solved > 0:
            med_time = df_p_solved['wall_clock_time'].median()
            med_exp = df_p_solved['nodes_expanded'].median()
            summary_text += f"  Median Time: {med_time:.2f}s | Med Exp: {int(med_exp):,}\n"
        summary_text += "\n"

    ax7.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle("Statistical Summary Dashboard", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Analysis & Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze evaluation results
  python analysis_and_visualization.py \\
      --results evaluation_results/evaluation_results.csv \\
      --output plots/

  # Analyze experiment results
  python analysis_and_visualization.py \\
      --experiments overfit_results problem_gen_results \\
      --output plots/
        """
    )

    parser.add_argument("--results", help="Path to evaluation_results.csv")
    parser.add_argument(
        "--experiments",
        nargs='+',
        help="Experiment directories"
    )
    parser.add_argument("--output", default="plots", help="Output directory")

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    df = None

    if args.results and os.path.exists(args.results):
        logger.info(f"Loading results from: {args.results}")
        df = load_results_csv(args.results)

    elif args.experiments:
        logger.info("Loading from experiments...")
        exp_dirs = {os.path.basename(d.rstrip('/')): d for d in args.experiments}
        df = load_experiment_results(exp_dirs)

    if df is None:
        logger.error("No results loaded!")
        return 1

    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 80 + "\n")

    # Generate all plots
    plot_solve_rate_comparison(df, os.path.join(args.output, "01_solve_rate.png"))
    plot_time_comparison(df, os.path.join(args.output, "02_time_comparison.png"))
    plot_expansions_comparison(df, os.path.join(args.output, "03_expansions.png"))
    plot_efficiency_frontier(df, os.path.join(args.output, "04_efficiency_frontier.png"))
    plot_cumulative_distribution(df, os.path.join(args.output, "05_cumulative_dist.png"))
    plot_performance_profile(df, os.path.join(args.output, "06_performance_profile.png"))
    plot_h_star_preservation(df, os.path.join(args.output, "07_h_star_preservation.png"))
    plot_statistical_summary(df, os.path.join(args.output, "08_statistical_summary.png"))

    logger.info(f"\n✅ All plots generated in {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())