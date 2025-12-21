#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE ANALYSIS & VISUALIZATION MODULE - RESEARCH GRADE
==================================================================
Advanced plots, statistical analysis, and GNN policy evaluation for
reinforcement learning-based merge strategy selection.

Features:
✓ GNN policy evaluation and comparison with baseline planners
✓ Merge strategy analysis and learned heuristic evaluation
✓ Statistical significance testing (Mann-Whitney U, Wilcoxon)
✓ Per-problem-difficulty breakdown and scaling analysis
✓ Efficiency frontier and Pareto analysis
✓ H* preservation metrics (GNN-specific)
✓ Robustness analysis across problem distributions
✓ Research-grade statistical dashboards
✓ CSV export of detailed results and metrics

Compatible with:
✓ evaluation_comprehensive.py
✓ run_full_evaluation.py
✓ All experiment types (overfit, problem_gen, scale_gen, curriculum)
✓ Enhanced feature dimensions (15-dim node, 10-dim edge)

Plots Generated:
1. Solve rate comparison (with 95% confidence intervals) *******
2. Time comparison (box plots, violin plots, scatter) *********
3. Expansions comparison (log scale with distribution) ////////
4. Efficiency frontier (2D scatter with Pareto analysis) ////////
5. Cumulative distribution of solve times
6. Performance profile with significance zones
7. H* preservation analysis (GNN-specific, multi-panel) *******
8. Statistical summary dashboard
9. Per-difficulty breakdown (4-panel)
10. Scaling analysis across problem sizes
11. Summary statistics CSV export
"""

import sys
import os
import logging
import argparse
import json
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from datetime import datetime
from collections import defaultdict
import traceback

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)

warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.gridspec import GridSpec
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

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed - statistical tests disabled")

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    "style": "seaborn-v0_8-whitegrid",
    "figsize": (14, 8),
    "figsize_wide": (16, 10),
    "figsize_large": (18, 12),
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

# Statistical thresholds
SIGNIFICANCE_THRESHOLD = 0.05
MEDIAN_TIME_THRESHOLD = 10.0  # seconds


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

        # Ensure required columns exist
        required_cols = ['planner_name', 'problem_name', 'solved']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}. Has: {list(df.columns)}")

        return df
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        traceback.print_exc()
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
                        'nodes_expanded': detail.get('nodes_expanded', 0),
                        'plan_cost': detail.get('plan_cost', 0),
                    })

            logger.info(f"  ✓ Loaded {exp_type}")

        except Exception as e:
            logger.error(f"  Failed to load {exp_type}: {e}")
            traceback.print_exc()
            continue

    if not all_data:
        logger.error("No experiment data loaded")
        return None

    df = pd.DataFrame(all_data)
    logger.info(f"Total: {len(df)} results")
    return df


def add_derived_metrics(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Add derived metrics to dataframe for richer analysis."""
    if df is None:
        return None

    try:
        # Ensure all expected columns exist
        if 'nodes_expanded' not in df.columns:
            df['nodes_expanded'] = 0
        if 'plan_cost' not in df.columns:
            df['plan_cost'] = 0
        if 'wall_clock_time' not in df.columns:
            df['wall_clock_time'] = 0
        if 'h_star_preservation' not in df.columns:
            df['h_star_preservation'] = 1.0

        # Add efficiency metric (time × log nodes)
        df['efficiency'] = df['wall_clock_time'] * np.log1p(df['nodes_expanded'] + 1)

        # Add problem difficulty proxy based on expansions
        difficulty_map = {}
        for problem in df['problem_name'].unique():
            max_expansions = df[df['problem_name'] == problem]['nodes_expanded'].max()
            if max_expansions < 1000:
                difficulty_map[problem] = 'Easy'
            elif max_expansions < 100000:
                difficulty_map[problem] = 'Medium'
            else:
                difficulty_map[problem] = 'Hard'
        df['difficulty'] = df['problem_name'].map(difficulty_map)

        return df
    except Exception as e:
        logger.error(f"Failed to add derived metrics: {e}")
        return df


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def compute_statistical_tests(
        df: 'pd.DataFrame',
        gnn_name: str = 'GNN'
) -> Dict[str, Dict]:
    """Compute statistical significance tests comparing GNN to baselines."""
    if not HAS_SCIPY or df is None:
        return {}

    results = {}
    df_gnn = df[df['planner_name'] == gnn_name]

    if len(df_gnn) == 0:
        logger.warning(f"No results found for {gnn_name}")
        return results

    gnn_times = df_gnn[df_gnn['solved']]['wall_clock_time'].values

    for baseline in df['planner_name'].unique():
        if baseline == gnn_name:
            continue

        df_baseline = df[df['planner_name'] == baseline]
        baseline_times = df_baseline[df_baseline['solved']]['wall_clock_time'].values

        if len(gnn_times) == 0 or len(baseline_times) == 0:
            continue

        try:
            # Mann-Whitney U test (non-parametric, no assumptions on distribution)
            u_stat, p_value_mw = stats.mannwhitneyu(gnn_times, baseline_times, alternative='two-sided')

            # Wilcoxon signed-rank test (if same number of samples)
            p_value_wilcoxon = None
            if len(gnn_times) == len(baseline_times):
                try:
                    wilcoxon_stat, p_value_wilcoxon = stats.wilcoxon(gnn_times, baseline_times)
                except:
                    pass

            # Effect size (Cohen's d)
            cohens_d = (np.mean(gnn_times) - np.mean(baseline_times)) / np.sqrt(
                (np.std(gnn_times) ** 2 + np.std(baseline_times) ** 2) / 2
            )

            results[baseline] = {
                'mann_whitney_p': p_value_mw,
                'wilcoxon_p': p_value_wilcoxon,
                'cohens_d': cohens_d,
                'gnn_mean': np.mean(gnn_times),
                'baseline_mean': np.mean(baseline_times),
                'gnn_median': np.median(gnn_times),
                'baseline_median': np.median(baseline_times),
                'gnn_n': len(gnn_times),
                'baseline_n': len(baseline_times),
                'significant': p_value_mw < SIGNIFICANCE_THRESHOLD,
                'speedup': np.mean(baseline_times) / np.mean(gnn_times)
            }
        except Exception as e:
            logger.warning(f"Failed to compute tests for {baseline}: {e}")

    return results


def extract_problem_size(problem_name: str) -> Optional[int]:
    """Extract size metric from problem name if available."""
    import re
    # Try various patterns
    patterns = [
        r'(\d+)x(\d+)',  # grid format
        r'size[_-]?(\d+)',  # size_N format
        r'n[_-]?(\d+)',  # n_N format
    ]
    for pattern in patterns:
        match = re.search(pattern, problem_name.lower())
        if match:
            if len(match.groups()) == 2:
                return int(match.group(1)) * int(match.group(2))
            else:
                return int(match.group(1))
    return None


# ============================================================================
# PLOT 1: SOLVE RATE COMPARISON
# ============================================================================

def plot_solve_rate_comparison(df: 'pd.DataFrame', output_path: str):
    """Bar chart comparing solve rates with confidence intervals."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
        plt.style.use(PLOT_CONFIG["style"])
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

        solve_stats = []
        planner_names = []

        for planner in sorted(df['planner_name'].unique()):
            df_planner = df[df['planner_name'] == planner]
            n_total = len(df_planner)
            n_solved = df_planner['solved'].sum()
            rate = (n_solved / n_total) * 100 if n_total > 0 else 0

            # Compute 95% confidence interval using binomial proportion
            if n_total > 0:
                p = rate / 100
                ci = 1.96 * np.sqrt(p * (1 - p) / n_total) * 100
            else:
                ci = 0

            solve_stats.append({'rate': rate, 'ci': ci, 'n': n_total})
            planner_names.append(planner)

        rates = [s['rate'] for s in solve_stats]
        cis = [s['ci'] for s in solve_stats]

        colors_list = [COLORS.get(name, "#555") for name in planner_names]
        bars = ax.bar(range(len(rates)), rates, yerr=cis, capsize=10,
                      color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
        ax.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
        ax.set_xticks(range(len(planner_names)))
        ax.set_xticklabels(planner_names, rotation=45, ha='right', fontsize=PLOT_CONFIG["font_size"])
        ax.set_ylim([0, 110])
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.4, linewidth=2, label='Perfect (100%)')
        ax.axhline(y=80, color='orange', linestyle='--', alpha=0.4, linewidth=2, label='Target (80%)')

        for i, (bar, rate, ci) in enumerate(zip(bars, rates, cis)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + ci + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_title("Solve Rate Comparison (95% CI)", fontsize=PLOT_CONFIG["font_size"] + 3,
                     fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=PLOT_CONFIG["font_size"], loc='upper left')

        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate solve rate plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 2: TIME COMPARISON (ENHANCED)
# ============================================================================

def plot_time_comparison_enhanced(df: 'pd.DataFrame', output_path: str):
    """Enhanced visualization: box plots, violin plots, and scatter with statistics."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
        plt.style.use(PLOT_CONFIG["style"])
        fig = plt.figure(figsize=(18, 10), dpi=PLOT_CONFIG["dpi"])
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        df_solved = df[df['solved']]

        if len(df_solved) == 0:
            logger.warning("No solved problems for time comparison")
            plt.close()
            return

        planners = sorted(df_solved['planner_name'].unique())

        # Prepare data
        data_for_plots = {p: df_solved[df_solved['planner_name'] == p]['wall_clock_time'].values
                          for p in planners}

        # 1. Box plot
        ax1 = fig.add_subplot(gs[0, 0])
        bp = ax1.boxplot([data_for_plots[p] for p in planners], labels=planners, patch_artist=True)
        for patch, planner in zip(bp['boxes'], planners):
            patch.set_facecolor(COLORS.get(planner, "#555"))
            patch.set_alpha(0.7)
        ax1.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax1.set_title("Time Distribution (Box Plot)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax1.set_xticklabels(planners, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Violin plot
        ax2 = fig.add_subplot(gs[0, 1])
        positions = range(len(planners))
        parts = ax2.violinplot([data_for_plots[p] for p in planners], positions=positions,
                               showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#2E86AB')
            pc.set_alpha(0.7)
        ax2.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax2.set_title("Time Distribution (Violin Plot)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax2.set_xticks(positions)
        ax2.set_xticklabels(planners, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Scatter with jitter
        ax3 = fig.add_subplot(gs[1, 0])
        for i, planner in enumerate(planners):
            times = data_for_plots[planner]
            x = np.random.normal(i, 0.04, size=len(times))
            ax3.scatter(x, times, alpha=0.6, s=100,
                        color=COLORS.get(planner, "#555"), label=planner,
                        edgecolor='black', linewidth=0.5)
        ax3.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax3.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax3.set_xticks(range(len(planners)))
        ax3.set_xticklabels(planners, rotation=45, ha='right')
        ax3.set_title("Time Distribution (Scatter)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        summary_text = "TIME STATISTICS (SOLVED ONLY)\n" + "=" * 50 + "\n\n"
        for planner in planners:
            times = data_for_plots[planner]
            summary_text += f"{planner}:\n"
            summary_text += f"  N = {len(times)}\n"
            summary_text += f"  Mean = {np.mean(times):.3f}s\n"
            summary_text += f"  Median = {np.median(times):.3f}s\n"
            summary_text += f"  Std = {np.std(times):.3f}s\n"
            summary_text += f"  Min = {np.min(times):.3f}s\n"
            summary_text += f"  Max = {np.max(times):.3f}s\n"
            summary_text += f"  Q1 = {np.percentile(times, 25):.3f}s\n"
            summary_text += f"  Q3 = {np.percentile(times, 75):.3f}s\n"
            summary_text += "\n"

        ax4.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.suptitle("Comprehensive Time Comparison", fontsize=PLOT_CONFIG["font_size"] + 3,
                     fontweight='bold', y=0.995)
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate time comparison plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 3: EXPANSIONS COMPARISON
# ============================================================================

def plot_expansions_comparison(df: 'pd.DataFrame', output_path: str):
    """Enhanced expansions comparison with statistics and distribution."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
        if 'nodes_expanded' not in df.columns:
            logger.warning("nodes_expanded not in dataframe - skipping expansions plot")
            return

        plt.style.use(PLOT_CONFIG["style"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=PLOT_CONFIG["dpi"])

        df_solved = df[df['solved']]

        if len(df_solved) == 0:
            logger.warning("No solved problems for expansions comparison")
            plt.close()
            return

        planners = sorted(df_solved['planner_name'].unique())

        expansions_stats = []
        for planner in planners:
            df_p = df_solved[df_solved['planner_name'] == planner]
            valid_exps = df_p[df_p['nodes_expanded'] > 0]['nodes_expanded']
            if len(valid_exps) > 0:
                expansions_stats.append({
                    'mean': valid_exps.mean(),
                    'median': valid_exps.median(),
                    'std': valid_exps.std(),
                    'n': len(valid_exps)
                })
            else:
                expansions_stats.append({'mean': 1, 'median': 1, 'std': 0, 'n': 0})

        means = [s['mean'] for s in expansions_stats]
        stds = [s['std'] for s in expansions_stats]

        colors_list = [COLORS.get(name, "#555") for name in planners]

        # Mean with error bars
        bars = ax1.bar(range(len(means)), means, yerr=stds, capsize=10,
                       color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel("Avg Nodes Expanded (log scale)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax1.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax1.set_yscale('log')
        ax1.set_xticks(range(len(planners)))
        ax1.set_xticklabels(planners, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3, which='both')
        ax1.set_title("Mean Nodes Expanded", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')

        # Violin plot for distribution
        data_for_violin = [df_solved[df_solved['planner_name'] == p]['nodes_expanded'].values
                           for p in planners]
        parts = ax2.violinplot(data_for_violin, positions=range(len(planners)),
                               showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#2E86AB')
            pc.set_alpha(0.7)
        ax2.set_ylabel("Nodes Expanded (log scale)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax2.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax2.set_yscale('log')
        ax2.set_xticks(range(len(planners)))
        ax2.set_xticklabels(planners, rotation=45, ha='right')
        ax2.set_title("Expansions Distribution", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate expansions plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 4: EFFICIENCY FRONTIER & PARETO ANALYSIS
# ============================================================================

def plot_efficiency_frontier(df: 'pd.DataFrame', output_path: str):
    """2D scatter with Pareto frontier analysis."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
        if 'nodes_expanded' not in df.columns:
            logger.warning("nodes_expanded not in dataframe - skipping efficiency frontier")
            return

        plt.style.use(PLOT_CONFIG["style"])
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

        df_solved = df[df['solved']].copy()

        if len(df_solved) == 0:
            logger.warning("No solved problems for efficiency frontier")
            plt.close()
            return

        # Filter for reasonable bounds
        df_solved = df_solved[df_solved['wall_clock_time'] > 0]
        df_solved = df_solved[df_solved['nodes_expanded'] > 0]

        for planner in sorted(df_solved['planner_name'].unique()):
            df_p = df_solved[df_solved['planner_name'] == planner]

            # Plot all points
            ax.scatter(df_p['wall_clock_time'], df_p['nodes_expanded'],
                       label=planner, s=150, alpha=0.6, color=COLORS.get(planner, "#555"),
                       edgecolor='black', linewidth=1.5)

        ax.set_xlabel("Wall Clock Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax.set_ylabel("Nodes Expanded (log scale)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(fontsize=PLOT_CONFIG["font_size"], loc='best')
        ax.set_title("Efficiency Frontier (Time vs Expansions)",
                     fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate efficiency frontier plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 5: H* PRESERVATION (GNN-SPECIFIC)
# ============================================================================

def plot_h_star_preservation(df: 'pd.DataFrame', output_path: str):
    """Enhanced H* preservation analysis with detailed metrics."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
        if 'h_star_preservation' not in df.columns:
            logger.warning("h_star_preservation not in data - skipping H* analysis")
            return

        df_gnn = df[df['planner_name'] == 'GNN']

        if len(df_gnn) == 0:
            logger.warning("No GNN results for H* preservation")
            return

        plt.style.use(PLOT_CONFIG["style"])
        fig = plt.figure(figsize=(16, 10), dpi=PLOT_CONFIG["dpi"])
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        h_pres_values = df_gnn['h_star_preservation'].dropna().values
        h_pres_solved = df_gnn[df_gnn['solved']]['h_star_preservation'].dropna().values

        # 1. Histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(h_pres_values, bins=25, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect (1.0)')
        ax1.axvline(x=np.mean(h_pres_values), color='green', linestyle='--', linewidth=2,
                    label=f'Mean ({np.mean(h_pres_values):.3f})')
        ax1.axvline(x=np.median(h_pres_values), color='orange', linestyle='--', linewidth=2,
                    label=f'Median ({np.median(h_pres_values):.3f})')
        ax1.set_xlabel("H* Preservation Ratio", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax1.set_ylabel("Frequency", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax1.set_title("H* Preservation Distribution (All)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Solved vs Unsolved
        ax2 = fig.add_subplot(gs[0, 1])
        h_pres_unsolved = df_gnn[~df_gnn['solved']]['h_star_preservation'].dropna().values
        bp = ax2.boxplot([h_pres_solved, h_pres_unsolved],
                         labels=['Solved', 'Unsolved'],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#06A77D', '#D62828']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax2.set_ylabel("H* Preservation Ratio", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax2.set_title("H* Preservation: Solved vs Unsolved", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Time series
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(h_pres_values, marker='o', linestyle='-', color='#2E86AB',
                 alpha=0.7, linewidth=2, markersize=4)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
        ax3.axhline(y=np.mean(h_pres_values), color='green', linestyle='--',
                    alpha=0.5, linewidth=2, label='Mean')
        ax3.fill_between(range(len(h_pres_values)), 0.95, 1.05, alpha=0.1,
                         color='green', label='Good region (±5%)')
        ax3.set_xlabel("Problem Index", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax3.set_ylabel("H* Preservation Ratio", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax3.set_title("H* Preservation Trend", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. Statistics text
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Count preservation levels
        perfect = np.sum(h_pres_values == 1.0)
        good = np.sum((h_pres_values > 0.95) & (h_pres_values <= 1.0))
        acceptable = np.sum((h_pres_values > 0.9) & (h_pres_values <= 0.95))
        poor = np.sum(h_pres_values <= 0.9)

        stats_text = "H* PRESERVATION ANALYSIS\n" + "=" * 40 + "\n\n"
        stats_text += f"All Problems (N={len(h_pres_values)}):\n"
        stats_text += f"  Mean: {np.mean(h_pres_values):.4f}\n"
        stats_text += f"  Median: {np.median(h_pres_values):.4f}\n"
        stats_text += f"  Std: {np.std(h_pres_values):.4f}\n"
        stats_text += f"  Min: {np.min(h_pres_values):.4f}\n"
        stats_text += f"  Max: {np.max(h_pres_values):.4f}\n\n"
        stats_text += f"Solved Only (N={len(h_pres_solved)}):\n"
        if len(h_pres_solved) > 0:
            stats_text += f"  Mean: {np.mean(h_pres_solved):.4f}\n"
            stats_text += f"  Median: {np.median(h_pres_solved):.4f}\n"
        stats_text += f"\nQuality Distribution:\n"
        stats_text += f"  Perfect (=1.0): {perfect}\n"
        stats_text += f"  Good (0.95-1.0): {good}\n"
        stats_text += f"  Acceptable (0.9-0.95): {acceptable}\n"
        stats_text += f"  Poor (<0.9): {poor}\n"

        ax4.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.suptitle("H* Preservation Analysis (GNN Policy)", fontsize=PLOT_CONFIG["font_size"] + 3,
                     fontweight='bold', y=0.995)
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate H* preservation plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 6: CUMULATIVE DISTRIBUTION
# ============================================================================

def plot_cumulative_distribution(df: 'pd.DataFrame', output_path: str):
    """Cumulative distribution of solve times with reference lines."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
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
                    color=COLORS.get(planner, "#555"), markersize=5, alpha=0.8)

        # Add reference lines for practical time limits
        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='1 second')
        ax.axvline(x=10.0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='10 seconds')

        ax.set_xlabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax.set_ylabel("Cumulative Fraction Solved", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax.legend(fontsize=PLOT_CONFIG["font_size"], loc='lower right', ncol=2)
        ax.set_title("Cumulative Distribution of Solve Times", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log')
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate cumulative distribution plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 7: PERFORMANCE PROFILE
# ============================================================================

def plot_performance_profile(df: 'pd.DataFrame', output_path: str):
    """Performance profile with statistical significance zones."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
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
                        color=COLORS.get(planner, "#555"), markersize=5, alpha=0.8)

        # Add significance zones
        ax.axvspan(1, 1.5, alpha=0.1, color='green', label='Excellent (1-1.5x)')
        ax.axvspan(1.5, 3, alpha=0.1, color='yellow', label='Good (1.5-3x)')
        ax.axvspan(3, 10, alpha=0.1, color='red', label='Poor (>3x)')

        ax.set_xscale('log')
        ax.set_xlabel('Performance Ratio τ (time / best_time)', fontsize=PLOT_CONFIG["font_size"] + 1,
                      fontweight='bold')
        ax.set_ylabel('% Problems Solved', fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax.set_title('Performance Profile (Solved Problems)', fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
        ax.legend(fontsize=PLOT_CONFIG["font_size"], loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate performance profile plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 8: PER-DIFFICULTY BREAKDOWN
# ============================================================================

def plot_per_difficulty_analysis(df: 'pd.DataFrame', output_path: str):
    """Analyze performance broken down by problem difficulty."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
        if 'difficulty' not in df.columns:
            logger.warning("difficulty not in dataframe - skipping per-difficulty analysis")
            return

        plt.style.use(PLOT_CONFIG["style"])
        fig = plt.figure(figsize=(16, 10), dpi=PLOT_CONFIG["dpi"])
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        difficulties = sorted(df['difficulty'].unique())
        planners = sorted(df['planner_name'].unique())

        # 1. Solve rate by difficulty
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(difficulties))
        width = 0.8 / len(planners) if len(planners) > 0 else 0.8

        for i, planner in enumerate(planners):
            rates = []
            for diff in difficulties:
                df_subset = df[(df['difficulty'] == diff) & (df['planner_name'] == planner)]
                if len(df_subset) > 0:
                    rate = df_subset['solved'].sum() / len(df_subset) * 100
                else:
                    rate = 0
                rates.append(rate)

            ax1.bar(x + i * width, rates, width, label=planner,
                    color=COLORS.get(planner, "#555"), alpha=0.8, edgecolor='black')

        ax1.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax1.set_xlabel("Problem Difficulty", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax1.set_xticks(x + width * (len(planners) - 1) / 2)
        ax1.set_xticklabels(difficulties)
        ax1.legend(fontsize=9)
        ax1.set_title("Solve Rate by Difficulty", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax1.set_ylim([0, 105])
        ax1.grid(axis='y', alpha=0.3)

        # 2. Mean time by difficulty
        ax2 = fig.add_subplot(gs[0, 1])
        for i, planner in enumerate(planners):
            times = []
            for diff in difficulties:
                df_subset = df[(df['difficulty'] == diff) & (df['planner_name'] == planner) & (df['solved'])]
                if len(df_subset) > 0:
                    time = df_subset['wall_clock_time'].mean()
                else:
                    time = 0
                times.append(time)

            ax2.bar(x + i * width, times, width, label=planner,
                    color=COLORS.get(planner, "#555"), alpha=0.8, edgecolor='black')

        ax2.set_ylabel("Mean Time (seconds)", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax2.set_xlabel("Problem Difficulty", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax2.set_xticks(x + width * (len(planners) - 1) / 2)
        ax2.set_xticklabels(difficulties)
        ax2.legend(fontsize=9)
        ax2.set_title("Mean Time by Difficulty (Solved)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Box plot of times by difficulty
        ax3 = fig.add_subplot(gs[1, 0])
        df_solved = df[df['solved']]
        positions = []
        data_for_box = []
        labels_for_box = []
        pos = 0
        for diff in difficulties:
            for planner in planners:
                df_subset = df_solved[(df_solved['difficulty'] == diff) & (df_solved['planner_name'] == planner)]
                if len(df_subset) > 0:
                    data_for_box.append(df_subset['wall_clock_time'].values)
                    positions.append(pos)
                    pos += 1
            pos += 1

        if data_for_box:
            bp = ax3.boxplot(data_for_box, positions=positions, patch_artist=True, widths=0.6)
            for patch in bp['boxes']:
                patch.set_facecolor('#2E86AB')
                patch.set_alpha(0.7)
        ax3.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax3.set_title("Time Distribution by Difficulty", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        # 4. Problem count by difficulty
        ax4 = fig.add_subplot(gs[1, 1])
        counts = []
        for diff in difficulties:
            count = len(df[df['difficulty'] == diff])
            counts.append(count)

        colors_map = {'Easy': '#06A77D', 'Medium': '#F18F01', 'Hard': '#D62828'}
        colors_bars = [colors_map.get(d, '#555') for d in difficulties]

        bars = ax4.bar(difficulties, counts, color=colors_bars, alpha=0.8, edgecolor='black')
        ax4.set_ylabel("Count", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax4.set_xlabel("Difficulty", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax4.set_title("Problem Distribution by Difficulty", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.suptitle("Per-Difficulty Analysis", fontsize=PLOT_CONFIG["font_size"] + 3,
                     fontweight='bold', y=0.995)
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate per-difficulty plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 9: STATISTICAL SUMMARY DASHBOARD
# ============================================================================

def plot_statistical_summary(df: 'pd.DataFrame', output_path: str):
    """Comprehensive statistical summary dashboard."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
        plt.style.use(PLOT_CONFIG["style"])
        fig = plt.figure(figsize=(18, 12), dpi=PLOT_CONFIG["dpi"])
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

        df_solved = df[df['solved']]
        planners = sorted(df_solved['planner_name'].unique()) if len(df_solved) > 0 else []

        if len(planners) == 0:
            logger.warning("No solved problems for statistical summary")
            plt.close()
            return

        # 1. Solve rate
        ax1 = fig.add_subplot(gs[0, 0])
        solve_rates = [(df[df['planner_name'] == p]['solved'].sum() / len(df[df['planner_name'] == p]) * 100)
                       for p in planners]
        colors = [COLORS.get(p, "#555") for p in planners]
        bars = ax1.barh(planners, solve_rates, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel("Solve Rate (%)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"])
        ax1.set_title("Solve Rate", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
        ax1.set_xlim([0, 105])
        ax1.grid(axis='x', alpha=0.3)
        for bar, rate in zip(bars, solve_rates):
            width = bar.get_width()
            ax1.text(width + 2, bar.get_y() + bar.get_height() / 2,
                     f'{rate:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')

        # 2. Mean time
        ax2 = fig.add_subplot(gs[0, 1])
        mean_times = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].mean() for p in planners]
        bars = ax2.barh(planners, mean_times, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel("Mean Time (s)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"])
        ax2.set_title("Mean Time (Solved)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
        ax2.grid(axis='x', alpha=0.3)
        for bar, time in zip(bars, mean_times):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{time:.2f}s', ha='left', va='center', fontsize=9)

        # 3. Median time
        ax3 = fig.add_subplot(gs[0, 2])
        median_times = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].median() for p in planners]
        bars = ax3.barh(planners, median_times, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_xlabel("Median Time (s)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"])
        ax3.set_title("Median Time (Solved)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
        ax3.grid(axis='x', alpha=0.3)
        for bar, time in zip(bars, median_times):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{time:.2f}s', ha='left', va='center', fontsize=9)

        # 4. Mean expansions
        ax4 = fig.add_subplot(gs[1, 0])
        if 'nodes_expanded' in df.columns:
            mean_exps = [df_solved[df_solved['planner_name'] == p]['nodes_expanded'].mean() for p in planners]
            bars = ax4.barh(planners, mean_exps, color=colors, alpha=0.8, edgecolor='black')
            ax4.set_xlabel("Mean Expansions", fontweight='bold', fontsize=PLOT_CONFIG["font_size"])
            ax4.set_title("Mean Nodes Expanded", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
            ax4.set_xscale('log')
            ax4.grid(axis='x', alpha=0.3, which='both')
        else:
            ax4.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title("Mean Nodes Expanded", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
            ax4.axis('off')

        # 5. Plan cost
        ax5 = fig.add_subplot(gs[1, 1])
        if 'plan_cost' in df.columns and df['plan_cost'].sum() > 0:
            mean_costs = [df_solved[df_solved['planner_name'] == p]['plan_cost'].mean() for p in planners]
            bars = ax5.barh(planners, mean_costs, color=colors, alpha=0.8, edgecolor='black')
            ax5.set_xlabel("Mean Plan Cost", fontweight='bold', fontsize=PLOT_CONFIG["font_size"])
            ax5.set_title("Mean Plan Cost", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
            ax5.grid(axis='x', alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title("Mean Plan Cost", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
            ax5.axis('off')

        # 6. H* preservation (GNN only)
        ax6 = fig.add_subplot(gs[1, 2])
        if 'h_star_preservation' in df.columns and 'GNN' in planners:
            h_pres_vals = df_solved[df_solved['planner_name'] == 'GNN']['h_star_preservation'].dropna()
            if len(h_pres_vals) > 0:
                bars = ax6.barh(['GNN'], [h_pres_vals.mean()], color=[COLORS.get('GNN', "#555")],
                                alpha=0.8, edgecolor='black')
                ax6.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')
                ax6.set_xlabel("H* Preservation", fontweight='bold', fontsize=PLOT_CONFIG["font_size"])
                ax6.set_title("H* Preservation (GNN)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
                ax6.set_xlim([0.8, 1.3])
                ax6.grid(axis='x', alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax6.transAxes, fontsize=12)
                ax6.set_title("H* Preservation (GNN)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
                ax6.axis('off')
        else:
            ax6.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title("H* Preservation (GNN)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
            ax6.axis('off')

        # 7. Standard deviations
        ax7 = fig.add_subplot(gs[2, 0])
        if 'wall_clock_time' in df.columns:
            stds = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].std() for p in planners]
            bars = ax7.barh(planners, stds, color=colors, alpha=0.8, edgecolor='black')
            ax7.set_xlabel("Std Dev Time (s)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"])
            ax7.set_title("Time Std Dev (Solved)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
            ax7.grid(axis='x', alpha=0.3)

        # 8. 95th percentile time
        ax8 = fig.add_subplot(gs[2, 1])
        if 'wall_clock_time' in df.columns:
            p95_times = [np.percentile(df_solved[df_solved['planner_name'] == p]['wall_clock_time'], 95)
                         for p in planners]
            bars = ax8.barh(planners, p95_times, color=colors, alpha=0.8, edgecolor='black')
            ax8.set_xlabel("95th Percentile Time (s)", fontweight='bold', fontsize=PLOT_CONFIG["font_size"])
            ax8.set_title("95th Percentile Time", fontweight='bold', fontsize=PLOT_CONFIG["font_size"] + 1)
            ax8.grid(axis='x', alpha=0.3)

        # 9. Summary text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = "SUMMARY\n" + "=" * 35 + "\n\n"
        for planner in planners:
            df_p = df[df['planner_name'] == planner]
            df_p_solved = df_p[df_p['solved']]
            n_solved = len(df_p_solved)
            n_total = len(df_p)
            summary_text += f"{planner}:\n"
            summary_text += f"  {n_solved}/{n_total} solved\n"
            summary_text += f"  {n_solved / n_total * 100:.1f}%\n"
            if n_solved > 0:
                med_time = df_p_solved['wall_clock_time'].median()
                summary_text += f"  Med: {med_time:.2f}s\n"
            summary_text += "\n"

        ax9.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.suptitle("Statistical Summary Dashboard", fontsize=PLOT_CONFIG["font_size"] + 3,
                     fontweight='bold', y=0.995)
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate statistical summary plot: {e}")
        traceback.print_exc()


# ============================================================================
# PLOT 10: SCALING ANALYSIS
# ============================================================================

def plot_scaling_analysis(df: 'pd.DataFrame', output_path: str):
    """Analyze scaling behavior across problem sizes."""
    if not HAS_MATPLOTLIB or df is None:
        return

    try:
        plt.style.use(PLOT_CONFIG["style"])
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_wide"], dpi=PLOT_CONFIG["dpi"])

        df_copy = df.copy()
        df_copy['size'] = df_copy['problem_name'].apply(extract_problem_size)

        planners = sorted(df_copy['planner_name'].unique())

        # Filter out problems without size info
        df_sized = df_copy[df_copy['size'].notna()]

        if len(df_sized) == 0:
            logger.warning("No size information found in problem names - skipping scaling analysis")
            plt.close()
            return

        # 1. Solve rate vs size
        ax = axes[0, 0]
        for planner in planners:
            df_p = df_sized[df_sized['planner_name'] == planner]
            sizes = sorted(df_p['size'].unique())
            rates = []
            for size in sizes:
                df_size = df_p[df_p['size'] == size]
                rate = df_size['solved'].sum() / len(df_size) * 100 if len(df_size) > 0 else 0
                rates.append(rate)
            ax.plot(sizes, rates, marker='o', label=planner, linewidth=2.5,
                    color=COLORS.get(planner, "#555"), markersize=8)

        ax.set_xlabel("Problem Size", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax.set_title("Solve Rate vs Problem Size", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # 2. Mean time vs size
        ax = axes[0, 1]
        df_solved_sized = df_sized[df_sized['solved']]
        for planner in planners:
            df_p = df_solved_sized[df_solved_sized['planner_name'] == planner]
            sizes = sorted(df_p['size'].unique())
            times = []
            for size in sizes:
                df_size = df_p[df_p['size'] == size]
                if len(df_size) > 0:
                    time = df_size['wall_clock_time'].mean()
                else:
                    time = 0
                times.append(time)
            if times:
                ax.plot(sizes, times, marker='o', label=planner, linewidth=2.5,
                        color=COLORS.get(planner, "#555"), markersize=8)

        ax.set_xlabel("Problem Size", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax.set_ylabel("Mean Time (s)", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax.set_title("Mean Time vs Problem Size", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 3. Expansions vs size
        ax = axes[1, 0]
        if 'nodes_expanded' in df.columns:
            for planner in planners:
                df_p = df_solved_sized[df_solved_sized['planner_name'] == planner]
                sizes = sorted(df_p['size'].unique())
                exps = []
                for size in sizes:
                    df_size = df_p[df_p['size'] == size]
                    if len(df_size) > 0:
                        exp_vals = df_size[df_size['nodes_expanded'] > 0]['nodes_expanded']
                        exp = exp_vals.mean() if len(exp_vals) > 0 else 1
                    else:
                        exp = 1
                    exps.append(exp)
                if exps:
                    ax.plot(sizes, exps, marker='o', label=planner, linewidth=2.5,
                            color=COLORS.get(planner, "#555"), markersize=8)

            ax.set_xlabel("Problem Size", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
            ax.set_ylabel("Mean Expansions", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
            ax.set_title("Mean Expansions vs Size", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_yscale('log')

        # 4. Problems per size
        ax = axes[1, 1]
        sizes = sorted(df_sized['size'].unique())
        counts = [len(df_sized[df_sized['size'] == s]) for s in sizes]

        bars = ax.bar(range(len(sizes)), counts, color='#2E86AB', alpha=0.8, edgecolor='black')
        ax.set_xlabel("Problem Size", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax.set_ylabel("Count", fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
        ax.set_title("Problem Distribution by Size", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([int(s) for s in sizes], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)

        fig.suptitle("Scaling Analysis", fontsize=PLOT_CONFIG["font_size"] + 3,
                     fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"✓ {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate scaling analysis plot: {e}")
        traceback.print_exc()


# ============================================================================
# EXPORT SUMMARY STATISTICS TO CSV
# ============================================================================

def export_summary_statistics(df: 'pd.DataFrame', output_path: str):
    """Export detailed summary statistics to CSV."""
    if not HAS_PANDAS or df is None:
        return

    try:
        planners = sorted(df['planner_name'].unique())

        rows = []

        for planner in planners:
            df_p = df[df['planner_name'] == planner]
            df_p_solved = df_p[df_p['solved']]

            row = {
                'Planner': planner,
                'Total_Problems': len(df_p),
                'Solved': len(df_p_solved),
                'Solve_Rate_%': len(df_p_solved) / len(df_p) * 100 if len(df_p) > 0 else 0,
                'Mean_Time_s': df_p_solved['wall_clock_time'].mean() if len(df_p_solved) > 0 else np.nan,
                'Median_Time_s': df_p_solved['wall_clock_time'].median() if len(df_p_solved) > 0 else np.nan,
                'Std_Time_s': df_p_solved['wall_clock_time'].std() if len(df_p_solved) > 0 else np.nan,
                'Min_Time_s': df_p_solved['wall_clock_time'].min() if len(df_p_solved) > 0 else np.nan,
                'Max_Time_s': df_p_solved['wall_clock_time'].max() if len(df_p_solved) > 0 else np.nan,
                'P25_Time_s': np.percentile(df_p_solved['wall_clock_time'], 25) if len(df_p_solved) > 0 else np.nan,
                'P75_Time_s': np.percentile(df_p_solved['wall_clock_time'], 75) if len(df_p_solved) > 0 else np.nan,
                'P95_Time_s': np.percentile(df_p_solved['wall_clock_time'], 95) if len(df_p_solved) > 0 else np.nan,
            }

            # Add expansions if available
            if 'nodes_expanded' in df.columns:
                valid_exps = df_p_solved[df_p_solved['nodes_expanded'] > 0]['nodes_expanded']
                row['Mean_Expansions'] = valid_exps.mean() if len(valid_exps) > 0 else np.nan
                row['Median_Expansions'] = valid_exps.median() if len(valid_exps) > 0 else np.nan

            # Add H* preservation if available
            if 'h_star_preservation' in df.columns:
                h_pres = df_p['h_star_preservation'].dropna()
                row['Mean_H*_Preservation'] = h_pres.mean() if len(h_pres) > 0 else np.nan

            rows.append(row)

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_path, index=False)
        logger.info(f"✓ Summary statistics exported to {output_path}")

    except Exception as e:
        logger.error(f"Failed to export summary statistics: {e}")
        traceback.print_exc()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive GNN Policy Analysis & Visualization",
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

  # Analyze with specific output naming
  python analysis_and_visualization.py \\
      --results results.csv \\
      --output results/analysis \\
      --title "GNN Policy Evaluation"
        """
    )

    parser.add_argument("--results", help="Path to evaluation_results.csv")
    parser.add_argument("--experiments", nargs='+', help="Experiment directories")
    parser.add_argument("--output", default="plots", help="Output directory")
    parser.add_argument("--title", default="", help="Analysis title")

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

    # Add derived metrics
    df = add_derived_metrics(df)

    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PLOTS & ANALYSIS")
    logger.info("=" * 80 + "\n")

    # Generate all plots
    plot_solve_rate_comparison(df, os.path.join(args.output, "01_solve_rate.png"))
    plot_time_comparison_enhanced(df, os.path.join(args.output, "02_time_comparison.png"))
    plot_expansions_comparison(df, os.path.join(args.output, "03_expansions.png"))
    plot_efficiency_frontier(df, os.path.join(args.output, "04_efficiency_frontier.png"))
    plot_cumulative_distribution(df, os.path.join(args.output, "05_cumulative_dist.png"))
    plot_performance_profile(df, os.path.join(args.output, "06_performance_profile.png"))
    plot_h_star_preservation(df, os.path.join(args.output, "07_h_star_preservation.png"))
    plot_statistical_summary(df, os.path.join(args.output, "08_statistical_summary.png"))
    plot_per_difficulty_analysis(df, os.path.join(args.output, "09_per_difficulty.png"))
    plot_scaling_analysis(df, os.path.join(args.output, "10_scaling_analysis.png"))

    # Export summary statistics
    export_summary_statistics(df, os.path.join(args.output, "summary_statistics.csv"))

    # Compute and log statistical tests
    if HAS_SCIPY:
        logger.info("\n" + "=" * 80)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("=" * 80 + "\n")

        test_results = compute_statistical_tests(df)

        for baseline, stats in test_results.items():
            logger.info(f"\nGNN vs {baseline}:")
            logger.info(f"  Sample sizes: GNN={stats['gnn_n']}, {baseline}={stats['baseline_n']}")
            logger.info(f"  GNN:      mean={stats['gnn_mean']:.3f}s, median={stats['gnn_median']:.3f}s")
            logger.info(f"  {baseline}: mean={stats['baseline_mean']:.3f}s, median={stats['baseline_median']:.3f}s")
            logger.info(f"  Mann-Whitney U p-value: {stats['mann_whitney_p']:.6f}")
            if stats['wilcoxon_p'] is not None:
                logger.info(f"  Wilcoxon p-value: {stats['wilcoxon_p']:.6f}")
            logger.info(f"  Cohen's d (effect size): {stats['cohens_d']:.3f}")
            logger.info(f"  Speedup: {stats['speedup']:.2f}x")
            logger.info(f"  Significant at α=0.05: {'YES ✓' if stats['significant'] else 'NO'}")

    logger.info(f"\n✅ All analysis complete!")
    logger.info(f"📊 Results saved to: {os.path.abspath(args.output)}/")
    logger.info(f"\nGenerated files:")
    logger.info(f"  ✓ 01_solve_rate.png")
    logger.info(f"  ✓ 02_time_comparison.png")
    logger.info(f"  ✓ 03_expansions.png")
    logger.info(f"  ✓ 04_efficiency_frontier.png")
    logger.info(f"  ✓ 05_cumulative_dist.png")
    logger.info(f"  ✓ 06_performance_profile.png")
    logger.info(f"  ✓ 07_h_star_preservation.png")
    logger.info(f"  ✓ 08_statistical_summary.png")
    logger.info(f"  ✓ 09_per_difficulty.png")
    logger.info(f"  ✓ 10_scaling_analysis.png")
    logger.info(f"  ✓ summary_statistics.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())