#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 14: LITERATURE ALIGNMENT CHECKLIST
========================================
Validate implementation against research papers.

✅ FIXED: Handle empty checklist
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import logging

from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, create_plots_directory, save_plot_safely,
)

logger = logging.getLogger(__name__)


def plot_literature_alignment(
        checklist: Dict[str, bool],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create literature alignment checklist visualization.

    ✅ FIXED: Handle empty or None checklist
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = create_plots_directory(output_dir)

    # ✅ FIX: Handle empty checklist
    if not checklist:
        logger.warning("No literature checklist data - generating placeholder plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'Literature Alignment Checklist\n\n'
                          'No checklist data available.\n\n'
                          'This occurs when:\n'
                          '1. Training data is insufficient\n'
                          '2. Correlation analysis was not computed\n'
                          '3. Bisimulation analysis was not computed\n\n'
                          'Try running with more training episodes.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plot_path = plots_dir / "14_literature_alignment.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    # ====================================================================
    # ORGANIZE CHECKS BY CATEGORY
    # ====================================================================

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

    # ====================================================================
    # CREATE FIGURE
    # ====================================================================

    fig, ax = plt.subplots(figsize=(14, 8))

    helmert_checks = [(n, p) for n, p, c in zip(check_names, check_passes, categories)
                      if c == 'Helmert et al. 2014']
    nissim_checks = [(n, p) for n, p, c in zip(check_names, check_passes, categories)
                     if c == 'Nissim et al. 2011']
    gnn_checks = [(n, p) for n, p, c in zip(check_names, check_passes, categories)
                  if c == 'GNN Architecture']
    val_checks = [(n, p) for n, p, c in zip(check_names, check_passes, categories)
                  if c == 'Validation']

    all_groups = [
        ('Helmert et al. 2014', helmert_checks, '#1f77b4'),
        ('Nissim et al. 2011', nissim_checks, '#ff7f0e'),
        ('GNN Architecture', gnn_checks, '#2ca02c'),
        ('Validation', val_checks, '#d62728'),
    ]

    y_pos = 0

    for group_name, checks, color in all_groups:
        if checks:
            # Group label
            ax.text(-0.1, y_pos + len(checks) / 2, group_name,
                    fontweight='bold', fontsize=11, ha='right', va='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

            # Individual checks
            for check_name, passes in checks:
                color_check = '#27ae60' if passes else '#e74c3c'
                # ✅ FIX: Use ASCII checkmarks instead of emoji
                marker = '[OK]' if passes else '[X]'

                ax.barh(y_pos, passes, color=color_check, alpha=0.7, edgecolor='black')
                ax.text(-0.05, y_pos, marker, fontsize=10, ha='right', va='center',
                        fontweight='bold', color=color_check)
                ax.text(0.5, y_pos, check_name, fontsize=10, va='center')

                y_pos += 1

            y_pos += 0.5

    # ====================================================================
    # FORMATTING
    # ====================================================================

    ax.set_xlim([-0.3, 1.3])
    ax.set_ylim([-1, y_pos])
    ax.set_xlabel('Implementation Status', fontweight='bold', fontsize=12)
    ax.set_title('Literature Alignment Checklist\n' +
                 'Validates implementation against Helmert et al. 2014 & Nissim et al. 2011',
                 fontweight='bold', fontsize=14)
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Implemented', 'Implemented'], fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ====================================================================
    # SAVE
    # ====================================================================

    plt.tight_layout()
    plot_path = plots_dir / "14_literature_alignment.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None