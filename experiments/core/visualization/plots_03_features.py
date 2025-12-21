#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 5: FEATURE ANALYSIS
========================
Analyze how input features correlate with GNN decisions and rewards.

Shows:
- Feature importance ranking
- Correlation with confidence
- Feature-reward relationships
"""

from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import logging  # ✅ ADD THIS


from experiments.core.visualization.plotting_utils import (
    setup_matplotlib, validate_data, format_plot_labels,
    create_plots_directory, save_plot_safely,
)

logger = logging.getLogger(__name__)  # ✅ ADD THIS


def plot_feature_importance(
        feature_importance_analysis: Dict[str, Any],
        correlation_analysis: Dict[str, Any],
        output_dir: Path,
) -> Optional[Path]:
    """
    Create feature importance ranking plot (with proper fallback).
    """
    plt = setup_matplotlib()
    if not plt:
        return None

    plots_dir = create_plots_directory(output_dir)

    feature_imp = feature_importance_analysis.get('feature_importance', {})

    # ✅ FIX: Handle missing data gracefully
    if not feature_imp or len(feature_imp) == 0:
        logger.warning("No feature importance data available - generating empty plot")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No feature importance data generated\n\n'
                          '(This occurs when GNN outputs lack discriminative features)\n'
                          'Try:\n'
                          '1. Increasing training episodes\n'
                          '2. Improving edge feature signals from C++\n'
                          '3. Checking GNN architecture',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')

        plt.tight_layout()
        plot_path = plots_dir / "05_feature_importance.png"
        if save_plot_safely(fig, plot_path):
            return plot_path
        return None

    # ✅ PREPARE DATA with fallback
    try:
        sorted_features = sorted(feature_imp.items(),
                                 key=lambda x: x[1].get('importance', 0) if isinstance(x[1], dict) else 0,
                                 reverse=True)

        feature_names = [f[0] for f in sorted_features]
        importances = []
        significances = []

        for f in sorted_features:
            if isinstance(f[1], dict):
                importances.append(f[1].get('importance', 0))
                significances.append(f[1].get('significant', False))
            else:
                importances.append(float(f[1]))
                significances.append(False)

        # Filter valid values
        valid_data = []
        for name, imp, sig in zip(feature_names, importances, significances):
            if isinstance(imp, (int, float)) and np.isfinite(imp) and imp >= 0:
                valid_data.append((name, imp, sig))

        if not valid_data:
            raise ValueError("All importance values invalid")

        feature_names, importances, significances = zip(*valid_data)

    except Exception as e:
        logger.warning(f"Failed to parse feature importance: {e}")
        return None

    # ✅ CREATE FIGURE
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2ecc71' if sig else '#f39c12' for sig in significances]

    bars = ax.barh(range(len(feature_names)), importances, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp, bar.get_y() + bar.get_height() / 2, f'{imp:.3f}',
                va='center', ha='left', fontweight='bold', fontsize=9)

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=10)

    format_plot_labels(ax, 'Feature Importance (|Correlation| with GNN Confidence)',
                       'Feature',
                       'Feature Importance Ranking\n(Green = Significant at p<0.05, '
                       'Orange = Not significant)')

    # Set xlim safely
    max_importance = max(importances) if importances else 1.0
    if np.isfinite(max_importance) and max_importance > 0:
        ax.set_xlim([0, max_importance * 1.15])

    # ✅ SAVE
    plt.tight_layout()
    plot_path = plots_dir / "05_feature_importance.png"

    if save_plot_safely(fig, plot_path):
        return plot_path

    return None