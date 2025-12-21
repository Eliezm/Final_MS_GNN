#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOTTING UTILITIES - Shared infrastructure for all plots
Provides matplotlib setup, colors, formatting, validation.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any  # ✅ ADD: Dict, Any
import numpy as np
import logging  # ✅ ADD THIS

logger = logging.getLogger(__name__)


def setup_matplotlib() -> Optional['plt']:
    """
    Setup matplotlib with safe backend.

    Returns:
        matplotlib.pyplot module or None if unavailable
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return None


# ============================================================================
# COLOR SCHEMES - Research-appropriate palettes
# ============================================================================

COLOR_PALETTES = {
    'components': {
        'h_preservation': '#2ecc71',  # Green
        'transition_control': '#3498db',  # Blue
        'operator_projection': '#e67e22',  # Orange
        'label_combinability': '#e74c3c',  # Red
        'bonus_signals': '#9b59b6',  # Purple
    },
    'quality': {
        'excellent': '#27ae60',  # Dark green
        'good': '#2ecc71',  # Light green
        'moderate': '#f39c12',  # Orange
        'poor': '#e74c3c',  # Light red
        'bad': '#c0392b',  # Dark red
    },
    'status': {
        'success': '#27ae60',
        'warning': '#f39c12',
        'error': '#e74c3c',
        'neutral': '#95a5a6',
    },
}

COMPONENT_COLORS = [
    '#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6'
]

QUALITY_COLORS = {
    'excellent': '#27ae60',
    'good': '#2ecc71',
    'moderate': '#f39c12',
    'poor': '#e74c3c',
    'bad': '#c0392b',
}


# ============================================================================
# VALIDATION & SANITIZATION
# ============================================================================

def validate_data(
        values: np.ndarray,
        name: str = "data",
        allow_empty: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate array data for plotting safety.

    Args:
        values: Array to validate
        name: Variable name for error messages
        allow_empty: Whether empty arrays are OK

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(values) == 0:
        if allow_empty:
            return True, None
        return False, f"{name} is empty"

    if np.any(~np.isfinite(values)):
        return False, f"{name} contains NaN or inf values"

    if np.std(values) < 1e-10:
        return False, f"{name} has no variance (constant)"

    return True, None


def safe_mean_and_rolling(
        values: np.ndarray,
        window: int = 10
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Safely compute mean and rolling average.

    Args:
        values: Data array
        window: Rolling window size

    Returns:
        Tuple of (mean, rolling_average)
    """
    if len(values) < 2:
        return (float(values[0]) if len(values) > 0 else 0.0, None)

    mean_val = float(np.mean(values))

    # Adjust window to data size
    window = min(window, max(2, len(values) // 4))

    if len(values) > window:
        rolling_avg = np.convolve(values, np.ones(window) / window, mode='valid')
        return mean_val, rolling_avg

    return mean_val, None


# ============================================================================
# COMMON PLOT ELEMENTS
# ============================================================================

def add_target_line(
        ax,
        y_value: float,
        label: str,
        color: str = 'green',
        linestyle: str = '--',
        alpha: float = 0.5,
):
    """Add horizontal target line to plot."""
    ax.axhline(y=y_value, color=color, linestyle=linestyle, alpha=alpha,
               linewidth=2, label=label)


def add_phase_backgrounds(
        ax,
        num_episodes: int,
        phases: int = 3,
        colors: list = None,
        alpha: float = 0.05,
):
    """Add background shading for training phases."""
    if colors is None:
        colors = ['green', 'yellow', 'red']

    phase_width = num_episodes / phases
    for i in range(phases):
        ax.axvspan(i * phase_width, (i + 1) * phase_width,
                   color=colors[i], alpha=alpha)


def format_plot_labels(
        ax,
        xlabel: str,
        ylabel: str,
        title: str,
        fontsize_title: int = 12,
        fontsize_labels: int = 11,
):
    """Apply consistent formatting to plot."""
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=fontsize_labels)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=fontsize_labels)
    ax.set_title(title, fontweight='bold', fontsize=fontsize_title)
    ax.grid(True, alpha=0.3)


# ============================================================================
# FIGURE & DIRECTORY MANAGEMENT
# ============================================================================

def create_plots_directory(output_dir: Path) -> Path:
    """Ensure plots directory exists and return path."""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def save_plot_safely(
        fig,
        plot_path: Path,
        dpi: int = 150,
) -> bool:
    """
    Save plot with error handling.

    Returns:
        True if successful, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save plot {plot_path}: {e}")
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except:
            pass
        return False


# ============================================================================
# COMPARISON TABLE GENERATION
# ============================================================================

def create_comparison_table(
        gnn_stats: Optional[Dict[str, Any]] = None,
        random_stats: Optional[Dict[str, Any]] = None,
        baseline_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Create detailed comparison table (CSV + formatted text).

    ✅ FIXED: Handle missing/None parameters gracefully

    Returns:
        Path to saved CSV file, or None if failed
    """
    if output_dir is None:
        return None

    import csv
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ FIX: Guard all parameters
    gnn_stats = gnn_stats or {}
    random_stats = random_stats or {}
    baseline_stats = baseline_stats or {}

    table_data = []

    # GNN row
    if gnn_stats:
        table_data.append({
            'Strategy': 'GNN (Learned)',
            'Solve Rate (%)': f"{gnn_stats.get('solve_rate_pct', 0):.1f}",
            'Mean Time (s)': f"{gnn_stats.get('mean_time_sec', 0):.3f}",
            'Median Time (s)': f"{gnn_stats.get('median_time_sec', 0):.3f}",
            'Mean Expansions': f"{gnn_stats.get('mean_expansions', 0):,}",
            'H* Preservation': f"{gnn_stats.get('mean_h_preservation', 1.0):.4f}",
        })

    # Random row
    if random_stats:
        table_data.append({
            'Strategy': 'Random Merge',
            'Solve Rate (%)': f"{random_stats.get('solve_rate_pct', 0):.1f}",
            'Mean Time (s)': f"{random_stats.get('mean_time_sec', 0):.3f}",
            'Median Time (s)': f"{random_stats.get('median_time_sec', 0):.3f}",
            'Mean Expansions': f"{random_stats.get('mean_expansions', 0):,}",
            'H* Preservation': f"{random_stats.get('mean_h_preservation', 1.0):.4f}",
        })

    # Baseline rows
    for baseline_name, baseline_data in baseline_stats.items():
        if baseline_data:
            table_data.append({
                'Strategy': baseline_name[:30],
                'Solve Rate (%)': f"{baseline_data.get('solve_rate_%', 0):.1f}",
                'Mean Time (s)': f"{baseline_data.get('avg_time_total_s', 0):.3f}",
                'Median Time (s)': f"{baseline_data.get('avg_time_total_s', 0):.3f}",
                'Mean Expansions': f"{baseline_data.get('avg_expansions', 0):,}",
                'H* Preservation': '1.0000',
            })

    if not table_data:
        logger.warning("No data for comparison table")
        return None

    # Save as CSV
    try:
        csv_path = output_dir / "strategy_comparison_summary.csv"
        df = pd.DataFrame(table_data)
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        logger.error(f"Failed to create comparison table: {e}")
        return None