#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OUTPUT STRUCTURE - Defines and creates organized experiment output directories
==============================================================================

Structure:
results/{experiment_name}/
├── training/
│   ├── model.zip                    # Final trained model
│   ├── training_log.jsonl           # Raw episode metrics
│   └── checkpoints/                 # Intermediate models
│       └── model_step_*.zip
├── analysis/
│   ├── training_analysis.json       # Main training analysis
│   ├── component_analysis.json      # Reward component breakdown
│   ├── feature_analysis.json        # Feature importance
│   ├── quality_analysis.json        # H* preservation, bisimulation
│   ├── decision_analysis.json       # GNN decision quality
│   └── safety_analysis.json         # Dead-end, solvability
├── plots/
│   ├── training/                    # Training dynamics plots
│   │   ├── 01_learning_curves.png
│   │   └── ...
│   ├── components/                  # Reward component plots
│   │   ├── 02_component_trajectories.png
│   │   └── ...
│   ├── quality/                     # Heuristic quality plots
│   │   ├── 06_bisimulation.png
│   │   └── ...
│   └── comparison/                  # Strategy comparison plots
│       ├── 13_three_way_comparison.png
│       └── ...
├── evaluation/
│   ├── gnn_vs_random/
│   │   ├── results.json
│   │   └── plots/
│   └── baselines/
│       ├── results.json
│       └── plots/
├── testing/
│   ├── {test_set_name}/
│   │   ├── results.json
│   │   ├── comparison.json
│   │   ├── summary.txt
│   │   └── plots/
│   └── ...
├── reports/
│   ├── experiment_report.json       # THE unified report
│   ├── experiment_report.txt        # Human-readable version
│   └── tables/
│       ├── strategy_comparison.csv
│       └── test_results.csv
└── logs/
    └── training.log                 # Full training log
"""

from pathlib import Path
from typing import Dict, Optional
import shutil
import logging

logger = logging.getLogger(__name__)


class ExperimentOutputManager:
    """
    Manages organized output structure for experiments.
    """

    def __init__(self, experiment_name: str, base_dir: str = "results"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.root = self.base_dir / experiment_name

        # Define directory structure
        self.dirs = {
            # Training
            "training": self.root / "training",
            "checkpoints": self.root / "training" / "checkpoints",

            # Analysis
            "analysis": self.root / "analysis",

            # Plots by category
            "plots": self.root / "plots",
            "plots_training": self.root / "plots" / "training",
            "plots_components": self.root / "plots" / "components",
            "plots_quality": self.root / "plots" / "quality",
            "plots_comparison": self.root / "plots" / "comparison",

            # Evaluation
            "evaluation": self.root / "evaluation",
            "eval_gnn_random": self.root / "evaluation" / "gnn_vs_random",
            "eval_baselines": self.root / "evaluation" / "baselines",

            # Testing (per-test-set subdirs created dynamically)
            "testing": self.root / "testing",

            # Reports
            "reports": self.root / "reports",
            "tables": self.root / "reports" / "tables",

            # Logs
            "logs": self.root / "logs",
        }

    def create_structure(self) -> None:
        """Create all directories."""
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")

        logger.info(f"Created experiment output structure: {self.root}")

    def get_dir(self, name: str) -> Path:
        """Get path to named directory."""
        if name not in self.dirs:
            raise ValueError(f"Unknown directory: {name}. Available: {list(self.dirs.keys())}")
        return self.dirs[name]

    def create_test_set_dir(self, test_set_name: str) -> Path:
        """Create dedicated directory for a test set."""
        test_dir = self.dirs["testing"] / test_set_name
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "plots").mkdir(exist_ok=True)
        return test_dir

    def get_plot_category_dir(self, plot_name: str) -> Path:
        """Get the appropriate plot subdirectory based on plot name."""

        # Map plot names to categories
        training_plots = ["01_learning", "07_dead_end", "12_merge_quality"]
        component_plots = ["02_component", "03_component", "04_merge_quality", "08_label", "09_transition"]
        quality_plots = ["05_feature", "06_bisim", "10_causal", "11_gnn_decision"]
        comparison_plots = ["13_three", "14_per_problem", "15_cumulative", "16_speedup", "17_literature"]

        for prefix in training_plots:
            if plot_name.startswith(prefix):
                return self.dirs["plots_training"]

        for prefix in component_plots:
            if plot_name.startswith(prefix):
                return self.dirs["plots_components"]

        for prefix in quality_plots:
            if plot_name.startswith(prefix):
                return self.dirs["plots_quality"]

        for prefix in comparison_plots:
            if plot_name.startswith(prefix):
                return self.dirs["plots_comparison"]

        # Default
        return self.dirs["plots"]

    def organize_plots(self, source_plots_dir: Path) -> Dict[str, Path]:
        """
        Move plots from flat directory to organized subdirectories.

        Returns mapping of plot_name -> new_path
        """
        moved = {}

        if not source_plots_dir.exists():
            return moved

        for plot_file in source_plots_dir.glob("*.png"):
            plot_name = plot_file.stem
            target_dir = self.get_plot_category_dir(plot_name)
            target_path = target_dir / plot_file.name

            try:
                shutil.copy2(plot_file, target_path)
                moved[plot_name] = target_path
                logger.debug(f"Organized plot: {plot_name} -> {target_dir.name}/")
            except Exception as e:
                logger.warning(f"Failed to organize {plot_name}: {e}")

        return moved

    def print_structure(self) -> None:
        """Print the directory structure."""
        print(f"\n📁 Experiment Output Structure: {self.root}")
        print("=" * 60)

        def print_tree(path: Path, prefix: str = ""):
            if not path.exists():
                return

            entries = sorted(path.iterdir())
            dirs = [e for e in entries if e.is_dir()]
            files = [e for e in entries if e.is_file()]

            # Print files
            for f in files[:5]:  # Limit to 5 files
                print(f"{prefix}├── {f.name}")
            if len(files) > 5:
                print(f"{prefix}├── ... ({len(files) - 5} more files)")

            # Print directories
            for i, d in enumerate(dirs):
                is_last = (i == len(dirs) - 1)
                print(f"{prefix}{'└' if is_last else '├'}── {d.name}/")
                print_tree(d, prefix + ("    " if is_last else "│   "))

        print_tree(self.root)
        print("=" * 60)


def setup_experiment_output(experiment_name: str, base_dir: str = "results") -> ExperimentOutputManager:
    """
    Factory function to create and setup experiment output structure.
    """
    manager = ExperimentOutputManager(experiment_name, base_dir)
    manager.create_structure()
    return manager