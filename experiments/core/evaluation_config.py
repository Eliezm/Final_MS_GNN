#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION CONFIG - Centralized configuration (FIXED)
===================================================
All settings for baseline evaluation and FD integration.

✅ FIXED: Robust Fast Downward path detection
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any


class EvaluationConfig:
    """Configuration for evaluation framework."""

    # Feature dimensions (match ThinMergeEnv)
    NODE_FEATURE_DIM = 9
    EDGE_FEATURE_DIM = 11

    # Reward weights (match thin_merge_env)
    REWARD_WEIGHTS = {
        'w_h_preservation': 0.40,
        'w_shrinkability': 0.25,
        'w_state_control': 0.20,
        'w_solvability': 0.15,
    }

    # Timeouts
    FD_TIMEOUT_SEC = 300
    GNN_TIMEOUT_SEC = 300

    # Time limits
    TIME_LIMIT_PER_RUN_S: int = 300  # 5 minutes per problem

    # Reproducibility
    RANDOM_SEED: int = 42

    # ✅ FIXED: Robust Fast Downward path detection
    @staticmethod
    def _get_fd_path() -> str:
        """Find Fast Downward installation directory."""
        # Try multiple paths in order of likelihood
        candidates = [
            Path(__file__).parent.parent.parent / "downward",
            Path.cwd() / "downward",
            Path(__file__).parent.parent.parent.parent / "downward",
            Path.home() / "downward",
            Path("/opt/downward"),
            Path("/usr/local/downward"),
        ]

        for candidate in candidates:
            if candidate.exists() and (candidate / "builds" / "release" / "bin").exists():
                return str(candidate.absolute())

        # Return default and log warning
        default = str(Path(__file__).parent.parent.parent / "downward")
        import logging
        logging.warning(f"Fast Downward not found in candidates, using default: {default}")
        return default

    @staticmethod
    def _get_translate_bin(downward_dir: str) -> str:
        """Get path to Fast Downward translate binary."""
        path = Path(downward_dir) / "builds" / "release" / "bin" / "translate" / "translate.py"
        return str(path.absolute())

    @staticmethod
    def _get_downward_bin(downward_dir: str) -> str:
        """Get path to Fast Downward main binary."""
        base_path = Path(downward_dir) / "builds" / "release" / "bin"
        candidates = [
            base_path / "downward.exe",
            base_path / "downward",
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.absolute())

        # Return most likely path (will error later with clear message)
        return str(base_path / "downward")

    # Set class variables in order
    DOWNWARD_DIR = _get_fd_path()
    FD_TRANSLATE_BIN = _get_translate_bin(DOWNWARD_DIR)
    FD_DOWNWARD_BIN = _get_downward_bin(DOWNWARD_DIR)

    # Output configuration
    OUTPUT_DIR = "evaluation_results"
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "baseline_performance_summary.csv")
    OUTPUT_DETAILED = os.path.join(OUTPUT_DIR, "detailed_results.json")

    # Working directories
    FD_TEMP_DIR = "evaluation_temp"

    # Problem discovery configuration
    BENCHMARK_DOMAINS = {
        "blocksworld": {"path": "benchmarks/blocksworld", "problem_prefix": "problem_"},
        "logistics": {"path": "benchmarks/logistics", "problem_prefix": "problem_"},
    }

    PROBLEM_SIZES = ["small", "medium", "large"]

    # Baseline Fast Downward configurations
    BASELINE_CONFIGS = [
        {
            "name": "FD ASTAR LM-Cut",
            "search_config": "astar(lmcut())",
            "category": "heuristic"
        },
        {
            "name": "FD ASTAR Linear Merge (Reverse Level)",
            "search_config": (
                "astar(merge_and_shrink("
                "merge_strategy=merge_precomputed(merge_tree=linear(variable_order=reverse_level)),"
                "shrink_strategy=shrink_bisimulation(greedy=false),"
                "label_reduction=exact(before_shrinking=true,before_merging=false),"
                "max_states=200000,threshold_before_merge=1))"
            ),
            "category": "m&s"
        },
        {
            "name": "FD ASTAR SCC-DFP (Topological)",
            "search_config": (
                "astar(merge_and_shrink("
                "merge_strategy=merge_sccs("
                "order_of_sccs=topological,"
                "merge_selector=score_based_filtering("
                "scoring_functions=[goal_relevance(),dfp(),total_order()])),"
                "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
                "label_reduction=exact(before_shrinking=true,before_merging=false),"
                "max_states=200000,threshold_before_merge=1))"
            ),
            "category": "m&s"
        },
        {
            "name": "FD ASTAR Add Heuristic",
            "search_config": "astar(add())",
            "category": "heuristic"
        },
        {
            "name": "FD ASTAR FF Heuristic",
            "search_config": "astar(ff())",
            "category": "heuristic"
        },

        {
            "name": "FD_GBFS_FF",
            "search_config": "eager_greedy([ff()])",
            "category": "satisficing"
        },
    ]

    @staticmethod
    def get_baselines(seed: int) -> List[Dict[str, Any]]:
        """Get baseline configurations with seed support."""
        return EvaluationConfig.BASELINE_CONFIGS