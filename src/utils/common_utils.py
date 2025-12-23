#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMMON UTILITIES - Updated for Enhanced Features
=================================================
"""

import os
import logging
from pathlib import Path

import torch

# ✅ FORCE CPU MODE (no GPU on server)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)  # Use 4 CPU threads

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
DOWNWARD_DIR = PROJECT_ROOT / "downward"
FD_OUTPUT_DIR = DOWNWARD_DIR / "fd_output"
GNN_OUTPUT_DIR = DOWNWARD_DIR / "gnn_output"

FD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GNN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# THIN CLIENT CONFIG - UPDATED DIMENSIONS
# ============================================================================

class ThinClientConfig:
    """Configuration for thin client architecture."""

    # Node features (15 features from C++ - expanded!)
    NODE_FEATURE_DIM = 15  # Was 7, now 15

    FEATURE_NAMES = [
        # Original 7 features
        "normalized_size",
        "is_atomic",
        "mean_h_value",
        "solvability",
        "diameter",
        "transition_density",
        "label_count",
        # New 8 features
        "init_h_value",  # Feature 7: h* for init state
        "dead_end_ratio",  # Feature 8
        "goal_state_ratio",  # Feature 9
        "h_value_variance",  # Feature 10
        "avg_trans_per_label",  # Feature 11
        "self_loop_ratio",  # Feature 12
        "num_variables",  # Feature 13
        "determinism_indicator",  # Feature 14
    ]

    # Edge features (10 features from C++ - NEW!)
    EDGE_FEATURE_DIM = 10

    EDGE_FEATURE_NAMES = [
        "label_jaccard",
        "shared_label_ratio",
        "sync_factor",
        "product_size_log",
        "shares_variables",
        "combined_h_star",
        "h_compatibility",
        "both_solvable",
        "size_balance",
        "density_ratio",
    ]

    # Observation limits
    MAX_NODES = 100
    MAX_EDGES = 1000

    # Timeouts
    OBSERVATION_TIMEOUT = 120.0
    ACK_TIMEOUT = 30.0
    POLL_INTERVAL = 0.05


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        FD_OUTPUT_DIR,
        GNN_OUTPUT_DIR,
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "tb_logs",
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)


def cleanup_signal_files():
    """Remove all signal files."""
    deleted = 0
    for directory in [FD_OUTPUT_DIR, GNN_OUTPUT_DIR]:
        if not directory.exists():
            continue
        import glob
        for pattern in ["*.json", "*.tmp"]:
            for filepath in glob.glob(str(directory / pattern)):
                try:
                    os.remove(filepath)
                    deleted += 1
                except:
                    pass
    return deleted


ensure_directories()