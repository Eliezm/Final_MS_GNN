#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CENTRALIZED PATH MANAGEMENT
============================
Single source of truth for all paths.
"""

from pathlib import Path
import os

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Downward
DOWNWARD_DIR = PROJECT_ROOT / "downward"
FD_OUTPUT_DIR = DOWNWARD_DIR / "fd_output"
GNN_OUTPUT_DIR = DOWNWARD_DIR / "gnn_output"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
TB_LOGS_DIR = PROJECT_ROOT / "tb_logs"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# ============================================================================
# ENSURE ALL DIRECTORIES EXIST
# ============================================================================

for dir_path in [
    FD_OUTPUT_DIR, GNN_OUTPUT_DIR, RESULTS_DIR, LOGS_DIR,
    MODELS_DIR, TB_LOGS_DIR, BENCHMARKS_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)