#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONFIGURATION - PROBLEM GENERATION PARAMETERS
=============================================
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
DOMAINS_DIR = PROJECT_ROOT / "domains"
PROBLEMS_DIR = PROJECT_ROOT / "problems"

# Fast Downward
FD_BUILD_DIR = os.environ.get("FD_BUILD_DIR", "./fast-downward")
FD_TIMEOUT = 480  # 8 minutes

# Generation parameters
GENERATION_CONFIG = {
    "blocksworld": {
        "target_problems_per_difficulty": 10,
        "max_generation_attempts": 50,
        "solution_length_ranges": {
            "small": (6, 8),      # 6-8 steps
            "medium": (11, 13),   # 11-13 steps
            "large": (16, 18),    # 16-18 steps
        },
        "difficulty_thresholds": {
            "small": (0, 60),     # 0-1 minute
            "medium": (60, 180),  # 1-3 minutes
            "large": (180, 420),  # 3-7 minutes
        }
    },
    "logistics": {
        "target_problems_per_difficulty": 10,
        "max_generation_attempts": 50,
    },
    "parking": {
        "target_problems_per_difficulty": 10,
        "max_generation_attempts": 50,
    }
}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Random seed for reproducibility
DEFAULT_SEED = 42