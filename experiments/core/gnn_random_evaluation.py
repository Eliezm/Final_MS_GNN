#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN AND RANDOM MERGE POLICY EVALUATION MODULE (REFACTORED)
==========================================================
Main entry point with backward compatibility.

This module provides:
✓ Random merge strategy implementation
✓ GNN policy evaluation (trained models)
✓ Fast Downward integration and output parsing
✓ Detailed metrics collection (25+ metrics per run)
✓ Seamless compatibility with evaluation.py framework
✓ Analysis and visualization ready results

INTERNAL STRUCTURE:
- gnn_evaluation_policies.py      → Policy implementations
- gnn_evaluation_parser.py        → FD output parsing
- gnn_evaluation_executor.py      → Execution engine
- gnn_evaluation_evaluators.py    → High-level evaluators
- gnn_evaluation_framework.py     → Main orchestrator
- gnn_random_evaluation.py        → This file (re-exports for backward compat)

USAGE (unchanged - backward compatible):
    from experiments.core.gnn_random_evaluation import (
        GNNRandomEvaluationFramework,
        GNNPolicyEvaluator,
        RandomMergeEvaluator,
        RandomMergePolicy,
        FastDownwardOutputParser,
    )

    framework = GNNRandomEvaluationFramework(
        model_path="results/blocksworld_exp_1/model.zip",
        domain_file="benchmarks/blocksworld/domain.pddl",
        problem_files=[...],
        output_dir="evaluation_results"
    )

    gnn_results, random_results = framework.evaluate()
    summary = framework.to_summary()
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ✅ FIXED: Use absolute imports with full module paths
from experiments.core.gnn_evaluation_policies import (
    RandomMergePolicy,
    GNNMergePolicy,
)

from experiments.core.gnn_evaluation_parser import (
    FastDownwardOutputParser,
)

from experiments.core.gnn_evaluation_executor import (
    ExecutorConfig,
    MergeExecutor,
)

from experiments.core.gnn_evaluation_evaluators import (
    GNNPolicyEvaluator,
    RandomMergeEvaluator,
)

from experiments.core.gnn_evaluation_framework import (
    GNNRandomEvaluationFramework,
)

# ============================================================================
# PUBLIC API - Backward Compatibility
# ============================================================================

__all__ = [
    # Policies
    "RandomMergePolicy",
    "GNNMergePolicy",

    # Parsing
    "FastDownwardOutputParser",

    # Execution
    "ExecutorConfig",
    "MergeExecutor",

    # Evaluators
    "GNNPolicyEvaluator",
    "RandomMergeEvaluator",

    # Framework
    "GNNRandomEvaluationFramework",
]