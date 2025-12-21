"""
GNN-Guided Merge-and-Shrink Planning Library
============================================
Core components for RL-based merge strategy selection.

Packages:
  - environments: ThinMergeEnv with 15-dim node + 10-dim edge features
  - models: GNN architecture and RL policy
  - rewards: Enhanced reward functions
  - communication: Python ↔ C++ IPC protocol
  - utils: Common utilities and configuration
"""

__version__ = "1.0.0"
__all__ = [
    "environments",
    "models",
    "rewards",
    "communication",
    "utils",
]
