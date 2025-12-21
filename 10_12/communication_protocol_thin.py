#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMMUNICATION PROTOCOL - Thin Client Version
=============================================
Simplified protocol for the Thin Client / Fat Server architecture.

Only two message types:
1. observation_{N}.json (C++ → Python): Pre-computed features + reward signals
2. merge_{N}.json (Python → C++): Chosen merge pair indices

All feature computation happens in C++.
"""

import os
import json
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class ThinCommConfig:
    """Communication configuration for thin client."""

    # Timeouts (seconds)
    OBSERVATION_TIMEOUT: float = 120.0
    ACK_TIMEOUT: float = 30.0

    # Polling interval (milliseconds)
    POLL_INTERVAL_MS: int = 50

    # File patterns
    OBSERVATION_PATTERN: str = "observation_{}.json"
    MERGE_DECISION_PATTERN: str = "merge_{}.json"

    # Directories (set at runtime)
    fd_output_dir: str = ""
    gnn_output_dir: str = ""


# ===========================================================================
# ATOMIC FILE I/O
# ===========================================================================

def write_json_atomic(data: Dict[str, Any], filepath: str) -> None:
    """
    Write JSON atomically using temp file + rename.

    Guarantees:
    - Complete write before visibility
    - No partial files
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file
    fd, temp_path = tempfile.mkstemp(
        dir=str(filepath.parent),
        suffix='.tmp',
        prefix=filepath.stem + '_'
    )

    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.replace(temp_path, filepath)

    except Exception:
        try:
            os.remove(temp_path)
        except:
            pass
        raise


def read_json_robust(
        filepath: str,
        timeout: float = 30.0,
        poll_interval: float = 0.05
) -> Optional[Dict[str, Any]]:
    """
    Read JSON with retries until file is complete.

    Returns:
        Parsed JSON dict, or None if timeout
    """
    filepath = Path(filepath)
    start = time.time()

    while time.time() - start < timeout:
        if not filepath.exists():
            time.sleep(poll_interval)
            continue

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            if not content.strip():
                time.sleep(poll_interval)
                continue

            return json.loads(content)

        except (json.JSONDecodeError, IOError):
            time.sleep(poll_interval)
            continue

    return None


# ===========================================================================
# MESSAGE TYPES
# ===========================================================================

@dataclass
class Observation:
    """
    C++ → Python: Pre-computed observation with features.

    Contains:
    - x: Node feature matrix [N, 7]
    - edge_index: Adjacency in COO format [[sources], [targets]]
    - num_active_systems: Number of remaining transition systems
    - reward_signals: Metrics for reward computation
    - iteration: Current iteration number
    - is_terminal: Whether episode should end
    """
    iteration: int
    x: list  # [[f1, f2, ..., f7], ...]
    edge_index: list  # [[src1, src2, ...], [tgt1, tgt2, ...]]
    num_active_systems: int
    reward_signals: Dict[str, float]
    is_terminal: bool = False
    timestamp: float = 0.0

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'Observation':
        return Observation(
            iteration=data.get('iteration', -1),
            x=data.get('x', []),
            edge_index=data.get('edge_index', [[], []]),
            num_active_systems=data.get('num_active_systems', 0),
            reward_signals=data.get('reward_signals', {}),
            is_terminal=data.get('is_terminal', False),
            timestamp=data.get('timestamp', time.time()),
        )


@dataclass
class MergeDecision:
    """
    Python → C++: Chosen merge pair.

    Contains:
    - iteration: Which iteration this decision is for
    - merge_pair: [node_idx_1, node_idx_2] indices into edge_index
    """
    iteration: int
    merge_pair: Tuple[int, int]
    timestamp: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'merge_pair': list(self.merge_pair),
            'timestamp': self.timestamp or time.time(),
        }


# ===========================================================================
# COMMUNICATION FUNCTIONS
# ===========================================================================

def wait_for_observation(
        iteration: int,
        fd_output_dir: str,
        timeout: float = 120.0
) -> Optional[Observation]:
    """
    Wait for C++ to export observation_{iteration}.json

    Args:
        iteration: Expected iteration number
        fd_output_dir: Directory where C++ writes observations
        timeout: Maximum wait time in seconds

    Returns:
        Observation object, or None if timeout/error
    """
    filepath = Path(fd_output_dir) / f"observation_{iteration}.json"

    logger.debug(f"[COMM] Waiting for: {filepath.name}")

    data = read_json_robust(str(filepath), timeout=timeout)

    if data is None:
        logger.error(f"[COMM] Timeout waiting for observation_{iteration}")
        return None

    # Validate iteration
    if data.get('iteration') != iteration:
        logger.error(f"[COMM] Iteration mismatch: expected {iteration}, got {data.get('iteration')}")
        return None

    return Observation.from_json(data)


def send_merge_decision(
        iteration: int,
        merge_pair: Tuple[int, int],
        gnn_output_dir: str
) -> bool:
    """
    Send merge decision to C++.

    Args:
        iteration: Current iteration
        merge_pair: (node_idx_1, node_idx_2)
        gnn_output_dir: Directory where Python writes decisions

    Returns:
        True if successful
    """
    decision = MergeDecision(
        iteration=iteration,
        merge_pair=merge_pair,
        timestamp=time.time()
    )

    filepath = Path(gnn_output_dir) / f"merge_{iteration}.json"

    try:
        write_json_atomic(decision.to_json(), str(filepath))
        logger.debug(f"[COMM] Sent merge decision: {merge_pair}")
        return True

    except Exception as e:
        logger.error(f"[COMM] Failed to send decision: {e}")
        return False


def ensure_directories(fd_output_dir: str, gnn_output_dir: str) -> None:
    """Ensure communication directories exist."""
    Path(fd_output_dir).mkdir(parents=True, exist_ok=True)
    Path(gnn_output_dir).mkdir(parents=True, exist_ok=True)


def cleanup_signal_files(fd_output_dir: str, gnn_output_dir: str) -> int:
    """
    Remove all signal files.

    Returns:
        Number of files deleted
    """
    deleted = 0

    for directory in [fd_output_dir, gnn_output_dir]:
        for pattern in ["*.json", "*.tmp"]:
            import glob
            for filepath in glob.glob(os.path.join(directory, pattern)):
                try:
                    os.remove(filepath)
                    deleted += 1
                except:
                    pass

    return deleted