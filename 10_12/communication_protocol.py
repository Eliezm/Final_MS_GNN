#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMMUNICATION PROTOCOL - Thin Client Architecture
==================================================
Simplified protocol for Thin Client / Fat Server communication.

Protocol:
1. C++ exports observation_{N}.json with pre-computed features + reward signals
2. Python reads observation, makes decision
3. Python writes merge_{N}.json with chosen merge pair indices
4. C++ reads decision, executes merge, exports next observation

This module handles all I/O between Python and C++.
"""

import os
import json
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from common_utils import FD_OUTPUT_DIR, GNN_OUTPUT_DIR, ThinClientConfig

logger = logging.getLogger(__name__)


# ============================================================================
# ATOMIC FILE I/O
# ============================================================================

def write_json_atomic(data: Dict[str, Any], filepath: str) -> None:
    """
    Write JSON atomically using temp file + rename.

    Guarantees:
    - Complete write before visibility
    - No partial files visible to readers
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
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
        # Clean up temp file on error
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
    Read JSON with retries until file is complete and valid.

    Args:
        filepath: Path to JSON file
        timeout: Maximum wait time in seconds
        poll_interval: Time between retries in seconds

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


# ============================================================================
# MESSAGE DATA STRUCTURES
# ============================================================================

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
        """Parse observation from JSON dict."""
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
    - merge_pair: [node_idx_1, node_idx_2] - indices into current node list
    """
    iteration: int
    merge_pair: Tuple[int, int]
    timestamp: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'iteration': self.iteration,
            'merge_pair': list(self.merge_pair),
            'timestamp': self.timestamp or time.time(),
        }


# ============================================================================
# CORE COMMUNICATION FUNCTIONS
# ============================================================================

def wait_for_observation(
        iteration: int,
        timeout: float = None
) -> Optional[Observation]:
    """
    Wait for C++ to export observation_{iteration}.json

    Args:
        iteration: Expected iteration number
        timeout: Maximum wait time (uses config default if None)

    Returns:
        Observation object, or None if timeout/error
    """
    if timeout is None:
        timeout = ThinClientConfig.OBSERVATION_TIMEOUT

    filepath = FD_OUTPUT_DIR / f"observation_{iteration}.json"

    logger.debug(f"[COMM] Waiting for: {filepath.name}")

    data = read_json_robust(str(filepath), timeout=timeout)

    if data is None:
        logger.error(f"[COMM] Timeout waiting for observation_{iteration}")
        return None

    # Validate iteration
    if data.get('iteration') != iteration:
        logger.error(
            f"[COMM] Iteration mismatch: expected {iteration}, "
            f"got {data.get('iteration')}"
        )
        return None

    return Observation.from_json(data)


def send_merge_decision(
        iteration: int,
        merge_pair: Tuple[int, int]
) -> bool:
    """
    Send merge decision to C++.

    Args:
        iteration: Current iteration
        merge_pair: (node_idx_1, node_idx_2)

    Returns:
        True if successful
    """
    decision = MergeDecision(
        iteration=iteration,
        merge_pair=merge_pair,
        timestamp=time.time()
    )

    filepath = GNN_OUTPUT_DIR / f"merge_{iteration}.json"

    try:
        write_json_atomic(decision.to_json(), str(filepath))
        logger.debug(f"[COMM] Sent merge decision: {merge_pair}")
        return True

    except Exception as e:
        logger.error(f"[COMM] Failed to send decision: {e}")
        return False


def cleanup_communication_files() -> int:
    """
    Remove all signal files from communication directories.

    Returns:
        Number of files deleted
    """
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

    logger.debug(f"[COMM] Cleaned up {deleted} files")
    return deleted


def ensure_communication_directories() -> None:
    """Ensure communication directories exist."""
    FD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GNN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)