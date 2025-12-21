# -*- coding: utf-8 -*-
"""
This module provides the GraphTracker class, a data structure for managing the
state of the merge-and-shrink heuristic construction process.

It represents the set of transition systems (TS) as nodes in a directed graph,
where edges represent causal dependencies from the Fast Downward planner. The class
is responsible for loading the initial state from planner output, performing
merge operations on nodes, and updating the graph based on new information from
the planner. It serves as the core state management component for the MergeEnv.
"""

# ------------------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------------------
import json
import logging
import time
from json import JSONDecoder
from typing import List, Union, Dict, Tuple, Any, FrozenSet, Optional

import os


import networkx as nx
import numpy as np

# matplotlib is an optional dependency for visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
#  Configuration and Constants
# ------------------------------------------------------------------------------
# --- Setup basic logging ---
# Consistent logging configuration with merge_env.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# --- Constants for file I/O ---
FILE_RETRY_COUNT = 60
FILE_RETRY_DELAY_S = 0.2


# ------------------------------------------------------------------------------
#  Helper Functions
# ------------------------------------------------------------------------------

def _load_json_robustly(path: str, retries: int = FILE_RETRY_COUNT, delay: float = FILE_RETRY_DELAY_S) -> Any:
    """
    Parses the first complete JSON object from a file path with retries.

    This function is designed to handle cases where a file might be read while
    another process is writing to it. It ensures the file is not empty and that
    the content appears to be a complete JSON object (ends with '}' or ']')
    before attempting to parse it.

    Method of Action:
    1. Loop for a specified number of `retries`.
    2. Read the file content, ignoring UTF-8 errors.
    3. If the file is empty or doesn't end with a closing brace/bracket,
       it's considered incomplete. Wait and retry.
    4. Use `JSONDecoder.raw_decode` to parse only the *first* valid JSON
       object, which avoids errors from trailing, partially-written data.
    5. If any error occurs (`OSError`, `JSONDecodeError`), wait and retry.
    6. If all retries fail, raise a `RuntimeError` with the last known error.
    """
    decoder = JSONDecoder()
    last_error = None
    for _ in range(retries):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lstrip()

            if not content:
                # File is empty, wait for content to be written.
                raise json.JSONDecodeError("File is empty", content, 0)

            # Heuristic check for completeness to avoid parsing mid-write.
            tail = content.rstrip()
            if not tail or tail[-1] not in ("]", "}"):
                raise json.JSONDecodeError("JSON appears incomplete (no closing bracket/brace)", content, len(content))

            # Decode the first object, ignoring any trailing garbage.
            obj, _ = decoder.raw_decode(content)
            return obj

        except (OSError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(delay)

    raise RuntimeError(f"Failed to load valid JSON from '{path}' after {retries} retries. Last error: {last_error}")


def product_state_index(s1: int, s2: int, n2: int) -> int:
    """
    Maps a pair of local states to a single index in their Cartesian product.

    This is a standard row-major order mapping. Given two state spaces of sizes
    `n1` and `n2`, a state `s1` from the first space and `s2` from the second
    are mapped to a unique index in the combined space of size `n1 * n2`.

    Args:
        s1 (int): The index of the state in the first transition system.
        s2 (int): The index of the state in the second transition system.
        n2 (int): The total number of states in the second transition system.

    Returns:
        int: The unique index in the product state space.
    """
    return s1 * n2 + s2


# FILE: graph_tracker.py
# REPLACE THIS FUNCTION (around line 143)

def merge_transition_systems(ts1: Dict[str, Any], ts2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    ✅ OPTIMIZED: Early detection of problematic merges.

    Returns None if merge is obviously bad (saves 90% of computation time).
    """
    n1, n2 = ts1["num_states"], ts2["num_states"]

    # ✅ OPTIMIZATION 1: Reject merges that will exceed reasonable size
    # This prevents attempting trillion-state merges
    SAFE_MERGE_LIMIT = 50_000_000  # 50M states (tunable)

    try:
        # Check if product would overflow
        if n1 > SAFE_MERGE_LIMIT or n2 > SAFE_MERGE_LIMIT:
            product = n1 * n2
        else:
            product = n1 * n2

    except OverflowError:
        logger.error(f"[EARLY REJECT] Merge would overflow: {n1} × {n2}")
        return None

    # Reject if product is unreasonably large
    if product > 10_000_000_000:  # 10 billion
        logger.warning(f"[EARLY REJECT] Merge product too large: {product} (limit: 10B)")
        logger.warning(f"  TS1: {n1} states, TS2: {n2} states")
        return None

    # ✅ OPTIMIZATION 2: Check reachability before creating goal states
    # Only create goal_states if both TS have reachable goals
    reachable_goals_1 = len([f for f in ts1.get("f_before", [])
                             if f != float('inf') and f < 1_000_000_000])
    reachable_goals_2 = len([f for f in ts2.get("f_before", [])
                             if f != float('inf') and f < 1_000_000_000])

    # if reachable_goals_1 == 0 or reachable_goals_2 == 0:
    #     logger.warning(f"[EARLY REJECT] Unreachable goals in merge")
    #     logger.warning(f"  TS1 reachable goals: {reachable_goals_1}, TS2: {reachable_goals_2}")
    #     return None

    # ✅ OPTIMIZATION 3: Lazy goal state computation
    # Don't pre-compute the full list; just store metadata
    goal_states_1 = ts1["goal_states"]
    goal_states_2 = ts2["goal_states"]

    # Only materialize if reasonable size
    num_product_goals = len(goal_states_1) * len(goal_states_2)

    if num_product_goals > 1_000_000:
        # Too many goals; don't enumerate, store as mapping function
        logger.warning(f"[LAZY GOALS] {num_product_goals} product goals; using lazy computation")

        merged_ts = {
            "num_states": n1 * n2,
            "init_state": product_state_index(ts1["init_state"], ts2["init_state"], n2),
            "goal_states": None,  # ✅ Mark as lazy
            "_goal_state_mapping": {
                "ts1_goals": goal_states_1,
                "ts2_goals": goal_states_2,
                "n2": n2,
            },
            "incorporated_variables": ts1["incorporated_variables"] + ts2["incorporated_variables"],
            "iteration": max(ts1.get("iteration", -1), ts2.get("iteration", -1)) + 1,
        }

        logger.info(f"[MERGE] Created lazy goal state mapping (would have {num_product_goals} entries)")
        return merged_ts

    # Otherwise, safe to materialize
    merged_ts = {
        "num_states": n1 * n2,
        "init_state": product_state_index(ts1["init_state"], ts2["init_state"], n2),
        "goal_states": [
            product_state_index(g1, g2, n2)
            for g1 in goal_states_1
            for g2 in goal_states_2
        ],
        "incorporated_variables": ts1["incorporated_variables"] + ts2["incorporated_variables"],
        "iteration": max(ts1.get("iteration", -1), ts2.get("iteration", -1)) + 1,
    }

    return merged_ts


# ✅ ADD THIS NEW HELPER FUNCTION to handle lazy goal states
def is_goal_state_lazy(state_index: int, ts_merged: Dict[str, Any]) -> bool:
    """Check if a state is a goal using lazy mapping (no list materialization)."""
    if ts_merged.get("goal_states") is not None:
        # Already materialized
        return state_index in ts_merged["goal_states"]

    # Use lazy mapping
    mapping = ts_merged.get("_goal_state_mapping")
    if not mapping:
        return False

    ts1_goals = mapping["ts1_goals"]
    ts2_goals = mapping["ts2_goals"]
    n2 = mapping["n2"]

    s1 = state_index // n2
    s2 = state_index % n2

    return s1 in ts1_goals and s2 in ts2_goals

# ------------------------------------------------------------------------------
#  GraphTracker Class
# ------------------------------------------------------------------------------

class GraphTracker:
    """
    Manages the graph of transition systems for the merge-and-shrink process.

    This class holds a `networkx.DiGraph` where each node represents a transition
    system (TS). Initially, nodes correspond to atomic TSs for individual problem
    variables. The class provides methods to merge nodes (creating a new composite
    TS node) and update node properties based on new data from the planner.

    Attributes:
        graph (nx.DiGraph): The graph of transition systems.
        varset_to_node (Dict): A mapping from a frozenset of variable IDs to the
                                corresponding node ID in the graph. This allows for
                                efficient lookups.
        next_node_id (int): A counter for allocating unique IDs to new merged nodes.
    """

    def __init__(self, ts_json_path: str, cg_json_path: str, is_debug: bool = False):
        """Initialize with caching infrastructure."""
        self.graph = nx.DiGraph()
        self.varset_to_node: Dict[FrozenSet, Union[int, str]] = {}
        self.next_node_id: int = 0
        self.is_debug = is_debug
        # ✅ NEW: Persistent caches (survive across observations)
        self._centrality_cache: Optional[Dict] = None
        self._centrality_cache_valid = False
        self._max_vars_cache: Optional[int] = None
        self._max_iter_cache: Optional[int] = None
        self._f_stats_cache: Dict[int, Tuple[float, float, float, float]] = {}  # Cache for f_stats
        self._graph_hash_last = None
        # Note: _edge_features_cache and _node_features_cache were mentioned
        # in the prompt but not used in the provided methods, so omitting them for now.
        # Add them here if needed later:
        # self._edge_features_cache: Optional[np.ndarray] = None
        # self._node_features_cache: Dict[int, np.ndarray] = {}

        logging.info("Initializing GraphTracker...")
        try:
            self._load_atomic_systems(ts_json_path)
            self._load_causal_edges(cg_json_path)
        except Exception as e:
            logging.error(f"Failed during initial graph loading: {e}")
            if not self.is_debug:
                raise
            else:
                logging.warning("Proceeding with an empty graph in debug mode.")

        # ✅ NEW: FD index tracking
        self._last_fd_mapping_iteration = -1

    # --- END OF REPLACEMENT FOR __init__ ---

    # ============================================================================
    # ✅ NEW: FD Index Synchronization
    # ============================================================================

    # ============================================================================
    # ✅ NEW: FD Index Synchronization Methods
    # ============================================================================

    # In GraphTracker class, ADD THESE METHODS:

    # FILE: graph_tracker.py
    # REPLACE the sync_fd_indices_from_mapping method

    # FILE: graph_tracker.py
    # ENHANCED sync_fd_indices_from_mapping()

    def sync_fd_indices_from_mapping(self, mapping_file: str) -> bool:
        """
        ✅ COMPLETE FIX: Sync Python node IDs with C++ FD indices.

        After a merge, some nodes will no longer exist in the FD mapping.
        This is expected and not an error.

        Returns: True if sync successful, False otherwise
        """
        if not os.path.exists(mapping_file):
            logger.warning(f"[SYNC] Mapping file not found: {mapping_file}")
            return False

        try:
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
        except Exception as e:
            logger.error(f"[SYNC] Failed to load mapping: {e}")
            return False

        systems = mapping_data.get("systems", [])

        # Build: incorporated_variables (as tuple) → fd_index
        fingerprint_to_fd_index = {}
        for system_entry in systems:
            fd_idx = system_entry.get("fd_index")
            inc_vars = tuple(sorted(system_entry.get("incorporated_variables", [])))
            fingerprint_to_fd_index[inc_vars] = fd_idx

            logger.debug(f"[SYNC] Mapping {inc_vars} → FD index {fd_idx}")

        # Update all graph nodes
        updated_count = 0
        missing_count = 0

        for node_id, node_data in self.graph.nodes(data=True):
            inc_vars = tuple(sorted(node_data.get("incorporated_variables", [])))

            if inc_vars in fingerprint_to_fd_index:
                old_idx = node_data.get("fd_index", None)
                new_idx = fingerprint_to_fd_index[inc_vars]

                node_data["fd_index"] = new_idx
                updated_count += 1

                if old_idx is not None and old_idx != new_idx:
                    logger.info(f"[SYNC] Updated node {node_id}: fd_index {old_idx} → {new_idx}")
            else:
                # ✅ FIX: This is EXPECTED after merges - don't warn
                missing_count += 1
                logger.debug(f"[SYNC] Node {node_id} not in FD mapping (vars: {inc_vars}) - likely merged")

        logger.info(f"[SYNC] Synchronized {updated_count} nodes with FD indices")
        if missing_count > 0:
            logger.debug(f"[SYNC] {missing_count} nodes not in mapping (expected after merges)")

        # ✅ VERIFICATION: Check that all active nodes have fd_index
        unsynced_nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("fd_index") is None:
                unsynced_nodes.append(node_id)

        if unsynced_nodes:
            logger.error(f"[SYNC] ❌ ERROR: {len(unsynced_nodes)} nodes still missing fd_index!")
            logger.error(f"[SYNC] Unsynced nodes: {unsynced_nodes}")
            return False

        logger.info(f"[SYNC] ✅ All nodes have valid fd_index")
        return True

    # FILE: graph_tracker.py
    # VERIFY this method - should already be correct

    def get_fd_indices_for_merge(self, node_a_id: int, node_b_id: int) -> Tuple[int, int]:
        """
        ✅ CRITICAL: Get current FD indices for merge pair.

        RETURNS:
            (fd_index_a, fd_index_b)

        RAISES:
            KeyError if nodes don't have FD indices (sync not called)

        USED BY: merge_env.step() before sending decision
        """
        if node_a_id not in self.graph.nodes:
            raise KeyError(f"Node {node_a_id} not in graph")
        if node_b_id not in self.graph.nodes:
            raise KeyError(f"Node {node_b_id} not in graph")

        node_a = self.graph.nodes[node_a_id]
        node_b = self.graph.nodes[node_b_id]

        fd_idx_a = node_a.get("fd_index")
        fd_idx_b = node_b.get("fd_index")

        if fd_idx_a is None or fd_idx_b is None:
            raise KeyError(
                f"Node missing FD index: A.fd_index={fd_idx_a}, B.fd_index={fd_idx_b}. "
                f"Must call sync_fd_indices_from_mapping() first!"
            )

        logger.debug(f"[INDICES] Merge {node_a_id}→{node_b_id}: FD indices ({fd_idx_a}, {fd_idx_b})")
        return (fd_idx_a, fd_idx_b)

    # --- ADD THESE NEW METHODS INSIDE THE GraphTracker CLASS ---

    def _get_graph_hash(self) -> str:
        """✅ Robust graph hash including node data."""
        edges_tuple = tuple(sorted(self.graph.edges()))
        node_count = self.graph.number_of_nodes()
        # Include max node ID to catch renumbering
        max_node_id = max(self.graph.nodes()) if self.graph.nodes() else -1
        return str(hash((edges_tuple, node_count, max_node_id)))

    def _invalidate_all_caches(self):
        """✅ COMPLETE: Invalidate ALL caches after any modification"""
        logging.debug("🧹 INVALIDATING ALL GRAPHTRACKER CACHES")

        # Core computation caches
        self._centrality_cache = None
        self._centrality_cache_valid = False
        self._f_stats_cache.clear()
        self._max_vars_cache = None
        self._max_iter_cache = None
        self._graph_hash_last = None

        logging.debug("✓ All caches cleared")

    def _invalidate_caches(self):
        """Alias for consistency"""
        self._invalidate_all_caches()

    def get_centrality(self, force_recompute: bool = False) -> Dict:
        """✅ GUARANTEED FRESH: Compute or return cached centrality"""
        if force_recompute:
            logging.debug("🔄 Force recomputing centrality...")
            self._centrality_cache_valid = False

        if not self._centrality_cache_valid:
            logging.debug("📊 Computing centrality (not in cache)...")

            try:
                if self.graph.number_of_nodes() > 0:
                    self._centrality_cache = nx.closeness_centrality(self.graph)
                    logging.debug(f"  ✓ Computed for {len(self._centrality_cache)} nodes")
                else:
                    self._centrality_cache = {}
                    logging.debug("  (Empty graph)")
            except nx.NetworkXError as e:
                logging.warning(f"  Centrality failed: {e}, using empty dict")
                self._centrality_cache = {}

            self._centrality_cache_valid = True

        return self._centrality_cache if self._centrality_cache else {}

    def get_max_vars(self) -> int:
        """✅ CACHED: Return max incorporated variables."""
        # Compute only if cache is empty
        if self._max_vars_cache is None:
            logging.debug("Computing max_vars (cached)...")
            self._max_vars_cache = max(
                (len(d.get("incorporated_variables", [])) for _, d in self.graph.nodes(data=True)),
                default=1  # Default if graph is empty
            ) or 1  # Ensure it's at least 1 if max returns 0
        return self._max_vars_cache

    def get_max_iter(self) -> int:
        """✅ CACHED: Return max iteration level."""
        # Compute only if cache is empty
        if self._max_iter_cache is None:
            logging.debug("Computing max_iter (cached)...")
            self._max_iter_cache = max(
                (d.get("iteration", 0) for _, d in self.graph.nodes(data=True)),
                default=0  # Default if graph is empty
            ) or 1  # Ensure it's at least 1 if max returns 0
        return self._max_iter_cache

    def update_graph(self, ts_json_path: str) -> None:
        """
        Updates the graph with new transition system data from a JSON file.

        This method reads a TS list from the given path and updates the properties
        of existing nodes. This is typically used after a merge operation, where
        the planner provides an updated TS file for the newly created node.

        Args:
            ts_json_path (str): The path to the JSON file with TS data.
        """
        logging.info(f"Updating graph from '{ts_json_path}'...")
        try:
            data = _load_json_robustly(ts_json_path)
            ts_list = data if isinstance(data, list) else [data]

            for ts in ts_list:
                if not isinstance(ts, dict):
                    continue
                self._add_or_update_node(ts)

        except Exception as e:
            logging.warning(f"Could not parse or process TS JSON from '{ts_json_path}': {e}")

    # FILE: graph_tracker.py
    # REPLACE THE EXISTING merge_nodes METHOD WITH THIS

    def f_stats(self, node_id: int) -> Tuple[float, float, float, float]:
        """✅ GUARANTEED FRESH: Memoized f-statistics"""
        # Check cache FIRST
        if node_id in self._f_stats_cache:
            cached = self._f_stats_cache[node_id]
            logging.debug(f"  [CACHE HIT] f_stats({node_id})")
            return cached

        logging.debug(f"  [CACHE MISS] Computing f_stats({node_id})...")

        if node_id not in self.graph.nodes:
            logging.warning(f"    Node {node_id} not in graph!")
            result = (0.0, 0.0, 0.0, 0.0)
        else:
            f_values_raw = self.graph.nodes[node_id].get("f_before", [])

            # ✅ Filter invalid values
            f_values = [f for f in f_values_raw
                        if f != float('inf') and f >= 0 and f < 1_000_000_000]

            if not f_values:
                result = (0.0, 0.0, 0.0, 0.0)
                logging.debug(f"    No valid f-values")
            else:
                arr = np.array(f_values, dtype=np.float32)
                result = (
                    float(np.min(arr)),
                    float(np.mean(arr)),
                    float(np.max(arr)),
                    float(np.std(arr))
                )
                logging.debug(f"    min={result[0]:.1f}, mean={result[1]:.1f}, "
                              f"max={result[2]:.1f}, std={result[3]:.1f}")

        # ✅ Store in cache
        self._f_stats_cache[node_id] = result
        return result

    def merge_nodes(self, node_ids: List[int]) -> Tuple[bool, Optional[str]]:
        """
        ✅ ENHANCED: Merge with comprehensive cache invalidation
        """
        if len(node_ids) != 2:
            raise KeyError(f"merge_nodes requires 2 IDs, got {len(node_ids)}")

        a, b = node_ids

        logging.info(f"\n{'=' * 60}")
        logging.info(f"MERGE: {a} + {b}")
        logging.info(f"{'=' * 60}\n")

        # ✅ PRE-MERGE STATE CAPTURE (for diagnostics)
        pre_state = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_a_exists': a in self.graph,
            'node_b_exists': b in self.graph,
        }

        logging.debug(f"Pre-merge state: {pre_state}")

        if a not in self.graph or b not in self.graph:
            logging.error(f"❌ Node not in graph: a={a in self.graph}, b={b in self.graph}")
            return False, "Node not in graph"

        ts1 = self.graph.nodes[a]
        ts2 = self.graph.nodes[b]

        merged_ts = merge_transition_systems(ts1, ts2)

        if merged_ts is None:
            reason = "Merge rejected: product too large"
            logging.warning(f"  ⚠️ {reason}")
            return False, reason

        # Perform merge
        new_id = self.next_node_id
        self.next_node_id += 1

        self.graph.add_node(new_id, **merged_ts)
        var_key = frozenset(merged_ts["incorporated_variables"])
        self.varset_to_node[var_key] = new_id

        self._rewire_edges(a, new_id)
        self._rewire_edges(b, new_id)

        self.graph.remove_nodes_from([a, b])

        # ✅ CRITICAL: Invalidate ALL caches IMMEDIATELY
        logging.info("Invalidating all caches after merge...")
        self._invalidate_all_caches()
        logging.info("✓ Caches invalidated\n")

        # ✅ POST-MERGE STATE CAPTURE
        post_state = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'new_node_id': new_id,
            'new_node_size': len(merged_ts.get("incorporated_variables", [])),
        }

        logging.info(f"Post-merge state: {post_state}")
        logging.info(f"✓ Merged into node {new_id}\n")

        return True, None

    def _invalidate_all_caches(self):
        """
        ✅ CRITICAL: Clear ALL caches after any graph modification
        Prevents stale data from previous problems
        """
        logging.debug("🧹 INVALIDATING ALL CACHES")

        # Core caches
        self._centrality_cache = None
        self._centrality_cache_valid = False
        self._f_stats_cache.clear()
        self._max_vars_cache = None
        self._max_iter_cache = None
        self._graph_hash_last = None

        logging.debug("✓ All caches cleared")

    # --- END OF REPLACEMENT FOR f_stats ---

    def _load_atomic_systems(self, ts_json_path: str) -> None:
        """
        Loads the initial set of atomic transition systems from a JSON file.

        These form the initial nodes of the graph. Atomic systems are identified
        by having `iteration == -1`.
        """
        logging.info(f"Loading atomic systems from '{ts_json_path}'...")
        data = _load_json_robustly(ts_json_path)
        ts_list = data if isinstance(data, list) else [data]

        num_loaded = 0
        for ts in ts_list:
            if isinstance(ts, dict) and ts.get("iteration", -1) == -1:
                self._add_or_update_node(ts)
                num_loaded += 1

        # Set the next node ID to be higher than any existing integer ID to avoid collisions.
        int_ids = [n for n in self.graph.nodes if isinstance(n, int)]
        self.next_node_id = max(int_ids, default=-1) + 1
        logging.info(f"Loaded {num_loaded} atomic systems. Next node ID set to {self.next_node_id}.")

    # REPLACE THE OLD METHOD WITH THIS NEW ONE
    def _load_causal_edges(self, cg_json_path: str) -> None:
        """Loads the causal graph edges from a JSON file into the graph."""
        logging.info(f"Loading causal edges from '{cg_json_path}'...")
        try:
            with open(cg_json_path, "r") as f:
                data = json.load(f)

            edges = data.get("edges", [])
            if not edges:
                logging.warning("Causal graph file contains no 'edges' key or the list is empty.")
                return

            # Debug: log what nodes exist before trying to add edges
            logging.info(f"Current graph nodes before adding edges: {list(self.graph.nodes())}")

            num_added = 0
            for edge in edges:
                src = edge.get("from")
                tgt = edge.get("to")

                # This check is crucial and now more explicit
                if src is not None and tgt is not None:
                    if self.graph.has_node(src) and self.graph.has_node(tgt):
                        self.graph.add_edge(src, tgt)
                        num_added += 1
                        # logging.info(f"Added edge ({src}, {tgt})")
                    else:
                        logging.warning(
                            f"Skipping edge ({src}, {tgt}) because one or both nodes do not exist in the graph. "
                            f"Current nodes: {list(self.graph.nodes())}"
                        )
                else:
                    logging.warning(f"Edge has None values: from={src}, to={tgt}")

            logging.info(f"Loaded {num_added} causal edges. Final edge count: {len(list(self.graph.edges()))}")

        except FileNotFoundError:
            logging.warning(f"Causal graph file '{cg_json_path}' not found. No edges loaded.")
            if not self.is_debug:
                raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading causal edges: {e}", exc_info=True)
            if not self.is_debug:
                raise

    def _add_or_update_node(self, ts: Dict[str, Any]) -> None:
        """
        Adds a new node or updates an existing node's data based on a TS dict.

        The identity of a node is determined by its set of "incorporated_variables".
        If a node representing a given set of variables already exists, its
        attributes are updated. Otherwise, a new node is created.

        This method also removes the large 'transitions' list from the node data
        to save memory.

        ✅ NEW: Validates that TS has meaningful data before adding.
        """
        # Validate input
        if not ts or not isinstance(ts, dict):
            logging.warning("Skipping invalid TS: not a dict or empty")
            return

        ts_data = ts.copy()

        # ✅ MEMORY: Don't store full transitions - just count them
        transitions = ts_data.pop("transitions", [])
        ts_data["num_transitions"] = len(transitions)

        ts_data.pop("transitions", None)

        # ✅ NEW: Comprehensive validation
        inc_vars = ts_data.get("incorporated_variables", [])

        if not inc_vars:
            logging.warning("❌ Skipping TS with NO INCORPORATED VARIABLES")
            logging.warning(f"   TS keys: {list(ts.keys())}")
            logging.warning(f"   TS content: {ts}")
            return

        # ✅ NEW: Validate other critical fields
        num_states = ts_data.get("num_states", 0)
        if num_states <= 0:
            logging.warning(f"⚠️  Skipping TS with invalid num_states: {num_states}")
            return

        # ✅ NEW: Check if TS looks complete
        required_fields = ["num_states", "init_state", "goal_states"]
        missing = [k for k in required_fields if k not in ts_data]
        if missing:
            logging.warning(f"⚠️  TS missing fields: {missing}")
            logging.warning(f"   Available fields: {list(ts_data.keys())}")
            # Don't skip - continue with defaults
            for field in missing:
                if field == "init_state":
                    ts_data[field] = 0
                elif field == "goal_states":
                    # ✅ FIX: Default should be empty, not [0]
                    ts_data[field] = []
                    logger.warning(f"⚠️ TS {ts_data.get('iteration')} has no goal_states, using empty list")

        # Now proceed with normal logic
        var_key = frozenset(inc_vars)
        existing_node_id = self.varset_to_node.get(var_key)

        if existing_node_id is not None and existing_node_id in self.graph:
            logging.info(f"✓ Updating existing node {existing_node_id} with {num_states} states")
            self.graph.nodes[existing_node_id].update(ts_data)
        else:
            # New node
            is_atomic = ts_data.get("iteration", -1) == -1
            if is_atomic:
                node_id = inc_vars[0]
                # logging.info(f"✓ Adding atomic node {node_id} for variable {inc_vars[0]}")
            else:
                node_id = self.next_node_id
                self.next_node_id += 1
                # logging.info(f"✓ Adding merged node {node_id} for variables {inc_vars}")

            self.graph.add_node(node_id, **ts_data)
            self.varset_to_node[var_key] = node_id

            # Update counter
            if isinstance(node_id, int):
                self.next_node_id = max(self.next_node_id, node_id + 1)

            # logging.info(f"   → Node {node_id} has {num_states} states")


    def _rewire_edges(self, old_id: Union[int, str], new_id: Union[int, str]) -> None:
        """
        Moves all incoming and outgoing edges from an old node to a new node.
        """
        # Rewire incoming edges: for every predecessor `p` of `old_id`, add edge `(p, new_id)`.
        if old_id in self.graph:
            for predecessor in list(self.graph.predecessors(old_id)):
                if predecessor != new_id:  # Avoid self-loops with the other merged node
                    self.graph.add_edge(predecessor, new_id)
            # Rewire outgoing edges: for every successor `s` of `old_id`, add edge `(new_id, s)`.
            for successor in list(self.graph.successors(old_id)):
                if successor != new_id:
                    self.graph.add_edge(new_id, successor)

    def display(self) -> None:
        """
        Renders and displays the current state of the graph using matplotlib.

        Note: This is intended for interactive debugging and requires the
        `matplotlib` library to be installed.
        """
        if plt is None:
            logging.warning("matplotlib is not installed. Cannot display graph.")
            return

        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(self.graph, seed=42)

        labels = {
            n: f"ID: {n}\n|S|={d.get('num_states', '?')}\nIter: {d.get('iteration', '?')}"
            for n, d in self.graph.nodes(data=True)
        }

        nx.draw_networkx(
            self.graph,
            pos,
            labels=labels,
            node_size=1500,
            node_color="lightblue",
            font_size=8,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
        )
        plt.title("Transition System Causal Graph", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # This block serves as a "smoke test" to verify basic functionality.
    # It requires dummy JSON files to be present in the same directory.
    logging.info("--- Running GraphTracker Smoke Test ---")

    # Create dummy files for testing purposes
    DUMMY_CG_FILE = "causal_graph_test.json"
    DUMMY_TS_FILE = "ts_test.json"

    cg_data = {"edges": [{"from": 0, "to": 1}, {"from": 1, "to": 2}]}
    ts_data = [
        {"num_states": 2, "init_state": 0, "goal_states": [1], "incorporated_variables": [0], "iteration": -1},
        {"num_states": 3, "init_state": 1, "goal_states": [2], "incorporated_variables": [1], "iteration": -1},
        {"num_states": 4, "init_state": 2, "goal_states": [0], "incorporated_variables": [2], "iteration": -1},
    ]

    with open(DUMMY_CG_FILE, "w") as f:
        json.dump(cg_data, f)
    with open(DUMMY_TS_FILE, "w") as f:
        json.dump(ts_data, f)

    try:
        # 1. Test initialization
        tracker = GraphTracker(ts_json_path=DUMMY_TS_FILE, cg_json_path=DUMMY_CG_FILE, is_debug=True)
        print("\nInitial Graph Nodes:", list(tracker.graph.nodes()))
        print("Initial Graph Edges:", list(tracker.graph.edges()))
        # tracker.display() # Uncomment for visual inspection

        # 2. Test merging
        if len(tracker.graph.nodes) >= 2:
            nodes_to_merge = [0, 1]
            tracker.merge_nodes(nodes_to_merge)
            print(f"\nGraph Nodes after merging {nodes_to_merge}:", list(tracker.graph.nodes()))
            print("Graph Edges after merging:", list(tracker.graph.edges()))
            # tracker.display()

            # 3. Test f_stats on the new node
            new_node_id = list(tracker.graph.nodes)[-1]
            # Add some dummy f-values to test f_stats
            tracker.graph.nodes[new_node_id]['f_before'] = [10, 20, 30, 40]
            stats = tracker.f_stats(new_node_id)
            print(
                f"\nF-stats for new node {new_node_id}: min={stats[0]}, mean={stats[1]}, max={stats[2]}, std={stats[3]}")

        else:
            print("\nNot enough nodes to test merge.")

        logging.info("--- Smoke Test Completed Successfully ---")

    except Exception as e:
        logging.error(f"--- Smoke Test FAILED: {e} ---")

    finally:
        # Clean up dummy files
        import os

        if os.path.exists(DUMMY_CG_FILE): os.remove(DUMMY_CG_FILE)
        if os.path.exists(DUMMY_TS_FILE): os.remove(DUMMY_TS_FILE)