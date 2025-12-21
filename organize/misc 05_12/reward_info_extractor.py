# FILE: reward_info_extractor.py (PHASE 2 - ENHANCED)
import os
import json
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - MUST MATCH C++ DEFINITIONS
# ============================================================================

# Fast Downward's INF constant
LARGE_VALUE_THRESHOLD = 1_000_000_000
FD_INF_CANDIDATES = [2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1, 999999999, 1_000_000_000, float('inf')]


def is_unreachable_value(value: Any) -> bool:
    """
    ✅ CRITICAL: Properly detect unreachable states.

    A value is unreachable if:
    - It's NaN
    - It's infinity (float or large int)
    - It's negative (error code)
    """
    try:
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return True

        if isinstance(value, int):
            if value < 0:
                return True
            if value in FD_INF_CANDIDATES:
                return True
            if value > LARGE_VALUE_THRESHOLD:
                return True

        return False
    except (TypeError, ValueError):
        return True


def is_valid_fvalue(value: Any) -> bool:
    """Validate that an F-value is actually valid."""
    if is_unreachable_value(value):
        return False

    try:
        num_value = float(value)
        if num_value < 0 or num_value > LARGE_VALUE_THRESHOLD:
            return False
        return True
    except (TypeError, ValueError):
        return False


# ============================================================================
# MERGE INFO DATACLASS - COMPLETE
# ============================================================================

@dataclass
class MergeInfo:
    """Complete container for merge information with ALL necessary fields."""

    # ✅ Iteration & IDs
    iteration: int
    ts1_id: int
    ts2_id: int

    # ✅ Sizes (BEFORE merge)
    states_before: int
    ts1_size: int
    ts2_size: int

    # ✅ Sizes (AFTER merge)
    states_after: int

    # ✅ F-values (RAW)
    f_before: List[int]
    f_after: List[int]

    # ✅ F-value statistics
    f_value_stability: float  # [0, 1] how stable F-values are
    f_preservation_score: float  # [0, 1] preservation of heuristic info

    # ✅ State changes
    delta_states: int
    state_explosion_penalty: float

    # ✅ Transition properties
    ts1_transitions: int = 0
    ts2_transitions: int = 0
    merged_transitions: int = 0
    transition_density_change: float = 0.0

    # ✅ Change detection
    num_significant_f_changes: int = 0
    avg_f_change: float = 0.0
    max_f_change: float = 0.0

    # ✅ A* Search signals
    nodes_expanded: int = 0
    search_depth: int = 0
    solution_cost: int = 0
    branching_factor: float = 1.0
    solution_found: bool = False

    # ✅ Reachability
    reachable_states: int = 0
    unreachable_states: int = 0
    reachability_ratio: float = 0.0

    # ✅ Goal states
    merged_goal_states: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'iteration': self.iteration,
            'ts1_id': self.ts1_id,
            'ts2_id': self.ts2_id,
            'states_before': self.states_before,
            'states_after': self.states_after,
            'delta_states': self.delta_states,
            'f_value_stability': float(self.f_value_stability),
            'f_preservation_score': float(self.f_preservation_score),
            'state_explosion_penalty': float(self.state_explosion_penalty),
            'transition_density_change': float(self.transition_density_change),
            'num_significant_f_changes': self.num_significant_f_changes,
            'avg_f_change': float(self.avg_f_change),
            'max_f_change': float(self.max_f_change),
            'nodes_expanded': self.nodes_expanded,
            'search_depth': self.search_depth,
            'solution_cost': self.solution_cost,
            'branching_factor': float(self.branching_factor),
            'solution_found': self.solution_found,
            'reachable_states': self.reachable_states,
            'unreachable_states': self.unreachable_states,
            'reachability_ratio': float(self.reachability_ratio),
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """Comprehensive validation."""
        issues = []

        if not (0 <= self.f_value_stability <= 1):
            issues.append(f"f_value_stability out of range: {self.f_value_stability}")
            self.f_value_stability = np.clip(self.f_value_stability, 0, 1)

        if self.states_before <= 0 or self.states_after <= 0:
            issues.append(f"Invalid state counts: {self.states_before} → {self.states_after}")

        if self.branching_factor < 1.0:
            self.branching_factor = 1.0
            issues.append("Corrected branching_factor to 1.0")

        for field in ['f_value_stability', 'avg_f_change', 'max_f_change',
                      'state_explosion_penalty', 'transition_density_change']:
            val = getattr(self, field)
            if isinstance(val, float):
                if np.isnan(val):
                    setattr(self, field, 0.0)
                    issues.append(f"Replaced NaN in {field}")
                elif np.isinf(val):
                    setattr(self, field, 1.0 if 'stability' in field or 'preservation' in field else 0.0)
                    issues.append(f"Replaced Inf in {field}")

        return len(issues) == 0, issues


# ============================================================================
# SIGNAL EXTRACTION - CORE LOGIC
# ============================================================================

class SignalParser:
    """Parses C++ exported JSON signals."""

    @staticmethod
    def safe_compute_average(values: List[Any], context: str = "") -> Tuple[float, int]:
        """Safely compute average with validation."""
        if not values:
            logger.warning(f"[{context}] Empty value list for averaging")
            return 0.0, 0

        valid_values = [v for v in values if is_valid_fvalue(v)]
        removed_count = len(values) - len(valid_values)

        if removed_count > 0:
            logger.debug(f"[{context}] Removed {removed_count}/{len(values)} invalid F-values")

        if not valid_values:
            logger.error(f"[{context}] ALL values were invalid! Using 0.5 as default")
            return 0.5, 0

        avg = float(np.mean(np.array(valid_values, dtype=np.float32)))

        if np.isnan(avg) or np.isinf(avg):
            logger.error(f"[{context}] Computed average is NaN/Inf! Defaulting to 0.5")
            return 0.5, len(valid_values)

        return avg, len(valid_values)

    @staticmethod
    def compute_f_stability(
            f_before_ts1: List[int],
            f_before_ts2: List[int],
            f_after: List[int],
            ts1_size: int,
            ts2_size: int
    ) -> float:
        """
        ✅ CRITICAL: Compute F-stability with robust handling.

        This is the MOST IMPORTANT metric for reward computation.
        """
        # Filter to valid values
        f_before_ts1_valid = [f for f in f_before_ts1 if is_valid_fvalue(f)]
        f_before_ts2_valid = [f for f in f_before_ts2 if is_valid_fvalue(f)]
        f_after_valid = [f for f in f_after if is_valid_fvalue(f)]

        logger.debug(f"F-value validation: {len(f_before_ts1_valid)}/{len(f_before_ts1)} valid in TS1, "
                     f"{len(f_before_ts2_valid)}/{len(f_before_ts2)} valid in TS2, "
                     f"{len(f_after_valid)}/{len(f_after)} valid after")

        if not f_before_ts1_valid or not f_before_ts2_valid or not f_after_valid:
            logger.warning("Insufficient valid F-values for stability computation, using neutral 0.5")
            return 0.5

        # Compute medians (robust to outliers)
        before_median = float(np.median(f_before_ts1_valid + f_before_ts2_valid))
        after_median = float(np.median(f_after_valid))

        before_std = float(np.std(f_before_ts1_valid + f_before_ts2_valid)) if len(
            f_before_ts1_valid + f_before_ts2_valid) > 1 else 0.0
        after_std = float(np.std(f_after_valid)) if len(f_after_valid) > 1 else 0.0

        # Normalized change in median
        if before_median > 0:
            median_change = abs(after_median - before_median) / before_median
        else:
            median_change = 0.0

        # Normalized change in variance
        if before_std > 0:
            std_change = abs(after_std - before_std) / before_std
        else:
            std_change = 0.0 if after_std == 0 else 1.0

        # Weighted combination (median is more important than std)
        change_metric = 0.7 * np.clip(median_change, 0, 1) + 0.3 * np.clip(std_change, 0, 1)

        # Convert to stability score: lower change = higher stability
        f_stability = max(0.0, 1.0 - change_metric)

        logger.debug(f"F-stability: median {before_median:.1f}→{after_median:.1f}, "
                     f"std {before_std:.1f}→{after_std:.1f}, stability={f_stability:.3f}")

        return float(np.clip(f_stability, 0.0, 1.0))

    @staticmethod
    def compute_significant_changes(
            f_before: List[int],
            f_after: List[int],
            threshold: float = 5.0
    ) -> Tuple[int, float, float]:
        """Count significant F-value changes."""
        f_before_valid = [f for f in f_before if is_valid_fvalue(f)]
        f_after_valid = [f for f in f_after if is_valid_fvalue(f)]

        if not f_before_valid or not f_after_valid:
            return 0, 0.0, 0.0

        # Compare first min(len) values
        min_len = min(len(f_before_valid), len(f_after_valid))

        try:
            abs_changes = np.abs(
                np.array(f_before_valid[:min_len], dtype=np.float32) -
                np.array(f_after_valid[:min_len], dtype=np.float32)
            )
            num_significant = int(np.sum(abs_changes > threshold))
            avg_change = float(np.mean(abs_changes))
            max_change = float(np.max(abs_changes))

            return num_significant, avg_change, max_change
        except Exception as e:
            logger.warning(f"Error computing changes: {e}")
            return 0, 0.0, 0.0


# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class RewardInfoExtractor:
    """Extracts and validates merge information from C++ signals."""

    def __init__(self, fd_output_dir: str = "downward/fd_output"):
        self.fd_output_dir = fd_output_dir
        self.parser = SignalParser()

    def extract_merge_info(
            self,
            iteration: int,
            timeout: float = 30.0,
            expected_launch_time: Optional[float] = None  # ✅ NEW
    ) -> Optional[MergeInfo]:
        """
        ✅ ENHANCED: Extract merge info with freshness validation

        NEW PARAM: expected_launch_time - when FD process was launched
                   Ensures signals are fresh and not from previous problem
        """
        logger.info(f"\n[EXTRACT::ITERATION {iteration}] Extracting merge info")
        logger.info(f"[EXTRACT::ITERATION {iteration}] Timeout: {timeout}s")

        if expected_launch_time:
            logger.info(f"[EXTRACT::ITERATION {iteration}] FD launch time: {expected_launch_time}")

        try:
            before_path = os.path.join(self.fd_output_dir, f"merge_before_{iteration}.json")
            after_path = os.path.join(self.fd_output_dir, f"merge_after_{iteration}.json")

            logger.info(f"[EXTRACT::ITERATION {iteration}] Waiting for signals...")

            start_time = time.time()

            # ✅ Wait for files with freshness validation
            while time.time() - start_time < timeout:
                before_exists = os.path.exists(before_path)
                after_exists = os.path.exists(after_path)

                if before_exists and after_exists:
                    # ✅ NEW: Verify files are FRESH
                    if expected_launch_time:
                        before_mtime = os.path.getmtime(before_path)
                        after_mtime = os.path.getmtime(after_path)

                        before_fresh = before_mtime >= expected_launch_time - 1.0
                        after_fresh = after_mtime >= expected_launch_time - 1.0

                        logger.debug(f"[EXTRACT::ITERATION {iteration}] before_fresh: {before_fresh}")
                        logger.debug(f"[EXTRACT::ITERATION {iteration}] after_fresh: {after_fresh}")

                        if not (before_fresh and after_fresh):
                            logger.warning(f"[EXTRACT::ITERATION {iteration}] Signals appear stale, re-waiting...")
                            time.sleep(0.2)
                            continue

                    # Try to load
                    try:
                        before_data = self._load_json_with_retry(before_path, timeout=5.0)
                        after_data = self._load_json_with_retry(after_path, timeout=5.0)

                        if before_data is not None and after_data is not None:
                            logger.info(f"[EXTRACT::ITERATION {iteration}] ✓ Both files loaded")
                            break
                        else:
                            logger.debug("[EXTRACT] Files exist but not yet readable")
                            time.sleep(0.2)
                            continue

                    except Exception as e:
                        logger.debug(f"[EXTRACT] Parse error: {e}")
                        time.sleep(0.2)
                        continue
                else:
                    logger.debug(f"[EXTRACT::ITERATION {iteration}] Files not yet ready...")
                    time.sleep(0.2)
                    continue

            elapsed = time.time() - start_time

            if before_data is None or after_data is None:
                logger.error(f"[EXTRACT::ITERATION {iteration}] ✗ Timeout after {elapsed:.1f}s")
                return None

            # ✅ VALIDATION: Verify iteration matches
            if before_data.get('iteration') != iteration or after_data.get('iteration') != iteration:
                logger.error(f"[EXTRACT::ITERATION {iteration}] Iteration mismatch!")
                logger.error(f"  Expected: {iteration}")
                logger.error(f"  Before: {before_data.get('iteration')}")
                logger.error(f"  After: {after_data.get('iteration')}")
                return None

            # ✅ REST OF EXTRACTION...

            logger.info(f"[LOAD] ✓ Both files loaded successfully")

            # ====================================================================
            # EXTRACT F-VALUES
            # ====================================================================
            logger.info(f"\n[F-VALUES] Extracting F-values...")

            ts1_f = before_data.get("ts1_f_values", [])
            ts2_f = before_data.get("ts2_f_values", [])
            f_after_raw = after_data.get("f_values", [])

            ts1_f_valid = [f for f in ts1_f if is_valid_fvalue(f)]
            ts2_f_valid = [f for f in ts2_f if is_valid_fvalue(f)]
            f_after_valid = [f for f in f_after_raw if is_valid_fvalue(f)]

            logger.info(f"  - ts1_valid: {len(ts1_f_valid)} / {len(ts1_f)}")
            logger.info(f"  - ts2_valid: {len(ts2_f_valid)} / {len(ts2_f)}")
            logger.info(f"  - after_valid: {len(f_after_valid)} / {len(f_after_raw)}")

            # ====================================================================
            # EXTRACT IDS & SIZES
            # ====================================================================
            logger.info(f"\n[IDS] Extracting IDs and sizes...")

            ts1_id = before_data.get("ts1_id", -1)
            ts2_id = before_data.get("ts2_id", -1)
            ts1_size = before_data.get("ts1_size", 0)
            ts2_size = before_data.get("ts2_size", 0)

            logger.info(f"  - ts1_id: {ts1_id}, size: {ts1_size}")
            logger.info(f"  - ts2_id: {ts2_id}, size: {ts2_size}")

            # ====================================================================
            # COMPUTE F-STABILITY
            # ====================================================================
            logger.info(f"\n[STABILITY] Computing F-stability...")

            f_stability = self.parser.compute_f_stability(
                ts1_f, ts2_f, f_after_raw, ts1_size, ts2_size
            )
            logger.info(f"  - f_stability: {f_stability:.3f}")

            # ====================================================================
            # COMPUTE SIGNIFICANT CHANGES
            # ====================================================================
            logger.info(f"\n[CHANGES] Computing F-value changes...")

            num_changes, avg_change, max_change = self.parser.compute_significant_changes(
                ts1_f + ts2_f, f_after_raw, threshold=5.0
            )
            logger.info(f"  - significant changes: {num_changes}")
            logger.info(f"  - avg change: {avg_change:.2f}")
            logger.info(f"  - max change: {max_change:.2f}")

            # ====================================================================
            # EXTRACT SIZES & DELTAS
            # ====================================================================
            logger.info(f"\n[SIZES] Extracting state counts...")

            states_before = max(1, ts1_size * ts2_size)
            states_after = max(1, len(f_after_raw))
            delta_states = states_after - states_before

            state_explosion = self._compute_state_explosion(delta_states, states_before)

            logger.info(f"  - states_before: {states_before}")
            logger.info(f"  - states_after: {states_after}")
            logger.info(f"  - delta_states: {delta_states}")
            logger.info(f"  - explosion_penalty: {state_explosion:.4f}")

            # ====================================================================
            # EXTRACT REACHABILITY
            # ====================================================================
            logger.info(f"\n[REACHABILITY] Computing reachability...")

            reachable_states = after_data.get("reachable_states", 0)
            unreachable_states = after_data.get("unreachable_states", 0)
            reachability_ratio = after_data.get("reachability_ratio", 0.0)

            logger.info(f"  - reachable: {reachable_states}")
            logger.info(f"  - unreachable: {unreachable_states}")
            logger.info(f"  - ratio: {reachability_ratio:.2%}")

            # ====================================================================
            # EXTRACT A* SIGNALS
            # ====================================================================
            logger.info(f"\n[A*] Extracting search signals...")

            search_signals = after_data.get("search_signals", {})

            nodes_expanded = int(search_signals.get("nodes_expanded", 0))
            search_depth = int(search_signals.get("search_depth", 0))
            solution_cost = int(search_signals.get("solution_cost", 0))
            branching_factor = float(search_signals.get("branching_factor", 1.0))
            solution_found = bool(search_signals.get("solution_found", False))

            # Safety checks
            if nodes_expanded < 0:
                nodes_expanded = 0
            if search_depth < 0:
                search_depth = 0
            if solution_cost < 0:
                solution_cost = 0
            if branching_factor < 1.0 or np.isnan(branching_factor) or np.isinf(branching_factor):
                branching_factor = 1.0

            logger.info(f"  - nodes_expanded: {nodes_expanded}")
            logger.info(f"  - search_depth: {search_depth}")
            logger.info(f"  - solution_cost: {solution_cost}")
            logger.info(f"  - branching_factor: {branching_factor:.3f}")
            logger.info(f"  - solution_found: {solution_found}")

            # ====================================================================
            # CREATE MERGE INFO
            # ====================================================================
            logger.info(f"\n[CREATE] Creating MergeInfo object...")

            merge_info = MergeInfo(
                iteration=iteration,
                ts1_id=ts1_id,
                ts2_id=ts2_id,
                states_before=states_before,
                ts1_size=ts1_size,
                ts2_size=ts2_size,
                states_after=states_after,
                f_before=ts1_f + ts2_f,
                f_after=f_after_raw,
                f_value_stability=f_stability,
                f_preservation_score=f_stability,
                delta_states=delta_states,
                state_explosion_penalty=state_explosion,
                ts1_transitions=before_data.get("ts1_num_transitions", 0),
                ts2_transitions=before_data.get("ts2_num_transitions", 0),
                merged_transitions=after_data.get("num_transitions", 0),
                transition_density_change=self._compute_transition_density_change(
                    before_data.get("ts1_num_transitions", 0),
                    before_data.get("ts2_num_transitions", 0),
                    after_data.get("num_transitions", 0),
                    ts1_size, ts2_size, states_after
                ),
                num_significant_f_changes=num_changes,
                avg_f_change=avg_change,
                max_f_change=max_change,
                nodes_expanded=nodes_expanded,
                search_depth=search_depth,
                solution_cost=solution_cost,
                branching_factor=branching_factor,
                solution_found=solution_found,
                reachable_states=reachable_states,
                unreachable_states=unreachable_states,
                reachability_ratio=reachability_ratio,
                merged_goal_states=after_data.get("num_goal_states", 0),
            )

            logger.info(f"[CREATE] ✓ MergeInfo created")

            # Validate
            is_valid, issues = merge_info.validate()
            if issues:
                for issue in issues:
                    logger.info(f"  [VALIDATE] {issue}")

            logger.info(f"\n[SUMMARY] Iteration {iteration}:")
            logger.info(f"  - TS{ts1_id}({ts1_size}) + TS{ts2_id}({ts2_size}) → {states_after} states")
            logger.info(f"  - f_stability={f_stability:.3f}")
            logger.info(f"  - A* nodes_expanded={nodes_expanded}")
            logger.info(f"  - solution_found={solution_found}")

            return merge_info

        except Exception as e:
            logger.error(f"\n[ERROR] Failed to extract merge info: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _load_json_with_retry(
            self,
            path: str,
            timeout: float = 30.0
    ) -> Optional[Dict]:
        """Load JSON with retries and validation."""
        start_time = time.time()
        last_size = -1

        while time.time() - start_time < timeout:
            try:
                if not os.path.exists(path):
                    time.sleep(0.1)
                    continue

                with open(path, 'r') as f:
                    content = f.read()

                if not content.strip():
                    time.sleep(0.1)
                    continue

                data = json.loads(content)
                return data

            except json.JSONDecodeError:
                time.sleep(0.1)
                continue
            except IOError:
                time.sleep(0.1)
                continue

        logger.error(f"Timeout loading {path}")
        return None

    def _compute_state_explosion(self, delta_states: int, states_before: int) -> float:
        """Compute state explosion penalty."""
        if states_before <= 0:
            return 0.0
        pct_increase = delta_states / float(max(states_before, 1))
        penalty = min(1.0, max(0.0, pct_increase / 0.5))
        return float(penalty)

    def _compute_transition_density_change(
            self,
            ts1_trans: int,
            ts2_trans: int,
            merged_trans: int,
            ts1_size: int,
            ts2_size: int,
            merged_size: int
    ) -> float:
        """Change in transition density."""
        if ts1_size <= 0 or ts2_size <= 0 or merged_size <= 0:
            return 0.0
        try:
            density_before_ts1 = ts1_trans / float(ts1_size)
            density_before_ts2 = ts2_trans / float(ts2_size)
            density_after = merged_trans / float(merged_size)
            expected_density = (ts1_trans + ts2_trans) / float(max(merged_size, 1))
            density_change = density_after - expected_density
            return float(density_change)
        except Exception:
            return 0.0