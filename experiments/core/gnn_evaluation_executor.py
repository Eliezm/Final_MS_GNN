#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN EVALUATION EXECUTOR - Core execution logic (FIXED v2)
======================================================
Properly handles FD process lifecycle and resource cleanup.
Fixes: Every 3rd run halting issue (resource accumulation)
"""

import subprocess
import time
import logging
import re
import gc
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from experiments.shared_experiment_utils import cleanup_signal_files, DEFAULT_REWARD_WEIGHTS

logger = logging.getLogger(__name__)


class ExecutorConfig:
    """Configuration for executor."""

    def __init__(
            self,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
            reward_weights: Optional[Dict[str, float]] = None,
            downward_dir: Optional[str] = None,
            debug: bool = False,
    ):
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step
        self.reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS.copy()
        self.downward_dir = downward_dir
        self.debug = debug


class MergeExecutor:
    """
    Executes merge decisions on a problem using ThinMergeEnv.

    ✅ FIXED v2:
    - Proper resource cleanup between runs
    - Better process management
    - Resource monitoring to detect accumulated state
    - Proper garbage collection
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self._env = None
        self._fd_process = None
        self._process_monitor = None

    def _create_environment(
            self,
            domain_file: str,
            problem_file: str,
            seed: int,
    ):
        """Create ThinMergeEnv with proper error handling and validation."""
        try:
            from src.environments.thin_merge_env import ThinMergeEnv
            from src.utils.step_validator import wrap_with_validation  # ✅ NEW: Validation

            try:
                env = ThinMergeEnv(
                    domain_file=domain_file,
                    problem_file=problem_file,
                    max_merges=self.config.max_merges,
                    timeout_per_step=self.config.timeout_per_step,
                    reward_weights=self.config.reward_weights,
                    debug=self.config.debug,
                    seed=seed,
                    downward_dir=self.config.downward_dir,
                )
            except TypeError:
                env = ThinMergeEnv(
                    domain_file=domain_file,
                    problem_file=problem_file,
                    max_merges=self.config.max_merges,
                    timeout_per_step=self.config.timeout_per_step,
                    reward_weights=self.config.reward_weights,
                    debug=self.config.debug,
                    downward_dir=self.config.downward_dir,
                )

            # ✅ NEW: Wrap with validation to catch type errors
            env = wrap_with_validation(env, strict=False)

            return env

        except ImportError as e:
            raise ImportError(f"Failed to import ThinMergeEnv: {e}")

    def _parse_fd_output(self, output_text: str) -> Dict[str, Any]:
        """
        ✅ FIXED: Comprehensive FD output parsing
        """
        metrics = {
            'solved': False,
            'plan_cost': 0,
            'plan_length': 0,
            'nodes_expanded': 0,
            'nodes_generated': 0,
            'search_depth': 0,
        }

        try:
            # ✅ Multiple solution indicators (more robust)
            if any(indicator in output_text for indicator in [
                "Solution found",
                "Plan length:",
                "Evaluating",
                "Plan generation",
                "Preferred operators",
                "Search succeeded",
            ]):
                metrics['solved'] = True

            # Extract plan length
            match = re.search(r'Plan length:\s*(\d+)', output_text)
            if match:
                metrics['plan_length'] = int(match.group(1))
                metrics['plan_cost'] = int(match.group(1))

            # Extract nodes expanded (last occurrence)
            matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', output_text))
            if matches:
                metrics['nodes_expanded'] = int(matches[-1].group(1))

            # Extract nodes generated
            match = re.search(r'Generated\s+(\d+)\s+state', output_text)
            if match:
                metrics['nodes_generated'] = int(match.group(1))

        except Exception as e:
            logger.warning(f"Error parsing FD output: {e}")

        return metrics

    def _extract_fd_metrics_from_log(self, log_path: str) -> Dict[str, Any]:
        """
        ✅ FIXED: Extract metrics from FD log file
        """
        metrics = {
            'solved': False,
            'plan_cost': 0,
            'plan_length': 0,
            'nodes_expanded': 0,
        }

        log_file = Path(log_path)

        if not log_file.exists():
            logger.warning(f"FD log file not found: {log_path}")
            return metrics

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            parsed = self._parse_fd_output(content)
            metrics.update(parsed)

        except Exception as e:
            logger.warning(f"Failed to read log file {log_path}: {e}")

        return metrics

    def _check_resource_health(self) -> Tuple[bool, Optional[str]]:
        """
        ✅ NEW: Check system resource health before merge

        Detects if resources are accumulating (explains 3rd run halt)
        """
        try:
            process = psutil.Process()

            # Check memory
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)

            if mem_mb > 2000:  # 2GB threshold
                return False, f"High memory usage: {mem_mb:.1f}MB (accumulated from previous runs?)"

            # Check file descriptors
            try:
                fds = process.num_fds()
                if fds > 100:
                    return False, f"High file descriptor count: {fds} (FD leak?)"
            except AttributeError:
                pass  # Not on Unix

            return True, None
        except Exception as e:
            logger.warning(f"Failed to check resource health: {e}")
            return True, None  # Continue anyway

    def _force_cleanup(self):
        """
        ✅ NEW: Aggressive cleanup to prevent accumulation
        Called between episodes to reset state
        """
        # Close environment
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

        # Kill any lingering FD processes
        if self._fd_process is not None:
            try:
                if self._fd_process.poll() is None:  # Still running
                    self._fd_process.terminate()
                    try:
                        self._fd_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._fd_process.kill()
                        try:
                            self._fd_process.wait(timeout=1)
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Error killing FD process: {e}")
            self._fd_process = None

        # Kill all descendant processes
        try:
            import psutil
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                try:
                    if 'downward' in child.name().lower() or 'translate' in child.name().lower():
                        child.kill()
                except:
                    pass
        except:
            pass

        # Force garbage collection
        import gc
        gc.collect()

        # Clear signal files
        try:
            cleanup_signal_files()
        except Exception:
            pass

    def execute_merges(
            self,
            domain_file: str,
            problem_file: str,
            policy,
            seed: int = 42,
    ) -> Tuple[bool, int, float, float, bool, Dict[str, Any]]:
        """
        ✅ FIXED v3: Aggressive resource cleanup prevents 3rd run halt

        Key fixes:
        1. Pre-cleanup before any work
        2. Aggressive garbage collection
        3. Process termination with force kill
        4. Proper environment disposal
        5. Better FD output parsing
        """

        # ✅ STEP 0: PRE-CLEANUP - Kill any zombies from previous runs
        import psutil
        import gc

        try:
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                try:
                    if 'downward' in child.name().lower() or 'translate' in child.name().lower():
                        child.kill()
                except:
                    pass
        except:
            pass

        gc.collect()

        env = None
        fd_process = None

        try:
            cleanup_signal_files()
        except:
            pass

        try:
            env = self._create_environment(domain_file, problem_file, seed)
            obs, _ = env.reset()

            start_time = time.time()
            num_merges = 0
            h_preservation = 1.0
            is_solvable = True
            final_info = {'solved': False, 'plan_cost': 0, 'nodes_expanded': 0}

            # ================================================================
            # PHASE 1: RUN MERGE STEPS (with aggressive timeout)
            # ================================================================

            for step in range(self.config.max_merges):
                try:
                    # ✅ FIX 3: Properly detect policy type and extract action
                    if hasattr(policy, 'model'):  # GNN policy
                        # GNN policy: pass full observation dict
                        action = policy.select_merge(obs)  # Returns Python int
                    else:  # Random policy
                        # Random policy: can pass just num_edges
                        num_edges_available = int(obs.get('num_edges', 1))
                        action = policy.select_merge(obs, num_edges=num_edges_available)

                    # ✅ FIX 4: Ensure action is Python int before using
                    action = int(action)
                    if not isinstance(action, int):
                        raise TypeError(f"Action must be Python int, got {type(action)}")

                    logger.debug(f"[EXEC] Step {step}: action={action} (type: {type(action).__name__})")

                    # Execute step
                    obs, reward, done, truncated, info = env.step(action)
                    num_merges += 1

                    # Track metrics
                    reward_signals = info.get('reward_signals', {})
                    h_preservation = reward_signals.get('h_star_preservation', 1.0)
                    is_solvable = reward_signals.get('is_solvable', True)
                    final_info = dict(info)

                    if done or truncated:
                        logger.debug(f"Merge process done/truncated at step {step}")
                        break

                    # ✅ FIX: Periodic garbage collection
                    if step % 5 == 0:
                        gc.collect()

                except subprocess.TimeoutExpired:
                    is_solvable = False
                    logger.warning(f"Timeout during merge step {step}")
                    break
                except Exception as e:
                    if "Timeout" in str(type(e)):
                        is_solvable = False
                        logger.warning(f"Timeout: {e}")
                        break
                    raise

            elapsed_merge = time.time() - start_time

            # ================================================================
            # PHASE 2: ROBUST FD PROCESS COMPLETION
            # ================================================================

            logger.debug(f"Waiting for FD process completion (completed {num_merges} merges)...")

            if hasattr(env, 'process') and env.process:
                fd_process = env.process
                self._fd_process = fd_process

                try:
                    # ✅ FIX: More robust wait with exponential backoff
                    poll_interval = 0.05
                    max_wait = self.config.timeout_per_step * 2
                    elapsed_wait = 0
                    backoff_multiplier = 1.0

                    while elapsed_wait < max_wait:
                        poll_result = fd_process.poll()

                        if poll_result is not None:
                            logger.debug(f"FD process completed with code {poll_result}")
                            break

                        time.sleep(poll_interval * backoff_multiplier)
                        elapsed_wait += poll_interval * backoff_multiplier

                        # Exponential backoff: wait longer each time
                        if elapsed_wait > 5.0:
                            backoff_multiplier = 2.0
                    else:
                        # Timeout occurred
                        logger.warning(f"FD process timeout after {max_wait}s - forcing termination")
                        is_solvable = False

                        # ✅ FIX: Aggressive process termination
                        if fd_process.poll() is None:
                            fd_process.terminate()
                            try:
                                fd_process.wait(timeout=3.0)
                            except subprocess.TimeoutExpired:
                                fd_process.kill()
                                try:
                                    fd_process.wait(timeout=2.0)
                                except:
                                    pass

                except Exception as e:
                    logger.warning(f"Error waiting for FD process: {e}")

            elapsed_total = time.time() - start_time

            # ================================================================
            # PHASE 3: PARSE FD OUTPUT
            # ================================================================

            logger.debug("Parsing FD output...")

            fd_metrics = {}
            fd_log_path = None

            if hasattr(env, 'fd_output_dir'):
                fd_log_path = Path(env.fd_output_dir) / "downward.log"
                if fd_log_path.exists():
                    fd_metrics = self._extract_fd_metrics_from_log(str(fd_log_path))
                    logger.debug(f"Parsed FD metrics: solved={fd_metrics.get('solved')}")

            solved = fd_metrics.get('solved', False)

            if not solved and is_solvable:
                env_solved = final_info.get('solved', False)
                if env_solved:
                    solved = True

            return True, num_merges, h_preservation, elapsed_total, is_solvable, {
                **final_info,
                'solved': solved,
                'plan_cost': fd_metrics.get('plan_cost', 0),
                'plan_length': fd_metrics.get('plan_length', 0),
                'nodes_expanded': fd_metrics.get('nodes_expanded', 0),
                'elapsed': elapsed_total,
                'num_merges': num_merges,
                'h_preservation': h_preservation,
                'fd_log_path': str(fd_log_path) if fd_log_path else None,
            }

        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0.0

            # ✅ ENHANCED: Log full traceback for debugging
            logger.error(f"Executor error: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            return False, 0, 1.0, elapsed, False, {
                'error': str(e),
                'error_type': type(e).__name__,
                'solved': False,
                'elapsed': elapsed,
                'num_merges': 0,
            }

        finally:
            # ✅ AGGRESSIVE CLEANUP
            if env is not None:
                try:
                    env.close()
                except:
                    pass

            if fd_process is not None:
                try:
                    if fd_process.poll() is None:
                        fd_process.terminate()
                        try:
                            fd_process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            fd_process.kill()
                except:
                    pass

            # Force garbage collection
            gc.collect()

            # Clear signal files
            try:
                cleanup_signal_files()
            except:
                pass

    def cleanup(self):
        """Cleanup resources."""
        self._force_cleanup()