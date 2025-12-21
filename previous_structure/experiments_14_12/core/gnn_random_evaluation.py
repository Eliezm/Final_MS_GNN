# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# GNN AND RANDOM MERGE POLICY EVALUATION MODULE
# ==============================================
# Comprehensive evaluation of GNN-guided and random merge strategies
# with A* search integration.
#
# This module provides:
# ✓ Random merge strategy implementation
# ✓ GNN policy evaluation (trained models)
# ✓ Fast Downward integration and output parsing
# ✓ Detailed metrics collection (25+ metrics per run)
# ✓ Seamless compatibility with evaluation.py framework
# ✓ Analysis and visualization ready results
#
# USAGE:
#     from experiments.core.gnn_random_evaluation import GNNRandomEvaluationFramework
#
#     framework = GNNRandomEvaluationFramework(
#         model_path="results/blocksworld_exp_1/model.zip",
#         domain_file="benchmarks/blocksworld/domain.pddl",
#         problem_files=[...],
#         output_dir="evaluation_results"
#     )
#
#     gnn_results, random_results = framework.evaluate()
# """
#
# import sys
# import os
# import json
# import glob
# import random
# import subprocess
# import tempfile
# import time
# import re
# import logging
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
# from collections import defaultdict
# from datetime import datetime
# import numpy as np
# from tqdm import tqdm
#
# PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
# sys.path.insert(0, str(PROJECT_ROOT))
#
# # ✅ CHANGE: Use TYPE_CHECKING to avoid circular import
# if TYPE_CHECKING:
#     from experiments.core.evaluation import DetailedMetrics
#
# from experiments.shared_experiment_utils import cleanup_signal_files, DEFAULT_REWARD_WEIGHTS
# from src.environments.thin_merge_env import ThinMergeEnv
#
#
#
#
# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)-8s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler("gnn_random_evaluation.log", encoding='utf-8'),
#     ]
# )
# logger = logging.getLogger(__name__)
#
#
# # ============================================================================
# # RANDOM MERGE STRATEGY
# # ============================================================================
#
# class RandomMergePolicy:
#     """
#     Random merge strategy - selects merge edges randomly.
#
#     Used as baseline comparison against GNN-guided strategy.
#     """
#
#     def __init__(self, seed: int = 42):
#         """
#         Initialize random merge policy.
#
#         Args:
#             seed: Random seed for reproducibility
#         """
#         self.rng = np.random.RandomState(seed)
#         self.merge_count = 0
#
#     def select_merge(self, edge_logits: np.ndarray, num_edges: int) -> int:
#         """
#         Select a random merge edge.
#
#         Args:
#             edge_logits: Unused (for compatibility with GNN)
#             num_edges: Number of available merge edges
#
#         Returns:
#             Random edge index to merge
#         """
#         if num_edges == 0:
#             return 0
#         action = self.rng.randint(0, num_edges)
#         self.merge_count += 1
#         return action
#
#     def reset(self):
#         """Reset merge counter for new problem."""
#         self.merge_count = 0
#
#
# # ============================================================================
# # FAST DOWNWARD OUTPUT PARSER
# # ============================================================================
#
# class FastDownwardOutputParser:
#     """
#     Parse Fast Downward output to extract comprehensive metrics.
#
#     Handles:
#     - Plan cost and length
#     - Search statistics (expansions, generations, depth)
#     - Timing breakdown
#     - M&S specific metrics
#     """
#
#     @staticmethod
#     def parse_search_output(output_text: str) -> Dict[str, Any]:
#         """
#         Extract comprehensive metrics from FD search output.
#
#         Args:
#             output_text: Combined stdout + stderr from FD process
#
#         Returns:
#             Dictionary of parsed metrics
#         """
#         metrics = {
#             'solved': False,
#             'plan_cost': 0,
#             'plan_length': 0,
#             'nodes_expanded': 0,
#             'nodes_generated': 0,
#             'search_depth': 0,
#             'branching_factor': 1.0,
#             'search_time': 0.0,
#             'peak_memory_kb': 0,
#             'plan_found': False,
#         }
#
#         try:
#             # Check if solution was found
#             if "Solution found" in output_text or "Plan length:" in output_text:
#                 metrics['solved'] = True
#                 metrics['plan_found'] = True
#
#             # Extract plan length/cost
#             match = re.search(r'Plan length:\s*(\d+)', output_text)
#             if match:
#                 metrics['plan_length'] = int(match.group(1))
#                 metrics['plan_cost'] = int(match.group(1))
#
#             # Extract nodes expanded (last occurrence is most relevant)
#             matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', output_text))
#             if matches:
#                 metrics['nodes_expanded'] = int(matches[-1].group(1))
#
#             # Extract nodes generated
#             match = re.search(r'Generated\s+(\d+)\s+state', output_text)
#             if match:
#                 metrics['nodes_generated'] = int(match.group(1))
#
#             # Extract search depth
#             match = re.search(r'Search depth:\s*(\d+)', output_text)
#             if match:
#                 metrics['search_depth'] = int(match.group(1))
#
#             # Extract branching factor
#             match = re.search(r'Branching factor:\s*([\d.]+)', output_text)
#             if match:
#                 metrics['branching_factor'] = float(match.group(1))
#
#             # Extract search time
#             match = re.search(r'Search time:\s*([\d.]+)s', output_text)
#             if match:
#                 metrics['search_time'] = float(match.group(1))
#
#             # Extract peak memory
#             match = re.search(r'Peak memory:\s*(\d+)\s*KB', output_text)
#             if match:
#                 metrics['peak_memory_kb'] = int(match.group(1))
#
#         except Exception as e:
#             logger.warning(f"Error parsing FD output: {e}")
#
#         return metrics
#
#     @staticmethod
#     def parse_fd_log_file(log_path: str) -> Dict[str, Any]:
#         """
#         Parse complete FD log file.
#
#         Args:
#             log_path: Path to downward.log file
#
#         Returns:
#             Parsed metrics dictionary
#         """
#         metrics = {
#             'solved': False,
#             'plan_cost': 0,
#             'plan_length': 0,
#             'nodes_expanded': 0,
#             'search_time': 0.0,
#         }
#
#         try:
#             if Path(log_path).exists():
#                 with open(log_path, 'r') as f:
#                     content = f.read()
#                 metrics.update(FastDownwardOutputParser.parse_search_output(content))
#         except Exception as e:
#             logger.warning(f"Failed to parse log file {log_path}: {e}")
#
#         return metrics
#
#
# # ============================================================================
# # GNN POLICY EVALUATOR
# # ============================================================================
#
# class GNNPolicyEvaluator:
#     """
#     Evaluate a trained GNN policy on test problems.
#
#     Process:
#     1. Load trained GNN model (PPO from stable-baselines3)
#     2. Run merge-and-shrink with GNN decisions to build abstraction
#     3. Execute A* search with the built heuristic
#     4. Extract solution metrics and h* preservation
#     """
#
#     def __init__(
#             self,
#             model_path: str,
#             downward_dir: Optional[str] = None,
#             max_merges: int = 50,
#             timeout_per_step: float = 120.0,
#     ):
#         """
#         Initialize GNN policy evaluator.
#
#         Args:
#             model_path: Path to trained model.zip file
#             downward_dir: Path to Fast Downward installation
#             max_merges: Maximum merge steps
#             timeout_per_step: Timeout per merge step (seconds)
#         """
#         self.model_path = model_path
#         self.downward_dir = downward_dir
#         self.max_merges = max_merges
#         self.timeout_per_step = timeout_per_step
#
#         logger.info(f"Loading GNN model from: {model_path}")
#         try:
#             from stable_baselines3 import PPO
#             self.model = PPO.load(model_path)
#             logger.info("✅ Model loaded successfully")
#         except Exception as e:
#             logger.error(f"Failed to load model: {e}")
#             raise
#
#     def run_problem(
#             self,
#             domain_file: str,
#             problem_file: str,
#             seed: int = 42,
#     ) -> DetailedMetrics:
#         """
#         Run GNN-guided merge-and-shrink with A* search.
#
#         Args:
#             domain_file: Path to domain.pddl
#             problem_file: Path to problem.pddl
#             seed: Random seed for reproducibility
#
#         Returns:
#             DetailedMetrics with solution cost, nodes expanded, h* preservation, etc.
#         """
#         from experiments.core.evaluation import DetailedMetrics
#
#         problem_name = os.path.basename(problem_file)
#         logger.info(f"\n[GNN] Evaluating: {problem_name}")
#
#         metrics = DetailedMetrics(
#             problem_name=problem_name,
#             planner_name="GNN"
#         )
#
#         env = None
#         try:
#             cleanup_signal_files()
#         except Exception:
#             pass
#
#         try:
#             from src.environments.thin_merge_env import ThinMergeEnv
#
#             # Create environment (runs M&S with GNN decisions)
#             env = ThinMergeEnv(
#                 domain_file=domain_file,
#                 problem_file=problem_file,
#                 max_merges=self.max_merges,
#                 timeout_per_step=self.timeout_per_step,
#                 reward_weights=DEFAULT_REWARD_WEIGHTS.copy(),
#                 debug=False,
#                 downward_dir=self.downward_dir,
#             )
#
#             obs, _ = env.reset()
#             start_time = time.time()
#             h_preservation = 1.0
#             is_solvable = True
#             eval_steps = 0
#
#             logger.info(f"[GNN] PHASE 1: Building abstraction with GNN decisions...")
#
#             # PHASE 1: Run merge decisions
#             for step in range(self.max_merges):
#                 try:
#                     # Get GNN decision
#                     action, _ = self.model.predict(obs, deterministic=True)
#
#                     # Execute merge
#                     obs, reward, done, truncated, info = env.step(int(action))
#                     eval_steps += 1
#
#                     # Track metrics
#                     reward_signals = info.get('reward_signals', {})
#                     h_preservation = reward_signals.get('h_star_preservation', 1.0)
#                     is_solvable = reward_signals.get('is_solvable', True)
#
#                     logger.debug(f"  Merge step {step}: h*={h_preservation:.3f}")
#
#                     if done or truncated:
#                         logger.info(f"[GNN] Merge process completed at step {step}")
#                         break
#
#                 except subprocess.TimeoutExpired:
#                     is_solvable = False
#                     logger.warning(f"  ⏱️ Timeout during merge step {step}")
#                     break
#                 except Exception as e:
#                     logger.error(f"  ❌ Step failed: {e}")
#                     break
#
#             # PHASE 2: Wait for A* search completion
#             logger.info(f"[GNN] PHASE 2: Waiting for A* search to complete...")
#
#             if env.process:
#                 try:
#                     env.process.wait(timeout=300)  # 5 minute timeout
#                     logger.info(f"[GNN] FD process completed")
#                 except subprocess.TimeoutExpired:
#                     logger.warning(f"[GNN] FD process timeout")
#                     is_solvable = False
#                     if env.process.poll() is None:
#                         env.process.terminate()
#                         try:
#                             env.process.wait(timeout=5)
#                         except:
#                             pass
#
#             # PHASE 3: Extract solution metrics
#             logger.info(f"[GNN] PHASE 3: Extracting solution metrics...")
#
#             fd_log_path = env.fd_output_dir / "downward.log" if hasattr(env, 'fd_output_dir') else None
#             fd_metrics = FastDownwardOutputParser.parse_fd_log_file(str(fd_log_path)) if fd_log_path else {}
#
#             wall_clock_time = time.time() - start_time
#
#             # Populate metrics
#             metrics.solved = fd_metrics.get('solved', False)
#             metrics.plan_cost = fd_metrics.get('plan_cost', 0)
#             metrics.plan_length = fd_metrics.get('plan_length', 0)
#             metrics.nodes_expanded = fd_metrics.get('nodes_expanded', 0)
#             metrics.search_time = fd_metrics.get('search_time', 0.0)
#             metrics.wall_clock_time = wall_clock_time
#             metrics.h_star_preservation = h_preservation
#             metrics.merge_episodes = eval_steps
#             metrics.solvable = is_solvable
#             metrics.timeout_occurred = False
#
#             if metrics.solved:
#                 logger.info(f"  ✅ SOLVED in {wall_clock_time:.2f}s")
#                 logger.info(f"     Plan cost: {metrics.plan_cost}")
#                 logger.info(f"     Nodes expanded: {metrics.nodes_expanded}")
#                 logger.info(f"     Merge steps: {eval_steps}")
#                 logger.info(f"     h* preservation: {h_preservation:.3f}")
#             else:
#                 logger.warning(f"  ❌ NOT SOLVED after {wall_clock_time:.2f}s")
#                 logger.warning(f"     Merge steps: {eval_steps}")
#                 logger.warning(f"     Is solvable: {is_solvable}")
#
#         except Exception as e:
#             logger.error(f"  ❌ ERROR: {e}")
#             metrics.solved = False
#             metrics.error_type = "exception"
#             metrics.error_message = str(e)[:500]
#
#         finally:
#             if env is not None:
#                 try:
#                     env.close()
#                 except Exception:
#                     pass
#
#         return metrics
#
#     def evaluate_problems(
#             self,
#             domain_file: str,
#             problem_files: List[str],
#             num_runs_per_problem: int = 1,
#     ) -> List['DetailedMetrics']:
#         """
#         Evaluate GNN on multiple problems.
#
#         Args:
#             domain_file: Path to domain.pddl
#             problem_files: List of problem.pddl paths
#             num_runs_per_problem: Number of runs per problem
#
#         Returns:
#             List of DetailedMetrics for all runs
#         """
#         all_results = []
#
#         for problem_file in tqdm(problem_files, desc="GNN Evaluation", unit="problem"):
#             for run_idx in range(num_runs_per_problem):
#                 seed = 42 + run_idx * 10000
#                 result = self.run_problem(domain_file, problem_file, seed=seed)
#                 all_results.append(result)
#
#         return all_results
#
#
# # ============================================================================
# # RANDOM MERGE EVALUATOR
# # ============================================================================
#
# class RandomMergeEvaluator:
#     """
#     Evaluate random merge baseline on test problems.
#
#     Process:
#     1. Run merge-and-shrink with RANDOM merge decisions
#     2. Execute A* search with the built abstraction heuristic
#     3. Collect solution metrics
#     """
#
#     def __init__(
#             self,
#             downward_dir: Optional[str] = None,
#             max_merges: int = 50,
#             timeout_per_step: float = 120.0,
#     ):
#         """
#         Initialize random merge evaluator.
#
#         Args:
#             downward_dir: Path to Fast Downward installation
#             max_merges: Maximum merge steps
#             timeout_per_step: Timeout per merge step (seconds)
#         """
#         self.downward_dir = downward_dir
#         self.max_merges = max_merges
#         self.timeout_per_step = timeout_per_step
#
#     def run_problem(
#             self,
#             domain_file: str,
#             problem_file: str,
#             seed: int = 42,
#     ) -> DetailedMetrics:
#         """
#         Run random merge strategy with A* search.
#
#         Args:
#             domain_file: Path to domain.pddl
#             problem_file: Path to problem.pddl
#             seed: Random seed for reproducibility
#
#         Returns:
#             DetailedMetrics with solution metrics
#         """
#         from experiments.core.evaluation import DetailedMetrics
#
#         problem_name = os.path.basename(problem_file)
#         logger.info(f"\n[RANDOM] Evaluating: {problem_name}")
#
#         metrics = DetailedMetrics(
#             problem_name=problem_name,
#             planner_name="Random"
#         )
#
#         random_policy = RandomMergePolicy(seed=seed)
#         env = None
#
#         try:
#             cleanup_signal_files()
#         except Exception:
#             pass
#
#         try:
#             from src.environments.thin_merge_env import ThinMergeEnv
#
#             # Create environment
#             env = ThinMergeEnv(
#                 domain_file=domain_file,
#                 problem_file=problem_file,
#                 max_merges=self.max_merges,
#                 timeout_per_step=self.timeout_per_step,
#                 reward_weights=DEFAULT_REWARD_WEIGHTS.copy(),
#                 debug=False,
#                 downward_dir=self.downward_dir,
#             )
#
#             obs, _ = env.reset()
#             start_time = time.time()
#             eval_steps = 0
#             h_preservation = 1.0
#             is_solvable = True
#
#             logger.info(f"[RANDOM] PHASE 1: Building abstraction with random decisions...")
#
#             # PHASE 1: Run random merge decisions
#             for step in range(self.max_merges):
#                 try:
#                     if not isinstance(obs, dict):
#                         logger.error("Observation is not a dict")
#                         break
#
#                     num_edges = obs.get('num_edges', 0)
#                     if num_edges <= 0:
#                         logger.info(f"[RANDOM] No more merges possible")
#                         break
#
#                     # Random policy picks a random edge
#                     action = random_policy.select_merge(
#                         np.zeros(num_edges),
#                         num_edges
#                     )
#
#                     # Execute merge
#                     obs, reward, done, truncated, info = env.step(int(action))
#                     eval_steps += 1
#
#                     reward_signals = info.get('reward_signals', {})
#                     h_preservation = reward_signals.get('h_star_preservation', 1.0)
#                     is_solvable = reward_signals.get('is_solvable', True)
#
#                     if done or truncated:
#                         logger.info(f"[RANDOM] Merge process completed at step {step}")
#                         break
#
#                 except subprocess.TimeoutExpired:
#                     is_solvable = False
#                     logger.warning(f"  ⏱️ Timeout during merge step {step}")
#                     break
#                 except Exception as e:
#                     logger.error(f"  ❌ Step failed: {e}")
#                     break
#
#             # PHASE 2: Wait for A* search completion
#             logger.info(f"[RANDOM] PHASE 2: Waiting for A* search to complete...")
#
#             if env.process:
#                 try:
#                     env.process.wait(timeout=300)
#                     logger.info(f"[RANDOM] FD process completed")
#                 except subprocess.TimeoutExpired:
#                     logger.warning(f"[RANDOM] FD process timeout")
#                     is_solvable = False
#                     if env.process.poll() is None:
#                         env.process.terminate()
#                         try:
#                             env.process.wait(timeout=5)
#                         except:
#                             pass
#
#             # PHASE 3: Extract solution metrics
#             logger.info(f"[RANDOM] PHASE 3: Extracting solution metrics...")
#
#             fd_log_path = env.fd_output_dir / "downward.log" if hasattr(env, 'fd_output_dir') else None
#             fd_metrics = FastDownwardOutputParser.parse_fd_log_file(str(fd_log_path)) if fd_log_path else {}
#
#             wall_clock_time = time.time() - start_time
#
#             # Populate metrics
#             metrics.solved = fd_metrics.get('solved', False)
#             metrics.plan_cost = fd_metrics.get('plan_cost', 0)
#             metrics.plan_length = fd_metrics.get('plan_length', 0)
#             metrics.nodes_expanded = fd_metrics.get('nodes_expanded', 0)
#             metrics.search_time = fd_metrics.get('search_time', 0.0)
#             metrics.wall_clock_time = wall_clock_time
#             metrics.h_star_preservation = h_preservation
#             metrics.merge_episodes = eval_steps
#             metrics.solvable = is_solvable
#             metrics.timeout_occurred = False
#
#             if metrics.solved:
#                 logger.info(f"  ✅ SOLVED in {wall_clock_time:.2f}s")
#                 logger.info(f"     Plan cost: {metrics.plan_cost}")
#                 logger.info(f"     Nodes expanded: {metrics.nodes_expanded}")
#                 logger.info(f"     Merge steps: {eval_steps}")
#             else:
#                 logger.warning(f"  ❌ NOT SOLVED after {wall_clock_time:.2f}s")
#                 logger.warning(f"     Merge steps: {eval_steps}")
#
#         except Exception as e:
#             logger.error(f"  ❌ ERROR: {e}")
#             metrics.solved = False
#             metrics.error_type = "exception"
#             metrics.error_message = str(e)[:500]
#
#         finally:
#             if env is not None:
#                 try:
#                     env.close()
#                 except Exception:
#                     pass
#
#         return metrics
#
#     def evaluate_problems(
#             self,
#             domain_file: str,
#             problem_files: List[str],
#             num_runs_per_problem: int = 1,
#     ) -> List['DetailedMetrics']:
#         """
#         Evaluate random on multiple problems.
#
#         Args:
#             domain_file: Path to domain.pddl
#             problem_files: List of problem.pddl paths
#             num_runs_per_problem: Number of runs per problem
#
#         Returns:
#             List of DetailedMetrics for all runs
#         """
#         all_results = []
#
#         for problem_file in tqdm(problem_files, desc="Random Evaluation", unit="problem"):
#             for run_idx in range(num_runs_per_problem):
#                 seed = 42 + run_idx * 10000
#                 result = self.run_problem(domain_file, problem_file, seed=seed)
#                 all_results.append(result)
#
#         return all_results
#
#
# # ============================================================================
# # MAIN ORCHESTRATOR
# # ============================================================================
#
# class GNNRandomEvaluationFramework:
#     """
#     Orchestrate evaluation of both GNN and Random merge strategies.
#
#     Returns results in format compatible with evaluation.py framework.
#     """
#
#     def __init__(
#             self,
#             model_path: str,
#             domain_file: str,
#             problem_files: List[str],
#             output_dir: str = "evaluation_results",
#             num_runs_per_problem: int = 1,
#             downward_dir: Optional[str] = None,
#             max_merges: int = 50,
#             timeout_per_step: float = 120.0,
#     ):
#         """
#         Initialize evaluation framework.
#
#         Args:
#             model_path: Path to trained model.zip
#             domain_file: Path to domain.pddl
#             problem_files: List of problem.pddl paths
#             output_dir: Output directory for results
#             num_runs_per_problem: Number of runs per problem
#             downward_dir: Path to Fast Downward
#             max_merges: Maximum merge steps
#             timeout_per_step: Timeout per merge step
#         """
#         self.model_path = model_path
#         self.domain_file = domain_file
#         self.problem_files = problem_files
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.num_runs_per_problem = num_runs_per_problem
#         self.downward_dir = downward_dir
#         self.max_merges = max_merges
#         self.timeout_per_step = timeout_per_step
#
#         self.all_results: List[DetailedMetrics] = []
#
#     def evaluate(
#             self,
#             include_gnn: bool = True,
#             include_random: bool = True,
#     ) -> Tuple[List['DetailedMetrics'], List['DetailedMetrics']]:
#         """
#         Execute full evaluation pipeline.
#
#         Args:
#             include_gnn: Whether to evaluate GNN policy
#             include_random: Whether to evaluate random baseline
#
#         Returns:
#             Tuple of (gnn_results, random_results)
#         """
#         logger.info("\n" + "=" * 100)
#         logger.info("GNN vs RANDOM MERGE STRATEGY EVALUATION")
#         logger.info("=" * 100)
#         logger.info(f"\n📋 Configuration:")
#         logger.info(f"   Model: {self.model_path}")
#         logger.info(f"   Domain: {self.domain_file}")
#         logger.info(f"   Test problems: {len(self.problem_files)}")
#         logger.info(f"   Runs per problem: {self.num_runs_per_problem}")
#
#         gnn_results = []
#         random_results = []
#
#         # Evaluate GNN
#         if include_gnn:
#             logger.info("\n" + "-" * 100)
#             logger.info("PHASE 1: GNN-GUIDED MERGE-AND-SHRINK + A* SEARCH")
#             logger.info("-" * 100 + "\n")
#
#             gnn_evaluator = GNNPolicyEvaluator(
#                 model_path=self.model_path,
#                 downward_dir=self.downward_dir,
#                 max_merges=self.max_merges,
#                 timeout_per_step=self.timeout_per_step,
#             )
#
#             gnn_results = gnn_evaluator.evaluate_problems(
#                 domain_file=self.domain_file,
#                 problem_files=self.problem_files,
#                 num_runs_per_problem=self.num_runs_per_problem,
#             )
#
#             self.all_results.extend(gnn_results)
#
#         # Evaluate Random
#         if include_random:
#             logger.info("\n" + "-" * 100)
#             logger.info("PHASE 2: RANDOM MERGE-AND-SHRINK + A* SEARCH")
#             logger.info("-" * 100 + "\n")
#
#             random_evaluator = RandomMergeEvaluator(
#                 downward_dir=self.downward_dir,
#                 max_merges=self.max_merges,
#                 timeout_per_step=self.timeout_per_step,
#             )
#
#             random_results = random_evaluator.evaluate_problems(
#                 domain_file=self.domain_file,
#                 problem_files=self.problem_files,
#                 num_runs_per_problem=self.num_runs_per_problem,
#             )
#
#             self.all_results.extend(random_results)
#
#         logger.info("\n" + "=" * 100)
#         logger.info("✅ EVALUATION COMPLETE")
#         logger.info("=" * 100)
#
#         return gnn_results, random_results
#
#     def get_all_results(self) -> List['DetailedMetrics']:
#         """
#         Get all evaluation results combined.
#
#         Returns:
#             List of all DetailedMetrics
#         """
#         return self.all_results
#
#     def to_summary(self) -> Dict[str, Any]:
#         """
#         Convert results to summary format.
#
#         Returns:
#             Summary dictionary with statistics
#         """
#         from experiments.core.evaluation import ComparisonAnalyzer
#
#         if not self.all_results:
#             return {}
#
#         analyzer = ComparisonAnalyzer(self.all_results)
#
#         summary = {}
#         for planner_name in set(r.planner_name for r in self.all_results):
#             stats = analyzer.get_aggregate_statistics(planner_name)
#             summary[planner_name] = stats.to_dict()
#
#         return summary

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN AND RANDOM MERGE POLICY EVALUATION MODULE
==============================================
Comprehensive evaluation of GNN-guided and random merge strategies
with A* search integration.

This module provides:
✓ Random merge strategy implementation
✓ GNN policy evaluation (trained models)
✓ Fast Downward integration and output parsing
✓ Detailed metrics collection (25+ metrics per run)
✓ Seamless compatibility with evaluation.py framework
✓ Analysis and visualization ready results

USAGE:
    from experiments.core.gnn_random_evaluation import GNNRandomEvaluationFramework

    framework = GNNRandomEvaluationFramework(
        model_path="results/blocksworld_exp_1/model.zip",
        domain_file="benchmarks/blocksworld/domain.pddl",
        problem_files=[...],
        output_dir="evaluation_results"
    )

    gnn_results, random_results = framework.evaluate()
"""

# ✅ FIX 1: Add this at the very top to defer type annotation evaluation
from __future__ import annotations

import sys
import os
import subprocess
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ✅ FIX 2: Keep TYPE_CHECKING import for type checkers (mypy, pyright)
if TYPE_CHECKING:
    from experiments.core.evaluation import DetailedMetrics

from shared_experiment_utils import cleanup_signal_files, DEFAULT_REWARD_WEIGHTS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("gnn_random_evaluation.log", encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# RANDOM MERGE STRATEGY
# ============================================================================

class RandomMergePolicy:
    """
    Random merge strategy - selects merge edges randomly.

    Used as baseline comparison against GNN-guided strategy.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize random merge policy.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.merge_count = 0

    def select_merge(self, edge_logits: np.ndarray, num_edges: int) -> int:
        """
        Select a random merge edge.

        Args:
            edge_logits: Unused (for compatibility with GNN)
            num_edges: Number of available merge edges

        Returns:
            Random edge index to merge
        """
        if num_edges == 0:
            return 0
        action = self.rng.randint(0, num_edges)
        self.merge_count += 1
        return action

    def reset(self):
        """Reset merge counter for new problem."""
        self.merge_count = 0


# ============================================================================
# FAST DOWNWARD OUTPUT PARSER
# ============================================================================

class FastDownwardOutputParser:
    """
    Parse Fast Downward output to extract comprehensive metrics.

    Handles:
    - Plan cost and length
    - Search statistics (expansions, generations, depth)
    - Timing breakdown
    - M&S specific metrics
    """

    @staticmethod
    def parse_search_output(output_text: str) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from FD search output.

        Args:
            output_text: Combined stdout + stderr from FD process

        Returns:
            Dictionary of parsed metrics
        """
        metrics = {
            'solved': False,
            'plan_cost': 0,
            'plan_length': 0,
            'nodes_expanded': 0,
            'nodes_generated': 0,
            'search_depth': 0,
            'branching_factor': 1.0,
            'search_time': 0.0,
            'peak_memory_kb': 0,
            'plan_found': False,
        }

        try:
            # Check if solution was found
            if "Solution found" in output_text or "Plan length:" in output_text:
                metrics['solved'] = True
                metrics['plan_found'] = True

            # Extract plan length/cost
            match = re.search(r'Plan length:\s*(\d+)', output_text)
            if match:
                metrics['plan_length'] = int(match.group(1))
                metrics['plan_cost'] = int(match.group(1))

            # Extract nodes expanded (last occurrence is most relevant)
            matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', output_text))
            if matches:
                metrics['nodes_expanded'] = int(matches[-1].group(1))

            # Extract nodes generated
            match = re.search(r'Generated\s+(\d+)\s+state', output_text)
            if match:
                metrics['nodes_generated'] = int(match.group(1))

            # Extract search depth
            match = re.search(r'Search depth:\s*(\d+)', output_text)
            if match:
                metrics['search_depth'] = int(match.group(1))

            # Extract branching factor
            match = re.search(r'Branching factor:\s*([\d.]+)', output_text)
            if match:
                metrics['branching_factor'] = float(match.group(1))

            # Extract search time
            match = re.search(r'Search time:\s*([\d.]+)s', output_text)
            if match:
                metrics['search_time'] = float(match.group(1))

            # Extract peak memory
            match = re.search(r'Peak memory:\s*(\d+)\s*KB', output_text)
            if match:
                metrics['peak_memory_kb'] = int(match.group(1))

        except Exception as e:
            logger.warning(f"Error parsing FD output: {e}")

        return metrics

    @staticmethod
    def parse_fd_log_file(log_path: str) -> Dict[str, Any]:
        """
        Parse complete FD log file.

        Args:
            log_path: Path to downward.log file

        Returns:
            Parsed metrics dictionary
        """
        metrics = {
            'solved': False,
            'plan_cost': 0,
            'plan_length': 0,
            'nodes_expanded': 0,
            'search_time': 0.0,
        }

        try:
            if Path(log_path).exists():
                with open(log_path, 'r') as f:
                    content = f.read()
                metrics.update(FastDownwardOutputParser.parse_search_output(content))
        except Exception as e:
            logger.warning(f"Failed to parse log file {log_path}: {e}")

        return metrics


# ============================================================================
# GNN POLICY EVALUATOR
# ============================================================================

class GNNPolicyEvaluator:
    """
    Evaluate a trained GNN policy on test problems.

    Process:
    1. Load trained GNN model (PPO from stable-baselines3)
    2. Run merge-and-shrink with GNN decisions to build abstraction
    3. Execute A* search with the built heuristic
    4. Extract solution metrics and h* preservation
    """

    def __init__(
            self,
            model_path: str,
            downward_dir: Optional[str] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
    ):
        """
        Initialize GNN policy evaluator.

        Args:
            model_path: Path to trained model.zip file
            downward_dir: Path to Fast Downward installation
            max_merges: Maximum merge steps
            timeout_per_step: Timeout per merge step (seconds)
        """
        self.model_path = model_path

        # ✅ FIX: Provide explicit default if not specified
        if downward_dir is None:
            downward_dir = str(PROJECT_ROOT / "downward")

        self.downward_dir = downward_dir
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step

        logger.info(f"Loading GNN model from: {model_path}")
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def run_problem(
            self,
            domain_file: str,
            problem_file: str,
            seed: int = 42,
    ) -> DetailedMetrics:  # ✅ FIX 3: String annotation works with __future__ import
        """
        Run GNN-guided merge-and-shrink with A* search.

        Args:
            domain_file: Path to domain.pddl
            problem_file: Path to problem.pddl
            seed: Random seed for reproducibility

        Returns:
            DetailedMetrics with solution cost, nodes expanded, h* preservation, etc.
        """
        # ✅ FIX 4: Import DetailedMetrics locally to avoid circular import at module level
        from experiments.core.evaluation import DetailedMetrics

        problem_name = os.path.basename(problem_file)
        logger.info(f"\n[GNN] Evaluating: {problem_name}")

        metrics = DetailedMetrics(
            problem_name=problem_name,
            planner_name="GNN"
        )

        env = None
        try:
            cleanup_signal_files()
        except Exception:
            pass

        try:
            from src.environments.thin_merge_env import ThinMergeEnv

            # Create environment (runs M&S with GNN decisions)
            env = ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
                reward_weights=DEFAULT_REWARD_WEIGHTS.copy(),
                debug=False,
                downward_dir=self.downward_dir,
            )

            obs, _ = env.reset()
            start_time = time.time()
            h_preservation = 1.0
            is_solvable = True
            eval_steps = 0

            logger.info(f"[GNN] PHASE 1: Building abstraction with GNN decisions...")

            # PHASE 1: Run merge decisions
            for step in range(self.max_merges):
                try:
                    # Get GNN decision
                    action, _ = self.model.predict(obs, deterministic=True)

                    # Execute merge
                    obs, reward, done, truncated, info = env.step(int(action))
                    eval_steps += 1

                    # Track metrics
                    reward_signals = info.get('reward_signals', {})
                    h_preservation = reward_signals.get('h_star_preservation', 1.0)
                    is_solvable = reward_signals.get('is_solvable', True)

                    logger.debug(f"  Merge step {step}: h*={h_preservation:.3f}")

                    if done or truncated:
                        logger.info(f"[GNN] Merge process completed at step {step}")
                        break

                except subprocess.TimeoutExpired:
                    is_solvable = False
                    logger.warning(f"  ⏱️ Timeout during merge step {step}")
                    break
                except Exception as e:
                    logger.error(f"  ❌ Step failed: {e}")
                    break

            # PHASE 2: Wait for A* search completion
            logger.info(f"[GNN] PHASE 2: Waiting for A* search to complete...")

            if env.process:
                try:
                    env.process.wait(timeout=300)  # 5 minute timeout
                    logger.info(f"[GNN] FD process completed")
                except subprocess.TimeoutExpired:
                    logger.warning(f"[GNN] FD process timeout")
                    is_solvable = False
                    if env.process.poll() is None:
                        env.process.terminate()
                        try:
                            env.process.wait(timeout=5)
                        except:
                            pass

            # PHASE 3: Extract solution metrics
            logger.info(f"[GNN] PHASE 3: Extracting solution metrics...")

            fd_log_path = env.fd_output_dir / "downward.log" if hasattr(env, 'fd_output_dir') else None
            fd_metrics = FastDownwardOutputParser.parse_fd_log_file(str(fd_log_path)) if fd_log_path else {}

            wall_clock_time = time.time() - start_time

            # Populate metrics
            metrics.solved = fd_metrics.get('solved', False)
            metrics.plan_cost = fd_metrics.get('plan_cost', 0)
            metrics.plan_length = fd_metrics.get('plan_length', 0)
            metrics.nodes_expanded = fd_metrics.get('nodes_expanded', 0)
            metrics.search_time = fd_metrics.get('search_time', 0.0)
            metrics.wall_clock_time = wall_clock_time
            metrics.h_star_preservation = h_preservation
            metrics.merge_episodes = eval_steps
            metrics.solvable = is_solvable
            metrics.timeout_occurred = False

            if metrics.solved:
                logger.info(f"  ✅ SOLVED in {wall_clock_time:.2f}s")
                logger.info(f"     Plan cost: {metrics.plan_cost}")
                logger.info(f"     Nodes expanded: {metrics.nodes_expanded}")
                logger.info(f"     Merge steps: {eval_steps}")
                logger.info(f"     h* preservation: {h_preservation:.3f}")
            else:
                logger.warning(f"  ❌ NOT SOLVED after {wall_clock_time:.2f}s")
                logger.warning(f"     Merge steps: {eval_steps}")
                logger.warning(f"     Is solvable: {is_solvable}")

        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}")
            metrics.solved = False
            metrics.error_type = "exception"
            metrics.error_message = str(e)[:500]

        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        return metrics

    def evaluate_problems(
            self,
            domain_file: str,
            problem_files: List[str],
            num_runs_per_problem: int = 1,
    ) -> List[DetailedMetrics]:  # ✅ FIX 5: String annotation in List
        """
        Evaluate GNN on multiple problems.

        Args:
            domain_file: Path to domain.pddl
            problem_files: List of problem.pddl paths
            num_runs_per_problem: Number of runs per problem

        Returns:
            List of DetailedMetrics for all runs
        """
        all_results = []

        for problem_file in tqdm(problem_files, desc="GNN Evaluation", unit="problem"):
            for run_idx in range(num_runs_per_problem):
                seed = 42 + run_idx * 10000
                result = self.run_problem(domain_file, problem_file, seed=seed)
                all_results.append(result)

        return all_results


# ============================================================================
# RANDOM MERGE EVALUATOR
# ============================================================================

class RandomMergeEvaluator:
    """
    Evaluate random merge baseline on test problems.

    Process:
    1. Run merge-and-shrink with RANDOM merge decisions
    2. Execute A* search with the built abstraction heuristic
    3. Collect solution metrics
    """

    def __init__(
            self,
            downward_dir: Optional[str] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
    ):
        """
        Initialize random merge evaluator.

        Args:
            downward_dir: Path to Fast Downward installation
            max_merges: Maximum merge steps
            timeout_per_step: Timeout per merge step (seconds)
        """
        self.downward_dir = downward_dir
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step

    def run_problem(
            self,
            domain_file: str,
            problem_file: str,
            seed: int = 42,
    ) -> DetailedMetrics:  # ✅ FIX 6: String annotation
        """
        Run random merge strategy with A* search.

        Args:
            domain_file: Path to domain.pddl
            problem_file: Path to problem.pddl
            seed: Random seed for reproducibility

        Returns:
            DetailedMetrics with solution metrics
        """
        # ✅ FIX 7: Local import to avoid circular import
        from experiments.core.evaluation import DetailedMetrics

        problem_name = os.path.basename(problem_file)
        logger.info(f"\n[RANDOM] Evaluating: {problem_name}")

        metrics = DetailedMetrics(
            problem_name=problem_name,
            planner_name="Random"
        )

        random_policy = RandomMergePolicy(seed=seed)
        env = None

        try:
            cleanup_signal_files()
        except Exception:
            pass

        try:
            from src.environments.thin_merge_env import ThinMergeEnv

            # Create environment
            env = ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
                reward_weights=DEFAULT_REWARD_WEIGHTS.copy(),
                debug=False,
                downward_dir=self.downward_dir,
            )

            obs, _ = env.reset()
            start_time = time.time()
            eval_steps = 0
            h_preservation = 1.0
            is_solvable = True

            logger.info(f"[RANDOM] PHASE 1: Building abstraction with random decisions...")

            # PHASE 1: Run random merge decisions
            for step in range(self.max_merges):
                try:
                    if not isinstance(obs, dict):
                        logger.error("Observation is not a dict")
                        break

                    num_edges = obs.get('num_edges', 0)
                    if num_edges <= 0:
                        logger.info(f"[RANDOM] No more merges possible")
                        break

                    # Random policy picks a random edge
                    action = random_policy.select_merge(
                        np.zeros(num_edges),
                        num_edges
                    )

                    # Execute merge
                    obs, reward, done, truncated, info = env.step(int(action))
                    eval_steps += 1

                    reward_signals = info.get('reward_signals', {})
                    h_preservation = reward_signals.get('h_star_preservation', 1.0)
                    is_solvable = reward_signals.get('is_solvable', True)

                    if done or truncated:
                        logger.info(f"[RANDOM] Merge process completed at step {step}")
                        break

                except subprocess.TimeoutExpired:
                    is_solvable = False
                    logger.warning(f"  ⏱️ Timeout during merge step {step}")
                    break
                except Exception as e:
                    logger.error(f"  ❌ Step failed: {e}")
                    break

            # PHASE 2: Wait for A* search completion
            logger.info(f"[RANDOM] PHASE 2: Waiting for A* search to complete...")

            if env.process:
                try:
                    env.process.wait(timeout=300)
                    logger.info(f"[RANDOM] FD process completed")
                except subprocess.TimeoutExpired:
                    logger.warning(f"[RANDOM] FD process timeout")
                    is_solvable = False
                    if env.process.poll() is None:
                        env.process.terminate()
                        try:
                            env.process.wait(timeout=5)
                        except:
                            pass

            # PHASE 3: Extract solution metrics
            logger.info(f"[RANDOM] PHASE 3: Extracting solution metrics...")

            fd_log_path = env.fd_output_dir / "downward.log" if hasattr(env, 'fd_output_dir') else None
            fd_metrics = FastDownwardOutputParser.parse_fd_log_file(str(fd_log_path)) if fd_log_path else {}

            wall_clock_time = time.time() - start_time

            # Populate metrics
            metrics.solved = fd_metrics.get('solved', False)
            metrics.plan_cost = fd_metrics.get('plan_cost', 0)
            metrics.plan_length = fd_metrics.get('plan_length', 0)
            metrics.nodes_expanded = fd_metrics.get('nodes_expanded', 0)
            metrics.search_time = fd_metrics.get('search_time', 0.0)
            metrics.wall_clock_time = wall_clock_time
            metrics.h_star_preservation = h_preservation
            metrics.merge_episodes = eval_steps
            metrics.solvable = is_solvable
            metrics.timeout_occurred = False

            if metrics.solved:
                logger.info(f"  ✅ SOLVED in {wall_clock_time:.2f}s")
                logger.info(f"     Plan cost: {metrics.plan_cost}")
                logger.info(f"     Nodes expanded: {metrics.nodes_expanded}")
                logger.info(f"     Merge steps: {eval_steps}")
            else:
                logger.warning(f"  ❌ NOT SOLVED after {wall_clock_time:.2f}s")
                logger.warning(f"     Merge steps: {eval_steps}")

        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}")
            metrics.solved = False
            metrics.error_type = "exception"
            metrics.error_message = str(e)[:500]

        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        return metrics

    def evaluate_problems(
            self,
            domain_file: str,
            problem_files: List[str],
            num_runs_per_problem: int = 1,
    ) -> List[DetailedMetrics]:  # ✅ FIX 8: String annotation
        """
        Evaluate random on multiple problems.

        Args:
            domain_file: Path to domain.pddl
            problem_files: List of problem.pddl paths
            num_runs_per_problem: Number of runs per problem

        Returns:
            List of DetailedMetrics for all runs
        """
        all_results = []

        for problem_file in tqdm(problem_files, desc="Random Evaluation", unit="problem"):
            for run_idx in range(num_runs_per_problem):
                seed = 42 + run_idx * 10000
                result = self.run_problem(domain_file, problem_file, seed=seed)
                all_results.append(result)

        return all_results


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class GNNRandomEvaluationFramework:
    """
    Orchestrate evaluation of both GNN and Random merge strategies.

    Returns results in format compatible with evaluation.py framework.
    """

    def __init__(
            self,
            model_path: str,
            domain_file: str,
            problem_files: List[str],
            output_dir: str = "evaluation_results",
            num_runs_per_problem: int = 1,
            downward_dir: Optional[str] = None,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
    ):
        """
        Initialize evaluation framework.

        Args:
            model_path: Path to trained model.zip
            domain_file: Path to domain.pddl
            problem_files: List of problem.pddl paths
            output_dir: Output directory for results
            num_runs_per_problem: Number of runs per problem
            downward_dir: Path to Fast Downward
            max_merges: Maximum merge steps
            timeout_per_step: Timeout per merge step
        """
        self.model_path = model_path
        self.domain_file = domain_file
        self.problem_files = problem_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_runs_per_problem = num_runs_per_problem
        self.downward_dir = downward_dir
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step

        self.all_results: List[DetailedMetrics] = []

    def evaluate(
            self,
            include_gnn: bool = True,
            include_random: bool = True,
    ) -> Tuple[List[DetailedMetrics], List[DetailedMetrics]]:  # ✅ FIX 9: String annotations
        """
        Execute full evaluation pipeline.

        Args:
            include_gnn: Whether to evaluate GNN policy
            include_random: Whether to evaluate random baseline

        Returns:
            Tuple of (gnn_results, random_results)
        """
        logger.info("\n" + "=" * 100)
        logger.info("GNN vs RANDOM MERGE STRATEGY EVALUATION")
        logger.info("=" * 100)
        logger.info(f"\n📋 Configuration:")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Domain: {self.domain_file}")
        logger.info(f"   Test problems: {len(self.problem_files)}")
        logger.info(f"   Runs per problem: {self.num_runs_per_problem}")

        gnn_results = []
        random_results = []

        # Evaluate GNN
        if include_gnn:
            logger.info("\n" + "-" * 100)
            logger.info("PHASE 1: GNN-GUIDED MERGE-AND-SHRINK + A* SEARCH")
            logger.info("-" * 100 + "\n")

            gnn_evaluator = GNNPolicyEvaluator(
                model_path=self.model_path,
                downward_dir=self.downward_dir,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
            )

            gnn_results = gnn_evaluator.evaluate_problems(
                domain_file=self.domain_file,
                problem_files=self.problem_files,
                num_runs_per_problem=self.num_runs_per_problem,
            )

            self.all_results.extend(gnn_results)

        # Evaluate Random
        if include_random:
            logger.info("\n" + "-" * 100)
            logger.info("PHASE 2: RANDOM MERGE-AND-SHRINK + A* SEARCH")
            logger.info("-" * 100 + "\n")

            random_evaluator = RandomMergeEvaluator(
                downward_dir=self.downward_dir,
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
            )

            random_results = random_evaluator.evaluate_problems(
                domain_file=self.domain_file,
                problem_files=self.problem_files,
                num_runs_per_problem=self.num_runs_per_problem,
            )

            self.all_results.extend(random_results)

        logger.info("\n" + "=" * 100)
        logger.info("✅ EVALUATION COMPLETE")
        logger.info("=" * 100)

        return gnn_results, random_results

    def get_all_results(self) -> List[DetailedMetrics]:  # ✅ FIX 10: String annotation
        """
        Get all evaluation results combined.

        Returns:
            List of all DetailedMetrics
        """
        return self.all_results

    def to_summary(self) -> Dict[str, Any]:
        """
        Convert results to summary format.

        Returns:
            Summary dictionary with statistics
        """
        from experiments.core.evaluation import ComparisonAnalyzer

        if not self.all_results:
            return {}

        analyzer = ComparisonAnalyzer(self.all_results)

        summary = {}
        for planner_name in set(r.planner_name for r in self.all_results):
            stats = analyzer.get_aggregate_statistics(planner_name)
            summary[planner_name] = stats.to_dict()

        return summary