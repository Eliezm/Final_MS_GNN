#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POST-TRAINING ANALYSIS SCRIPT
=============================
Run complete analysis on a trained experiment.

Usage:
    python run_post_training_analysis.py blocksworld_exp_1
    python run_post_training_analysis.py blocksworld_exp_2
    python run_post_training_analysis.py blocksworld_exp_3_curriculum

This script:
1. Loads trained model and training logs from experiment directory
2. Defines test sets (seen/unseen, different sizes, logistics transfer)
3. Runs GNN evaluation on all test sets
4. Runs Random baseline on all test sets
5. Runs FD baselines on all test sets (efficiently - once per unique problem)
6. Generates comparison plots (GNN vs Random, GNN vs Baselines, 3-way)
7. Generates comparison tables
8. Generates training log analysis plots
9. Saves everything to experiment directory

All outputs go to: results/{experiment_name}/post_analysis/
"""

import sys
import os
import json
import argparse
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Disable GPU for consistency
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# ============================================================================
# IMPORTS FROM EXISTING FRAMEWORK
# ============================================================================

from experiments.configs.experiment_configs import (
    get_experiment, list_experiments, ExperimentConfig, TestConfig, Domain, ProblemSize
)
from experiments.core.gnn_random_evaluation import (
    GNNPolicyEvaluator, RandomMergeEvaluator, GNNRandomEvaluationFramework
)
from experiments.core.evaluation import EvaluationFramework
from experiments.core.evaluation_analyzer import ComparisonAnalyzer
from experiments.core.evaluation_plots import GenerateEvaluationPlots
from experiments.core.evaluation_metrics import DetailedMetrics, AggregateStatistics
from experiments.core.baseline_runner import BaselineRunner
from experiments.core.analysis import (
    analyze_training_results,
    analyze_component_trajectories,
    analyze_feature_reward_correlation,
    analyze_feature_importance_from_decisions,
    analyze_causal_alignment,
    analyze_transition_explosion_risk,
    analyze_gnn_decision_quality,
    analyze_bisimulation_preservation,
    analyze_dead_end_creation,
    generate_literature_alignment_report,
)
from experiments.core.visualization import generate_all_plots
from experiments.core.unified_reporting import UnifiedReporter
from experiments.core.logging import EpisodeMetrics
from experiments.shared_experiment_utils import DEFAULT_REWARD_WEIGHTS


# ============================================================================
# TEST SET DEFINITIONS
# ============================================================================

@dataclass
class TestSetDefinition:
    """Definition of a test set for evaluation."""
    name: str
    domain: str
    size: str
    problem_pattern: str
    num_problems: int
    is_seen: bool
    description: str

    def get_domain_file(self) -> str:
        """Get domain file path."""
        base_patterns = [
            f"benchmarks/{self.domain}/{self.size}/domain.pddl",
            f"benchmarks/{self.domain}/domain.pddl",
        ]
        for pattern in base_patterns:
            if Path(pattern).exists():
                return str(Path(pattern).absolute())
        raise FileNotFoundError(f"Domain file not found for {self.domain}/{self.size}")

    def get_problem_files(self, seed: int = 42) -> List[str]:
        """Get list of problem files."""
        import random

        all_problems = sorted(glob.glob(self.problem_pattern))
        if not all_problems:
            raise ValueError(f"No problems found matching: {self.problem_pattern}")

        random.seed(seed)
        selected = random.sample(all_problems, min(self.num_problems, len(all_problems)))
        return sorted([str(Path(p).absolute()) for p in selected])


def get_test_sets_for_experiment(experiment_name: str) -> List[TestSetDefinition]:
    """
    Get test set definitions based on experiment type.

    Returns test sets that cover:
    - Seen problems (from training)
    - Unseen problems of same size
    - Unseen problems of different sizes
    - Domain transfer (logistics) for curriculum
    """

    # Common blocksworld test sets
    blocksworld_small_seen = TestSetDefinition(
        name="blocksworld_small_seen",
        domain="blocksworld",
        size="small",
        problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
        num_problems=5,
        is_seen=True,
        description="Blocksworld SMALL (training problems)"
    )

    blocksworld_small_unseen = TestSetDefinition(
        name="blocksworld_small_unseen",
        domain="blocksworld",
        size="small",
        problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
        num_problems=5,
        is_seen=False,
        description="Blocksworld SMALL (unseen)"
    )

    blocksworld_medium_seen = TestSetDefinition(
        name="blocksworld_medium_seen",
        domain="blocksworld",
        size="medium",
        problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
        num_problems=5,
        is_seen=True,
        description="Blocksworld MEDIUM (training problems)"
    )

    blocksworld_medium_unseen = TestSetDefinition(
        name="blocksworld_medium_unseen",
        domain="blocksworld",
        size="medium",
        problem_pattern="benchmarks/blocksworld/medium_new/problem_medium_*.pddl",
        num_problems=5,
        is_seen=False,
        description="Blocksworld MEDIUM (unseen)"
    )

    blocksworld_large_seen = TestSetDefinition(
        name="blocksworld_large_seen",
        domain="blocksworld",
        size="large",
        problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
        num_problems=5,
        is_seen=True,
        description="Blocksworld LARGE (training problems)"
    )

    blocksworld_large_unseen = TestSetDefinition(
        name="blocksworld_large_unseen",
        domain="blocksworld",
        size="large",
        problem_pattern="benchmarks/blocksworld/large_new/problem_large_*.pddl",
        num_problems=5,
        is_seen=False,
        description="Blocksworld LARGE (unseen)"
    )

    # Logistics transfer test sets
    logistics_small = TestSetDefinition(
        name="logistics_small_transfer",
        domain="logistics",
        size="small",
        problem_pattern="benchmarks/logistics/small/problem_small_*.pddl",
        num_problems=5,
        is_seen=False,
        description="Logistics SMALL (domain transfer)"
    )

    logistics_medium = TestSetDefinition(
        name="logistics_medium_transfer",
        domain="logistics",
        size="medium",
        problem_pattern="benchmarks/logistics/medium/problem_medium_*.pddl",
        num_problems=5,
        is_seen=False,
        description="Logistics MEDIUM (domain transfer)"
    )

    # Define test sets per experiment
    if experiment_name in ["blocksworld_exp_1", "blocksworld_exp_1_medium_train"]:
        # Trained on MEDIUM
        return [
            blocksworld_small_unseen,  # Generalize DOWN
            blocksworld_medium_seen,  # Same size, seen
            blocksworld_medium_unseen,  # Same size, unseen
            blocksworld_large_unseen,  # Generalize UP
        ]

    elif experiment_name in ["blocksworld_exp_2", "blocksworld_exp_2_large_train"]:
        # Trained on LARGE
        return [
            blocksworld_small_unseen,  # Generalize DOWN
            blocksworld_medium_unseen,  # Generalize DOWN
            blocksworld_large_seen,  # Same size, seen
            blocksworld_large_unseen,  # Same size, unseen
        ]

    elif experiment_name in ["blocksworld_exp_3_curriculum", "blocksworld_curriculum"]:
        # Curriculum: S→M→L + logistics transfer
        return [
            blocksworld_small_unseen,
            blocksworld_medium_unseen,
            blocksworld_large_unseen,
            logistics_small,
            logistics_medium,
        ]

    else:
        # Default: test on all sizes
        return [
            blocksworld_small_unseen,
            blocksworld_medium_unseen,
            blocksworld_large_unseen,
        ]


# ============================================================================
# EXPERIMENT LOADER
# ============================================================================

@dataclass
class LoadedExperiment:
    """Container for loaded experiment data."""
    name: str
    config: ExperimentConfig
    model_path: str
    training_log_path: str
    output_dir: Path
    training_log: List[EpisodeMetrics] = field(default_factory=list)
    episode_reward_signals: Dict = field(default_factory=dict)


def find_experiment_directory(experiment_name: str, base_dir: str = "results") -> Path:
    """Find experiment directory by name."""
    base_path = Path(base_dir)

    # Try exact match
    exact_path = base_path / experiment_name
    if exact_path.exists():
        return exact_path

    # Try partial match
    for subdir in base_path.iterdir():
        if subdir.is_dir() and experiment_name in subdir.name:
            return subdir

    raise FileNotFoundError(
        f"Experiment directory not found for: {experiment_name}\n"
        f"Searched in: {base_path}\n"
        f"Available: {[d.name for d in base_path.iterdir() if d.is_dir()]}"
    )


def load_experiment(experiment_name: str, base_dir: str = "results") -> LoadedExperiment:
    """Load a trained experiment."""
    print(f"\n📂 Loading experiment: {experiment_name}")

    # Find directory
    exp_dir = find_experiment_directory(experiment_name, base_dir)
    print(f"   Found: {exp_dir}")

    # Get config
    try:
        config = get_experiment(experiment_name)
    except ValueError:
        # Try to infer config from directory name
        if "exp_1" in experiment_name or "medium" in experiment_name.lower():
            config = get_experiment("blocksworld_exp_1")
        elif "exp_2" in experiment_name or "large" in experiment_name.lower():
            config = get_experiment("blocksworld_exp_2")
        elif "curriculum" in experiment_name.lower() or "exp_3" in experiment_name:
            config = get_experiment("blocksworld_exp_3_curriculum")
        else:
            raise ValueError(f"Cannot determine config for: {experiment_name}")

    # Find model
    model_candidates = [
        exp_dir / "model.zip",
        exp_dir / "training" / "model.zip",
        exp_dir / "checkpoints" / "model_final.zip",
    ]
    # Also check for latest checkpoint
    checkpoint_dir = exp_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("model_step_*.zip"))
        if checkpoints:
            model_candidates.insert(0, checkpoints[-1])

    model_path = None
    for candidate in model_candidates:
        if candidate.exists():
            model_path = str(candidate)
            break

    if not model_path:
        raise FileNotFoundError(
            f"Model not found in {exp_dir}\n"
            f"Checked: {model_candidates}"
        )

    print(f"   Model: {Path(model_path).name}")

    # Find training log
    log_candidates = [
        exp_dir / "training_log.jsonl",
        exp_dir / "training" / "training_log.jsonl",
    ]

    training_log_path = None
    for candidate in log_candidates:
        if candidate.exists():
            training_log_path = str(candidate)
            break

    # Load training log
    training_log = []
    episode_reward_signals = {}

    if training_log_path:
        print(f"   Training log: {Path(training_log_path).name}")
        try:
            with open(training_log_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        metrics = EpisodeMetrics(**data)
                        training_log.append(metrics)

                        # Build episode_reward_signals
                        if hasattr(metrics, 'merge_decisions_per_step') and metrics.merge_decisions_per_step:
                            episode_reward_signals[metrics.episode] = {
                                'problem_name': metrics.problem_name,
                                'episode_reward': metrics.reward,
                                'reward_signals_per_step': metrics.merge_decisions_per_step,
                                'component_summary': {
                                    'avg_h_preservation': metrics.component_h_preservation,
                                    'avg_transition_control': metrics.component_transition_control,
                                    'avg_operator_projection': metrics.component_operator_projection,
                                    'avg_label_combinability': metrics.component_label_combinability,
                                    'avg_bonus_signals': metrics.component_bonus_signals,
                                    'avg_h_star_ratio': metrics.h_star_ratio,
                                    'avg_transition_growth': metrics.transition_growth_ratio,
                                    'avg_opp_score': metrics.opp_score,
                                    'avg_label_score': metrics.label_combinability_score,
                                    'min_reachability': metrics.reachability_ratio,
                                    'max_dead_end_penalty': metrics.penalty_dead_end,
                                    'max_solvability_penalty': metrics.penalty_solvability_loss,
                                }
                            }
                    except (json.JSONDecodeError, TypeError) as e:
                        continue

            print(f"   Loaded {len(training_log)} episodes from training log")
        except Exception as e:
            print(f"   ⚠️ Could not load training log: {e}")
    else:
        print(f"   ⚠️ No training log found")

    return LoadedExperiment(
        name=experiment_name,
        config=config,
        model_path=model_path,
        training_log_path=training_log_path or "",
        output_dir=exp_dir,
        training_log=training_log,
        episode_reward_signals=episode_reward_signals,
    )


# ============================================================================
# EVALUATION ENGINE - Runs all evaluations efficiently
# ============================================================================

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    test_set_name: str
    gnn_results: List[DetailedMetrics]
    random_results: List[DetailedMetrics]
    baseline_results: List[DetailedMetrics]
    gnn_stats: Dict[str, Any]
    random_stats: Dict[str, Any]
    baseline_stats: Dict[str, Dict[str, Any]]


class EfficientEvaluationEngine:
    """
    Runs evaluations efficiently by:
    1. Collecting all unique problems first
    2. Running baselines once per unique problem
    3. Caching and reusing results
    """

    def __init__(
            self,
            model_path: str,
            output_dir: Path,
            max_merges: int = 50,
            timeout_per_step: float = 120.0,
            baseline_timeout: int = 300,
            num_runs_per_problem: int = 1,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.max_merges = max_merges
        self.timeout_per_step = timeout_per_step
        self.baseline_timeout = baseline_timeout
        self.num_runs_per_problem = num_runs_per_problem

        # Create output directories
        self.eval_dir = output_dir / "post_analysis" / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Caches for efficiency
        self.baseline_cache: Dict[str, List[DetailedMetrics]] = {}
        self.problem_results_cache: Dict[str, Dict] = {}

        # Initialize evaluators
        self.gnn_evaluator = None
        self.random_evaluator = None
        self.baseline_runner = None

    def _init_evaluators(self):
        """Initialize evaluators lazily."""
        if self.gnn_evaluator is None:
            self.gnn_evaluator = GNNPolicyEvaluator(
                model_path=self.model_path,
                downward_dir=str(PROJECT_ROOT / "downward"),
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
            )

        if self.random_evaluator is None:
            self.random_evaluator = RandomMergeEvaluator(
                downward_dir=str(PROJECT_ROOT / "downward"),
                max_merges=self.max_merges,
                timeout_per_step=self.timeout_per_step,
            )

        if self.baseline_runner is None:
            self.baseline_runner = BaselineRunner(
                timeout_sec=self.baseline_timeout,
                downward_dir=str(PROJECT_ROOT / "downward"),
            )

    def evaluate_test_set(
            self,
            test_set: TestSetDefinition,
            run_baselines: bool = True,
            seed: int = 42,
    ) -> EvaluationResult:
        """
        Evaluate a single test set with GNN, Random, and Baselines.
        """
        print(f"\n📋 Evaluating: {test_set.name}")
        print(f"   {test_set.description}")

        self._init_evaluators()

        # Get problem files
        try:
            domain_file = test_set.get_domain_file()
            problem_files = test_set.get_problem_files(seed)
            print(f"   Domain: {Path(domain_file).name}")
            print(f"   Problems: {len(problem_files)}")
        except (FileNotFoundError, ValueError) as e:
            print(f"   ❌ Cannot load test set: {e}")
            return EvaluationResult(
                test_set_name=test_set.name,
                gnn_results=[],
                random_results=[],
                baseline_results=[],
                gnn_stats={},
                random_stats={},
                baseline_stats={},
            )

        # Run GNN evaluation
        print(f"   Running GNN evaluation...")
        gnn_results = self.gnn_evaluator.evaluate_problems(
            domain_file=domain_file,
            problem_files=problem_files,
            num_runs_per_problem=self.num_runs_per_problem,
        )
        gnn_solved = sum(1 for r in gnn_results if r.solved)
        print(f"   ✓ GNN: {gnn_solved}/{len(gnn_results)} solved")

        # Run Random evaluation
        print(f"   Running Random evaluation...")
        random_results = self.random_evaluator.evaluate_problems(
            domain_file=domain_file,
            problem_files=problem_files,
            num_runs_per_problem=self.num_runs_per_problem,
        )
        random_solved = sum(1 for r in random_results if r.solved)
        print(f"   ✓ Random: {random_solved}/{len(random_results)} solved")

        # Run Baseline evaluation (with caching)
        baseline_results = []
        baseline_stats = {}

        if run_baselines:
            baseline_results, baseline_stats = self._run_baselines_cached(
                domain_file, problem_files, test_set.name
            )

        # Compute statistics
        all_results = gnn_results + random_results + baseline_results

        gnn_stats = {}
        random_stats = {}

        if all_results:
            analyzer = ComparisonAnalyzer(all_results)

            try:
                stats = analyzer.get_aggregate_statistics("GNN")
                gnn_stats = stats.to_dict()
            except:
                pass

            try:
                stats = analyzer.get_aggregate_statistics("Random")
                random_stats = stats.to_dict()
            except:
                pass

        # Save results
        self._save_test_set_results(test_set.name, gnn_results, random_results,
                                    baseline_results, gnn_stats, random_stats, baseline_stats)

        return EvaluationResult(
            test_set_name=test_set.name,
            gnn_results=gnn_results,
            random_results=random_results,
            baseline_results=baseline_results,
            gnn_stats=gnn_stats,
            random_stats=random_stats,
            baseline_stats=baseline_stats,
        )

    def _run_baselines_cached(
            self,
            domain_file: str,
            problem_files: List[str],
            test_set_name: str,
    ) -> Tuple[List[DetailedMetrics], Dict[str, Dict]]:
        """Run baselines with caching to avoid duplicate runs."""
        from experiments.core.evaluation_config import EvaluationConfig

        print(f"   Running FD baselines...")

        all_baseline_results = []
        baseline_stats = {}

        baselines = EvaluationConfig.BASELINE_CONFIGS[:3]  # Use top 3 for speed

        for baseline_config in baselines:
            baseline_name = baseline_config["name"]
            search_config = baseline_config["search_config"]

            baseline_results = []

            for problem_file in problem_files:
                # Check cache
                cache_key = f"{problem_file}_{baseline_name}"

                if cache_key in self.baseline_cache:
                    baseline_results.extend(self.baseline_cache[cache_key])
                    continue

                # Run baseline
                result = self.baseline_runner.run(
                    domain_file=domain_file,
                    problem_file=problem_file,
                    search_config=search_config,
                    baseline_name=baseline_name,
                )

                baseline_results.append(result)
                self.baseline_cache[cache_key] = [result]

            all_baseline_results.extend(baseline_results)

            # Compute stats for this baseline
            solved = sum(1 for r in baseline_results if r.solved)
            total = len(baseline_results)

            baseline_stats[baseline_name] = {
                'solve_rate_%': (solved / total * 100) if total > 0 else 0,
                'solved': solved,
                'total': total,
                'avg_time_total_s': np.mean(
                    [r.wall_clock_time for r in baseline_results if r.solved]) if solved > 0 else 0,
                'avg_expansions': int(
                    np.mean([r.nodes_expanded for r in baseline_results if r.solved])) if solved > 0 else 0,
            }

            print(f"     {baseline_name[:25]}: {solved}/{total} solved")

        return all_baseline_results, baseline_stats

    def _save_test_set_results(
            self,
            test_set_name: str,
            gnn_results: List[DetailedMetrics],
            random_results: List[DetailedMetrics],
            baseline_results: List[DetailedMetrics],
            gnn_stats: Dict,
            random_stats: Dict,
            baseline_stats: Dict,
    ):
        """Save test set results to file."""
        test_dir = self.eval_dir / test_set_name
        test_dir.mkdir(parents=True, exist_ok=True)

        results_data = {
            "test_set": test_set_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "gnn_solved": sum(1 for r in gnn_results if r.solved),
                "gnn_total": len(gnn_results),
                "random_solved": sum(1 for r in random_results if r.solved),
                "random_total": len(random_results),
                "baseline_solved": sum(1 for r in baseline_results if r.solved),
                "baseline_total": len(baseline_results),
            },
            "gnn_stats": gnn_stats,
            "random_stats": random_stats,
            "baseline_stats": baseline_stats,
            "gnn_results": [r.to_dict() for r in gnn_results],
            "random_results": [r.to_dict() for r in random_results],
            "baseline_results": [r.to_dict() for r in baseline_results],
        }

        with open(test_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)


# ============================================================================
# ANALYSIS ENGINE - Generates all plots and tables
# ============================================================================

class AnalysisEngine:
    """Generates all analysis outputs from training logs and evaluation results."""

    def __init__(self, experiment: LoadedExperiment):
        self.experiment = experiment
        self.analysis_dir = experiment.output_dir / "post_analysis" / "analysis"
        self.plots_dir = experiment.output_dir / "post_analysis" / "plots"
        self.tables_dir = experiment.output_dir / "post_analysis" / "tables"
        self.reports_dir = experiment.output_dir / "post_analysis" / "reports"

        # Create directories
        for d in [self.analysis_dir, self.plots_dir, self.tables_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run_training_log_analysis(self) -> Dict[str, Any]:
        """Run all analysis on training logs."""
        print("\n🔍 Running training log analysis...")

        if not self.experiment.training_log:
            print("   ⚠️ No training log available")
            return {}

        training_log = self.experiment.training_log
        episode_reward_signals = self.experiment.episode_reward_signals

        all_analysis = {}

        # 1. Main summary
        print("   [1/10] Main training summary...")
        try:
            summary = analyze_training_results(
                training_log=training_log,
                eval_results=[],
                problem_names=[m.problem_name for m in training_log[:10]],
                benchmarks=[],
                experiment_id=self.experiment.name,
            )
            all_analysis['main_summary'] = summary.to_dict()
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 2. Component analysis
        print("   [2/10] Component trajectories...")
        try:
            all_analysis['component_analysis'] = analyze_component_trajectories(
                training_log, self.analysis_dir
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 3. Correlation analysis
        print("   [3/10] Feature-reward correlation...")
        try:
            all_analysis['correlation_analysis'] = analyze_feature_reward_correlation(
                episode_reward_signals, self.analysis_dir
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 4. Feature importance
        print("   [4/10] Feature importance...")
        try:
            all_analysis['feature_importance'] = analyze_feature_importance_from_decisions(
                training_log, self.analysis_dir
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 5. Causal alignment
        print("   [5/10] Causal alignment...")
        try:
            all_analysis['causal_alignment'] = analyze_causal_alignment(
                training_log, self.analysis_dir
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 6. Transition explosion
        print("   [6/10] Transition explosion risk...")
        try:
            all_analysis['explosion_analysis'] = analyze_transition_explosion_risk(
                training_log, self.analysis_dir
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 7. Decision quality
        print("   [7/10] GNN decision quality...")
        try:
            decision_traces = {
                i: m.merge_decisions_per_step
                for i, m in enumerate(training_log)
                if m.merge_decisions_per_step
            }
            all_analysis['decision_quality'] = analyze_gnn_decision_quality(
                decision_traces, self.analysis_dir
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 8. Bisimulation preservation
        print("   [8/10] Bisimulation preservation...")
        try:
            all_analysis['bisimulation'] = analyze_bisimulation_preservation(
                training_log, self.analysis_dir
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 9. Dead-end analysis
        print("   [9/10] Dead-end creation...")
        try:
            all_analysis['safety'] = analyze_dead_end_creation(
                training_log, self.analysis_dir
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # 10. Literature alignment
        print("   [10/10] Literature alignment...")
        try:
            all_analysis['literature_checklist'] = generate_literature_alignment_report(
                training_log,
                episode_reward_signals,
                all_analysis.get('correlation_analysis', {}),
                all_analysis.get('bisimulation', {}),
                self.analysis_dir,
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")

        # Save all analysis
        with open(self.analysis_dir / "all_analysis_results.json", 'w') as f:
            json.dump(all_analysis, f, indent=2, default=str)

        print(f"   ✓ Analysis saved to: {self.analysis_dir}")

        return all_analysis

    def generate_training_plots(self, analysis_data: Dict[str, Any]) -> Dict[str, Optional[Path]]:
        """Generate all plots from training logs."""
        print("\n📊 Generating training analysis plots...")

        if not self.experiment.training_log:
            print("   ⚠️ No training log available")
            return {}

        try:
            plot_results = generate_all_plots(
                training_log=self.experiment.training_log,
                eval_results={},
                output_dir=str(self.experiment.output_dir / "post_analysis"),
                component_analysis=analysis_data.get('component_analysis'),
                correlation_analysis=analysis_data.get('correlation_analysis'),
                feature_importance_analysis=analysis_data.get('feature_importance'),
                bisim_analysis=analysis_data.get('bisimulation'),
                causal_alignment_analysis=analysis_data.get('causal_alignment'),
                explosion_analysis=analysis_data.get('explosion_analysis'),
                decision_quality_analysis=analysis_data.get('decision_quality'),
                episode_reward_signals=self.experiment.episode_reward_signals,
                literature_checklist=analysis_data.get('literature_checklist'),
            )

            successful = sum(1 for p in plot_results.values() if p is not None)
            print(f"   ✓ Generated {successful}/{len(plot_results)} training plots")

            return plot_results

        except Exception as e:
            print(f"   ❌ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def generate_comparison_plots(
            self,
            evaluation_results: List[EvaluationResult],
    ) -> Dict[str, Optional[Path]]:
        """Generate comparison plots from evaluation results."""
        print("\n📈 Generating comparison plots...")

        # Aggregate all results
        all_gnn_results = []
        all_random_results = []
        all_baseline_results = []

        combined_gnn_stats = {}
        combined_random_stats = {}
        combined_baseline_stats = {}

        for eval_result in evaluation_results:
            all_gnn_results.extend(eval_result.gnn_results)
            all_random_results.extend(eval_result.random_results)
            all_baseline_results.extend(eval_result.baseline_results)

            # Merge stats (use first non-empty)
            if not combined_gnn_stats and eval_result.gnn_stats:
                combined_gnn_stats = eval_result.gnn_stats
            if not combined_random_stats and eval_result.random_stats:
                combined_random_stats = eval_result.random_stats
            combined_baseline_stats.update(eval_result.baseline_stats)

        # Recompute aggregate stats from all results
        all_results = all_gnn_results + all_random_results + all_baseline_results

        if all_results:
            analyzer = ComparisonAnalyzer(all_results)

            try:
                combined_gnn_stats = analyzer.get_aggregate_statistics("GNN").to_dict()
            except:
                pass

            try:
                combined_random_stats = analyzer.get_aggregate_statistics("Random").to_dict()
            except:
                pass

        # Generate plots
        comparison_plots_dir = self.plots_dir / "comparison"
        comparison_plots_dir.mkdir(exist_ok=True)

        try:
            plotter = GenerateEvaluationPlots(output_dir=str(comparison_plots_dir))

            plot_results = plotter.generate_all_plots(
                statistics={
                    **combined_gnn_stats,
                    **combined_random_stats,
                    **combined_baseline_stats,
                },
                results=all_results,
                gnn_results=combined_gnn_stats,
            )

            successful = sum(1 for p in plot_results.values() if p is not None)
            print(f"   ✓ Generated {successful}/{len(plot_results)} comparison plots")

            return plot_results

        except Exception as e:
            print(f"   ❌ Comparison plot generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def generate_per_test_set_plots(
            self,
            evaluation_results: List[EvaluationResult],
    ):
        """Generate separate plots for each test set."""
        print("\n📊 Generating per-test-set plots...")

        for eval_result in evaluation_results:
            if not eval_result.gnn_results and not eval_result.random_results:
                continue

            test_plot_dir = self.plots_dir / "test_sets" / eval_result.test_set_name
            test_plot_dir.mkdir(parents=True, exist_ok=True)

            all_results = eval_result.gnn_results + eval_result.random_results + eval_result.baseline_results

            if not all_results:
                continue

            try:
                plotter = GenerateEvaluationPlots(output_dir=str(test_plot_dir))

                stats = {
                    "GNN": eval_result.gnn_stats,
                    "Random": eval_result.random_stats,
                    **eval_result.baseline_stats,
                }

                plotter.generate_all_plots(
                    statistics=stats,
                    results=all_results,
                    gnn_results=eval_result.gnn_stats,
                )

                print(f"   ✓ {eval_result.test_set_name}")

            except Exception as e:
                print(f"   ⚠️ {eval_result.test_set_name}: {e}")

    def generate_comparison_tables(
            self,
            evaluation_results: List[EvaluationResult],
    ) -> List[Path]:
        """Generate comparison tables in multiple formats."""
        print("\n📋 Generating comparison tables...")

        saved_files = []

        # ========================================
        # TABLE 1: Per-Test-Set Results
        # ========================================
        per_test_rows = []

        for eval_result in evaluation_results:
            gnn_solved = sum(1 for r in eval_result.gnn_results if r.solved)
            gnn_total = len(eval_result.gnn_results)
            random_solved = sum(1 for r in eval_result.random_results if r.solved)
            random_total = len(eval_result.random_results)

            gnn_rate = (gnn_solved / gnn_total * 100) if gnn_total > 0 else 0
            random_rate = (random_solved / random_total * 100) if random_total > 0 else 0
            improvement = gnn_rate - random_rate

            per_test_rows.append({
                "Test Set": eval_result.test_set_name,
                "GNN Solved": f"{gnn_solved}/{gnn_total}",
                "GNN Rate (%)": f"{gnn_rate:.1f}",
                "Random Solved": f"{random_solved}/{random_total}",
                "Random Rate (%)": f"{random_rate:.1f}",
                "Improvement (%)": f"{improvement:+.1f}",
            })

        # Save as CSV
        if per_test_rows:
            import pandas as pd
            df = pd.DataFrame(per_test_rows)
            csv_path = self.tables_dir / "per_test_set_results.csv"
            df.to_csv(csv_path, index=False)
            saved_files.append(csv_path)
            print(f"   ✓ Saved: {csv_path.name}")

        # ========================================
        # TABLE 2: Strategy Comparison (Aggregate)
        # ========================================
        strategy_rows = []

        # Aggregate all results
        all_gnn = []
        all_random = []
        all_baselines = defaultdict(list)

        for eval_result in evaluation_results:
            all_gnn.extend(eval_result.gnn_results)
            all_random.extend(eval_result.random_results)
            for baseline_name, baseline_data in eval_result.baseline_stats.items():
                all_baselines[baseline_name].append(baseline_data)

        # GNN row
        if all_gnn:
            gnn_solved = sum(1 for r in all_gnn if r.solved)
            gnn_times = [r.wall_clock_time for r in all_gnn if r.solved]
            gnn_expansions = [r.nodes_expanded for r in all_gnn if r.solved]
            gnn_h_pres = [r.h_star_preservation for r in all_gnn if r.solved]

            strategy_rows.append({
                "Strategy": "GNN (Learned)",
                "Solve Rate (%)": f"{gnn_solved / len(all_gnn) * 100:.1f}",
                "Mean Time (s)": f"{np.mean(gnn_times):.3f}" if gnn_times else "N/A",
                "Mean Expansions": f"{int(np.mean(gnn_expansions)):,}" if gnn_expansions else "N/A",
                "H* Preservation": f"{np.mean(gnn_h_pres):.4f}" if gnn_h_pres else "N/A",
            })

        # Random row
        if all_random:
            random_solved = sum(1 for r in all_random if r.solved)
            random_times = [r.wall_clock_time for r in all_random if r.solved]
            random_expansions = [r.nodes_expanded for r in all_random if r.solved]
            random_h_pres = [r.h_star_preservation for r in all_random if r.solved]

            strategy_rows.append({
                "Strategy": "Random Merge",
                "Solve Rate (%)": f"{random_solved / len(all_random) * 100:.1f}",
                "Mean Time (s)": f"{np.mean(random_times):.3f}" if random_times else "N/A",
                "Mean Expansions": f"{int(np.mean(random_expansions)):,}" if random_expansions else "N/A",
                "H* Preservation": f"{np.mean(random_h_pres):.4f}" if random_h_pres else "N/A",
            })

        # Baseline rows
        for baseline_name, baseline_data_list in all_baselines.items():
            total_solved = sum(d.get('solved', 0) for d in baseline_data_list)
            total_problems = sum(d.get('total', 0) for d in baseline_data_list)
            avg_time = np.mean(
                [d.get('avg_time_total_s', 0) for d in baseline_data_list if d.get('avg_time_total_s', 0) > 0])
            avg_exp = np.mean(
                [d.get('avg_expansions', 0) for d in baseline_data_list if d.get('avg_expansions', 0) > 0])

            strategy_rows.append({
                "Strategy": baseline_name[:30],
                "Solve Rate (%)": f"{total_solved / max(1, total_problems) * 100:.1f}",
                "Mean Time (s)": f"{avg_time:.3f}" if avg_time > 0 else "N/A",
                "Mean Expansions": f"{int(avg_exp):,}" if avg_exp > 0 else "N/A",
                "H* Preservation": "1.0000 (optimal)",
            })

        # Save as CSV
        if strategy_rows:
            import pandas as pd
            df = pd.DataFrame(strategy_rows)
            csv_path = self.tables_dir / "strategy_comparison.csv"
            df.to_csv(csv_path, index=False)
            saved_files.append(csv_path)
            print(f"   ✓ Saved: {csv_path.name}")

        # ========================================
        # TABLE 3: Human-readable summary
        # ========================================
        summary_path = self.tables_dir / "comparison_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"COMPARISON SUMMARY: {self.experiment.name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 100 + "\n\n")

            f.write("PER-TEST-SET RESULTS\n")
            f.write("-" * 100 + "\n")
            for row in per_test_rows:
                f.write(f"\n{row['Test Set']}:\n")
                f.write(f"  GNN:    {row['GNN Solved']} ({row['GNN Rate (%)']}%)\n")
                f.write(f"  Random: {row['Random Solved']} ({row['Random Rate (%)']}%)\n")
                f.write(f"  Improvement: {row['Improvement (%)']}%\n")

            f.write("\n\nSTRATEGY COMPARISON (AGGREGATE)\n")
            f.write("-" * 100 + "\n")
            for row in strategy_rows:
                f.write(f"\n{row['Strategy']}:\n")
                f.write(f"  Solve Rate:     {row['Solve Rate (%)']}\n")
                f.write(f"  Mean Time:      {row['Mean Time (s)']}\n")
                f.write(f"  Mean Expansions:{row['Mean Expansions']}\n")
                f.write(f"  H* Preservation:{row['H* Preservation']}\n")

            f.write("\n" + "=" * 100 + "\n")

        saved_files.append(summary_path)
        print(f"   ✓ Saved: {summary_path.name}")

        return saved_files

    def create_unified_report(
            self,
            analysis_data: Dict[str, Any],
            evaluation_results: List[EvaluationResult],
    ) -> Path:
        """Create unified experiment report."""
        print("\n📝 Creating unified report...")

        reporter = UnifiedReporter(self.reports_dir)

        # Build test results dict
        test_results = {}
        for eval_result in evaluation_results:
            test_results[eval_result.test_set_name] = {
                "results": {
                    "summary": {
                        "gnn_total": len(eval_result.gnn_results),
                        "gnn_solved": sum(1 for r in eval_result.gnn_results if r.solved),
                        "random_total": len(eval_result.random_results),
                        "random_solved": sum(1 for r in eval_result.random_results if r.solved),
                    }
                },
                "gnn_stats": eval_result.gnn_stats,
                "random_stats": eval_result.random_stats,
                "baseline_stats": eval_result.baseline_stats,
            }

        # Build evaluation summary
        gnn_vs_random_summary = {}
        for eval_result in evaluation_results:
            if eval_result.gnn_stats:
                gnn_vs_random_summary['GNN'] = eval_result.gnn_stats
            if eval_result.random_stats:
                gnn_vs_random_summary['Random'] = eval_result.random_stats
            break

        # Build baseline summary
        baseline_summary = {}
        for eval_result in evaluation_results:
            baseline_summary.update(eval_result.baseline_stats)

        report_path = reporter.create_unified_report(
            config=self.experiment.config.to_dict(),
            training_summary=analysis_data.get('main_summary', {}),
            analysis_summary=analysis_data,
            evaluation_summary={"gnn_vs_random": gnn_vs_random_summary},
            test_results=test_results,
            baseline_summary={"baseline_configs": list(baseline_summary.items())},
        )

        print(f"   ✓ Report: {report_path}")

        return report_path


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def run_post_training_analysis(
        experiment_name: str,
        base_dir: str = "results",
        include_baselines: bool = True,
        num_runs_per_problem: int = 1,
        baseline_timeout: int = 300,
        seed: int = 42,
) -> Dict[str, Any]:
    """
    Run complete post-training analysis on an experiment.

    Args:
        experiment_name: Name of experiment to analyze
        base_dir: Base results directory
        include_baselines: Whether to run FD baseline comparisons
        num_runs_per_problem: Number of evaluation runs per problem
        baseline_timeout: Timeout for baseline evaluations (seconds)
        seed: Random seed for reproducibility

    Returns:
        Summary dict with all results
    """
    start_time = time.time()

    print("\n" + "=" * 100)
    print(f"🔬 POST-TRAINING ANALYSIS: {experiment_name}")
    print("=" * 100)

    # ========================================
    # STEP 1: Load Experiment
    # ========================================
    experiment = load_experiment(experiment_name, base_dir)

    # ========================================
    # STEP 2: Get Test Sets
    # ========================================
    test_sets = get_test_sets_for_experiment(experiment_name)
    print(f"\n📋 Test sets to evaluate: {len(test_sets)}")
    for ts in test_sets:
        print(f"   • {ts.name}: {ts.description}")

    # ========================================
    # STEP 3: Run Evaluations
    # ========================================
    print("\n" + "-" * 100)
    print("PHASE 1: EVALUATIONS")
    print("-" * 100)

    eval_engine = EfficientEvaluationEngine(
        model_path=experiment.model_path,
        output_dir=experiment.output_dir,
        num_runs_per_problem=num_runs_per_problem,
        baseline_timeout=baseline_timeout,
    )

    evaluation_results = []
    for test_set in test_sets:
        result = eval_engine.evaluate_test_set(
            test_set,
            run_baselines=include_baselines,
            seed=seed,
        )
        evaluation_results.append(result)

    # ========================================
    # STEP 4: Run Training Log Analysis
    # ========================================
    print("\n" + "-" * 100)
    print("PHASE 2: TRAINING LOG ANALYSIS")
    print("-" * 100)

    analysis_engine = AnalysisEngine(experiment)
    analysis_data = analysis_engine.run_training_log_analysis()

    # ========================================
    # STEP 5: Generate Training Plots
    # ========================================
    print("\n" + "-" * 100)
    print("PHASE 3: TRAINING PLOTS")
    print("-" * 100)

    training_plots = analysis_engine.generate_training_plots(analysis_data)

    # ========================================
    # STEP 6: Generate Comparison Plots
    # ========================================
    print("\n" + "-" * 100)
    print("PHASE 4: COMPARISON PLOTS")
    print("-" * 100)

    comparison_plots = analysis_engine.generate_comparison_plots(evaluation_results)
    analysis_engine.generate_per_test_set_plots(evaluation_results)

    # ========================================
    # STEP 7: Generate Tables
    # ========================================
    print("\n" + "-" * 100)
    print("PHASE 5: COMPARISON TABLES")
    print("-" * 100)

    table_files = analysis_engine.generate_comparison_tables(evaluation_results)

    # ========================================
    # STEP 8: Create Unified Report
    # ========================================
    print("\n" + "-" * 100)
    print("PHASE 6: UNIFIED REPORT")
    print("-" * 100)

    report_path = analysis_engine.create_unified_report(analysis_data, evaluation_results)

    # ========================================
    # SUMMARY
    # ========================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 100)
    print(f"✅ POST-TRAINING ANALYSIS COMPLETE: {experiment_name}")
    print("=" * 100)
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    print(f"\n📁 Output directory: {experiment.output_dir / 'post_analysis'}")
    print(f"\n📂 Generated outputs:")
    print(f"   ✓ evaluation/ - Raw evaluation results per test set")
    print(f"   ✓ analysis/ - Training log analysis (JSON)")
    print(f"   ✓ plots/")
    print(f"     ├── training/ - 22 training analysis plots")
    print(f"     ├── comparison/ - GNN vs Random vs Baselines plots")
    print(f"     └── test_sets/ - Per-test-set comparison plots")
    print(f"   ✓ tables/ - CSV and TXT comparison tables")
    print(f"   ✓ reports/ - Unified experiment report")

    # Print key results
    print(f"\n📊 KEY RESULTS:")
    for eval_result in evaluation_results:
        gnn_solved = sum(1 for r in eval_result.gnn_results if r.solved)
        gnn_total = len(eval_result.gnn_results)
        random_solved = sum(1 for r in eval_result.random_results if r.solved)

        if gnn_total > 0:
            gnn_rate = gnn_solved / gnn_total * 100
            random_rate = random_solved / max(1, len(eval_result.random_results)) * 100
            improvement = gnn_rate - random_rate

            status = "✓" if improvement > 0 else "✗" if improvement < 0 else "="
            print(f"   {status} {eval_result.test_set_name}: "
                  f"GNN {gnn_rate:.1f}% vs Random {random_rate:.1f}% "
                  f"({improvement:+.1f}%)")

    print("\n" + "=" * 100)

    return {
        "status": "success",
        "experiment": experiment_name,
        "output_dir": str(experiment.output_dir / "post_analysis"),
        "report_path": str(report_path),
        "elapsed_seconds": elapsed,
        "num_test_sets": len(evaluation_results),
        "num_training_episodes": len(experiment.training_log),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run post-training analysis on a trained GNN experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:

  Analyze experiment 1 (trained on medium):
    python run_post_training_analysis.py blocksworld_exp_1

  Analyze experiment 2 (trained on large):
    python run_post_training_analysis.py blocksworld_exp_2

  Analyze curriculum experiment:
    python run_post_training_analysis.py blocksworld_exp_3_curriculum

  Skip baselines (faster):
    python run_post_training_analysis.py blocksworld_exp_1 --no-baselines

  Custom results directory:
    python run_post_training_analysis.py blocksworld_exp_1 --results-dir my_results

OUTPUT STRUCTURE:
  results/{experiment_name}/post_analysis/
  ├── evaluation/          (raw results per test set)
  │   ├── blocksworld_small_unseen/
  │   │   └── results.json
  │   ├── blocksworld_medium_seen/
  │   └── ...
  ├── analysis/            (training log analysis)
  │   └── all_analysis_results.json
  ├── plots/
  │   ├── training/        (learning curves, components, etc.)
  │   ├── comparison/      (GNN vs Random vs Baselines)
  │   └── test_sets/       (per-test-set plots)
  ├── tables/
  │   ├── per_test_set_results.csv
  │   ├── strategy_comparison.csv
  │   └── comparison_summary.txt
  └── reports/
      ├── experiment_report.json
      └── experiment_report.txt
        """
    )

    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment name (e.g., blocksworld_exp_1, blocksworld_exp_3_curriculum)"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory (default: results)"
    )

    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip FD baseline comparisons (faster)"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs per problem (default: 1)"
    )

    parser.add_argument(
        "--baseline-timeout",
        type=int,
        default=300,
        help="Timeout for baseline evaluations in seconds (default: 300)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments"
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\n" + "=" * 100)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 100)

        for exp_name in list_experiments():
            exp = get_experiment(exp_name)
            curriculum_tag = " [CURRICULUM]" if exp.is_curriculum else ""
            print(f"\n  {exp_name}{curriculum_tag}")
            print(f"    {exp.description}")

        print("\n" + "=" * 100)
        return 0

    # Run analysis
    try:
        result = run_post_training_analysis(
            experiment_name=args.experiment,
            base_dir=args.results_dir,
            include_baselines=not args.no_baselines,
            num_runs_per_problem=args.runs,
            baseline_timeout=args.baseline_timeout,
            seed=args.seed,
        )

        return 0 if result["status"] == "success" else 1

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nTip: Make sure you've trained the experiment first:")
        print(f"  python run_full_experiment.py {args.experiment} --train-only")
        return 1

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())