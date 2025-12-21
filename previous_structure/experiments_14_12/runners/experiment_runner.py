#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENT RUNNER - Orchestrates training, evaluation, and analysis
"""

import sys
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs.experiment_configs import ExperimentConfig, TestConfig
from experiments.core.training import GNNTrainer, set_all_seeds, save_json_atomic
from experiments.core.analysis import analyze_training_results
from experiments.core.visualization import generate_all_plots
from shared_experiment_utils import DEFAULT_REWARD_WEIGHTS


def select_problems(
        problem_pattern: str,
        num_problems: int,
        seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Select K problems from pattern."""
    all_problems = sorted(glob.glob(problem_pattern))

    if not all_problems:
        raise ValueError(f"No problems found matching: {problem_pattern}")

    random.seed(seed)
    selected = random.sample(all_problems, min(num_problems, len(all_problems)))
    selected = sorted(selected)

    return selected, [Path(p).name for p in selected]


def get_domain_file(domain: str) -> str:
    """Get domain PDDL file path."""
    domain_lower = domain.lower()

    # Search for domain file
    possible_paths = [
        f"benchmarks/{domain_lower}/domain.pddl",
        f"benchmarks/{domain_lower.capitalize()}/domain.pddl",
        f"domains/{domain_lower}.pddl",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return path

    raise FileNotFoundError(f"Domain file not found for: {domain}")

class CurriculumExperimentRunner:
    """Orchestrates curriculum learning experiments."""

    def __init__(
            self,
            config: ExperimentConfig,
            output_base_dir: str = "results",
    ):
        self.config = config
        self.output_base_dir = Path(output_base_dir)
        self.output_dir = self.output_base_dir / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not config.is_curriculum:
            raise ValueError(f"Config {config.name} is not a curriculum config")

        set_all_seeds(config.seed)

    def run_curriculum(self) -> Dict:
        """Execute full curriculum learning pipeline."""
        print("\n" + "=" * 100)
        print(f"🎓 CURRICULUM LEARNING EXPERIMENT - {self.config.name}")
        print(f"   {self.config.description}")
        print(f"   Phases: {len(self.config.curriculum_phases)}")
        print("=" * 100)

        # Track models across phases
        phase_results = {}
        current_model_path = None

        # ====================================================================
        # PHASE LOOP: Train progressively larger problems
        # ====================================================================

        for phase_idx, phase in enumerate(self.config.curriculum_phases, 1):
            print(f"\n{'=' * 100}")
            print(f"📚 CURRICULUM PHASE {phase_idx}/{len(self.config.curriculum_phases)}")
            print(f"   {phase.name}: {phase.problem_size.value}")
            print(f"   Problems: {phase.num_problems}, Episodes: {phase.num_episodes}")
            print(f"{'=' * 100}\n")

            # Get problems for this phase
            problem_files, problem_names = select_problems(
                phase.problem_pattern,
                phase.num_problems,
                self.config.seed + phase_idx,
            )

            domain_file = get_domain_file(self.config.domain.value)
            benchmarks = [(domain_file, pf) for pf in problem_files]

            # ================================================================
            # TRAIN ON THIS PHASE
            # ================================================================

            phase_output_dir = self.output_dir / f"phase_{phase_idx}_{phase.name}"
            phase_output_dir.mkdir(parents=True, exist_ok=True)

            trainer = GNNTrainer(
                benchmarks=benchmarks,
                problem_names=problem_names,
                output_dir=str(phase_output_dir),
                reward_weights=DEFAULT_REWARD_WEIGHTS.copy(),
                max_merges=self.config.max_merges,
                timeout_per_step=self.config.timeout_per_step,
                checkpoint_interval=self.config.checkpoint_interval,
                min_episodes_per_problem=self.config.min_episodes_per_problem,
                seed=self.config.seed + phase_idx,
            )

            model_path = trainer.run_training(
                num_episodes=phase.num_episodes,
                timesteps_per_episode=self.config.timesteps_per_episode,
                resume_from=current_model_path,
            )

            if not model_path:
                print(f"\n❌ Phase {phase_idx} training failed")
                return {
                    "status": "failed",
                    "phase": phase_idx,
                    "phase_name": phase.name,
                }

            trainer.save_training_log()
            trainer.close_logger()

            # Save phase results
            phase_model_path = phase_output_dir / "model_final.zip"
            import shutil
            shutil.copy(model_path, phase_model_path)

            current_model_path = str(phase_model_path)

            phase_results[phase.name] = {
                "phase": phase_idx,
                "model_path": current_model_path,
                "training_log": trainer.episode_log,
                "output_dir": str(phase_output_dir),
            }

            print(f"\n✅ Phase {phase_idx} complete!")
            print(f"   Model: {current_model_path}")
            print(f"   Episodes: {len(trainer.episode_log)}")

        # ====================================================================
        # FINAL EVALUATION (on final curriculum model)
        # ====================================================================

        print(f"\n{'=' * 100}")
        print(f"📊 FINAL EVALUATION (on final curriculum model)")
        print(f"{'=' * 100}\n")

        # ✅ FIX: Use GNNPolicyEvaluator instead of GNNEvaluator
        from experiments.core.gnn_random_evaluation import GNNPolicyEvaluator

        test_results = {}
        for test_config in self.config.test_configurations:
            print(f"📋 Testing: {test_config.name}")

            problem_files, problem_names = select_problems(
                test_config.problem_pattern,
                test_config.num_problems,
                self.config.seed + 1000,
            )

            domain_file = get_domain_file(test_config.domain.value)

            # ✅ CORRECTED: Use GNNPolicyEvaluator API
            evaluator = GNNPolicyEvaluator(
                model_path=current_model_path,
                downward_dir=str(PROJECT_ROOT / "downward"),  # ✅ EXPLICIT PATH
                max_merges=self.config.max_merges,
                timeout_per_step=self.config.timeout_per_step,
            )

            # ✅ CORRECTED: Use evaluate_problems method with domain_file and problem_files
            eval_results = evaluator.evaluate_problems(
                domain_file=domain_file,
                problem_files=problem_files,
                num_runs_per_problem=test_config.num_runs_per_problem,
            )

            test_results[test_config.name] = {
                "results": eval_results,
                "num_solved": sum(1 for r in eval_results if r.solved),
                "num_problems": len(eval_results),
            }

        # ====================================================================
        # SAVE COMPREHENSIVE CURRICULUM REPORT
        # ====================================================================

        curriculum_summary = {
            "status": "success",
            "config": self.config.to_dict(),
            "num_phases": len(self.config.curriculum_phases),
            "phases": phase_results,
            "test_results": test_results,
            "final_model_path": current_model_path,
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = self.output_dir / "curriculum_summary.json"
        save_json_atomic(curriculum_summary, str(summary_path))

        print("\n" + "=" * 100)
        print(f"✅ CURRICULUM LEARNING COMPLETE - {self.config.name}")
        print("=" * 100)
        print(f"   Output: {self.output_dir.absolute()}")
        print(f"   Final Model: {current_model_path}")
        print(f"   Summary: {summary_path}")

        return curriculum_summary

class ExperimentRunner:
    """Orchestrates a complete experiment."""

    def __init__(
            self,
            config: ExperimentConfig,
            output_base_dir: str = "results",
    ):
        self.config = config
        self.output_base_dir = Path(output_base_dir)
        self.output_dir = self.output_base_dir / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_all_seeds(config.seed)

    def _prepare_training_problems(self) -> Tuple[str, List[Tuple[str, str]], List[str]]:
        """Prepare training problems."""
        domain_file = get_domain_file(self.config.domain.value)
        problem_files, problem_names = select_problems(
            self.config.train_problem_pattern,
            self.config.train_num_problems,
            self.config.seed
        )

        benchmarks = [(domain_file, pf) for pf in problem_files]

        print(f"\n📋 Training Configuration:")
        print(f"   Domain: {self.config.domain.value}")
        print(f"   Problem size: {self.config.train_problem_size.value}")
        print(f"   Selected {len(benchmarks)} training problems:")
        for name in problem_names:
            print(f"      • {name}")

        return domain_file, benchmarks, problem_names

    def run_training(self) -> Optional[Tuple[str, 'GNNTrainer']]:
        """Execute training phase."""
        print("\n" + "=" * 100)
        print(f"🚀 TRAINING PHASE - {self.config.name}")
        print("=" * 100)

        domain_file, benchmarks, problem_names = self._prepare_training_problems()

        trainer = GNNTrainer(
            benchmarks=benchmarks,
            problem_names=problem_names,
            output_dir=str(self.output_dir),
            reward_weights=DEFAULT_REWARD_WEIGHTS.copy(),
            max_merges=self.config.max_merges,
            timeout_per_step=self.config.timeout_per_step,
            checkpoint_interval=self.config.checkpoint_interval,
            min_episodes_per_problem=self.config.min_episodes_per_problem,
            seed=self.config.seed,
        )

        model_path = trainer.run_training(
            num_episodes=self.config.num_train_episodes,
            timesteps_per_episode=self.config.timesteps_per_episode,
        )

        trainer.save_training_log()
        trainer.close_logger()

        if not model_path:
            print("\n❌ Training failed")
            return None

        print(f"\n✅ Training complete!")
        print(f"   Model: {model_path}")
        print(f"   Checkpoints: {len(list(trainer.checkpoints_dir.glob('*.zip')))}")
        print(f"   Failed episodes: {trainer.failed_episode_count}")

        return model_path, trainer

    def run_evaluation(
            self,
            model_path: str,
            trainer: Optional['GNNTrainer'] = None,
    ) -> Dict:
        """Execute evaluation phase on training problems using GNN policy."""
        print("\n" + "=" * 100)
        print(f"📊 EVALUATION PHASE (Training Problems) - {self.config.name}")
        print("=" * 100)

        domain_file, benchmarks, problem_names = self._prepare_training_problems()
        problem_files = [pf for _, pf in benchmarks]

        # ✅ USE NEW GNNPolicyEvaluator
        from experiments.core.gnn_random_evaluation import GNNPolicyEvaluator

        evaluator = GNNPolicyEvaluator(
            model_path=model_path,
            downward_dir=str(PROJECT_ROOT / "downward"),  # ✅ EXPLICIT PATH
            max_merges=self.config.max_merges,
            timeout_per_step=self.config.timeout_per_step,
        )

        eval_results = evaluator.evaluate_problems(
            domain_file=domain_file,
            problem_files=problem_files,
            num_runs_per_problem=self.config.eval_runs_per_problem,
        )

        print("\n✅ Evaluation complete!")
        print(f"   Solved: {sum(1 for r in eval_results if r.solved)}/{len(eval_results)}")
        print(f"   Avg time: {sum(r.wall_clock_time for r in eval_results) / len(eval_results):.2f}s")

        return {
            "results": eval_results,
            "num_solved": sum(1 for r in eval_results if r.solved),
            "num_problems": len(eval_results),
        }

    def run_test(
            self,
            model_path: str,
            test_config: TestConfig,
    ) -> Dict:
        """Execute testing phase on test problems using GNN policy."""
        print(f"\n📋 Testing: {test_config.name}")
        print(f"   Domain: {test_config.domain.value}")
        print(f"   Size: {test_config.problem_size.value}")
        print(f"   Description: {test_config.description}")

        problem_files, problem_names = select_problems(
            test_config.problem_pattern,
            test_config.num_problems,
            self.config.seed + 1000  # Different seed for test
        )

        domain_file = get_domain_file(test_config.domain.value)

        # ✅ USE NEW GNNPolicyEvaluator
        from experiments.core.gnn_random_evaluation import GNNPolicyEvaluator

        evaluator = GNNPolicyEvaluator(
            model_path=model_path,
            downward_dir=str(PROJECT_ROOT / "downward"),  # ✅ EXPLICIT PATH
            max_merges=self.config.max_merges,
            timeout_per_step=self.config.timeout_per_step,
        )

        test_results = evaluator.evaluate_problems(
            domain_file=domain_file,
            problem_files=problem_files,
            num_runs_per_problem=test_config.num_runs_per_problem,
        )

        return {
            "test_config": test_config.name,
            "results": test_results,
            "num_solved": sum(1 for r in test_results if r.solved),
            "num_problems": len(test_results),
        }

    def run_analysis(
            self,
            trainer: 'GNNTrainer',
            eval_results: Dict,
    ) -> Dict:
        """Execute analysis phase."""
        print("\n" + "=" * 100)
        print(f"🔍 ANALYSIS PHASE - {self.config.name}")
        print("=" * 100)

        # Safely extract evaluation results list
        eval_results_list = None
        if isinstance(eval_results, dict):
            eval_results_list = eval_results.get("results", [])
        elif isinstance(eval_results, list):
            eval_results_list = eval_results
        else:
            eval_results_list = []

        # Run comprehensive analysis
        domain_file, benchmarks, problem_names = self._prepare_training_problems()

        summary = analyze_training_results(
            trainer.episode_log,
            eval_results_list,  # Pass the list explicitly
            problem_names,
            benchmarks,
            experiment_id=trainer.get_experiment_id(),
        )

        # Save summary
        summary_path = self.output_dir / "experiment_summary.json"
        save_json_atomic(summary.to_dict(), str(summary_path))

        print(f"\n✅ Analysis complete!")
        print(f"   Summary saved to: {summary_path}")

        return summary.to_dict()

    def run_visualization(
            self,
            trainer: 'GNNTrainer',
    ):
        """Generate all visualization plots."""
        print("\n📈 Generating visualizations...")

        plot_results = generate_all_plots(
            trainer.episode_log,
            {},
            str(self.output_dir),
            episode_reward_signals=trainer.episode_reward_signals if hasattr(trainer,
                                                                             'episode_reward_signals') else {},
        )

        print(f"✅ Visualizations complete!")
        return plot_results

    def run_full_experiment(self) -> Dict:
        """Execute complete experiment: train → evaluate → analyze → visualize."""
        print("\n" + "=" * 100)
        print(f"🔬 FULL EXPERIMENT - {self.config.name}")
        print(f"   {self.config.description}")
        print("=" * 100)

        # Phase 1: Training
        result = self.run_training()
        if not result:
            return {"status": "failed", "phase": "training"}

        model_path, trainer = result

        # Phase 2: Evaluation (training problems)
        eval_results = self.run_evaluation(model_path, trainer)

        # Phase 3: Test on test configurations
        test_results = {}
        for test_config in self.config.test_configurations:
            test_results[test_config.name] = self.run_test(model_path, test_config)

        # Phase 4: Analysis
        summary = self.run_analysis(trainer, eval_results)

        # Phase 5: Visualization
        self.run_visualization(trainer)

        # Save test results
        test_results_path = self.output_dir / "test_results.json"
        save_json_atomic(test_results, str(test_results_path))

        print("\n" + "=" * 100)
        print(f"✅ EXPERIMENT COMPLETE - {self.config.name}")
        print("=" * 100)
        print(f"   Output: {self.output_dir.absolute()}")

        return {
            "status": "success",
            "config": self.config.to_dict(),
            "summary": summary,
            "test_results": test_results,
        }