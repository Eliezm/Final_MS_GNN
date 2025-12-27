#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENT RUNNER - Orchestrates training, evaluation, and analysis
"""
import json
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
from experiments.shared_experiment_utils import DEFAULT_REWARD_WEIGHTS, REWARD_FUNCTION_CONFIG


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
    """
    Get domain PDDL file path (FIXED v2).

    ✅ FIXED: Multiple fallback strategies
    """
    domain_lower = domain.lower()
    domain_capitalized = domain.capitalize()

    # Try multiple patterns
    possible_paths = [
        # Pattern 1: benchmarks/{domain}/domain.pddl
        f"benchmarks/{domain_lower}/domain.pddl",
        f"benchmarks/{domain_capitalized}/domain.pddl",

        # Pattern 2: benchmarks/{domain}/small/domain.pddl (and other sizes)
        f"benchmarks/{domain_lower}/small/domain.pddl",
        f"benchmarks/{domain_lower}/medium/domain.pddl",
        f"benchmarks/{domain_lower}/large/domain.pddl",

        # Pattern 3: domains/{domain}.pddl
        f"domains/{domain_lower}.pddl",
        f"domains/{domain_capitalized}.pddl",

        # Pattern 4: From PROJECT_ROOT
        str(PROJECT_ROOT / "benchmarks" / domain_lower / "domain.pddl"),
        str(PROJECT_ROOT / "benchmarks" / domain_capitalized / "domain.pddl"),
    ]

    for path in possible_paths:
        if Path(path).exists():
            return str(Path(path).absolute())

    # Last resort: search in current directory
    import glob
    matches = glob.glob(f"**/domain.pddl", recursive=True)
    if matches:
        for m in matches:
            if domain_lower in m.lower():
                return str(Path(m).absolute())

    raise FileNotFoundError(
        f"Domain file not found for: {domain}\n"
        f"Tried: {possible_paths[:3]}\n"
        f"Make sure benchmarks are in benchmarks/ directory"
    )

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
        """Execute full curriculum learning pipeline with UNIFIED OUTPUTS."""
        print("\n" + "=" * 100)
        print(f"🎓 CURRICULUM LEARNING EXPERIMENT - {self.config.name}")
        print(f"   {self.config.description}")
        print(f"   Phases: {len(self.config.curriculum_phases)}")
        print("=" * 100)

        from experiments.core.analysis import (
            analyze_training_results,
            analyze_component_trajectories,
            analyze_feature_reward_correlation,
            analyze_feature_importance_from_decisions,
            analyze_causal_alignment,
            analyze_transition_explosion_risk,
            analyze_gnn_decision_quality,
            analyze_bisimulation_preservation,
            generate_literature_alignment_report,
        )

        # ✅ UNIFIED: Create same directory structure as regular runner
        (self.output_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "testing").mkdir(parents=True, exist_ok=True)

        phase_results = {}
        current_model_path = None
        all_episode_logs = []
        all_reward_signals = {}
        all_benchmarks = []
        all_problem_names = []

        # ====================================================================
        # PHASE LOOP: Train progressively larger problems
        # ====================================================================

        # Before the phase loop:
        cumulative_episodes = 0
        total_all_episodes = sum(phase.num_episodes for phase in self.config.curriculum_phases)

        for phase_idx, phase in enumerate(self.config.curriculum_phases, 1):
            print(f"\n{'=' * 100}")
            print(f"📚 CURRICULUM PHASE {phase_idx}/{len(self.config.curriculum_phases)}")
            print(f"   {phase.name}: {phase.problem_size.value}")
            print(f"   Problems: {phase.num_problems}, Episodes: {phase.num_episodes}")
            print(f"{'=' * 100}\n")

            problem_files, problem_names = select_problems(
                phase.problem_pattern,
                phase.num_problems,
                self.config.seed + phase_idx,
            )

            domain_file = get_domain_file(self.config.domain.value)
            benchmarks = [(domain_file, pf) for pf in problem_files]

            all_benchmarks.extend(benchmarks)
            all_problem_names.extend(problem_names)

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

            cumulative_episodes += phase.num_episodes

            if not model_path:
                print(f"\n❌ Phase {phase_idx} training failed")
                return {"status": "failed", "phase": phase_idx, "phase_name": phase.name}

            trainer.save_training_log()
            trainer.close_logger()

            # Save phase model
            phase_model_path = phase_output_dir / "model_final.zip"
            import shutil
            shutil.copy(model_path, phase_model_path)
            current_model_path = str(phase_model_path)

            # ✅ UNIFIED: Also save to checkpoints directory
            checkpoint_path = self.output_dir / "checkpoints" / f"model_phase_{phase_idx}.zip"
            shutil.copy(model_path, checkpoint_path)

            # Compute phase analysis
            phase_summary = analyze_training_results(
                trainer.episode_log,
                [],
                problem_names,
                benchmarks,
                experiment_id=trainer.get_experiment_id(),
            )

            phase_results[phase.name] = {
                "phase": phase_idx,
                "model_path": current_model_path,
                "training_log": [m.to_dict() for m in trainer.episode_log],
                "output_dir": str(phase_output_dir),
                "summary": phase_summary.to_dict(),
                "num_episodes": len(trainer.episode_log),
                "num_failed": trainer.failed_episode_count,
            }

            all_episode_logs.extend(trainer.episode_log)

            # Accumulate reward signals
            if hasattr(trainer, 'episode_reward_signals') and trainer.episode_reward_signals:
                phase_offset = phase_idx * 10000
                for episode_num, reward_data in trainer.episode_reward_signals.items():
                    all_reward_signals[episode_num + phase_offset] = reward_data

            print(f"\n✅ Phase {phase_idx} complete!")
            print(f"   Model: {current_model_path}")
            print(f"   Episodes: {len(trainer.episode_log)}")

        # ====================================================================
        # ✅ UNIFIED OUTPUT: Save model.zip at root level
        # ====================================================================

        final_model_path = self.output_dir / "model.zip"
        import shutil
        shutil.copy(current_model_path, final_model_path)
        print(f"\n✅ Final model saved: {final_model_path}")

        # ====================================================================
        # ✅ UNIFIED OUTPUT: Save training_log.jsonl at root level
        # ====================================================================

        training_log_jsonl = self.output_dir / "training_log.jsonl"
        with open(training_log_jsonl, 'w', encoding='utf-8') as f:
            for metrics in all_episode_logs:
                import json
                f.write(json.dumps(metrics.to_dict()) + '\n')
        print(f"✅ Training log saved: {training_log_jsonl}")

        # ====================================================================
        # FINAL EVALUATION
        # ====================================================================

        print(f"\n{'=' * 100}")
        print(f"📊 FINAL EVALUATION (on final curriculum model)")
        print(f"{'=' * 100}\n")

        from experiments.core.gnn_random_evaluation import GNNPolicyEvaluator

        test_results = {}
        for test_config in self.config.test_configurations:
            test_result = self.run_test_with_dedicated_output(model_path, test_config)
            test_results[test_config.name] = test_result

        # ====================================================================
        # COMPUTE ALL ANALYSES
        # ====================================================================

        print("\n" + "=" * 100)
        print("🔍 CURRICULUM ANALYSIS - COMPUTING ALL METRICS")
        print("=" * 100)

        curriculum_summary_data = analyze_training_results(
            all_episode_logs, [], all_problem_names, all_benchmarks,
            experiment_id="curriculum_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

        component_analysis = analyze_component_trajectories(all_episode_logs, self.output_dir)
        correlation_analysis = analyze_feature_reward_correlation(all_reward_signals, self.output_dir)
        feature_importance_analysis = analyze_feature_importance_from_decisions(all_episode_logs, self.output_dir)
        causal_alignment_analysis = analyze_causal_alignment(all_episode_logs, self.output_dir)
        explosion_analysis = analyze_transition_explosion_risk(all_episode_logs, self.output_dir)

        decision_quality_analysis = analyze_gnn_decision_quality(
            {i: m.merge_decisions_per_step for i, m in enumerate(all_episode_logs)
             if m.merge_decisions_per_step},
            self.output_dir,
        )

        bisim_analysis = analyze_bisimulation_preservation(all_episode_logs, self.output_dir)

        literature_checklist = generate_literature_alignment_report(
            all_episode_logs, all_reward_signals, correlation_analysis, bisim_analysis, self.output_dir,
        )

        print("✅ All 10 analysis methods completed!")

        # ====================================================================
        # ✅ UNIFIED: Save analysis to analysis/ directory
        # ====================================================================

        analysis_dir = self.output_dir / "analysis"
        save_json_atomic({
            'main_summary': curriculum_summary_data.to_dict(),
            'component_analysis': component_analysis,
            'correlation_analysis': correlation_analysis,
            'feature_importance': feature_importance_analysis,
            'causal_alignment': causal_alignment_analysis,
            'explosion_risk': explosion_analysis,
            'decision_quality': decision_quality_analysis,
            'bisimulation': bisim_analysis,
            'literature_checklist': literature_checklist,
        }, str(analysis_dir / "all_analysis_results.json"))

        # ====================================================================
        # GENERATE VISUALIZATIONS
        # ====================================================================

        print("\n📈 GENERATING VISUALIZATIONS")

        from experiments.core.visualization import generate_all_plots

        plot_results = generate_all_plots(
            all_episode_logs, {},
            str(self.output_dir),
            component_analysis=component_analysis,
            correlation_analysis=correlation_analysis,
            feature_importance_analysis=feature_importance_analysis,
            bisim_analysis=bisim_analysis,
            causal_alignment_analysis=causal_alignment_analysis,
            explosion_analysis=explosion_analysis,
            decision_quality_analysis=decision_quality_analysis,
            episode_reward_signals=all_reward_signals,
            literature_checklist=literature_checklist,
        )

        successful_plots = sum(1 for p in plot_results.values() if p is not None)
        print(f"✅ Generated {successful_plots}/{len(plot_results)} plots")

        # ====================================================================
        # ✅ UNIFIED: Save experiment_summary.json
        # ====================================================================

        experiment_summary = {
            "status": "success",
            "config": self.config.to_dict(),
            "num_phases": len(self.config.curriculum_phases),
            "phases": phase_results,
            "test_results": test_results,
            "final_model_path": str(final_model_path),
            "overall_summary": curriculum_summary_data.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "total_episodes": len(all_episode_logs),
            "total_failed": sum(p["num_failed"] for p in phase_results.values()),
        }

        save_json_atomic(experiment_summary, str(self.output_dir / "experiment_summary.json"))

        # ✅ UNIFIED: Save test_results.json
        save_json_atomic(test_results, str(self.output_dir / "test_results.json"))

        # ✅ UNIFIED: Save curriculum_summary.json (for backward compat)
        save_json_atomic(experiment_summary, str(self.output_dir / "curriculum_summary.json"))

        # ✅ UNIFIED: Save training.log
        training_log_path = self.output_dir / "training.log"
        with open(training_log_path, 'w') as f:
            f.write("CURRICULUM TRAINING LOG\n")
            f.write("=" * 80 + "\n\n")
            for phase_name, phase_data in phase_results.items():
                f.write(f"PHASE: {phase_name}\n")
                f.write(f"  Episodes: {phase_data['num_episodes']}\n")
                f.write(f"  Failed: {phase_data['num_failed']}\n")
                f.write(f"  Summary: {phase_data['summary']['avg_reward_over_all']:.4f} avg reward\n\n")

        print("\n" + "=" * 100)
        print(f"✅ CURRICULUM LEARNING COMPLETE - {self.config.name}")
        print("=" * 100)
        print(f"\n📁 Output directory: {self.output_dir.absolute()}")
        print(f"\n📂 UNIFIED OUTPUT FILES:")
        print(f"   ✓ model.zip (final trained model)")
        print(f"   ✓ training_log.jsonl (all episode metrics)")
        print(f"   ✓ experiment_summary.json (full summary)")
        print(f"   ✓ test_results.json (test set results)")
        print(f"   ✓ training.log (human-readable log)")
        print(f"   ✓ analysis/ (all analysis JSON)")
        print(f"   ✓ plots/ (all visualizations)")
        print(f"   ✓ checkpoints/ (phase models)")

        return experiment_summary

class ExperimentRunner:
    """Orchestrates a complete experiment."""

    def __init__(
            self,
            config: ExperimentConfig,
            output_base_dir: str = "results",
    ):
        self.config = config
        # self.reward_function_config = reward_function_config or REWARD_FUNCTION_CONFIG


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

    def run_test_with_dedicated_output(
            self,
            model_path: str,
            test_config: TestConfig,
    ) -> Dict:
        """
        Execute testing phase with DEDICATED output folder per test set.

        Creates:
            testing/{test_config.name}/
                ├── results.json
                ├── comparison.json
                ├── plots/
                │   ├── solve_rate.png
                │   └── time_comparison.png
                └── summary.txt
        """
        from experiments.core.gnn_random_evaluation import GNNPolicyEvaluator, RandomMergeEvaluator
        from experiments.core.evaluation_plots import GenerateEvaluationPlots
        from experiments.core.evaluation_analyzer import ComparisonAnalyzer

        print(f"\n📋 Testing: {test_config.name}")
        print(f"   Domain: {test_config.domain.value}")
        print(f"   Size: {test_config.problem_size.value}")
        print(f"   Description: {test_config.description}")

        # Create dedicated output directory for this test set
        test_output_dir = self.output_dir / "testing" / test_config.name
        test_output_dir.mkdir(parents=True, exist_ok=True)
        (test_output_dir / "plots").mkdir(exist_ok=True)

        problem_files, problem_names = select_problems(
            test_config.problem_pattern,
            test_config.num_problems,
            self.config.seed + 1000
        )

        domain_file = get_domain_file(test_config.domain.value)

        # =========================================================================
        # RUN GNN EVALUATION
        # =========================================================================

        gnn_evaluator = GNNPolicyEvaluator(
            model_path=model_path,
            downward_dir=str(PROJECT_ROOT / "downward"),
            max_merges=self.config.max_merges,
            timeout_per_step=self.config.timeout_per_step,
        )

        gnn_results = gnn_evaluator.evaluate_problems(
            domain_file=domain_file,
            problem_files=problem_files,
            num_runs_per_problem=test_config.num_runs_per_problem,
        )

        # =========================================================================
        # RUN RANDOM EVALUATION
        # =========================================================================

        random_evaluator = RandomMergeEvaluator(
            downward_dir=str(PROJECT_ROOT / "downward"),
            max_merges=self.config.max_merges,
            timeout_per_step=self.config.timeout_per_step,
        )

        random_results = random_evaluator.evaluate_problems(
            domain_file=domain_file,
            problem_files=problem_files,
            num_runs_per_problem=test_config.num_runs_per_problem,
        )

        # =========================================================================
        # COMPUTE STATISTICS
        # =========================================================================

        all_results = gnn_results + random_results

        gnn_stats = {}
        random_stats = {}

        if all_results:
            analyzer = ComparisonAnalyzer(all_results)

            try:
                gnn_stats = analyzer.get_aggregate_statistics("GNN").to_dict()
            except:
                gnn_stats = {}

            try:
                random_stats = analyzer.get_aggregate_statistics("Random").to_dict()
            except:
                random_stats = {}

        # =========================================================================
        # SAVE RESULTS
        # =========================================================================

        results_data = {
            "test_config": test_config.to_dict(),
            "gnn_results": [r.to_dict() for r in gnn_results],
            "random_results": [r.to_dict() for r in random_results],
            "gnn_statistics": gnn_stats,
            "random_statistics": random_stats,
            "summary": {
                "gnn_solved": sum(1 for r in gnn_results if r.solved),
                "gnn_total": len(gnn_results),
                "random_solved": sum(1 for r in random_results if r.solved),
                "random_total": len(random_results),
            }
        }

        # Save detailed results
        with open(test_output_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        # Save comparison
        comparison_data = {
            "GNN": gnn_stats,
            "Random": random_stats,
            "improvement": {
                "solve_rate_diff": gnn_stats.get('solve_rate_pct', 0) - random_stats.get('solve_rate_pct', 0),
                "time_ratio": (random_stats.get('mean_time_sec', 1) / max(0.001, gnn_stats.get('mean_time_sec', 1)))
                if gnn_stats.get('mean_time_sec', 0) > 0 else 0,
            }
        }

        with open(test_output_dir / "comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)

        # =========================================================================
        # GENERATE TEST-SPECIFIC PLOTS
        # =========================================================================

        try:
            plotter = GenerateEvaluationPlots(output_dir=str(test_output_dir / "plots"))
            plotter.generate_all_plots(
                statistics={"GNN": gnn_stats, "Random": random_stats},
                results=all_results,
                gnn_results=gnn_stats,
            )
            print(f"   ✓ Plots saved to: {test_output_dir / 'plots'}")
        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")

        # =========================================================================
        # SAVE HUMAN-READABLE SUMMARY
        # =========================================================================

        summary_text = f"""
    {'=' * 80}
    TEST SET: {test_config.name}
    {'=' * 80}

    Description: {test_config.description}
    Domain: {test_config.domain.value}
    Problem Size: {test_config.problem_size.value}
    Problems Tested: {len(problem_files)}
    Runs per Problem: {test_config.num_runs_per_problem}

    {'=' * 80}
    RESULTS SUMMARY
    {'=' * 80}

    GNN Policy:
      - Solved: {results_data['summary']['gnn_solved']}/{results_data['summary']['gnn_total']}
      - Solve Rate: {gnn_stats.get('solve_rate_pct', 0):.1f}%
      - Mean Time: {gnn_stats.get('mean_time_sec', 0):.3f}s
      - Mean Expansions: {gnn_stats.get('mean_expansions', 0):,}
      - H* Preservation: {gnn_stats.get('mean_h_preservation', 1.0):.4f}

    Random Merge:
      - Solved: {results_data['summary']['random_solved']}/{results_data['summary']['random_total']}
      - Solve Rate: {random_stats.get('solve_rate_pct', 0):.1f}%
      - Mean Time: {random_stats.get('mean_time_sec', 0):.3f}s
      - Mean Expansions: {random_stats.get('mean_expansions', 0):,}

    {'=' * 80}
    COMPARISON
    {'=' * 80}

    GNN vs Random Improvement:
      - Solve Rate: {comparison_data['improvement']['solve_rate_diff']:+.1f}%
      - Speedup: {comparison_data['improvement']['time_ratio']:.2f}x

    {'=' * 80}
    """

        with open(test_output_dir / "summary.txt", 'w') as f:
            f.write(summary_text)

        print(f"   ✓ Results saved to: {test_output_dir}")

        return {
            "test_config": test_config.name,
            "output_dir": str(test_output_dir),
            "results": results_data,
            "num_problems": len(problem_files),
            "num_solved": results_data['summary']['gnn_solved'],
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

        # ✅ FIX: Ensure all directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "evaluation").mkdir(parents=True, exist_ok=True)

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
            test_results[test_config.name] = self.run_test_with_dedicated_output(model_path, test_config)

        # ====================================================================
        # ✅ COMPLETE ANALYSIS - USE ALL METHODS
        # ====================================================================

        print("\n" + "=" * 100)
        print("🔬 COMPREHENSIVE ANALYSIS - CALLING ALL ANALYSIS METHODS")
        print("=" * 100 + "\n")

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

        domain_file, benchmarks, problem_names = self._prepare_training_problems()

        print("[1/10] Main summary...")
        summary = analyze_training_results(
            trainer.episode_log,
            eval_results.get("results", []),
            problem_names,
            benchmarks,
            experiment_id=trainer.get_experiment_id(),
        )

        print("[2/10] Component trajectories...")
        component_analysis = analyze_component_trajectories(
            trainer.episode_log,
            self.output_dir,
        )

        print("[3/10] Feature-reward correlation...")
        correlation_analysis = analyze_feature_reward_correlation(
            trainer.episode_reward_signals if hasattr(trainer, 'episode_reward_signals') else {},
            self.output_dir,
        )

        print("[4/10] Feature importance...")
        feature_importance_analysis = analyze_feature_importance_from_decisions(
            trainer.episode_log,
            self.output_dir,
        )

        print("[5/10] Causal alignment...")
        causal_alignment_analysis = analyze_causal_alignment(
            trainer.episode_log,
            self.output_dir,
        )

        print("[6/10] Transition explosion risk...")
        explosion_analysis = analyze_transition_explosion_risk(
            trainer.episode_log,
            self.output_dir,
        )

        print("[7/10] GNN decision quality...")
        decision_quality_analysis = analyze_gnn_decision_quality(
            {i: m.merge_decisions_per_step for i, m in enumerate(trainer.episode_log)
             if m.merge_decisions_per_step},
            self.output_dir,
        )

        print("[8/10] Bisimulation preservation...")
        bisim_analysis = analyze_bisimulation_preservation(
            trainer.episode_log,
            self.output_dir,
        )

        print("[9/10] Dead-end creation analysis...")
        safety_analysis = analyze_dead_end_creation(
            trainer.episode_log,
            self.output_dir,
        )

        print("[10/10] Literature alignment report...")
        literature_checklist = generate_literature_alignment_report(
            trainer.episode_log,
            trainer.episode_reward_signals if hasattr(trainer, 'episode_reward_signals') else {},
            correlation_analysis,
            bisim_analysis,
            self.output_dir,
        )

        print("\n✅ All 10 analysis methods completed!\n")

        # ====================================================================
        # ✅ SAVE ALL ANALYSIS RESULTS
        # ====================================================================

        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        save_json_atomic({
            'main_summary': summary.to_dict(),
            'component_analysis': component_analysis,
            'correlation_analysis': correlation_analysis,
            'feature_importance': feature_importance_analysis,
            'causal_alignment': causal_alignment_analysis,
            'explosion_risk': explosion_analysis,
            'decision_quality': decision_quality_analysis,
            'bisimulation': bisim_analysis,
            'safety': safety_analysis,
            'literature_checklist': literature_checklist,
        }, str(analysis_dir / "all_analysis_results.json"))

        print(f"✅ Analysis results saved to: {analysis_dir / 'all_analysis_results.json'}\n")

        # ====================================================================
        # Phase 5: Visualization - ✅ PASS ALL ANALYSES AND TEST RESULTS
        # ====================================================================

        print("📈 Generating visualizations (32 plots)...")
        from experiments.core.visualization import generate_all_plots

        plot_results = generate_all_plots(
            trainer.episode_log,
            {},
            str(self.output_dir),
            component_analysis=component_analysis,
            correlation_analysis=correlation_analysis,
            feature_importance_analysis=feature_importance_analysis,
            bisim_analysis=bisim_analysis,
            causal_alignment_analysis=causal_alignment_analysis,
            explosion_analysis=explosion_analysis,
            decision_quality_analysis=decision_quality_analysis,
            episode_reward_signals=trainer.episode_reward_signals if hasattr(trainer, 'episode_reward_signals') else {},
            literature_checklist=literature_checklist,
            # ✅ NEW: Pass test results for generalization plots
            test_results=test_results,
            is_curriculum=False,
        )

        successful_plots = sum(1 for p in plot_results.values() if p is not None)
        print(f"✅ Visualizations complete: {successful_plots}/{len(plot_results)} plots generated\n")

        # Save summary
        summary_path = self.output_dir / "experiment_summary.json"
        save_json_atomic(summary.to_dict(), str(summary_path))

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
            "summary": summary.to_dict(),
            "test_results": test_results,
        }

    def run_test_with_dedicated_output(
            self,
            model_path: str,
            test_config: TestConfig,
    ) -> Dict:
        """
        Execute testing phase with DEDICATED output folder per test set.

        Creates:
            testing/{test_config.name}/
                ├── results.json
                ├── comparison.json
                ├── plots/
                │   ├── solve_rate.png
                │   └── time_comparison.png
                └── summary.txt
        """
        from experiments.core.gnn_random_evaluation import GNNPolicyEvaluator, RandomMergeEvaluator
        from experiments.core.evaluation_plots import GenerateEvaluationPlots
        from experiments.core.evaluation_analyzer import ComparisonAnalyzer

        print(f"\n📋 Testing: {test_config.name}")
        print(f"   Domain: {test_config.domain.value}")
        print(f"   Size: {test_config.problem_size.value}")
        print(f"   Description: {test_config.description}")

        # Create dedicated output directory for this test set
        test_output_dir = self.output_dir / "testing" / test_config.name
        test_output_dir.mkdir(parents=True, exist_ok=True)
        (test_output_dir / "plots").mkdir(exist_ok=True)

        problem_files, problem_names = select_problems(
            test_config.problem_pattern,
            test_config.num_problems,
            self.config.seed + 1000
        )

        domain_file = get_domain_file(test_config.domain.value)

        # =========================================================================
        # RUN GNN EVALUATION
        # =========================================================================

        gnn_evaluator = GNNPolicyEvaluator(
            model_path=model_path,
            downward_dir=str(PROJECT_ROOT / "downward"),
            max_merges=self.config.max_merges,
            timeout_per_step=self.config.timeout_per_step,
        )

        gnn_results = gnn_evaluator.evaluate_problems(
            domain_file=domain_file,
            problem_files=problem_files,
            num_runs_per_problem=test_config.num_runs_per_problem,
        )

        # =========================================================================
        # RUN RANDOM EVALUATION
        # =========================================================================

        random_evaluator = RandomMergeEvaluator(
            downward_dir=str(PROJECT_ROOT / "downward"),
            max_merges=self.config.max_merges,
            timeout_per_step=self.config.timeout_per_step,
        )

        random_results = random_evaluator.evaluate_problems(
            domain_file=domain_file,
            problem_files=problem_files,
            num_runs_per_problem=test_config.num_runs_per_problem,
        )

        # =========================================================================
        # COMPUTE STATISTICS
        # =========================================================================

        all_results = gnn_results + random_results

        gnn_stats = {}
        random_stats = {}

        if all_results:
            analyzer = ComparisonAnalyzer(all_results)

            try:
                gnn_stats = analyzer.get_aggregate_statistics("GNN").to_dict()
            except:
                gnn_stats = {}

            try:
                random_stats = analyzer.get_aggregate_statistics("Random").to_dict()
            except:
                random_stats = {}

        # =========================================================================
        # SAVE RESULTS
        # =========================================================================

        results_data = {
            "test_config": test_config.to_dict(),
            "gnn_results": [r.to_dict() for r in gnn_results],
            "random_results": [r.to_dict() for r in random_results],
            "gnn_statistics": gnn_stats,
            "random_statistics": random_stats,
            "summary": {
                "gnn_solved": sum(1 for r in gnn_results if r.solved),
                "gnn_total": len(gnn_results),
                "random_solved": sum(1 for r in random_results if r.solved),
                "random_total": len(random_results),
            }
        }

        # Save detailed results
        with open(test_output_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        # Save comparison
        comparison_data = {
            "GNN": gnn_stats,
            "Random": random_stats,
            "improvement": {
                "solve_rate_diff": gnn_stats.get('solve_rate_pct', 0) - random_stats.get('solve_rate_pct', 0),
                "time_ratio": (random_stats.get('mean_time_sec', 1) / max(0.001, gnn_stats.get('mean_time_sec', 1)))
                if gnn_stats.get('mean_time_sec', 0) > 0 else 0,
            }
        }

        with open(test_output_dir / "comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)

        # =========================================================================
        # GENERATE TEST-SPECIFIC PLOTS
        # =========================================================================

        try:
            plotter = GenerateEvaluationPlots(output_dir=str(test_output_dir / "plots"))
            plotter.generate_all_plots(
                statistics={"GNN": gnn_stats, "Random": random_stats},
                results=all_results,
                gnn_results=gnn_stats,
            )
            print(f"   ✓ Plots saved to: {test_output_dir / 'plots'}")
        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")

        # =========================================================================
        # SAVE HUMAN-READABLE SUMMARY
        # =========================================================================

        summary_text = f"""
    {'=' * 80}
    TEST SET: {test_config.name}
    {'=' * 80}

    Description: {test_config.description}
    Domain: {test_config.domain.value}
    Problem Size: {test_config.problem_size.value}
    Problems Tested: {len(problem_files)}
    Runs per Problem: {test_config.num_runs_per_problem}

    {'=' * 80}
    RESULTS SUMMARY
    {'=' * 80}

    GNN Policy:
      - Solved: {results_data['summary']['gnn_solved']}/{results_data['summary']['gnn_total']}
      - Solve Rate: {gnn_stats.get('solve_rate_pct', 0):.1f}%
      - Mean Time: {gnn_stats.get('mean_time_sec', 0):.3f}s
      - Mean Expansions: {gnn_stats.get('mean_expansions', 0):,}
      - H* Preservation: {gnn_stats.get('mean_h_preservation', 1.0):.4f}

    Random Merge:
      - Solved: {results_data['summary']['random_solved']}/{results_data['summary']['random_total']}
      - Solve Rate: {random_stats.get('solve_rate_pct', 0):.1f}%
      - Mean Time: {random_stats.get('mean_time_sec', 0):.3f}s
      - Mean Expansions: {random_stats.get('mean_expansions', 0):,}

    {'=' * 80}
    COMPARISON
    {'=' * 80}

    GNN vs Random Improvement:
      - Solve Rate: {comparison_data['improvement']['solve_rate_diff']:+.1f}%
      - Speedup: {comparison_data['improvement']['time_ratio']:.2f}x

    {'=' * 80}
    """

        with open(test_output_dir / "summary.txt", 'w') as f:
            f.write(summary_text)

        print(f"   ✓ Results saved to: {test_output_dir}")

        return {
            "test_config": test_config.name,
            "output_dir": str(test_output_dir),
            "results": results_data,
            "num_problems": len(problem_files),
            "num_solved": results_data['summary']['gnn_solved'],
        }