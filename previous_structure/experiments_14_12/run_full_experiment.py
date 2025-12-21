#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASTER SCRIPT - Run complete end-to-end experiment
Trains GNN, analyzes, visualizes, evaluates, tests, and compares against baselines.
All outputs saved to structured directory.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs.experiment_configs import get_experiment, list_experiments
from experiments.runners.experiment_runner import ExperimentRunner, CurriculumExperimentRunner
from experiments.core.evaluation import EvaluationFramework


def create_final_report(
        experiment_dir: Path,
        config_dict: dict,
        training_summary: dict,
        eval_summary: dict,
        test_summaries: dict,
        baseline_summary: dict,
) -> Path:
    """Create master summary report combining all experiment phases."""

    report = {
        "experiment": {
            "name": config_dict.get("name", "unknown"),
            "description": config_dict.get("description", ""),
            "timestamp": datetime.now().isoformat(),
            "config": config_dict,
        },
        "training": training_summary,
        "gnn_vs_random": eval_summary,
        "testing": test_summaries,
        "baselines": baseline_summary,
        "output_directory": str(experiment_dir.absolute()),
        "summary": {
            "total_train_episodes": training_summary.get("num_train_episodes", 0),
            "training_duration_sec": training_summary.get("duration_seconds", 0),
            "gnn_solve_rate": eval_summary.get("GNN", {}).get("solve_rate_pct", 0),
            "gnn_vs_random_improvement": (
                    eval_summary.get("GNN", {}).get("solve_rate_pct", 0) -
                    eval_summary.get("Random", {}).get("solve_rate_pct", 0)
            ),
            "num_test_configs": len(test_summaries),
            "num_baseline_configs": len(baseline_summary.get("baseline_configs", [])),
        }
    }

    report_path = experiment_dir / "final_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return report_path


def run_full_experiment_pipeline(
        experiment_name: str,
        output_base_dir: str = "results",
        include_baselines: bool = True,
        seed: int = 42,
        verbose: bool = False,
) -> dict:
    """
    Execute complete experiment pipeline: Train → Analyze → Visualize → Evaluate → Test → Baselines

    Args:
        experiment_name: Name of experiment config to run
        output_base_dir: Base output directory
        include_baselines: Whether to run Fast Downward baselines
        seed: Random seed
        verbose: Verbose logging

    Returns:
        Summary dictionary with all results
    """

    print("\n" + "=" * 100)
    print(f"🔬 MASTER EXPERIMENT PIPELINE - {experiment_name}")
    print("=" * 100)

    # Load config
    config = get_experiment(experiment_name)
    print(f"\n📋 Experiment: {config.description}")
    print(f"   Domain: {config.domain.value}")
    print(f"   Training: {config.train_num_problems} problems, {config.num_train_episodes} episodes")
    print(f"   Testing: {len(config.test_configurations)} test configs")

    experiment_dir = Path(output_base_dir) / config.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    try:
        # ====================================================================
        # PHASE 1: TRAINING + ANALYSIS + VISUALIZATION
        # ====================================================================
        print("\n" + "-" * 100)
        print("PHASE 1: TRAINING, ANALYSIS & VISUALIZATION")
        print("-" * 100)

        if config.is_curriculum:
            print(f"\n🎓 Using CurriculumExperimentRunner (phases: {len(config.curriculum_phases)})")
            runner = CurriculumExperimentRunner(config, output_base_dir=output_base_dir)
            phase_result = runner.run_curriculum()
            all_results["training"] = phase_result

            # Extract final model path from curriculum
            final_model_path = phase_result.get("final_model_path")
            training_summary = phase_result
        else:
            print(f"\n🚀 Using standard ExperimentRunner")
            runner = ExperimentRunner(config, output_base_dir=output_base_dir)

            # Train
            train_result = runner.run_training()
            if not train_result:
                print("❌ Training failed!")
                return {"status": "failed", "phase": "training"}

            model_path, trainer = train_result
            final_model_path = model_path

            # Analyze & Visualize
            eval_results = runner.run_evaluation(model_path, trainer)
            runner.run_analysis(trainer, eval_results)
            runner.run_visualization(trainer)

            training_summary = {
                "status": "success",
                "model_path": str(final_model_path),
                "num_train_episodes": len(trainer.episode_log),
                "num_failed_episodes": trainer.failed_episode_count,
                "duration_seconds": trainer.resource_monitor.get_elapsed_ms() / 1000 if hasattr(trainer,
                                                                                                'resource_monitor') else 0,
            }
            all_results["training"] = training_summary

        # ====================================================================
        # PHASE 2: GNN vs RANDOM EVALUATION
        # ====================================================================
        print("\n" + "-" * 100)
        print("PHASE 2: GNN vs RANDOM MERGE STRATEGY")
        print("-" * 100)

        from experiments.core.gnn_random_evaluation import GNNRandomEvaluationFramework

        domain_file = f"benchmarks/{config.domain.value}/small/domain.pddl"
        problem_files = []

        for test_config in config.test_configurations[:1]:  # Use first test config for eval
            import glob
            problem_files = sorted(glob.glob(test_config.problem_pattern))[:5]  # Sample 5 problems
            break

        if problem_files and Path(final_model_path).exists():
            eval_fw = GNNRandomEvaluationFramework(
                model_path=final_model_path,
                domain_file=domain_file,
                problem_files=problem_files,
                output_dir=str(experiment_dir / "evaluation"),
                num_runs_per_problem=1,
                downward_dir=str(PROJECT_ROOT / "downward"),
            )

            gnn_results, random_results = eval_fw.evaluate()
            eval_summary = eval_fw.to_summary()

            all_results["evaluation"] = {
                "gnn_results": len(gnn_results) if gnn_results else 0,
                "random_results": len(random_results) if random_results else 0,
                "summary": eval_summary if eval_summary else {},
            }

            print(f"\n✅ Evaluation complete:")
            if eval_summary:
                if "GNN" in eval_summary and isinstance(eval_summary["GNN"], dict):
                    gnn_rate = eval_summary.get("GNN", {}).get("solve_rate_pct", 0)
                    print(f"   GNN solve rate: {gnn_rate:.1f}%")
                if "Random" in eval_summary and isinstance(eval_summary["Random"], dict):
                    random_rate = eval_summary.get("Random", {}).get("solve_rate_pct", 0)
                    print(f"   Random solve rate: {random_rate:.1f}%")
            else:
                print(f"   (No evaluation results available)")
        else:
            eval_summary = {}
            all_results["evaluation"] = {"status": "skipped", "reason": "No model or problems"}

        # ====================================================================
        # PHASE 3: TEST SET EVALUATION
        # ====================================================================
        print("\n" + "-" * 100)
        print("PHASE 3: TEST SET EVALUATION")
        print("-" * 100)

        test_runner = ExperimentRunner(config, output_base_dir=output_base_dir)
        test_summaries = {}

        for test_config in config.test_configurations:
            if Path(final_model_path).exists():
                test_result = test_runner.run_test(final_model_path, test_config)
                test_summaries[test_config.name] = {
                    "num_problems": test_result["num_problems"],
                    "num_solved": test_result["num_solved"],
                    "solve_rate": test_result["num_solved"] / max(1, test_result["num_problems"]) * 100,
                }
                print(f"✅ {test_config.name}: {test_summaries[test_config.name]['solve_rate']:.1f}% solved")

        all_results["testing"] = test_summaries

        # ====================================================================
        # PHASE 4: FAST DOWNWARD BASELINE EVALUATION (Optional)
        # ====================================================================
        if include_baselines:
            print("\n" + "-" * 100)
            print("PHASE 4: FAST DOWNWARD BASELINE COMPARISON")
            print("-" * 100)

            eval_framework = EvaluationFramework(output_dir=str(experiment_dir / "baselines"))
            baseline_result = eval_framework.run_baseline_evaluation(
                timeout_sec=300,
                max_problems=5,  # Sample for speed
                domains=config.domain.value,
                sizes="small,medium",
            )

            all_results["baselines"] = baseline_result
            print(f"\n✅ Baseline evaluation complete")
        else:
            all_results["baselines"] = {"status": "skipped", "reason": "include_baselines=False"}

        # ====================================================================
        # FINAL REPORT
        # ====================================================================
        print("\n" + "-" * 100)
        print("CREATING FINAL REPORT")
        print("-" * 100)

        report_path = create_final_report(
            experiment_dir,
            config.to_dict(),
            training_summary,
            eval_summary,
            test_summaries,
            all_results.get("baselines", {}),
        )

        print(f"✅ Final report saved: {report_path}")

        # ====================================================================
        # SUMMARY
        # ====================================================================
        print("\n" + "=" * 100)
        print(f"✅ EXPERIMENT COMPLETE - {experiment_name}")
        print("=" * 100)
        print(f"\n📁 Output directory: {experiment_dir.absolute()}")
        print(f"\n📊 Key Results:")
        print(f"   Training episodes: {training_summary.get('num_train_episodes', 0)}")
        print(f"   GNN solve rate: {eval_summary.get('GNN', {}).get('solve_rate_pct', 0):.1f}%")
        print(f"   Random solve rate: {eval_summary.get('Random', {}).get('solve_rate_pct', 0):.1f}%")
        print(f"   Test configs: {len(test_summaries)}")

        print(f"\n📂 Output Files:")
        print(f"   ✓ {(experiment_dir / 'training_log.jsonl').name}")
        print(f"   ✓ {(experiment_dir / 'model.zip').name}")
        print(f"   ✓ {(experiment_dir / 'analysis/').name}/")
        print(f"   ✓ {(experiment_dir / 'plots/').name}/ ({len(list((experiment_dir / 'plots').glob('*.png')))} plots)")
        print(f"   ✓ {(experiment_dir / 'evaluation/').name}/")
        print(f"   ✓ {(experiment_dir / 'testing/').name}/")
        if include_baselines:
            print(f"   ✓ {(experiment_dir / 'baselines/').name}/")
        print(f"   ✓ {report_path.name}")

        return {
            "status": "success",
            "experiment": experiment_name,
            "output_dir": str(experiment_dir.absolute()),
            "results": all_results,
        }

    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        return {"status": "interrupted", "experiment": experiment_name}

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "experiment": experiment_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Master script to run complete GNN Merge-and-Shrink experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:

  Run single experiment (full pipeline):
    python run_full_experiment.py blocksworld_exp_0

  Run with custom output directory:
    python run_full_experiment.py blocksworld_exp_1 --output my_results

  Run curriculum experiment:
    python run_full_experiment.py blocksworld_curriculum --curriculum

  Skip baselines (faster):
    python run_full_experiment.py blocksworld_exp_0 --no-baselines

  Run all experiments:
    python run_full_experiment.py --all

  List available experiments:
    python run_full_experiment.py --list

OUTPUT STRUCTURE:
  results/<experiment_name>/
    ├── training_log.jsonl          (raw training metrics)
    ├── model.zip                   (trained model)
    ├── analysis/                   (10+ JSON analysis files)
    ├── plots/                      (15+ PNG visualizations)
    ├── evaluation/                 (GNN vs Random comparison)
    ├── testing/                    (test set results)
    ├── baselines/                  (FD baseline comparison)
    └── final_report.json           (master summary)
        """
    )

    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name to run"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments sequentially"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output base directory (default: results)"
    )

    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip Fast Downward baseline evaluation (faster)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\n" + "=" * 100)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 100)

        for exp_name in list_experiments():
            exp = get_experiment(exp_name)
            print(f"\n  {exp_name}")
            print(f"    {exp.description}")
            if exp.is_curriculum:
                print(f"    [CURRICULUM] {len(exp.curriculum_phases)} phases")

        print("\n" + "=" * 100)
        return 0

    # Run experiments
    experiments = []

    if args.all:
        experiments = list_experiments()
    elif args.experiment:
        experiments = [args.experiment]
    else:
        parser.print_help()
        return 1

    results_summary = {}

    for exp_name in experiments:
        try:
            result = run_full_experiment_pipeline(
                exp_name,
                output_base_dir=args.output,
                include_baselines=not args.no_baselines,
                seed=args.seed,
                verbose=args.verbose,
            )
            results_summary[exp_name] = result
        except Exception as e:
            print(f"\n❌ {exp_name} failed: {e}")
            results_summary[exp_name] = {"status": "failed", "error": str(e)}

    # Final summary
    print("\n" + "=" * 100)
    print("EXPERIMENT BATCH SUMMARY")
    print("=" * 100)

    for exp_name, result in results_summary.items():
        status_icon = "✅" if result.get("status") == "success" else "❌"
        print(f"{status_icon} {exp_name}: {result.get('status', 'unknown')}")
        if "output_dir" in result:
            print(f"   → {result['output_dir']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())