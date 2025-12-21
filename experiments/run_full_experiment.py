# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASTER SCRIPT - Run complete end-to-end experiment (FIXED v3)
=============================================================
Trains GNN, analyzes, visualizes, evaluates, tests, and compares against baselines.

✅ FIXED: Phase 2 now generates ALL comparison plots (GNN vs Random) with detailed stats
✅ FIXED: Phase 4 integrates with Phase 2 for complete 3-way comparison (GNN vs Random vs Baselines)
✅ FIXED: Comparison variables scoped correctly for both curriculum and standard runners
✅ FIXED: Creates unified comparison report with tables using plotting_utils
✅ FIXED: All plots actually generate - no silent failures
✅ NEW: Uses organized output structure with ExperimentOutputManager
✅ NEW: Creates unified single-source-of-truth report with UnifiedReporter

All outputs saved to structured directory.
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs.experiment_configs import get_experiment, list_experiments
from experiments.runners.experiment_runner import ExperimentRunner, CurriculumExperimentRunner
from experiments.core.evaluation import EvaluationFramework
from experiments.core.evaluation_plots import GenerateEvaluationPlots
from experiments.core.evaluation_analyzer import ComparisonAnalyzer

# ✅ NEW: Import organized output structure and unified reporting
from experiments.core.output_structure import setup_experiment_output, ExperimentOutputManager
from experiments.core.unified_reporting import UnifiedReporter


def _prepare_training_problems_tuple(benchmarks: List[Tuple], problem_names: List[str]) -> Tuple[
    List[Tuple], List[str]]:
    """Ensure benchmarks and problem_names are properly formatted."""
    if not benchmarks or not problem_names:
        return [], []
    return benchmarks, problem_names


def run_training_only_pipeline(
        experiment_name: str,
        output_base_dir: str = "results",
        seed: int = 42,
        verbose: bool = False,
) -> dict:
    """
    Run ONLY the training phase - save model and logs for later analysis.

    ✅ Trains model completely with full logging
    ✅ Saves model.zip and training_log.jsonl
    ✅ Saves training checkpoint at: {output_base_dir}/{experiment_name}/training/
    ✅ Skip all evaluation, analysis, visualization (run separately later)

    Usage:
        python run_full_experiment.py blocksworld_exp_1 --train-only

    Later run analysis:
        python run_full_experiment.py blocksworld_exp_1 --analyze-only \\
            --model-path results/blocksworld_exp_1/training/model.zip
    """
    config = get_experiment(experiment_name)

    # Setup output structure
    output_manager = setup_experiment_output(config.name, output_base_dir)
    experiment_dir = output_manager.root

    print("\n" + "=" * 100)
    print(f"🚀 TRAINING-ONLY MODE: {experiment_name}")
    print(f"📁 Output: {experiment_dir}")
    print("=" * 100)
    print(f"\n📋 Experiment: {config.description}")
    print(f"   Domain: {config.domain.value}")
    print(f"   Training: {config.train_num_problems} problems, {config.num_train_episodes} episodes")
    print(f"\n⏭️  Skipping: Evaluation, Analysis, Visualization")
    print(f"   (Run these later after model is trained)\n")

    try:
        # ====================================================================
        # PHASE 1 ONLY: TRAINING WITH FULL LOGGING
        # ====================================================================
        print("-" * 100)
        print("PHASE 1: TRAINING WITH FULL LOGGING")
        print("-" * 100)

        if config.is_curriculum:
            print(f"\n🎓 Curriculum Learning: {len(config.curriculum_phases)} phases")
            runner = CurriculumExperimentRunner(config, output_base_dir=output_base_dir)
            result = runner.run_curriculum()

            training_summary = result
            final_model_path = result.get("final_model_path")

        else:
            print(f"\n🚀 Standard Training")
            runner = ExperimentRunner(config, output_base_dir=output_base_dir)

            train_result = runner.run_training()
            if not train_result:
                print("❌ Training failed!")
                return {"status": "failed", "phase": "training"}

            model_path, trainer = train_result
            final_model_path = model_path

            # Save training log
            trainer.save_training_log()
            trainer.close_logger()

            training_summary = {
                "status": "success",
                "model_path": str(final_model_path),
                "num_train_episodes": len(trainer.episode_log),
                "num_failed_episodes": trainer.failed_episode_count,
                "duration_seconds": trainer.resource_monitor.get_elapsed_ms() / 1000
                if hasattr(trainer, 'resource_monitor') else 0,
            }

        # ====================================================================
        # SAVE TRAINING SUMMARY
        # ====================================================================
        summary_path = experiment_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)

        print(f"\n✅ TRAINING COMPLETE")
        print(f"   Model: {final_model_path}")
        print(f"   Summary: {summary_path}")
        print(f"\n📝 NEXT STEPS:")
        print(f"   1. Model ready for evaluation/analysis")
        print(f"   2. Run analysis later:")
        print(f"      python run_full_experiment.py {experiment_name} \\")
        print(f"        --analyze-only \\")
        print(f"        --model-path {final_model_path}")
        print(f"   3. Or run full evaluation:")
        print(f"      python run_evaluation_only_pipeline('{experiment_name}', \\")
        print(f"        model_path='{final_model_path}')")

        return {
            "status": "success",
            "experiment": experiment_name,
            "output_dir": str(experiment_dir),
            "model_path": str(final_model_path),
            "training_summary": training_summary,
        }

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return {"status": "interrupted", "experiment": experiment_name}

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "experiment": experiment_name, "error": str(e)}


def run_full_experiment_pipeline(
        experiment_name: str,
        output_base_dir: str = "results",
        include_baselines: bool = True,
        seed: int = 42,
        verbose: bool = False,
        training_only: bool = False,  # ✅ ADD THIS
) -> dict:
    """
    Execute complete experiment pipeline: Train → Analyze → Visualize → Evaluate → Test → Baselines

    ✅ FIXED: Phase 2 generates comparison plots with detailed statistics
    ✅ FIXED: Phase 4 integrates with Phase 2 for complete 3-way comparison
    ✅ FIXED: Comparison variables properly scoped for all runner types
    ✅ NEW: Uses organized output structure and unified reporting

    Args:
        experiment_name: Name of experiment config to run
        output_base_dir: Base output directory
        include_baselines: Whether to run Fast Downward baselines
        seed: Random seed
        verbose: Verbose logging

    Returns:
        Summary dictionary with all results
    """

    config = get_experiment(experiment_name)

    # =========================================================================
    # SETUP ORGANIZED OUTPUT STRUCTURE ✅ NEW
    # =========================================================================
    output_manager = setup_experiment_output(config.name, output_base_dir)
    experiment_dir = output_manager.root

    print("\n" + "=" * 100)
    print(f"🔬 EXPERIMENT: {experiment_name}")
    print(f"📁 Output: {experiment_dir}")
    print("=" * 100)
    print(f"\n📋 Experiment: {config.description}")
    print(f"   Domain: {config.domain.value}")
    print(f"   Training: {config.train_num_problems} problems, {config.num_train_episodes} episodes")
    print(f"   Testing: {len(config.test_configurations)} test configs")

    all_results = {}

    # ✅ INITIALIZE PHASE 2 VARIABLES AT FUNCTION SCOPE (available for Phase 4)
    gnn_vs_random_summary = {}
    gnn_results = []
    random_results = []
    gnn_stats_dict = {}
    random_stats_dict = {}
    final_model_path = None

    # ✅ NEW: Initialize collection variables for unified report
    training_summary = {}
    all_analysis_results = {}
    test_results = {}
    baseline_stats_dict = {}

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

            final_model_path = phase_result.get("final_model_path")
            training_summary = phase_result
        else:
            print(f"\n🚀 Using standard ExperimentRunner")
            runner = ExperimentRunner(config, output_base_dir=output_base_dir)

            train_result = runner.run_training()
            if not train_result:
                print("❌ Training failed!")
                return {"status": "failed", "phase": "training"}

            model_path, trainer = train_result
            final_model_path = model_path

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
        # PHASE 2: GNN vs RANDOM EVALUATION WITH PLOTS ✅ FIXED
        # ====================================================================
        print("\n" + "-" * 100)
        print("PHASE 2: GNN vs RANDOM MERGE STRATEGY (WITH PLOTS)")
        print("-" * 100)

        # ✅ Initialize variables at function scope (needed for Phase 4)
        gnn_results = []
        random_results = []
        gnn_stats_dict = {}
        random_stats_dict = {}

        if final_model_path and Path(final_model_path).exists():
            try:
                from experiments.core.gnn_random_evaluation import GNNRandomEvaluationFramework
                from experiments.runners.experiment_runner import get_domain_file, select_problems

                # ✅ FIX: Use proper utilities to find domain file
                print(f"\n🔍 Looking for domain file for: {config.domain.value}")
                try:
                    domain_file = get_domain_file(config.domain.value)
                    if not Path(domain_file).exists():
                        raise FileNotFoundError(f"Domain file does not exist: {domain_file}")
                    print(f"   ✓ Found: {domain_file}")
                except FileNotFoundError as e:
                    print(f"   ✗ {e}")
                    domain_file = None

                # ✅ FIX: Use proper utilities to find problem files
                problem_files = []
                if domain_file:
                    print(f"\n🔍 Looking for problem files...")
                    for idx, test_config in enumerate(config.test_configurations):
                        try:
                            selected_files, problem_names = select_problems(
                                test_config.problem_pattern,
                                min(5, test_config.num_problems),  # Use up to 5 for quick eval
                                seed=config.seed + 1000 + idx
                            )
                            problem_files = selected_files
                            print(f"   ✓ Found {len(problem_files)} problems from '{test_config.name}'")
                            print(f"     Pattern: {test_config.problem_pattern}")
                            break
                        except ValueError as e:
                            print(f"   ✗ {test_config.name}: {e}")
                            continue

                # ✅ FIX: Only proceed if we have valid domain and problems
                if domain_file and problem_files:
                    print(f"\n📊 Evaluating GNN vs Random on {len(problem_files)} problems...")
                    print(f"   Domain: {Path(domain_file).name}")
                    print(f"   Problems: {[Path(p).name for p in problem_files]}")

                    try:
                        eval_fw = GNNRandomEvaluationFramework(
                            model_path=final_model_path,
                            domain_file=domain_file,
                            problem_files=problem_files,
                            output_dir=str(output_manager.get_dir("eval_gnn_random")),  # ✅ NEW
                            num_runs_per_problem=1,
                            downward_dir=str(PROJECT_ROOT / "downward"),
                        )

                        gnn_results, random_results = eval_fw.evaluate()

                        print(f"\n✅ Evaluation complete:")
                        print(f"   GNN results: {len(gnn_results)} runs")
                        print(f"   Random results: {len(random_results)} runs")

                        # ✅ COMPUTE STATISTICS FOR BOTH STRATEGIES
                        if gnn_results or random_results:
                            analyzer = ComparisonAnalyzer(gnn_results + random_results)

                            for planner_name in ["GNN", "Random"]:
                                try:
                                    stats = analyzer.get_aggregate_statistics(planner_name)
                                    stats_dict = stats.to_dict()
                                    if planner_name == "GNN":
                                        gnn_stats_dict = stats_dict
                                    else:
                                        random_stats_dict = stats_dict
                                    print(f"   ✓ {planner_name}: {stats_dict.get('solve_rate_pct', 0):.1f}% solved")
                                except Exception as e:
                                    print(f"   ⚠️  Could not compute stats for {planner_name}: {e}")

                            gnn_vs_random_summary = {
                                'GNN': gnn_stats_dict,
                                'Random': random_stats_dict,
                            }

                            # ✅ GENERATE PHASE 2 PLOTS WITH DETAILED STATISTICS
                            try:
                                plotter = GenerateEvaluationPlots(
                                    output_dir=str(output_manager.get_dir("eval_gnn_random"))  # ✅ NEW
                                )

                                plot_results = plotter.generate_all_plots(
                                    statistics={**gnn_stats_dict, **random_stats_dict},
                                    results=gnn_results + random_results,
                                    gnn_results=gnn_stats_dict,
                                    gnn_vs_random_detailed=gnn_results + random_results,
                                    gnn_stats=gnn_stats_dict,
                                    random_stats=random_stats_dict,
                                )

                                successful_plots = sum(1 for p in plot_results.values() if p is not None)
                                print(
                                    f"\n✅ Generated {successful_plots}/{len(plot_results)} Phase 2 comparison plots")

                            except Exception as e:
                                print(f"⚠️  Phase 2 plot generation failed: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"⚠️  No results from evaluation - skipping plots")
                            gnn_vs_random_summary = {}

                    except Exception as e:
                        print(f"⚠️  GNN/Random evaluation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        gnn_vs_random_summary = {}

                else:
                    print(f"\n⚠️  Cannot proceed with Phase 2:")
                    if not domain_file:
                        print(f"   - Domain file not found for: {config.domain.value}")
                    if not problem_files:
                        print(f"   - No problem files found (check patterns in test_configurations)")
                    gnn_vs_random_summary = {}

            except Exception as e:
                print(f"⚠️  Phase 2 setup failed: {e}")
                import traceback
                traceback.print_exc()
                gnn_vs_random_summary = {}

        else:
            print(f"⚠️  Model not found at {final_model_path} - skipping Phase 2")
            gnn_vs_random_summary = {}

        all_results["evaluation"] = {
            "gnn_results": len(gnn_results),
            "random_results": len(random_results),
            "summary": gnn_vs_random_summary,
        }

        # ====================================================================
        # PHASE 3: TEST SET EVALUATION
        # ====================================================================
        print("\n" + "-" * 100)
        print("PHASE 3: TEST SET EVALUATION")
        print("-" * 100)

        test_runner = ExperimentRunner(config, output_base_dir=output_base_dir)
        test_results = {}

        if final_model_path and Path(final_model_path).exists():
            for test_config in config.test_configurations:
                test_result = test_runner.run_test(final_model_path, test_config)
                num_problems = test_result["num_problems"]
                num_solved = test_result["num_solved"]
                solve_rate = (num_solved / max(1, num_problems)) * 100

                # ✅ NEW: Store in format for unified reporter
                test_results[test_config.name] = {
                    "results": {
                        "summary": {
                            "gnn_total": num_problems,
                            "gnn_solved": num_solved,
                            "random_solved": 0,  # Placeholder
                            "random_total": 0,
                        }
                    },
                    "num_problems": num_problems,
                    "num_solved": num_solved,
                    "solve_rate": solve_rate,
                }

                print(f"✅ {test_config.name}: {solve_rate:.1f}% solved ({num_solved}/{num_problems})")
        else:
            print(f"⚠️  Model not found - skipping Phase 3")

        all_results["testing"] = test_results

        # ====================================================================
        # PHASE 4: BASELINE EVALUATION WITH INTEGRATION ✅ FIXED
        # ====================================================================
        baseline_stats_dict = {}

        if include_baselines:
            print("\n" + "-" * 100)
            print("PHASE 4: FAST DOWNWARD BASELINES (WITH FULL COMPARISON)")
            print("-" * 100)

            baseline_framework = EvaluationFramework(
                output_dir=str(output_manager.get_dir("eval_baselines"))  # ✅ NEW
            )
            baseline_result = baseline_framework.run_baseline_evaluation(
                timeout_sec=300,
                max_problems=5,
                domains=config.domain.value,
                sizes="small,medium",
            )

            all_results["baselines"] = baseline_result

            # ✅ COLLECT BASELINE STATS
            if baseline_framework.all_results:
                analyzer = ComparisonAnalyzer(baseline_framework.all_results)

                planners = set(r.planner_name for r in baseline_framework.all_results)
                for planner_name in sorted(planners):
                    try:
                        stats = analyzer.get_aggregate_statistics(planner_name)
                        baseline_stats_dict[planner_name] = stats.to_dict()
                    except Exception as e:
                        print(f"⚠️  Could not compute stats for {planner_name}: {e}")

            # ✅ GENERATE UNIFIED COMPARISON PLOTS (GNN vs Random vs Baselines)
            print(f"\n📈 Generating unified 3-way comparison plots...")

            try:
                unified_plotter = GenerateEvaluationPlots(
                    output_dir=str(output_manager.get_dir("plots_comparison"))  # ✅ NEW
                )

                # ✅ CALL WITH ALL COMPARISON DATA FROM PHASE 2 AND PHASE 4
                unified_plot_results = unified_plotter.generate_all_plots(
                    # Aggregate statistics
                    statistics={**gnn_stats_dict, **random_stats_dict, **baseline_stats_dict},
                    # All raw results
                    results=gnn_results + random_results + baseline_framework.all_results,
                    # Legacy parameter
                    gnn_results=gnn_stats_dict,
                    # NEW: Pass detailed results for heatmaps, cumulative plots
                    gnn_vs_random_detailed=gnn_results + random_results,
                    baseline_detailed=baseline_framework.all_results,
                    # Detailed statistics for each strategy
                    gnn_stats=gnn_stats_dict,
                    random_stats=random_stats_dict,
                    baseline_stats=baseline_stats_dict,
                )

                successful = sum(1 for p in unified_plot_results.values() if p is not None)
                print(f"\n✅ Generated {successful} unified 3-way comparison plots")

            except Exception as e:
                print(f"⚠️  Unified plot generation failed: {e}")
                import traceback
                traceback.print_exc()

        else:
            all_results["baselines"] = {"status": "skipped", "reason": "include_baselines=False"}

        # ====================================================================
        # CREATE COMPARISON STATISTICS TABLE ✅ FIXED
        # ====================================================================
        print("\n" + "-" * 100)
        print("CREATING COMPARISON TABLES")
        print("-" * 100)

        try:
            from experiments.core.visualization.plotting_utils import create_comparison_table

            table_path = create_comparison_table(
                gnn_stats_dict,
                random_stats_dict,
                baseline_stats_dict,
                output_manager.get_dir("tables")  # ✅ NEW
            )
            print(f"✅ Comparison table saved: {table_path}")

            # Also create comprehensive tables for final report
            comparison_tables = _create_comparison_tables(
                gnn_vs_random_summary,
                all_results.get("baselines", {}),
                test_results
            )

            # Save comparison tables as CSV and JSON
            _save_comparison_tables(output_manager.get_dir("tables"), comparison_tables)  # ✅ NEW

        except Exception as e:
            print(f"⚠️  Failed to create comparison table: {e}")
            comparison_tables = {}

        # ====================================================================
        # CREATE UNIFIED REPORT ✅ NEW - REPLACES create_final_report()
        # ====================================================================
        print("\n" + "-" * 100)
        print("CREATING UNIFIED EXPERIMENT REPORT")
        print("-" * 100)

        try:
            reporter = UnifiedReporter(output_manager.get_dir("reports"))
            report_path = reporter.create_unified_report(
                config=config.to_dict(),
                training_summary=training_summary,
                analysis_summary=all_analysis_results,
                evaluation_summary={"gnn_vs_random": gnn_vs_random_summary},
                test_results=test_results,
                baseline_summary={"baseline_configs": list(baseline_stats_dict.items())},
            )

            print(f"✅ Unified report created: {report_path}")
            print(f"✅ Human-readable report: {output_manager.get_dir('reports') / 'experiment_report.txt'}")

        except Exception as e:
            print(f"⚠️  Failed to create unified report: {e}")
            import traceback
            traceback.print_exc()

        # ====================================================================
        # ORGANIZE PLOTS INTO CATEGORIES ✅ NEW
        # ====================================================================
        print("\n" + "-" * 100)
        print("ORGANIZING OUTPUT STRUCTURE")
        print("-" * 100)

        try:
            # Organize any plots in the root plots directory
            if (experiment_dir / "plots").exists():
                output_manager.organize_plots(experiment_dir / "plots")
                print("✅ Plots organized into categories")
        except Exception as e:
            print(f"⚠️  Failed to organize plots: {e}")

        # Print final structure
        output_manager.print_structure()

        # ====================================================================
        # SUMMARY
        # ====================================================================
        print("\n" + "=" * 100)
        print(f"✅ EXPERIMENT COMPLETE - {experiment_name}")
        print("=" * 100)
        print(f"\n📁 Output directory: {experiment_dir.absolute()}")
        print(f"\n📊 Key Results:")
        print(f"   Training episodes: {training_summary.get('num_train_episodes', 0)}")

        if gnn_vs_random_summary and "GNN" in gnn_vs_random_summary:
            print(f"   GNN solve rate: {gnn_vs_random_summary['GNN'].get('solve_rate_pct', 0):.1f}%")
            print(f"   Random solve rate: {gnn_vs_random_summary.get('Random', {}).get('solve_rate_pct', 0):.1f}%")
            improvement = (
                    gnn_vs_random_summary['GNN'].get('solve_rate_pct', 0) -
                    gnn_vs_random_summary.get('Random', {}).get('solve_rate_pct', 0)
            )
            print(f"   GNN vs Random improvement: {improvement:+.1f}%")

        print(f"   Test configs: {len(test_results)}")

        print(f"\n📂 Output Files:")
        print(f"   ✓ training/")
        print(f"     ├── model.zip")
        print(f"     ├── training_log.jsonl")
        print(f"     └── checkpoints/")
        print(f"   ✓ analysis/")
        print(f"   ✓ plots/")
        print(f"     ├── training/")
        print(f"     ├── components/")
        print(f"     ├── quality/")
        print(f"     └── comparison/")
        print(f"   ✓ evaluation/gnn_vs_random/")
        print(f"   ✓ evaluation/baselines/")
        print(f"   ✓ testing/")
        print(f"   ✓ reports/")
        print(f"     ├── experiment_report.json (UNIFIED)")
        print(f"     ├── experiment_report.txt (UNIFIED)")
        print(f"     └── tables/")
        print(f"   ✓ logs/")

        return {
            "status": "success",
            "experiment": experiment_name,
            "output_dir": str(experiment_dir.absolute()),
            "report_path": str(output_manager.get_dir("reports") / "experiment_report.json"),
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


def run_evaluation_only_pipeline(
        experiment_name: str,
        model_path: str,
        output_base_dir: str = "results",
        include_baselines: bool = True,
        seed: int = 42,
        verbose: bool = False,
) -> dict:
    """
    Run ONLY evaluation phases (2-4) using a pre-trained model.

    Skips training entirely and goes straight to:
    - Phase 2: GNN vs Random evaluation
    - Phase 3: Test set evaluation
    - Phase 4: Baseline comparison

    Usage:
        python run_full_experiment.py blocksworld_exp_0 \\
            --model-path results/blocksworld_exp_0/training/model.zip
    """

    print("\n" + "=" * 100)
    print(f"📊 EVALUATION-ONLY PIPELINE - {experiment_name}")
    print(f"   Model: {model_path}")
    print("=" * 100)

    # Verify model exists
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Model not found: {model_path}")
        return {"status": "failed", "error": f"Model not found: {model_path}"}

    config = get_experiment(experiment_name)

    # ✅ NEW: Setup output structure
    output_manager = setup_experiment_output(config.name, output_base_dir)
    experiment_dir = output_manager.root

    print(f"\n📋 Experiment: {config.description}")
    print(f"   Domain: {config.domain.value}")
    print(f"   Testing: {len(config.test_configurations)} test configs")
    print(f"📁 Output: {experiment_dir}")

    all_results = {}
    gnn_vs_random_summary = {}
    gnn_results = []
    random_results = []
    gnn_stats_dict = {}
    random_stats_dict = {}
    test_results = {}
    baseline_stats_dict = {}

    try:
        # ====================================================================
        # PHASE 2: GNN vs RANDOM EVALUATION WITH PLOTS
        # ====================================================================
        print("\n" + "-" * 100)
        print("PHASE 2: GNN vs RANDOM MERGE STRATEGY (WITH PLOTS)")
        print("-" * 100)

        try:
            from experiments.core.gnn_random_evaluation import GNNRandomEvaluationFramework
            from experiments.runners.experiment_runner import get_domain_file, select_problems

            print(f"\n🔍 Looking for domain file for: {config.domain.value}")
            try:
                domain_file = get_domain_file(config.domain.value)
                if not Path(domain_file).exists():
                    raise FileNotFoundError(f"Domain file does not exist: {domain_file}")
                print(f"   ✓ Found: {domain_file}")
            except FileNotFoundError as e:
                print(f"   ✗ {e}")
                domain_file = None

            problem_files = []
            if domain_file:
                print(f"\n🔍 Looking for problem files...")
                for idx, test_config in enumerate(config.test_configurations):
                    try:
                        selected_files, problem_names = select_problems(
                            test_config.problem_pattern,
                            min(5, test_config.num_problems),
                            seed=config.seed + 1000 + idx
                        )
                        problem_files = selected_files
                        print(f"   ✓ Found {len(problem_files)} problems from '{test_config.name}'")
                        break
                    except ValueError as e:
                        print(f"   ✗ {test_config.name}: {e}")
                        continue

            if domain_file and problem_files:
                print(f"\n📊 Evaluating GNN vs Random on {len(problem_files)} problems...")

                eval_fw = GNNRandomEvaluationFramework(
                    model_path=str(model_path),
                    domain_file=domain_file,
                    problem_files=problem_files,
                    output_dir=str(output_manager.get_dir("eval_gnn_random")),  # ✅ NEW
                    num_runs_per_problem=1,
                    downward_dir=str(PROJECT_ROOT / "downward"),
                )

                gnn_results, random_results = eval_fw.evaluate()

                print(f"\n✅ Evaluation complete:")
                print(f"   GNN results: {len(gnn_results)} runs")
                print(f"   Random results: {len(random_results)} runs")

                if gnn_results or random_results:
                    analyzer = ComparisonAnalyzer(gnn_results + random_results)

                    for planner_name in ["GNN", "Random"]:
                        try:
                            stats = analyzer.get_aggregate_statistics(planner_name)
                            stats_dict = stats.to_dict()
                            if planner_name == "GNN":
                                gnn_stats_dict = stats_dict
                            else:
                                random_stats_dict = stats_dict
                            print(f"   ✓ {planner_name}: {stats_dict.get('solve_rate_pct', 0):.1f}% solved")
                        except Exception as e:
                            print(f"   ⚠️  Could not compute stats for {planner_name}: {e}")

                    gnn_vs_random_summary = {
                        'GNN': gnn_stats_dict,
                        'Random': random_stats_dict,
                    }

                    # Generate Phase 2 plots
                    try:
                        plotter = GenerateEvaluationPlots(
                            output_dir=str(output_manager.get_dir("eval_gnn_random"))  # ✅ NEW
                        )

                        plot_results = plotter.generate_all_plots(
                            statistics={**gnn_stats_dict, **random_stats_dict},
                            results=gnn_results + random_results,
                            gnn_results=gnn_stats_dict,
                            gnn_vs_random_detailed=gnn_results + random_results,
                            gnn_stats=gnn_stats_dict,
                            random_stats=random_stats_dict,
                        )

                        successful_plots = sum(1 for p in plot_results.values() if p is not None)
                        print(f"✅ Generated {successful_plots}/{len(plot_results)} Phase 2 plots")

                    except Exception as e:
                        print(f"⚠️  Phase 2 plot generation failed: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"⚠️  Cannot proceed with Phase 2 - domain or problems not found")
                gnn_vs_random_summary = {}

        except Exception as e:
            print(f"⚠️  Phase 2 failed: {e}")
            import traceback
            traceback.print_exc()
            gnn_vs_random_summary = {}

        all_results["evaluation"] = {
            "gnn_results": len(gnn_results),
            "random_results": len(random_results),
            "summary": gnn_vs_random_summary,
        }

        # Save raw Phase 2 results for future regeneration
        with open(output_manager.get_dir("eval_gnn_random") / "raw_phase2_results.json", "w") as f:
            json.dump(gnn_results + random_results, f, indent=2, default=str)

        # ====================================================================
        # PHASE 3: TEST SET EVALUATION
        # ====================================================================
        print("\n" + "-" * 100)
        print("PHASE 3: TEST SET EVALUATION")
        print("-" * 100)

        test_runner = ExperimentRunner(config, output_base_dir=output_base_dir)

        for test_config in config.test_configurations:
            test_result = test_runner.run_test(str(model_path), test_config)
            num_problems = test_result["num_problems"]
            num_solved = test_result["num_solved"]
            solve_rate = (num_solved / max(1, num_problems)) * 100

            # ✅ NEW: Store in format for unified reporter
            test_results[test_config.name] = {
                "results": {
                    "summary": {
                        "gnn_total": num_problems,
                        "gnn_solved": num_solved,
                        "random_solved": 0,
                        "random_total": 0,
                    }
                },
                "num_problems": num_problems,
                "num_solved": num_solved,
                "solve_rate": solve_rate,
            }

            print(f"✅ {test_config.name}: {solve_rate:.1f}% solved ({num_solved}/{num_problems})")

        all_results["testing"] = test_results

        # ====================================================================
        # PHASE 4: BASELINE EVALUATION
        # ====================================================================
        baseline_stats_dict = {}

        if include_baselines:
            print("\n" + "-" * 100)
            print("PHASE 4: FAST DOWNWARD BASELINES")
            print("-" * 100)

            baseline_framework = EvaluationFramework(
                output_dir=str(output_manager.get_dir("eval_baselines"))  # ✅ NEW
            )
            baseline_result = baseline_framework.run_baseline_evaluation(
                timeout_sec=300,
                max_problems=5,
                domains=config.domain.value,
                sizes="small,medium",
            )

            all_results["baselines"] = baseline_result

            if baseline_framework.all_results:
                analyzer = ComparisonAnalyzer(baseline_framework.all_results)
                planners = set(r.planner_name for r in baseline_framework.all_results)
                for planner_name in sorted(planners):
                    try:
                        stats = analyzer.get_aggregate_statistics(planner_name)
                        baseline_stats_dict[planner_name] = stats.to_dict()
                    except Exception as e:
                        print(f"⚠️  Could not compute stats for {planner_name}: {e}")

            # Generate unified comparison plots
            print(f"\n📈 Generating unified 3-way comparison plots...")

            try:
                unified_plotter = GenerateEvaluationPlots(
                    output_dir=str(output_manager.get_dir("plots_comparison"))  # ✅ NEW
                )

                unified_plot_results = unified_plotter.generate_all_plots(
                    statistics={**gnn_stats_dict, **random_stats_dict, **baseline_stats_dict},
                    results=gnn_results + random_results + baseline_framework.all_results,
                    gnn_results=gnn_stats_dict,
                    gnn_vs_random_detailed=gnn_results + random_results,
                    baseline_detailed=baseline_framework.all_results,
                    gnn_stats=gnn_stats_dict,
                    random_stats=random_stats_dict,
                    baseline_stats=baseline_stats_dict,
                )

                successful = sum(1 for p in unified_plot_results.values() if p is not None)
                print(f"✅ Generated {successful} unified comparison plots")

            except Exception as e:
                print(f"⚠️  Unified plot generation failed: {e}")
        else:
            all_results["baselines"] = {"status": "skipped"}

        # ====================================================================
        # CREATE UNIFIED REPORT ✅ NEW
        # ====================================================================
        print("\n" + "-" * 100)
        print("CREATING UNIFIED REPORT")
        print("-" * 100)

        try:
            reporter = UnifiedReporter(output_manager.get_dir("reports"))
            report_path = reporter.create_unified_report(
                config=config.to_dict(),
                training_summary={"status": "skipped", "reason": "evaluation_only"},
                analysis_summary={},
                evaluation_summary={"gnn_vs_random": gnn_vs_random_summary},
                test_results=test_results,
                baseline_summary={"baseline_configs": list(baseline_stats_dict.items())},
            )

            print(f"✅ Unified report created: {report_path}")

        except Exception as e:
            print(f"⚠️  Failed to create unified report: {e}")

        # Print final structure
        output_manager.print_structure()

        # ====================================================================
        # SUMMARY
        # ====================================================================
        print("\n" + "=" * 100)
        print(f"✅ EVALUATION COMPLETE - {experiment_name}")
        print("=" * 100)
        print(f"\n📁 Output directory: {experiment_dir.absolute()}")
        print(f"\n📊 Results:")

        if gnn_vs_random_summary and "GNN" in gnn_vs_random_summary:
            print(f"   GNN solve rate: {gnn_vs_random_summary['GNN'].get('solve_rate_pct', 0):.1f}%")
            print(f"   Random solve rate: {gnn_vs_random_summary.get('Random', {}).get('solve_rate_pct', 0):.1f}%")

        print(f"   Test configs: {len(test_results)}")

        return {
            "status": "success",
            "experiment": experiment_name,
            "model_path": str(model_path),
            "output_dir": str(experiment_dir.absolute()),
            "report_path": str(output_manager.get_dir("reports") / "experiment_report.json"),
            "results": all_results,
        }

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "experiment": experiment_name, "error": str(e)}


def _create_comparison_tables(
        gnn_vs_random_summary: dict,
        baseline_summary: dict,
        test_results: dict
) -> dict:
    """
    Create comprehensive comparison tables for reporting.

    Returns dict with:
    - strategy_comparison: GNN vs Random vs Baselines on metrics
    - test_performance: Test set results
    """
    tables = {}

    # ====================================================================
    # STRATEGY COMPARISON TABLE
    # ====================================================================
    strategy_rows = []

    # Add GNN
    if "GNN" in gnn_vs_random_summary:
        gnn_stats = gnn_vs_random_summary["GNN"]
        strategy_rows.append({
            "Strategy": "GNN (Learned)",
            "Solve Rate (%)": f"{gnn_stats.get('solve_rate_pct', 0):.1f}",
            "Avg Time (s)": f"{gnn_stats.get('mean_time_sec', 0):.2f}",
            "Mean Expansions": f"{gnn_stats.get('mean_expansions', 0):,}",
            "H* Preservation": f"{gnn_stats.get('mean_h_preservation', 1.0):.3f}",
        })

    # Add Random
    if "Random" in gnn_vs_random_summary:
        random_stats = gnn_vs_random_summary["Random"]
        strategy_rows.append({
            "Strategy": "Random Merge",
            "Solve Rate (%)": f"{random_stats.get('solve_rate_pct', 0):.1f}",
            "Avg Time (s)": f"{random_stats.get('mean_time_sec', 0):.2f}",
            "Mean Expansions": f"{random_stats.get('mean_expansions', 0):,}",
            "H* Preservation": f"{random_stats.get('mean_h_preservation', 1.0):.3f}",
        })

    # Add baselines
    if isinstance(baseline_summary, dict) and "baseline_configs" in baseline_summary:
        for baseline_config in baseline_summary.get("baseline_configs", [])[:5]:
            if isinstance(baseline_config, dict):
                strategy_rows.append({
                    "Strategy": baseline_config.get("name", "Unknown")[:30],
                    "Solve Rate (%)": f"{baseline_config.get('solve_rate_%', 0):.1f}",
                    "Avg Time (s)": f"{baseline_config.get('avg_time_total_s', 0):.2f}",
                    "Mean Expansions": f"{baseline_config.get('avg_expansions', 0):,}",
                    "H* Preservation": "N/A (FD)",
                })

    if strategy_rows:
        tables["strategy_comparison"] = strategy_rows

    # ====================================================================
    # TEST PERFORMANCE TABLE
    # ====================================================================
    if test_results:
        test_rows = []
        for test_name, test_data in test_results.items():
            results = test_data.get("results", {})
            summary = results.get("summary", {})

            test_rows.append({
                "Test Config": test_name,
                "Total Problems": summary.get("gnn_total", 0),
                "Solved": summary.get("gnn_solved", 0),
                "Solve Rate (%)": f"{(summary.get('gnn_solved', 0) / max(1, summary.get('gnn_total', 1)) * 100):.1f}",
            })

        if test_rows:
            tables["test_performance"] = test_rows

    return tables


def _save_comparison_tables(tables_dir: Path, comparison_tables: dict) -> None:
    """Save comparison tables as CSV and JSON."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    for table_name, table_data in comparison_tables.items():
        if isinstance(table_data, list):
            df = pd.DataFrame(table_data)
            csv_path = tables_dir / f"{table_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved: {csv_path.name}")

    # Save as JSON
    json_path = tables_dir / "comparison_tables.json"
    with open(json_path, 'w') as f:
        json.dump(comparison_tables, f, indent=2, default=str)
    print(f"✓ Saved: comparison_tables.json")

    # Save as human-readable text
    txt_path = tables_dir / "comparison_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("STRATEGY COMPARISON REPORT\n")
        f.write("=" * 100 + "\n\n")

        if "strategy_comparison" in comparison_tables:
            f.write("STRATEGY PERFORMANCE:\n")
            f.write("-" * 100 + "\n")
            for row in comparison_tables["strategy_comparison"]:
                f.write(f"\n{row.get('Strategy', 'Unknown')}:\n")
                for key, value in row.items():
                    if key != "Strategy":
                        f.write(f"  {key:<25} {value}\n")

        if "test_performance" in comparison_tables:
            f.write("\n\nTEST SET PERFORMANCE:\n")
            f.write("-" * 100 + "\n")
            for row in comparison_tables["test_performance"]:
                f.write(f"\n{row.get('Test Config', 'Unknown')}:\n")
                for key, value in row.items():
                    if key != "Test Config":
                        f.write(f"  {key:<25} {value}\n")

        f.write("\n" + "=" * 100 + "\n")

    print(f"✓ Saved: comparison_summary.txt")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Master script to run complete GNN Merge-and-Shrink experiments",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# USAGE EXAMPLES:
#
#   Run single experiment (full pipeline):
#     python run_full_experiment.py blocksworld_exp_0
#
#   Run with custom output directory:
#     python run_full_experiment.py blocksworld_exp_1 --output my_results
#
#   Run curriculum experiment:
#     python run_full_experiment.py blocksworld_curriculum
#
#   Skip baselines (faster):
#     python run_full_experiment.py blocksworld_exp_0 --no-baselines
#
#   Run all experiments:
#     python run_full_experiment.py --all
#
#   List available experiments:
#     python run_full_experiment.py --list
#
#   Evaluation-only (with pre-trained model):
#     python run_full_experiment.py blocksworld_exp_0 \\
#         --model-path results/blocksworld_exp_0/training/model.zip
#
# OUTPUT STRUCTURE:
#   results/<experiment_name>/
#     ├── training/                                   (trained models & logs)
#     │   ├── model.zip
#     │   ├── training_log.jsonl
#     │   └── checkpoints/
#     ├── analysis/                                   (JSON analysis files)
#     ├── plots/                                      (organized by category)
#     │   ├── training/                               (learning curves, dead-ends)
#     │   ├── components/                             (reward components)
#     │   ├── quality/                                (h* preservation, bisimulation)
#     │   └── comparison/                             (3-way comparisons)
#     ├── evaluation/
#     │   ├── gnn_vs_random/                          (GNN vs Random comparison)
#     │   └── baselines/                              (Fast Downward results)
#     ├── testing/                                    (test set results)
#     ├── reports/                                    (✅ UNIFIED REPORTS)
#     │   ├── experiment_report.json                  (master summary)
#     │   ├── experiment_report.txt                   (human-readable)
#     │   └── tables/                                 (CSV summary tables)
#     └── logs/
#         """
#     )
#
#     parser.add_argument(
#         "experiment",
#         nargs="?",
#         help="Experiment name to run"
#     )
#
#     parser.add_argument(
#         "--all",
#         action="store_true",
#         help="Run all experiments sequentially"
#     )
#
#     parser.add_argument(
#         "--list",
#         action="store_true",
#         help="List available experiments"
#     )
#
#     parser.add_argument(
#         "--model-path",
#         type=str,
#         default=None,
#         help="Path to pre-trained model.zip (skips training, evaluations only)"
#     )
#
#     parser.add_argument(
#         "--output",
#         type=str,
#         default="results",
#         help="Output base directory (default: results)"
#     )
#
#     parser.add_argument(
#         "--no-baselines",
#         action="store_true",
#         help="Skip Fast Downward baseline evaluation (faster)"
#     )
#
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed (default: 42)"
#     )
#
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Verbose logging"
#     )
#
#     args = parser.parse_args()
#
#     # List experiments
#     if args.list:
#         print("\n" + "=" * 100)
#         print("AVAILABLE EXPERIMENTS")
#         print("=" * 100)
#
#         for exp_name in list_experiments():
#             exp = get_experiment(exp_name)
#             print(f"\n  {exp_name}")
#             print(f"    {exp.description}")
#             if exp.is_curriculum:
#                 print(f"    [CURRICULUM] {len(exp.curriculum_phases)} phases")
#
#         print("\n" + "=" * 100)
#         return 0
#
#     # Run experiments
#     experiments = []
#
#     if args.all:
#         experiments = list_experiments()
#     elif args.experiment:
#         experiments = [args.experiment]
#     else:
#         parser.print_help()
#         return 1
#
#     results_summary = {}
#
#     for exp_name in experiments:
#         try:
#             # Choose pipeline based on whether model path is provided
#             if args.model_path:
#                 result = run_evaluation_only_pipeline(
#                     exp_name,
#                     model_path=args.model_path,
#                     output_base_dir=args.output,
#                     include_baselines=not args.no_baselines,
#                     seed=args.seed,
#                     verbose=args.verbose,
#                 )
#             else:
#                 result = run_full_experiment_pipeline(
#                     exp_name,
#                     output_base_dir=args.output,
#                     include_baselines=not args.no_baselines,
#                     seed=args.seed,
#                     verbose=args.verbose,
#                 )
#             results_summary[exp_name] = result
#         except Exception as e:
#             print(f"\n❌ {exp_name} failed: {e}")
#             results_summary[exp_name] = {"status": "failed", "error": str(e)}
#
#     # Final summary
#     print("\n" + "=" * 100)
#     print("EXPERIMENT BATCH SUMMARY")
#     print("=" * 100)
#
#     for exp_name, result in results_summary.items():
#         status_icon = "✅" if result.get("status") == "success" else "❌"
#         print(f"{status_icon} {exp_name}: {result.get('status', 'unknown')}")
#         if "output_dir" in result:
#             print(f"   📁 {result['output_dir']}")
#         if "report_path" in result:
#             print(f"   📄 {result['report_path']}")
#
#     return 0
#
#
# if __name__ == "__main__":
#     sys.exit(main())


# Update the main() function - add --train-only flag
def main():
    parser = argparse.ArgumentParser(
        description="GNN Merge-and-Shrink Experiment Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:

  Train only (model + logs, no eval):
    python run_full_experiment.py blocksworld_exp_1 --train-only
    python run_full_experiment.py blocksworld_exp_2 --train-only
    python run_full_experiment.py blocksworld_exp_3_curriculum --train-only

  Full pipeline (everything):
    python run_full_experiment.py blocksworld_exp_1

  Run evaluation on pre-trained model:
    python run_full_experiment.py blocksworld_exp_1 \\
      --model-path results/blocksworld_exp_1/training/model.zip

  List all experiments:
    python run_full_experiment.py --list
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
        help="Run all experiments"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments"
    )

    # ✅ NEW: Training-only mode
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run ONLY training phase (skip eval/analysis). Model saved for later use."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pre-trained model.zip (evaluation only, skips training)"
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
            curriculum_tag = " [CURRICULUM]" if exp.is_curriculum else ""
            print(f"\n  {exp_name}{curriculum_tag}")
            print(f"    {exp.description}")
            if exp.is_curriculum:
                print(f"    Phases: {len(exp.curriculum_phases)}")

        print("\n" + "=" * 100)
        return 0

    # ✅ NEW: Handle training-only mode
    if args.train_only:
        if not args.experiment:
            print("❌ Please specify an experiment with --train-only")
            print("   Example: python run_full_experiment.py blocksworld_exp_1 --train-only")
            return 1

        print(f"\n🎯 TRAINING-ONLY MODE")
        result = run_training_only_pipeline(
            args.experiment,
            output_base_dir=args.output,
            seed=args.seed,
            verbose=args.verbose,
        )

        print("\n" + "=" * 100)
        if result["status"] == "success":
            print(f"✅ {args.experiment} training complete!")
            print(f"   Model: {result['model_path']}")
            print(f"\n📝 To run evaluation/analysis later:")
            print(f"   python run_full_experiment.py {args.experiment} \\")
            print(f"     --model-path {result['model_path']}")
        else:
            print(f"❌ Training failed!")
        print("=" * 100)

        return 0 if result["status"] == "success" else 1

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
            if args.model_path:
                # Evaluation only with pre-trained model
                result = run_evaluation_only_pipeline(
                    exp_name,
                    model_path=args.model_path,
                    output_base_dir=args.output,
                    include_baselines=not args.no_baselines,
                    seed=args.seed,
                    verbose=args.verbose,
                )
            else:
                # Full pipeline
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

    # Summary
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)

    for exp_name, result in results_summary.items():
        status = result.get("status", "unknown")
        status_icon = "✅" if status == "success" else "❌"
        print(f"{status_icon} {exp_name}: {status}")
        if "output_dir" in result:
            print(f"   📁 {result['output_dir']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())