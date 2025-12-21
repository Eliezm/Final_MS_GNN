#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUN ALL PAPER EXPERIMENTS
=========================
Runs the 3 main experiments for the paper:
1. Train on MEDIUM → Test S/M(seen)/M(unseen)/L
2. Train on LARGE → Test S/M/L(seen)/L(unseen)
3. Curriculum S→M→L → Test S/M/L + Logistics transfer

Usage:
    python run_paper_experiments.py --all
    python run_paper_experiments.py --exp 1
    python run_paper_experiments.py --exp 2
    python run_paper_experiments.py --exp 3
    python run_paper_experiments.py --quick-test
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs.experiment_configs import (
    get_experiment, get_paper_experiments, list_experiments
)
from experiments.runners.experiment_runner import ExperimentRunner, CurriculumExperimentRunner
from experiments.core.training import set_all_seeds


def run_single_experiment(exp_name: str, output_dir: str, seed: int = 42) -> dict:
    """Run a single experiment end-to-end."""
    print("\n" + "=" * 100)
    print(f"🔬 STARTING EXPERIMENT: {exp_name}")
    print("=" * 100)

    set_all_seeds(seed)
    config = get_experiment(exp_name)

    try:
        if config.is_curriculum:
            print(f"   Type: CURRICULUM ({len(config.curriculum_phases)} phases)")
            runner = CurriculumExperimentRunner(config, output_base_dir=output_dir)
            result = runner.run_curriculum()
        else:
            print(f"   Type: REGULAR")
            runner = ExperimentRunner(config, output_base_dir=output_dir)
            result = runner.run_full_experiment()

        return result

    except Exception as e:
        print(f"\n❌ Experiment {exp_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def run_all_paper_experiments(output_dir: str = "paper_results", seed: int = 42) -> dict:
    """Run all 3 paper experiments sequentially."""
    print("\n" + "=" * 100)
    print("🎯 RUNNING ALL PAPER EXPERIMENTS")
    print("=" * 100)
    print(f"\nOutput directory: {output_dir}")
    print(f"Seed: {seed}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    paper_experiments = get_paper_experiments()
    print(f"\nExperiments to run: {paper_experiments}")

    results = {}

    for i, exp_name in enumerate(paper_experiments, 1):
        print(f"\n{'#' * 100}")
        print(f"# EXPERIMENT {i}/3: {exp_name}")
        print(f"{'#' * 100}")

        result = run_single_experiment(exp_name, output_dir, seed)
        results[exp_name] = result

        # Save intermediate results
        results_path = Path(output_dir) / "paper_results_partial.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Save final results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "experiments": results,
        "summary": {
            "total": len(paper_experiments),
            "success": sum(1 for r in results.values() if r.get("status") == "success"),
            "failed": sum(1 for r in results.values() if r.get("status") != "success"),
        }
    }

    results_path = Path(output_dir) / "paper_results_final.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 100)
    print("📊 PAPER EXPERIMENTS SUMMARY")
    print("=" * 100)

    for exp_name, result in results.items():
        status = result.get("status", "unknown")
        icon = "✅" if status == "success" else "❌"
        print(f"   {icon} {exp_name}: {status}")

    print(f"\n   Total: {final_results['summary']['success']}/{final_results['summary']['total']} succeeded")
    print(f"   Results saved to: {results_path}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Run paper experiments for GNN Merge-and-Shrink",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all 3 paper experiments:
    python run_paper_experiments.py --all

  Run specific experiment:
    python run_paper_experiments.py --exp 1    # Medium training
    python run_paper_experiments.py --exp 2    # Large training
    python run_paper_experiments.py --exp 3    # Curriculum

  Quick test (validate pipeline):
    python run_paper_experiments.py --quick-test

  List available experiments:
    python run_paper_experiments.py --list
        """
    )

    parser.add_argument("--all", action="store_true", help="Run all 3 paper experiments")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], help="Run specific experiment (1-3)")
    parser.add_argument("--quick-test", action="store_true", help="Run quick validation test")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--output", type=str, default="paper_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.list:
        print("\n" + "=" * 80)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 80)

        paper_exps = get_paper_experiments()
        print("\n📜 PAPER EXPERIMENTS (main 3):")
        for exp in paper_exps:
            config = get_experiment(exp)
            print(f"   • {exp}")
            print(f"     {config.description}")

        print("\n🧪 ALL EXPERIMENTS:")
        for exp in list_experiments():
            config = get_experiment(exp)
            tag = "[PAPER]" if exp in paper_exps else "[TEST]" if "test" in exp.lower() else "[OTHER]"
            print(f"   {tag} {exp}")

        return 0

    if args.quick_test:
        print("\n🧪 Running quick validation test...")
        result = run_single_experiment("quick_test_regular", args.output + "_quick", args.seed)
        return 0 if result.get("status") == "success" else 1

    if args.exp:
        exp_map = {
            1: "blocksworld_exp_1",
            2: "blocksworld_exp_2",
            3: "blocksworld_exp_3_curriculum",
        }
        exp_name = exp_map[args.exp]
        result = run_single_experiment(exp_name, args.output, args.seed)
        return 0 if result.get("status") == "success" else 1

    if args.all:
        result = run_all_paper_experiments(args.output, args.seed)
        return 0 if result["summary"]["failed"] == 0 else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())