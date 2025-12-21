#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAIN ENTRY POINT - Run experiments
"""

import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs.experiment_configs import (
    get_experiment, list_experiments, ALL_EXPERIMENTS
)
from experiments.runners.experiment_runner import ExperimentRunner, CurriculumExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="GNN Merge-and-Shrink Experiment Framework",
        epilog="""
USAGE EXAMPLES:

  Run a single experiment:
    python run_experiment.py --exp blocksworld_exp_1

  Run all experiments:
    python run_experiment.py --all

  List available experiments:
    python run_experiment.py --list

  Custom output directory:
    python run_experiment.py --exp blocksworld_exp_1 --output my_results/
        """
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="Experiment name (see --list for available)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\n" + "=" * 100)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 100)

        for exp_name in list_experiments():
            exp = ALL_EXPERIMENTS[exp_name]
            print(f"\n  {exp_name}")
            print(f"    Description: {exp.description}")
            print(f"    Domain: {exp.domain.value}")
            print(f"    Train size: {exp.train_problem_size.value}")
            print(f"    Train problems: {exp.train_num_problems}")
            print(f"    Test configs: {len(exp.test_configurations)}")

        print("\n" + "=" * 100)
        return 0

    # Run experiments
    experiments_to_run = []

    if args.all:
        experiments_to_run = list_experiments()
    elif args.exp:
        experiments_to_run = [args.exp]
    else:
        parser.print_help()
        return 1

    results = {}

    for exp_name in experiments_to_run:
        try:
            exp_config = get_experiment(exp_name)

            # ⭐ Choose runner based on config type
            if exp_config.is_curriculum:
                print(f"\n🎓 Using CurriculumExperimentRunner for {exp_name}")
                runner = CurriculumExperimentRunner(exp_config, output_base_dir=args.output)
                result = runner.run_curriculum()
            else:
                print(f"\n🔬 Using regular ExperimentRunner for {exp_name}")
                runner = ExperimentRunner(exp_config, output_base_dir=args.output)
                result = runner.run_full_experiment()

            results[exp_name] = result

        except KeyboardInterrupt:
            print(f"\n⚠️  Experiment {exp_name} interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Experiment {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[exp_name] = {"status": "failed", "error": str(e)}

    # Summary
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)

    for exp_name, result in results.items():
        status = result.get("status", "unknown")
        status_icon = "✅" if status == "success" else "❌"
        print(f"{status_icon} {exp_name}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())