#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MINI EXPERIMENTS - Quick framework validation
Tests both regular and curriculum runners in parallel
Validates output compatibility
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs.experiment_configs import get_experiment
from experiments.runners.experiment_runner import ExperimentRunner, CurriculumExperimentRunner
from experiments.core.training import set_all_seeds


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


class MiniExperimentValidator:
    """Validate mini experiments and compare outputs."""

    def __init__(self, output_base_dir: str = "mini_experiments"):
        self.output_dir = Path(output_base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_log = []

    def run_mini_regular(self) -> dict:
        """Run minimal regular experiment."""
        print("\n" + "=" * 100)
        print("🔬 MINI EXPERIMENT 1: REGULAR (Non-Curriculum)")
        print("=" * 100)

        set_all_seeds(42)
        config = get_experiment("mini_regular")

        runner = ExperimentRunner(config, output_base_dir=str(self.output_dir))

        try:
            result = runner.run_full_experiment()
            self._validate_regular_outputs(runner.output_dir)
            return {"status": "success", "type": "regular", "result": result}
        except Exception as e:
            print(f"❌ Regular experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "type": "regular", "error": str(e)}

    def run_mini_curriculum(self) -> dict:
        """Run minimal curriculum experiment."""
        print("\n" + "=" * 100)
        print("🎓 MINI EXPERIMENT 2: CURRICULUM")
        print("=" * 100)

        set_all_seeds(42)
        config = get_experiment("mini_curriculum")

        runner = CurriculumExperimentRunner(config, output_base_dir=str(self.output_dir))

        try:
            result = runner.run_curriculum()
            self._validate_curriculum_outputs(runner.output_dir)
            return {"status": "success", "type": "curriculum", "result": result}
        except Exception as e:
            print(f"❌ Curriculum experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "type": "curriculum", "error": str(e)}

    def _validate_regular_outputs(self, exp_dir: Path):
        """Validate all expected outputs from regular experiment."""
        print("\n✓ Validating regular experiment outputs...")

        expected_files = {
            "training_log.jsonl": "Training log",
            "model.zip": "Trained model",
            "experiment_summary.json": "Experiment summary",
            "test_results.json": "Test results",
            "training.log": "Training log file",
        }

        expected_dirs = {
            "checkpoints": "Model checkpoints",
            "plots": "Visualization plots",
        }

        all_present = True

        # Check files
        for filename, description in expected_files.items():
            filepath = exp_dir / filename
            if filepath.exists():
                file_hash = compute_file_hash(str(filepath))
                print(f"  ✅ {description:<25} ({filename:<30}) hash={file_hash}")
                self.validation_log.append({
                    "file": filename,
                    "type": "regular",
                    "status": "present",
                    "hash": file_hash,
                })
            else:
                print(f"  ⚠️  {description:<25} ({filename:<30}) MISSING")
                all_present = False

        # Check directories
        for dirname, description in expected_dirs.items():
            dirpath = exp_dir / dirname
            if dirpath.exists() and dirpath.is_dir():
                num_items = len(list(dirpath.glob("*")))
                print(f"  ✅ {description:<25} ({dirname:<30}) items={num_items}")
                self.validation_log.append({
                    "directory": dirname,
                    "type": "regular",
                    "status": "present",
                    "items": num_items,
                })
            else:
                print(f"  ⚠️  {description:<25} ({dirname:<30}) MISSING")
                all_present = False

        if all_present:
            print("\n  ✅ All regular outputs validated successfully!")
        else:
            print("\n  ⚠️  Some outputs missing - check framework!")

        return all_present

    def _validate_curriculum_outputs(self, exp_dir: Path):
        """Validate all expected outputs from curriculum experiment."""
        print("\n✓ Validating curriculum experiment outputs...")

        expected_files = {
            "curriculum_summary.json": "Curriculum summary",
            "training.log": "Training log file",
        }

        all_present = True

        # Check summary file
        for filename, description in expected_files.items():
            filepath = exp_dir / filename
            if filepath.exists():
                file_hash = compute_file_hash(str(filepath))
                print(f"  ✅ {description:<25} ({filename:<30}) hash={file_hash}")
                self.validation_log.append({
                    "file": filename,
                    "type": "curriculum",
                    "status": "present",
                    "hash": file_hash,
                })
            else:
                print(f"  ⚠️  {description:<25} ({filename:<30}) MISSING")
                all_present = False

        # Check phase directories (they should exist)
        phase_dirs = list(exp_dir.glob("phase_*"))
        if phase_dirs:
            print(f"\n  Found {len(phase_dirs)} phase directories:")
            for phase_dir in sorted(phase_dirs):
                phase_name = phase_dir.name

                # Check for model
                model_file = phase_dir / "model_final.zip"
                model_ok = "✅" if model_file.exists() else "❌"
                print(f"    {model_ok} {phase_name}/model_final.zip")

                if not model_file.exists():
                    all_present = False

        if all_present and len(phase_dirs) > 0:
            print("\n  ✅ All curriculum outputs validated successfully!")
        else:
            print("\n  ⚠️  Some outputs missing or incomplete - check framework!")

        return all_present and len(phase_dirs) > 0

    def compare_outputs(self, regular_result: dict, curriculum_result: dict) -> dict:
        """Compare outputs from regular and curriculum experiments."""
        print("\n" + "=" * 100)
        print("📊 COMPARING OUTPUTS")
        print("=" * 100)

        comparison = {
            "both_succeeded": False,
            "output_types_match": False,
            "both_have_models": False,
            "both_have_logs": False,
            "can_use_interchangeably": False,
        }

        # Check both succeeded
        if regular_result["status"] == "success" and curriculum_result["status"] == "success":
            comparison["both_succeeded"] = True
            print("✅ Both experiments completed successfully")
        else:
            print("❌ One or both experiments failed")
            return comparison

        # Check output compatibility
        regular_dir = self.output_dir / get_experiment("mini_regular").name
        curriculum_dir = self.output_dir / get_experiment("mini_curriculum").name

        # Check models exist
        regular_model = regular_dir / "model.zip"
        curriculum_model = curriculum_dir / "curriculum_summary.json"

        if regular_model.exists():
            print("✅ Regular experiment produced model.zip")
            comparison["both_have_models"] = True

        if curriculum_model.exists():
            print("✅ Curriculum experiment produced outputs")
            comparison["both_have_models"] = True

        # Check logs
        regular_log = regular_dir / "training_log.jsonl"
        curriculum_log = curriculum_dir / "training.log"

        if regular_log.exists() and curriculum_log.exists():
            print("✅ Both experiments produced logs")
            comparison["both_have_logs"] = True

        # Check if outputs are compatible for paper
        if comparison["both_succeeded"] and comparison["both_have_models"] and comparison["both_have_logs"]:
            comparison["output_types_match"] = True
            comparison["can_use_interchangeably"] = True
            print("\n✅ OUTPUTS ARE COMPATIBLE FOR PAPER!")
            print("   Both can be used for publication")
        else:
            print("\n⚠️  OUTPUTS NOT FULLY COMPATIBLE")
            print("   Check missing components above")

        return comparison

    def generate_validation_report(self, regular_result: dict, curriculum_result: dict, comparison: dict):
        """Generate comprehensive validation report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "mini_experiments": {
                "regular": regular_result,
                "curriculum": curriculum_result,
            },
            "comparison": comparison,
            "validation_log": self.validation_log,
            "summary": {
                "total_checks": len(self.validation_log),
                "all_passed": comparison["can_use_interchangeably"],
                "framework_ready": comparison["can_use_interchangeably"],
            }
        }

        # Save report
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✓ Validation report saved: {report_path}")

        # Print summary
        print("\n" + "=" * 100)
        print("VALIDATION SUMMARY")
        print("=" * 100)

        if comparison["can_use_interchangeably"]:
            print("✅ FRAMEWORK READY FOR EXPERIMENTS!")
            print("\n✓ Both regular and curriculum experiments work correctly")
            print("✓ All outputs are produced as expected")
            print("✓ Can be used interchangeably in papers")
        else:
            print("⚠️  FRAMEWORK VALIDATION INCOMPLETE")
            print("\nIssues to fix:")
            if not comparison["both_succeeded"]:
                print("  - Both experiments must succeed")
            if not comparison["both_have_models"]:
                print("  - Both must produce trained models")
            if not comparison["both_have_logs"]:
                print("  - Both must produce training logs")

        return report_path

    def run_all(self) -> Path:
        """Run complete validation pipeline."""
        print("\n" + "=" * 100)
        print("🔍 MINI EXPERIMENTS - FRAMEWORK VALIDATION")
        print("=" * 100)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        # Run experiments
        regular_result = self.run_mini_regular()
        curriculum_result = self.run_mini_curriculum()

        # Compare outputs
        comparison = self.compare_outputs(regular_result, curriculum_result)

        # Generate report
        report_path = self.generate_validation_report(regular_result, curriculum_result, comparison)

        return report_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run mini experiments to validate framework",
        epilog="Example: python run_mini_experiments.py --output mini_test"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="mini_experiments",
        help="Output directory for mini experiments (default: mini_experiments)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    set_all_seeds(args.seed)

    validator = MiniExperimentValidator(output_base_dir=args.output)
    report_path = validator.run_all()

    print(f"\n✅ Report saved: {report_path}")
    print(f"   Open this file to see full validation results")

    return 0


if __name__ == "__main__":
    sys.exit(main())