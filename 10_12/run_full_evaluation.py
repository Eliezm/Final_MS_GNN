#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
END-TO-END EVALUATION ORCHESTRATOR - FULLY COMPATIBLE VERSION
==============================================================
Master script for complete evaluation pipeline.

Features:
✓ Input validation (comprehensive checks)
✓ Baseline evaluation (all FD variants)
✓ GNN evaluation (ThinMergeEnv compatible)
✓ Statistical analysis
✓ Visualization generation
✓ Report compilation
✓ Error recovery

Usage:
  # Full evaluation (baseline + GNN)
  python run_full_evaluation.py \\
      --model mvp_output/gnn_model.zip \\
      --domain domain.pddl \\
      --problems "problem_*.pddl" \\
      --output evaluation_results/

  # GNN only (skip baselines for speed)
  python run_full_evaluation.py \\
      --model model.zip \\
      --domain domain.pddl \\
      --problems "problem_*.pddl" \\
      --skip-baselines

  # Analyze completed experiments
  python run_full_evaluation.py \\
      --analyze-experiments \\
      --experiments exp1_results exp2_results \\
      --output analysis_results/
"""

import sys
import os
import logging
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_orchestrator.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class EvaluationValidator:
    """Comprehensive input validation."""

    @staticmethod
    def validate_model(model_path: str) -> bool:
        """Validate GNN model file."""
        if not model_path:
            logger.error("Model path not provided")
            return False

        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        if not model_path.endswith('.zip'):
            logger.warning(f"Model file should be ZIP: {model_path}")

        try:
            import zipfile
            with zipfile.ZipFile(model_path, 'r') as z:
                namelist = z.namelist()
                if 'data' not in namelist:
                    logger.warning(f"Model may not be valid PPO model")
        except Exception as e:
            logger.warning(f"Could not validate model ZIP: {e}")

        logger.info(f"✓ Model validated: {model_path}")
        return True

    @staticmethod
    def validate_domain(domain_path: str) -> bool:
        """Validate domain PDDL file."""
        if not domain_path:
            logger.error("Domain path not provided")
            return False

        if not os.path.exists(domain_path):
            logger.error(f"Domain file not found: {domain_path}")
            return False

        if not domain_path.endswith('.pddl'):
            logger.warning(f"Domain should be PDDL: {domain_path}")

        try:
            with open(domain_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '(define (domain' not in content:
                    logger.warning(f"Domain may not be valid PDDL")
        except Exception as e:
            logger.error(f"Could not read domain: {e}")
            return False

        logger.info(f"✓ Domain validated: {domain_path}")
        return True

    @staticmethod
    def validate_problems(problem_pattern: str) -> bool:
        """Validate problem files."""
        if not problem_pattern:
            logger.error("Problem pattern not provided")
            return False

        import glob
        problems = sorted(glob.glob(problem_pattern))

        if not problems:
            logger.error(f"No problems found: {problem_pattern}")
            return False

        logger.info(f"✓ Found {len(problems)} problem(s)")

        # Sample check first few
        for prob in problems[:min(3, len(problems))]:
            if not prob.endswith('.pddl'):
                logger.warning(f"Problem should be PDDL: {prob}")
            try:
                with open(prob, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '(define (problem' not in content:
                        logger.warning(f"May not be valid PDDL: {prob}")
            except Exception as e:
                logger.error(f"Could not read problem: {prob} - {e}")
                return False

        return True

    @staticmethod
    def validate_output_dir(output_dir: str) -> bool:
        """Validate output directory is writable."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)

            logger.info(f"✓ Output directory ready: {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Cannot write to output: {e}")
            return False

    @staticmethod
    def validate_experiment_dirs(experiment_dirs: list) -> bool:
        """Validate experiment directories."""
        if not experiment_dirs:
            logger.error("No experiment directories provided")
            return False

        for exp_dir in experiment_dirs:
            if not os.path.exists(exp_dir):
                logger.error(f"Experiment directory not found: {exp_dir}")
                return False

            results_file = os.path.join(exp_dir, "results.json")
            if not os.path.exists(results_file):
                logger.warning(f"Results file may not exist: {results_file}")

            logger.info(f"✓ Experiment validated: {exp_dir}")

        return True

    @staticmethod
    def validate_all_standalone(
            model_path: str,
            domain_path: str,
            problem_pattern: str,
            output_dir: str
    ) -> bool:
        """Run all validations for standalone mode."""
        print_section("INPUT VALIDATION (STANDALONE)")

        checks = [
            ("Model file", lambda: EvaluationValidator.validate_model(model_path)),
            ("Domain PDDL", lambda: EvaluationValidator.validate_domain(domain_path)),
            ("Problem files", lambda: EvaluationValidator.validate_problems(problem_pattern)),
            ("Output directory", lambda: EvaluationValidator.validate_output_dir(output_dir)),
        ]

        results = []
        for name, check in checks:
            try:
                result = check()
                results.append((name, result))
                if not result:
                    logger.error(f"✗ {name} validation failed")
            except Exception as e:
                logger.error(f"✗ {name} validation exception: {e}")
                results.append((name, False))

        passed = sum(1 for _, r in results if r)
        total = len(results)

        logger.info(f"\nValidation: {passed}/{total} passed")

        return all(r for _, r in results)

    @staticmethod
    def validate_all_experiment_analysis(
            experiment_dirs: list,
            output_dir: str
    ) -> bool:
        """Run validations for experiment analysis mode."""
        print_section("INPUT VALIDATION (EXPERIMENT ANALYSIS)")

        checks = [
            ("Experiment directories", lambda: EvaluationValidator.validate_experiment_dirs(experiment_dirs)),
            ("Output directory", lambda: EvaluationValidator.validate_output_dir(output_dir)),
        ]

        results = []
        for name, check in checks:
            try:
                result = check()
                results.append((name, result))
            except Exception as e:
                logger.error(f"✗ {name} exception: {e}")
                results.append((name, False))

        passed = sum(1 for _, r in results if r)
        logger.info(f"\nValidation: {passed}/{len(results)} passed")

        return all(r for _, r in results)


# ============================================================================
# PIPELINE STAGES
# ============================================================================

def run_comprehensive_evaluation(
        model_path: str,
        domain_path: str,
        problem_pattern: str,
        output_dir: str,
        timeout: int,
        skip_baselines: bool,
        baseline_names: list = None
) -> bool:
    """Run comprehensive evaluation."""
    print_section("STAGE 1: COMPREHENSIVE EVALUATION")

    try:
        cmd = [
            "python", "evaluation_comprehensive.py",
            "--domain", domain_path,
            "--problems", problem_pattern,
            "--output", output_dir,
            "--timeout", str(timeout)
        ]

        if model_path:
            cmd.extend(["--model", model_path])

        if skip_baselines:
            cmd.append("--skip-baselines")

        if baseline_names:
            cmd.extend(["--baselines"] + baseline_names)

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            logger.error(f"Evaluation failed (code {result.returncode})")
            return False

        logger.info("✅ Evaluation complete")
        return True

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_analysis_and_visualization(
        results_csv: str,
        output_dir: str
) -> bool:
    """Run analysis and visualization."""
    print_section("STAGE 2: ANALYSIS & VISUALIZATION")

    if not os.path.exists(results_csv):
        logger.warning(f"Results CSV not found: {results_csv}")
        logger.warning("Skipping analysis")
        return False

    try:
        plots_dir = os.path.join(output_dir, "plots")

        cmd = [
            "python", "analysis_and_visualization.py",
            "--results", results_csv,
            "--output", plots_dir
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            logger.warning(f"Analysis exited with code {result.returncode}")
            return False

        logger.info("✅ Analysis complete")
        return True

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False


def run_experiment_analysis(
        experiment_dirs: list,
        output_dir: str
) -> bool:
    """Analyze experiment results."""
    print_section("STAGE 1: EXPERIMENT ANALYSIS")

    try:
        plots_dir = os.path.join(output_dir, "plots")

        cmd = [
                  "python", "analysis_and_visualization.py",
                  "--experiments"] + experiment_dirs + [
                  "--output", plots_dir
              ]

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            logger.error(f"Analysis failed (code {result.returncode})")
            return False

        logger.info("✅ Analysis complete")
        return True

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False


def generate_final_report(output_dir: str) -> bool:
    """Generate final evaluation report."""
    print_section("FINAL REPORT GENERATION")

    try:
        report_path = os.path.join(output_dir, "EVALUATION_REPORT.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("GNN MERGE STRATEGY - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

            f.write("EVALUATION OVERVIEW\n")
            f.write("-" * 100 + "\n")
            f.write("Complete evaluation of GNN-guided merge-and-shrink planning.\n\n")

            f.write("KEY OUTPUT FILES\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Results CSV:          {os.path.join(output_dir, 'evaluation_results.csv')}\n")
            f.write(f"  Summary JSON:         {os.path.join(output_dir, 'evaluation_summary.json')}\n")
            f.write(f"  Comparison Report:    {os.path.join(output_dir, 'comparison_report.txt')}\n")
            f.write(f"  Plots:                {os.path.join(output_dir, 'plots/')}\n\n")

            f.write("METRICS EXPLAINED\n")
            f.write("-" * 100 + "\n")
            f.write("  Solve Rate: % of problems solved\n")
            f.write("  Time: Wall clock time (seconds)\n")
            f.write("  Expansions: Nodes expanded during search\n")
            f.write("  H* Preservation: Quality of heuristic function (GNN-specific, 1.0 = perfect)\n")
            f.write("  Efficiency Score: Combined metric (lower = better)\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 100 + "\n")
            f.write("  ✓ GOOD:    Solve rate >= 80% with reasonable time\n")
            f.write("  ⚠ FAIR:    Solve rate 60-80% or slow execution\n")
            f.write("  ✗ POOR:    Solve rate < 60%\n\n")

            f.write("=" * 100 + "\n")

        logger.info(f"✓ Report: {report_path}")

        logger.info("\n" + "=" * 100)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 100)

        logger.info(f"\n📁 Results: {os.path.abspath(output_dir)}")
        logger.info(f"📊 View: {report_path}")

        return True

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str, width: int = 100):
    """Print section header."""
    logger.info("")
    logger.info("=" * width)
    logger.info(f"// {title}")
    logger.info("=" * width)
    logger.info("")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Evaluation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Complete evaluation (baseline + GNN)
  python run_full_evaluation.py \\
      --model mvp_output/gnn_model.zip \\
      --domain domain.pddl \\
      --problems "problem_*.pddl" \\
      --output evaluation_results/

  # GNN only (skip baselines)
  python run_full_evaluation.py \\
      --model model.zip \\
      --domain domain.pddl \\
      --problems "problem_*.pddl" \\
      --skip-baselines \\
      --output results/

  # Analyze experiments
  python run_full_evaluation.py \\
      --analyze-experiments \\
      --experiments exp1 exp2 exp3 \\
      --output analysis_results/
        """
    )

    parser.add_argument("--analyze-experiments", action="store_true",
                        help="Analyze completed experiments")

    parser.add_argument("--model", help="Path to GNN model (ZIP)")
    parser.add_argument("--domain", help="Path to domain PDDL")
    parser.add_argument("--problems", help="Glob pattern for problems")
    parser.add_argument("--experiments", nargs='+', help="Experiment directories")

    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per problem")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baselines")
    parser.add_argument("--baselines", nargs='+', help="Specific baselines")
    parser.add_argument("--downward-dir", help="Fast Downward directory")

    args = parser.parse_args()

    logger.info("✅ Evaluation Orchestrator Started")

    # ========================================================================
    # EXPERIMENT ANALYSIS MODE
    # ========================================================================

    if args.analyze_experiments:
        if not args.experiments:
            logger.error("--experiments required with --analyze-experiments")
            return 1

        if not EvaluationValidator.validate_all_experiment_analysis(
                args.experiments,
                args.output
        ):
            logger.error("❌ Validation failed")
            return 1

        if not run_experiment_analysis(args.experiments, args.output):
            logger.error("❌ Experiment analysis failed")
            return 1

        if not generate_final_report(args.output):
            logger.error("❌ Report generation failed")
            return 1

        logger.info("✅ EXPERIMENT ANALYSIS PIPELINE COMPLETE")
        return 0

    # ========================================================================
    # STANDALONE EVALUATION MODE
    # ========================================================================

    if not args.domain or not args.problems:
        logger.error("--domain and --problems required for standalone evaluation")
        parser.print_help()
        return 1

    if not args.skip_baselines and not args.model:
        logger.error("--model required unless --skip-baselines")
        return 1

    # Validation
    if not EvaluationValidator.validate_all_standalone(
            args.model or "",
            args.domain,
            args.problems,
            args.output
    ):
        logger.error("❌ Validation failed")
        return 1

    # Run evaluation
    if not run_comprehensive_evaluation(
            args.model or "",
            args.domain,
            args.problems,
            args.output,
            args.timeout,
            args.skip_baselines,
            args.baselines
    ):
        logger.error("❌ Evaluation failed")
        return 1

    # Run analysis
    results_csv = os.path.join(args.output, "evaluation_results.csv")
    if not run_analysis_and_visualization(results_csv, args.output):
        logger.warning("⚠️ Analysis had issues (continuing)")

    # Generate report
    if not generate_final_report(args.output):
        logger.error("❌ Report generation failed")
        return 1

    logger.info("✅ EVALUATION PIPELINE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())


