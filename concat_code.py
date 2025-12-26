#!/usr/bin/env python3
import os
import glob

OUTPUT_FILE = "concat_code.txt"

def get_py_files():
    """
    Returns a list of all .py files in cwd, excluding this script itself.
    """
    all_py = glob.glob("*.py")
    this_script = os.path.basename(__file__)
    return [f for f in all_py if f != this_script]

def concatenate(files, out_path):
    with open(out_path, "w", encoding="utf-8") as out:
        for fname in files:
            out.write(f"The file with its path {fname} code is in the following block:\n")
            # out.write(f"Please understand how it is compatible with the rest of my codebase for the mission i asked you.\n")

            try:
                with open(fname, "r", encoding="utf-8") as inp:
                    out.write(inp.read())
            except Exception as e:
                out.write(f"[Error reading {fname}: {e}]\n")
            out.write(f"\n")
            # out.write(f"End: The file {fname} code is this:\n")
            out.write("\n" + "-" * 80 + "\n\n")

def main():

    infra_files = [

    "experiments/shared_experiment_utils.py",
    "experiments/__init__.py",
    "experiments/run_experiment.py",
    "experiments/run_full_experiment.py",
    "experiments/configs/__init__.py",
    "experiments/configs/experiment_configs.py",
    "experiments/runners/__init__.py",
    "experiments/runners/experiment_runner.py",
    "experiments/core/__init__.py",

    "experiments/core/evaluation.py",
    "experiments/core/benchmark_discovery.py",
    "experiments/core/evaluation_analyzer.py",
    "experiments/core/evaluation_config.py",
    "experiments/core/evaluation_export.py",
    "experiments/core/evaluation_metrics.py",
    "experiments/core/evaluation_orchestrator.py",
    "experiments/core/evaluation_plots.py",

    "experiments/core/gnn_random_evaluation.py",
    "experiments/core/gnn_evaluation_evaluators.py",
    "experiments/core/gnn_evaluation_executor.py",
    "experiments/core/gnn_evaluation_framework.py",
    "experiments/core/gnn_evaluation_parser.py",
    "experiments/core/gnn_evaluation_policies.py",
    "experiments/core/baseline_runner.py",

    "experiments/core/training.py",
    "experiments/core/logging.py",
    "experiments/core/unified_reporting.py",
    "experiments/core/output_structure.py",
    #
    "experiments/core/analysis.py",
    "experiments/core/analysis/__init__.py",
    "experiments/core/analysis/analysis_components.py",
    "experiments/core/analysis/analysis_decisions.py",
    "experiments/core/analysis/analysis_metrics.py",
    "experiments/core/analysis/analysis_features.py",
    "experiments/core/analysis/analysis_orchestrator.py",
    "experiments/core/analysis/analysis_quality.py",
    "experiments/core/analysis/analysis_safety.py",
    "experiments/core/analysis/analysis_training.py",
    "experiments/core/analysis/analysis_utils.py",
    "experiments/core/analysis/analysis_validation.py",

    "experiments/core/visualization.py",
    "experiments/core/visualization/plot_research_mapping.py",
    "experiments/core/visualization/__init__.py",
    "experiments/core/visualization/orchestrator.py",
    "experiments/core/visualization/plots_01_learning.py",
    "experiments/core/visualization/plots_02_components.py",
    "experiments/core/visualization/plots_03_features.py",
    "experiments/core/visualization/plots_04_quality.py",
    "experiments/core/visualization/plots_05_safety.py",
    "experiments/core/visualization/plots_06_transitions.py",
    "experiments/core/visualization/plots_07_decisions.py",
    "experiments/core/visualization/plots_08_baselines.py",
    "experiments/core/visualization/plots_09_literature.py",
    "experiments/core/visualization/plots_10_training_diagnostics.py",
    "experiments/core/visualization/plots_11_generalization.py",
    "experiments/core/visualization/plots_12_curriculum.py",
    "experiments/core/visualization/plotting_utils.py",

    # "src/environments/thin_merge_env.py",
    # "src/communication/communication_protocol.py",
    # "src/utils/common_utils.py",
    # "src/models/gnn_model.py",
    # "src/models/gnn_policy.py",
    # "src/rewards/reward_function_enhanced.py",


    # "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.cc",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.h",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_signals.cc",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_signals.h",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_signals_enhanced.cc",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_signals_enhanced.h",
    # #
    # "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.cc",
    # "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.h",
    # "downward/src/search/merge_and_shrink/merge_strategy_gnn.cc",
    # "downward/src/search/merge_and_shrink/merge_strategy_gnn.h",

    ]

    """
    PROBLEMS:
    1. THE EPISODE FAILED 'EXELLLENT' PROBLEM - ✗ Episode 1 failed: could not convert string to float: 'excellent'  
    2. THE MISSING PLOTS PROBLEM
    3. TNOT PRINTING PLOTS AT ALL IN CURRICULUM AND EMPTY TEST_RESULTS
    4. THE OUTPUT FOLDERS CONSISTENCY
    5. ENSURE EVALUATION AGAINST BASELINES
    """

    if not infra_files:
        print("No Python files found to concatenate.")
        return
    concatenate(infra_files, OUTPUT_FILE)
    print(f"✅ Concatenated {len(infra_files)} file(s) into '{OUTPUT_FILE}'")

    # py_files = get_py_files()
    # for p in py_files:
    #     print(f"{p}")

if __name__ == "__main__":
    main()