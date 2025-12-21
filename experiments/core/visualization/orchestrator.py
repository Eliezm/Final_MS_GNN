#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORCHESTRATOR - Main visualization pipeline (COMPLETE)
======================================================
✅ Includes all 32 plots organized by research question
✅ Training diagnostics (18-22) now integrated
✅ Generalization analysis (23-27) for size/seen-unseen
✅ Curriculum analysis (28-32) for phase transitions
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import numpy as np
import json

from experiments.core.logging import EpisodeMetrics

logger = logging.getLogger(__name__)

# Import all plot generators
from experiments.core.visualization.plots_01_learning import plot_learning_curves
from experiments.core.visualization.plots_02_components import (
    plot_component_trajectories, plot_component_stability, plot_merge_quality_heatmap
)
from experiments.core.visualization.plots_03_features import plot_feature_importance
from experiments.core.visualization.plots_04_quality import plot_bisimulation_preservation
from experiments.core.visualization.plots_05_safety import plot_dead_end_analysis
from experiments.core.visualization.plots_06_transitions import (
    plot_label_reduction_impact, plot_transition_explosion
)
from experiments.core.visualization.plots_07_decisions import (
    plot_causal_alignment, plot_gnn_decision_quality
)
from experiments.core.visualization.plots_08_baselines import (
    plot_merge_quality_distribution,
    plot_three_way_comparison,
    plot_per_problem_winners,
    plot_cumulative_solved,
    plot_speedup_analysis,
)
from experiments.core.visualization.plots_09_literature import plot_literature_alignment

# Training diagnostics (NEW - NOW INTEGRATED)
from experiments.core.visualization.plots_10_training_diagnostics import (
    plot_policy_entropy_evolution,
    plot_value_loss_evolution,
    plot_gradient_health,
    plot_inference_performance,
    plot_graph_compression,
)

# Generalization analysis (NEW)
from experiments.core.visualization.plots_11_generalization import (
    plot_performance_by_problem_size,
    plot_seen_vs_unseen_gap,
    plot_training_size_effect,
    plot_complexity_correlation,
    plot_generalization_heatmap,
)

# Curriculum analysis (NEW)
from experiments.core.visualization.plots_12_curriculum import (
    plot_curriculum_phase_transitions,
    plot_knowledge_transfer_analysis,
    plot_domain_transfer_results,
    plot_curriculum_vs_direct_training,
    plot_phase_difficulty_progression,
)


def _ensure_analysis_data(
        training_log: List[EpisodeMetrics],
        component_analysis: Optional[Dict],
        correlation_analysis: Optional[Dict],
        feature_importance_analysis: Optional[Dict],
        bisim_analysis: Optional[Dict],
        causal_alignment_analysis: Optional[Dict],
        explosion_analysis: Optional[Dict],
        decision_quality_analysis: Optional[Dict],
        episode_reward_signals: Optional[Dict],
        literature_checklist: Optional[Dict],
) -> Dict[str, Any]:
    """
    Compute any missing analysis data from training log.
    """
    from experiments.core.analysis import (
        analyze_component_trajectories,
        analyze_feature_reward_correlation,
        analyze_feature_importance_from_decisions,
        analyze_bisimulation_preservation,
        analyze_causal_alignment,
        analyze_transition_explosion_risk,
        analyze_gnn_decision_quality,
        generate_literature_alignment_report,
    )

    results = {}

    # Component analysis
    if component_analysis:
        results['component_analysis'] = component_analysis
    elif training_log:
        logger.info("Computing component_analysis from training log...")
        results['component_analysis'] = analyze_component_trajectories(training_log, None)
    else:
        results['component_analysis'] = {}

    # Correlation analysis
    if correlation_analysis:
        results['correlation_analysis'] = correlation_analysis
    elif episode_reward_signals:
        logger.info("Computing correlation_analysis from reward signals...")
        results['correlation_analysis'] = analyze_feature_reward_correlation(episode_reward_signals, None)
    else:
        results['correlation_analysis'] = {}

    # Feature importance
    if feature_importance_analysis:
        results['feature_importance_analysis'] = feature_importance_analysis
    elif training_log:
        logger.info("Computing feature_importance from training log...")
        results['feature_importance_analysis'] = analyze_feature_importance_from_decisions(training_log, None)
    else:
        results['feature_importance_analysis'] = {}

    # Bisimulation analysis
    if bisim_analysis:
        results['bisim_analysis'] = bisim_analysis
    elif training_log:
        logger.info("Computing bisim_analysis from training log...")
        results['bisim_analysis'] = analyze_bisimulation_preservation(training_log, None)
    else:
        results['bisim_analysis'] = {}

    # Causal alignment
    if causal_alignment_analysis:
        results['causal_alignment_analysis'] = causal_alignment_analysis
    elif training_log:
        logger.info("Computing causal_alignment from training log...")
        results['causal_alignment_analysis'] = analyze_causal_alignment(training_log, None)
    else:
        results['causal_alignment_analysis'] = {}

    # Explosion analysis
    if explosion_analysis:
        results['explosion_analysis'] = explosion_analysis
    elif training_log:
        logger.info("Computing explosion_analysis from training log...")
        results['explosion_analysis'] = analyze_transition_explosion_risk(training_log, None)
    else:
        results['explosion_analysis'] = {}

    # Decision quality
    if decision_quality_analysis:
        results['decision_quality_analysis'] = decision_quality_analysis
    elif training_log:
        logger.info("Computing decision_quality from training log...")
        decision_traces = {
            i: m.merge_decisions_per_step
            for i, m in enumerate(training_log)
            if m.merge_decisions_per_step
        }
        results['decision_quality_analysis'] = analyze_gnn_decision_quality(decision_traces, None)
    else:
        results['decision_quality_analysis'] = {}

    # Literature checklist
    if literature_checklist:
        results['literature_checklist'] = literature_checklist
    elif training_log and episode_reward_signals:
        logger.info("Generating literature_checklist...")
        results['literature_checklist'] = generate_literature_alignment_report(
            training_log,
            episode_reward_signals,
            results.get('correlation_analysis', {}),
            results.get('bisim_analysis', {}),
            None,
        )
    else:
        results['literature_checklist'] = {
            'h_preservation_preserved': bool(training_log),
            'transition_growth_penalized': bool(training_log),
            'label_combinability_extracted': bool(episode_reward_signals),
            'bisimulation_validation_exists': bool(results.get('bisim_analysis')),
        }

    results['episode_reward_signals'] = episode_reward_signals or {}

    return results


def generate_all_plots(
        training_log: List[EpisodeMetrics],
        eval_results: Dict,
        output_dir: Path,
        # Comparison data
        gnn_vs_random_detailed: Optional[List] = None,
        baseline_detailed: Optional[List] = None,
        gnn_stats: Optional[Dict] = None,
        random_stats: Optional[Dict] = None,
        baseline_stats: Optional[Dict] = None,
        statistics: Optional[Dict] = None,
        results: Optional[List] = None,
        gnn_results: Optional[Dict] = None,
        # Analysis data
        component_analysis: Optional[Dict] = None,
        correlation_analysis: Optional[Dict] = None,
        feature_importance_analysis: Optional[Dict] = None,
        bisim_analysis: Optional[Dict] = None,
        causal_alignment_analysis: Optional[Dict] = None,
        explosion_analysis: Optional[Dict] = None,
        decision_quality_analysis: Optional[Dict] = None,
        episode_reward_signals: Optional[Dict] = None,
        literature_checklist: Optional[Dict] = None,
        # Test results for generalization plots
        test_results: Optional[Dict] = None,
        # Curriculum data
        is_curriculum: bool = False,
        phase_results: Optional[Dict] = None,
        curriculum_results: Optional[Dict] = None,
        direct_training_results: Optional[Dict] = None,
) -> Dict[str, Optional[Path]]:
    """
    Generate ALL 32 visualization plots.

    Plots are organized by research question:
    - 01-04: Training dynamics
    - 05: Feature analysis
    - 06: Heuristic quality
    - 07: Safety
    - 08-09: Abstraction size
    - 10-11: Merge strategy
    - 12: Distribution
    - 13-16: Comparison
    - 17: Literature
    - 18-22: Training diagnostics (NEW - NOW INTEGRATED)
    - 23-27: Generalization (NEW)
    - 28-32: Curriculum (NEW)
    """
    output_path = Path(output_dir)
    results_plots = {}

    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 100)
    print("🎨 GENERATING ALL VISUALIZATION PLOTS (32 total)")
    print("=" * 100 + "\n")

    # =========================================================================
    # STEP 1: ENSURE ALL ANALYSIS DATA EXISTS
    # =========================================================================

    analysis_data = _ensure_analysis_data(
        training_log=training_log,
        component_analysis=component_analysis,
        correlation_analysis=correlation_analysis,
        feature_importance_analysis=feature_importance_analysis,
        bisim_analysis=bisim_analysis,
        causal_alignment_analysis=causal_alignment_analysis,
        explosion_analysis=explosion_analysis,
        decision_quality_analysis=decision_quality_analysis,
        episode_reward_signals=episode_reward_signals,
        literature_checklist=literature_checklist,
    )

    # =========================================================================
    # STEP 2: PREPARE COMPARISON DATA
    # =========================================================================

    gnn_vs_random_detailed = gnn_vs_random_detailed or []
    baseline_detailed = baseline_detailed or []
    gnn_stats = gnn_stats or {}
    random_stats = random_stats or {}
    baseline_stats = baseline_stats or {}
    test_results = test_results or {}
    phase_results = phase_results or {}

    all_detailed_results = gnn_vs_random_detailed + baseline_detailed

    # =========================================================================
    # STEP 3: DEFINE ALL 32 PLOTS
    # =========================================================================

    plots_to_generate = [
        # === TRAINING DYNAMICS (01-04) ===
        ("01_learning_curves", plot_learning_curves,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        ("02_component_trajectories", plot_component_trajectories,
         {"training_log": training_log,
          "component_analysis": analysis_data['component_analysis'],
          "output_dir": output_path},
         "training_log"),

        ("03_component_stability", plot_component_stability,
         {"component_analysis": analysis_data['component_analysis'],
          "output_dir": output_path},
         "component_analysis"),

        ("04_merge_quality_heatmap", plot_merge_quality_heatmap,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        # === FEATURE ANALYSIS (05) ===
        ("05_feature_importance", plot_feature_importance,
         {"feature_importance_analysis": analysis_data['feature_importance_analysis'],
          "correlation_analysis": analysis_data['correlation_analysis'],
          "output_dir": output_path},
         "feature_importance_analysis"),

        # === HEURISTIC QUALITY (06) ===
        ("06_bisimulation_preservation", plot_bisimulation_preservation,
         {"training_log": training_log,
          "bisim_analysis": analysis_data['bisim_analysis'],
          "output_dir": output_path},
         "training_log"),

        # === SAFETY (07) ===
        ("07_dead_end_timeline", plot_dead_end_analysis,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        # === ABSTRACTION SIZE (08-09) ===
        ("08_label_reduction_impact", plot_label_reduction_impact,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        ("09_transition_explosion", plot_transition_explosion,
         {"training_log": training_log,
          "explosion_analysis": analysis_data['explosion_analysis'],
          "output_dir": output_path},
         "training_log"),

        # === MERGE STRATEGY (10-11) ===
        ("10_causal_alignment", plot_causal_alignment,
         {"training_log": training_log,
          "causal_analysis": analysis_data['causal_alignment_analysis'],
          "output_dir": output_path},
         "training_log"),

        ("11_gnn_decision_quality", plot_gnn_decision_quality,
         {"decision_quality_analysis": analysis_data['decision_quality_analysis'],
          "output_dir": output_path},
         "decision_quality_analysis"),

        # === DISTRIBUTION (12) ===
        ("12_merge_quality_distribution", plot_merge_quality_distribution,
         {"episode_reward_signals": analysis_data['episode_reward_signals'],
          "output_dir": output_path},
         "episode_reward_signals"),

        # === COMPARISON (13-16) ===
        ("13_three_way_comparison", plot_three_way_comparison,
         {"gnn_stats": gnn_stats,
          "random_stats": random_stats,
          "baseline_stats": baseline_stats,
          "output_dir": output_path},
         "comparison_data"),

        ("14_per_problem_winners", plot_per_problem_winners,
         {"detailed_results": all_detailed_results,
          "output_dir": output_path},
         "comparison_data"),

        ("15_cumulative_solved", plot_cumulative_solved,
         {"detailed_results": all_detailed_results,
          "output_dir": output_path},
         "comparison_data"),

        ("16_speedup_analysis", plot_speedup_analysis,
         {"gnn_results": gnn_vs_random_detailed,
          "baseline_results": baseline_detailed,
          "output_dir": output_path},
         "comparison_data"),

        # === LITERATURE VALIDATION (17) ===
        ("17_literature_alignment", plot_literature_alignment,
         {"checklist": analysis_data['literature_checklist'],
          "output_dir": output_path},
         "literature_checklist"),

        # === TRAINING DIAGNOSTICS (18-22) - NOW INTEGRATED ===
        ("18_policy_entropy_evolution", plot_policy_entropy_evolution,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        ("19_value_loss_evolution", plot_value_loss_evolution,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        ("20_gradient_health", plot_gradient_health,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        ("21_inference_performance", plot_inference_performance,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        ("22_graph_compression", plot_graph_compression,
         {"training_log": training_log, "output_dir": output_path},
         "training_log"),

        # === GENERALIZATION ANALYSIS (23-27) - NEW ===
        ("23_performance_by_problem_size", plot_performance_by_problem_size,
         {"test_results": test_results, "output_dir": output_path},
         "test_results"),

        ("24_seen_vs_unseen_gap", plot_seen_vs_unseen_gap,
         {"test_results": test_results, "output_dir": output_path},
         "test_results"),

        ("25_training_size_effect", plot_training_size_effect,
         {"experiment_results": {},  # Requires multi-experiment data
          "output_dir": output_path},
         "multi_experiment"),

        ("26_complexity_correlation", plot_complexity_correlation,
         {"test_results": test_results,
          "detailed_results": all_detailed_results,
          "output_dir": output_path},
         "comparison_data"),

        ("27_generalization_heatmap", plot_generalization_heatmap,
         {"test_results": test_results, "output_dir": output_path},
         "test_results"),

        # === CURRICULUM ANALYSIS (28-32) - NEW ===
        ("28_curriculum_phase_transitions", plot_curriculum_phase_transitions,
         {"phase_results": phase_results, "output_dir": output_path},
         "curriculum_data"),

        ("29_knowledge_transfer_analysis", plot_knowledge_transfer_analysis,
         {"phase_results": phase_results, "output_dir": output_path},
         "curriculum_data"),

        ("30_domain_transfer_results", plot_domain_transfer_results,
         {"same_domain_results": {k: v for k, v in test_results.items()
                                  if 'blocksworld' in k.lower()},
          "transfer_domain_results": {k: v for k, v in test_results.items()
                                      if 'logistics' in k.lower()},
          "output_dir": output_path},
         "test_results"),

        ("31_curriculum_vs_direct_training", plot_curriculum_vs_direct_training,
         {"curriculum_results": curriculum_results or {},
          "direct_training_results": direct_training_results or {},
          "output_dir": output_path},
         "multi_experiment"),

        ("32_phase_difficulty_progression", plot_phase_difficulty_progression,
         {"phase_results": phase_results, "output_dir": output_path},
         "curriculum_data"),
    ]

    # =========================================================================
    # STEP 4: GENERATE EACH PLOT
    # =========================================================================

    successful = 0
    failed = 0
    skipped = 0

    for plot_name, plot_func, plot_args, data_requirement in plots_to_generate:
        try:
            # Check if we have required data
            has_data = True
            skip_reason = None

            if data_requirement == "training_log" and not training_log:
                has_data = False
                skip_reason = "No training log"
            elif data_requirement == "component_analysis" and not analysis_data.get('component_analysis'):
                has_data = False
                skip_reason = "No component analysis"
            elif data_requirement == "feature_importance_analysis" and not analysis_data.get(
                    'feature_importance_analysis', {}).get('feature_importance'):
                has_data = False
                skip_reason = "No feature importance data"
            elif data_requirement == "decision_quality_analysis" and not analysis_data.get('decision_quality_analysis'):
                has_data = False
                skip_reason = "No decision quality data"
            elif data_requirement == "episode_reward_signals" and not analysis_data.get('episode_reward_signals'):
                has_data = False
                skip_reason = "No reward signals"
            elif data_requirement == "comparison_data" and not (
                    gnn_stats or random_stats or baseline_stats or all_detailed_results):
                has_data = False
                skip_reason = "No comparison data"
            elif data_requirement == "literature_checklist" and not analysis_data.get('literature_checklist'):
                has_data = False
                skip_reason = "No literature checklist"
            elif data_requirement == "test_results" and not test_results:
                has_data = False
                skip_reason = "No test results (run evaluation)"
            elif data_requirement == "curriculum_data" and not phase_results:
                has_data = False
                skip_reason = "No curriculum phase data (not a curriculum experiment)"
            elif data_requirement == "multi_experiment":
                has_data = False
                skip_reason = "Requires multi-experiment comparison data"

            print(f"  [{successful + failed + skipped + 1:2d}/32] {plot_name:45s}", end=" ", flush=True)

            if not has_data:
                print(f"⏭️  SKIPPED: {skip_reason}")
                results_plots[plot_name] = None
                skipped += 1
                continue

            result = plot_func(**plot_args)

            if result is not None and Path(result).exists():
                results_plots[plot_name] = result
                successful += 1
                print("✅ SUCCESS")
            else:
                results_plots[plot_name] = None
                failed += 1
                print("⚠️  No output generated")

        except Exception as e:
            results_plots[plot_name] = None
            failed += 1
            error_msg = str(e)[:50]
            print(f"❌ {error_msg}")
            logger.error(f"Plot {plot_name} failed: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 100)
    print("📊 PLOT GENERATION SUMMARY")
    print("=" * 100)
    print(f"✅ Successfully generated: {successful} plots")
    print(f"⏭️  Skipped (no data):      {skipped} plots")
    print(f"❌ Failed:                  {failed} plots")
    print(f"📁 Output directory:        {plots_dir}")
    print("=" * 100 + "\n")

    return results_plots