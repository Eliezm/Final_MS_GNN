#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION EXPORT - Results export and reporting
==============================================
Exports results in CSV, JSON, and text formats.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import numpy as np

from experiments.core.evaluation_metrics import DetailedMetrics, AggregateStatistics
from experiments.core.evaluation_config import EvaluationConfig

logger = logging.getLogger(__name__)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results_to_csv(
        results: List[DetailedMetrics],
        output_path: str
) -> None:
    """Export detailed results to CSV with proper serialization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to export")
        return

    fieldnames = [
        "planner_name",
        "problem_name",
        "solved",
        "plan_cost",
        "wall_clock_time",
        "translate_time",
        "search_time",
        "nodes_expanded",
        "nodes_generated",
        "search_depth",
        "branching_factor",
        "peak_memory_kb",
        "h_star_preservation",
        "error_type",
        "timestamp"
    ]

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for result in results:
                if not isinstance(result, DetailedMetrics):
                    logger.warning(f"Skipping non-DetailedMetrics result: {type(result)}")
                    continue

                row = result.to_dict()

                # ✅ SANITIZE: Convert non-serializable types
                for field in fieldnames:
                    if field not in row:
                        row[field] = None
                    elif row[field] is None:
                        row[field] = None
                    elif isinstance(row[field], np.ndarray):
                        row[field] = row[field].tolist()  # Convert array to list
                    elif isinstance(row[field], (np.integer, np.floating)):
                        row[field] = float(row[field]) if isinstance(row[field], np.floating) else int(row[field])
                    elif isinstance(row[field], bool):
                        row[field] = "yes" if row[field] else "no"
                    elif isinstance(row[field], (list, dict)):
                        row[field] = str(row[field])[:100]  # Truncate complex types

                writer.writerow(row)

        logger.info(f"✓ CSV exported: {output_path} ({len(results)} results)")

    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())


def export_statistics_to_json(
        statistics: Dict[str, AggregateStatistics],
        output_path: str
) -> None:
    """
    Export statistics to JSON.

    Args:
        statistics: Dict mapping planner_name -> AggregateStatistics
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {name: stats.to_dict() for name, stats in statistics.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"✓ JSON exported: {output_path}")


def export_summary_report(
        statistics: Dict[str, AggregateStatistics],
        speedup_analysis: Dict[str, Any],
        winners: Dict[str, str],
        output_path: str
) -> None:
    """
    Export comprehensive text report.

    Includes:
    - Planner statistics
    - Per-problem winners
    - Speedup analysis
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        # Statistics section
        f.write("PLANNER STATISTICS\n")
        f.write("-" * 100 + "\n")
        for planner_name in sorted(statistics.keys()):
            stats = statistics[planner_name]
            f.write(f"\n{planner_name}:\n")
            f.write(f"  Solved: {stats.num_problems_solved}/{stats.num_problems_total} "
                    f"({stats.solve_rate_pct:.1f}%)\n")
            f.write(f"  Time: {stats.mean_time_sec:.2f}s (±{stats.std_time_sec:.2f}s)\n")
            f.write(f"  Expansions: {stats.mean_expansions:,}\n")
            if stats.mean_h_preservation < 1.0:
                f.write(f"  H* Preservation: {stats.mean_h_preservation:.3f}\n")

        # Winners
        if winners:
            f.write("\n\nPER-PROBLEM WINNERS\n")
            f.write("-" * 100 + "\n")
            for problem in sorted(winners.keys()):
                f.write(f"  {problem}: {winners[problem]}\n")

        # Speedup
        if speedup_analysis:
            f.write("\n\nGNN SPEEDUP ANALYSIS\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Mean Speedup: {speedup_analysis.get('mean_speedup', 0):.2f}x\n")
            f.write(f"  Geometric Mean Speedup: {speedup_analysis.get('geometric_mean_speedup', 0):.2f}x\n")
            f.write(f"  Comparisons: {speedup_analysis.get('num_speedup_comparisons', 0)}\n")

        f.write("\n" + "=" * 100 + "\n")

    logger.info(f"✓ Report exported: {output_path}")


def generate_reports(
        all_results: List[Dict[str, Any]],
        all_detailed: List[DetailedMetrics]
) -> None:
    """
    Generate CSV summary and detailed JSON reports.

    Creates:
    - CSV with summary statistics
    - JSON with detailed results
    - Text summary report
    """

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: GENERATE REPORTS")
    logger.info("=" * 80)

    if not all_results:
        logger.error("No results to report")
        return

    Path(EvaluationConfig.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ====== CSV SUMMARY ======
    fieldnames = [
        "name", "set_size", "solved", "solve_rate_%",
        "avg_time_total_s", "avg_time_translation_s", "avg_time_search_s",
        "avg_expansions", "avg_cost", "avg_generated_states",
        "avg_max_abstraction_size", "avg_final_abstraction_size",
        "avg_num_merges", "avg_num_shrinks", "avg_num_label_reductions",
        "errors"
    ]

    csv_path = EvaluationConfig.OUTPUT_CSV
    logger.info(f"\nWriting CSV report: {csv_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"  ✓ {len(all_results)} baseline configurations")

    # ====== JSON DETAILED ======
    json_path = EvaluationConfig.OUTPUT_DETAILED
    logger.info(f"\nWriting detailed JSON: {json_path}")

    detailed_dicts = [m.to_dict() for m in all_detailed]
    with open(json_path, "w") as f:
        json.dump(detailed_dicts, f, indent=2, default=str)

    logger.info(f"  ✓ {len(all_detailed)} individual problem results")

    # ====== PRINT SUMMARY TABLE ======
    logger.info("\n" + "=" * 100)
    logger.info("BASELINE PERFORMANCE SUMMARY")
    logger.info("=" * 100)

    header = (
        f"{'Planner':<25} "
        f"{'Solved':<10} "
        f"{'Total':<8} "
        f"{'Time':<10} "
        f"{'Trans':<10} "
        f"{'Search':<10} "
        f"{'Exp':<10} "
        f"{'Cost':<8}"
    )
    logger.info(header)
    logger.info("-" * 100)

    for result in all_results:
        name = result["name"][:23]
        solved = f"{result['solved']}/{result['set_size']}"
        rate = f"({result['solve_rate_%']:.0f}%)"
        time_str = f"{result['avg_time_total_s']:.3f}s"
        trans_str = f"{result['avg_time_translation_s']:.3f}s"
        search_str = f"{result['avg_time_search_s']:.3f}s"
        exp_str = f"{result['avg_expansions']:,}"
        cost_str = f"{result['avg_cost']}"

        logger.info(
            f"{name:<25} "
            f"{solved:<10} {rate:<8} "
            f"{time_str:<10} "
            f"{trans_str:<10} "
            f"{search_str:<10} "
            f"{exp_str:<10} "
            f"{cost_str:<8}"
        )

    logger.info("-" * 100)
    logger.info(f"\nFull reports:")
    logger.info(f"  CSV:      {Path(csv_path).absolute()}")
    logger.info(f"  Detailed: {Path(json_path).absolute()}")