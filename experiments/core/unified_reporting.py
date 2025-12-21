#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED REPORTING - Single source of truth for experiment results
=================================================================
Eliminates redundant reports and consolidates all outputs.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd


class UnifiedReporter:
    """
    Creates a single, comprehensive experiment report.

    Replaces:
    - experiment_summary.json
    - curriculum_summary.json
    - final_report.json
    - comparison_tables.json

    With ONE unified report.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_unified_report(
            self,
            config: Dict[str, Any],
            training_summary: Dict[str, Any],
            analysis_summary: Dict[str, Any],
            evaluation_summary: Dict[str, Any],
            test_results: Dict[str, Dict[str, Any]],
            baseline_summary: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Create THE single experiment report.

        Structure:
        {
            "metadata": {...},
            "training": {...},
            "analysis": {...},
            "evaluation": {
                "gnn_vs_random": {...},
                "baselines": {...},
            },
            "testing": {
                "test_set_1": {...},
                "test_set_2": {...},
            },
            "summary_tables": {...},
            "key_findings": [...],
        }
        """

        report = {
            "metadata": {
                "experiment_name": config.get("name", "unknown"),
                "description": config.get("description", ""),
                "timestamp": datetime.now().isoformat(),
                "config": config,
            },

            "training": training_summary,

            "analysis": analysis_summary,

            "evaluation": {
                "gnn_vs_random": evaluation_summary.get("gnn_vs_random", {}),
                "baselines": baseline_summary or {},
            },

            "testing": test_results,

            "summary_tables": self._create_summary_tables(
                training_summary, evaluation_summary, test_results, baseline_summary
            ),

            "key_findings": self._extract_key_findings(
                training_summary, evaluation_summary, test_results
            ),
        }

        # Save as JSON
        report_path = self.output_dir / "experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save human-readable version
        self._save_readable_report(report)

        # Save CSV tables
        self._save_csv_tables(report)

        return report_path

    def _create_summary_tables(
            self,
            training_summary: Dict,
            evaluation_summary: Dict,
            test_results: Dict,
            baseline_summary: Optional[Dict],
    ) -> Dict[str, List[Dict]]:
        """Create summary tables for paper."""

        tables = {}

        # Table 1: Strategy Comparison
        strategy_rows = []

        gnn_stats = evaluation_summary.get("gnn_vs_random", {}).get("GNN", {})
        if gnn_stats:
            strategy_rows.append({
                "Strategy": "GNN (Learned)",
                "Solve Rate (%)": f"{gnn_stats.get('solve_rate_pct', 0):.1f}",
                "Mean Time (s)": f"{gnn_stats.get('mean_time_sec', 0):.3f}",
                "Mean Expansions": f"{gnn_stats.get('mean_expansions', 0):,}",
                "H* Preservation": f"{gnn_stats.get('mean_h_preservation', 1.0):.4f}",
            })

        random_stats = evaluation_summary.get("gnn_vs_random", {}).get("Random", {})
        if random_stats:
            strategy_rows.append({
                "Strategy": "Random Merge",
                "Solve Rate (%)": f"{random_stats.get('solve_rate_pct', 0):.1f}",
                "Mean Time (s)": f"{random_stats.get('mean_time_sec', 0):.3f}",
                "Mean Expansions": f"{random_stats.get('mean_expansions', 0):,}",
                "H* Preservation": f"{random_stats.get('mean_h_preservation', 1.0):.4f}",
            })

        if baseline_summary:
            for name, stats in list(baseline_summary.items())[:5]:
                if isinstance(stats, dict):
                    strategy_rows.append({
                        "Strategy": name[:30],
                        "Solve Rate (%)": f"{stats.get('solve_rate_%', stats.get('solve_rate_pct', 0)):.1f}",
                        "Mean Time (s)": f"{stats.get('avg_time_total_s', stats.get('mean_time_sec', 0)):.3f}",
                        "Mean Expansions": f"{stats.get('avg_expansions', stats.get('mean_expansions', 0)):,}",
                        "H* Preservation": "N/A",
                    })

        tables["strategy_comparison"] = strategy_rows

        # Table 2: Test Set Results
        test_rows = []
        for test_name, test_data in test_results.items():
            results = test_data.get("results", {})
            summary = results.get("summary", {})

            test_rows.append({
                "Test Set": test_name,
                "GNN Solved": f"{summary.get('gnn_solved', 0)}/{summary.get('gnn_total', 0)}",
                "Random Solved": f"{summary.get('random_solved', 0)}/{summary.get('random_total', 0)}",
                "GNN Solve Rate": f"{(summary.get('gnn_solved', 0) / max(1, summary.get('gnn_total', 1)) * 100):.1f}%",
                "Improvement": f"{(summary.get('gnn_solved', 0) - summary.get('random_solved', 0)):+d}",
            })

        tables["test_results"] = test_rows

        return tables

    def _extract_key_findings(
            self,
            training_summary: Dict,
            evaluation_summary: Dict,
            test_results: Dict,
    ) -> List[str]:
        """Extract key findings for paper abstract/intro."""

        findings = []

        # Training findings
        num_episodes = training_summary.get("num_train_episodes", 0)
        if num_episodes > 0:
            findings.append(f"Successfully trained GNN on {num_episodes} episodes")

        # GNN vs Random
        gnn_stats = evaluation_summary.get("gnn_vs_random", {}).get("GNN", {})
        random_stats = evaluation_summary.get("gnn_vs_random", {}).get("Random", {})

        if gnn_stats and random_stats:
            gnn_rate = gnn_stats.get('solve_rate_pct', 0)
            random_rate = random_stats.get('solve_rate_pct', 0)
            improvement = gnn_rate - random_rate

            if improvement > 0:
                findings.append(f"GNN outperforms random by {improvement:.1f}% solve rate")

            gnn_h = gnn_stats.get('mean_h_preservation', 1.0)
            if gnn_h > 0.95:
                findings.append(f"GNN achieves {gnn_h:.1%} h* preservation")

        # Generalization
        if test_results:
            generalized_tests = [
                name for name, data in test_results.items()
                if "unseen" in name.lower() and
                   data.get("results", {}).get("summary", {}).get("gnn_solved", 0) > 0
            ]
            if generalized_tests:
                findings.append(f"GNN generalizes to {len(generalized_tests)} unseen test sets")

        return findings

    def _save_readable_report(self, report: Dict) -> Path:
        """Save human-readable text report."""

        output_path = self.output_dir / "experiment_report.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"EXPERIMENT REPORT: {report['metadata']['experiment_name']}\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Description: {report['metadata']['description']}\n")
            f.write(f"Generated: {report['metadata']['timestamp']}\n\n")

            f.write("-" * 100 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("-" * 100 + "\n")
            for i, finding in enumerate(report['key_findings'], 1):
                f.write(f"  {i}. {finding}\n")
            f.write("\n")

            f.write("-" * 100 + "\n")
            f.write("STRATEGY COMPARISON\n")
            f.write("-" * 100 + "\n")
            for row in report['summary_tables'].get('strategy_comparison', []):
                f.write(f"\n  {row['Strategy']}:\n")
                for key, value in row.items():
                    if key != 'Strategy':
                        f.write(f"    {key}: {value}\n")
            f.write("\n")

            f.write("-" * 100 + "\n")
            f.write("TEST SET RESULTS\n")
            f.write("-" * 100 + "\n")
            for row in report['summary_tables'].get('test_results', []):
                f.write(f"\n  {row['Test Set']}:\n")
                for key, value in row.items():
                    if key != 'Test Set':
                        f.write(f"    {key}: {value}\n")

            f.write("\n" + "=" * 100 + "\n")

        return output_path

    def _save_csv_tables(self, report: Dict) -> None:
        """Save tables as CSV for paper."""

        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        for table_name, table_data in report['summary_tables'].items():
            if table_data:
                df = pd.DataFrame(table_data)
                df.to_csv(tables_dir / f"{table_name}.csv", index=False)