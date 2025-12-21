#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSAL ANALYSIS SCRIPT FOR EXPERIMENTS 4 & 5
================================================

Implements the "Silent Training / Deep Analysis" workflow:
- Parses structured EVENT logs from training.log
- Works with any experiment (auto-detect from logs)
- Extracts metrics and generates post-mortem analysis
- Creates visualization plots
- Runs AFTER training is complete

Run with:
    python analyze_logs.py <experiment_id>
    python analyze_logs.py <experiment_id> --base-dir misc/experiment_outputs
    python analyze_logs.py <experiment_id> --experiment 5

Examples:
    python analyze_logs.py 20240115_143022
    python analyze_logs.py 20240115_143022 --experiment 5
    python analyze_logs.py 20240115_143022 --base-dir misc/experiment_outputs/experiment_5
"""

import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Warning: Matplotlib not available. Plots will be skipped.")


class UniversalLogAnalyzer:
    """
    Universal analyzer for all experiments.

    Features:
    - Auto-detect experiment number from directory or args
    - Flexible event type parsing (handles different schemas)
    - Generic metrics extraction
    - Compatible with both SilentTrainingLogger and DualLogger
    """

    def __init__(
            self,
            experiment_id: str,
            base_dir: str = "misc/experiment_outputs",
            experiment_num: Optional[int] = None
    ):
        self.experiment_id = experiment_id
        self.base_dir = base_dir
        self.experiment_num = experiment_num

        # Auto-detect experiment number if not provided
        if self.experiment_num is None:
            self.experiment_num = self._detect_experiment_number()

        if self.experiment_num is None:
            raise ValueError(f"Could not determine experiment number. Use --experiment flag.")

        # Build paths
        self.experiment_dir = os.path.join(
            base_dir,
            f"experiment_{self.experiment_num}"
        )
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.analysis_dir = os.path.join(self.experiment_dir, "analysis")

        # Find matching log files
        self.log_files = self._find_log_files()

        os.makedirs(self.analysis_dir, exist_ok=True)

        self.events: List[Dict[str, Any]] = []
        self.metrics_data = defaultdict(list)
        self.detected_event_types: set = set()

    def _detect_experiment_number(self) -> Optional[int]:
        """Auto-detect experiment number from directory structure."""
        # Try common patterns
        for num in [5, 4, 3, 2, 1]:
            exp_dir = os.path.join(self.base_dir, f"experiment_{num}", "logs")
            if os.path.exists(exp_dir):
                log_files = list(Path(exp_dir).glob("training_*.log"))
                if log_files:
                    for log_file in log_files:
                        if self.experiment_id in log_file.name or self.experiment_id[:8] in log_file.name:
                            print(f"✓ Auto-detected: Experiment {num}")
                            return num
        return None

    def _find_log_files(self) -> List[str]:
        """Find all log files matching the experiment ID."""
        if not os.path.exists(self.log_dir):
            print(f"⚠️  Log directory not found: {self.log_dir}")
            return []

        log_files = []
        for f in sorted(Path(self.log_dir).glob("training_*.log")):
            if self.experiment_id in f.name or self.experiment_id[:8] in f.name:
                log_files.append(str(f))

        return log_files

    def parse_log_file(self) -> bool:
        """Parse training log files and extract structured EVENTs."""
        if not self.log_files:
            print(f"❌ No log files found for experiment {self.experiment_num}: {self.experiment_id}")
            print(f"   Checked in: {self.log_dir}")
            return False

        total_events = 0
        for log_file in self.log_files:
            print(f"📖 Parsing log file: {os.path.basename(log_file)}")

            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Look for EVENT markers (handles both loggers)
                    if "EVENT: " in line:
                        try:
                            # Extract JSON after "EVENT: "
                            json_str = line.split("EVENT: ", 1)[1]
                            event = json.loads(json_str)
                            self.events.append(event)
                            self.detected_event_types.add(event.get('event_type', 'unknown'))
                            total_events += 1
                        except (json.JSONDecodeError, IndexError) as e:
                            pass

        print(f"  ✓ Found {total_events} structured events")
        print(f"  ✓ Event types: {', '.join(sorted(self.detected_event_types))}")
        return total_events > 0

    def extract_training_metrics(self) -> Dict[str, Any]:
        """Extract comprehensive metrics from events (flexible schema)."""
        metrics = {
            'experiment_id': self.experiment_id,
            'experiment_num': self.experiment_num,
            'events_found': len(self.events),
            'event_types_found': sorted(list(self.detected_event_types)),
            'event_types': defaultdict(int),
            'training_started': None,
            'training_ended': None,
            'total_steps': 0,
            'checkpoints_saved': [],
            'best_models': [],
            'problems_trained': set(),
            'domains_seen': set(),
            'difficulties_seen': set(),
            'training_conditions': set(),
            'physics_config': {},
            'errors': [],
        }

        for event in self.events:
            event_type = event.get('event_type', 'unknown')
            metrics['event_types'][event_type] += 1

            # Flexible event parsing (handles different schemas)
            if event_type in ['experiment_started', 'training_started']:
                metrics['training_started'] = event.get('timestamp')

            elif event_type in ['experiment_completed', 'training_completed']:
                metrics['training_ended'] = event.get('timestamp')
                metrics['total_steps'] = event.get('total_steps', event.get('step', 0))

            elif event_type == 'checkpoint_saved':
                metrics['checkpoints_saved'].append({
                    'step': event.get('step', event.get('training_step')),
                    'path': event.get('path', event.get('checkpoint_id')),
                    'difficulty': event.get('difficulty', 'unknown'),
                    'domain': event.get('domain', event.get('domain_name', 'unknown')),
                    'reward': event.get('reward', 0.0),
                    'timestamp': event.get('timestamp'),
                })

            elif event_type == 'best_model_saved':
                metrics['best_models'].append({
                    'reward': event.get('reward', 0.0),
                    'problem': event.get('problem', event.get('checkpoint_id')),
                    'step': event.get('step', event.get('training_step')),
                    'timestamp': event.get('timestamp'),
                })

            elif event_type == 'training_step_completed':
                problem = event.get('problem', event.get('problem_name', 'unknown'))
                metrics['problems_trained'].add(problem)
                domain = event.get('domain', event.get('domain_name', 'unknown'))
                if domain != 'unknown':
                    metrics['domains_seen'].add(domain)

            elif event_type == 'difficulty_transition':
                difficulty = event.get('difficulty', 'unknown')
                metrics['difficulties_seen'].add(difficulty)

            elif event_type in ['experiment_failed', 'training_failed']:
                metrics['errors'].append({
                    'type': event_type,
                    'details': event.get('error', event.get('reason', 'unknown')),
                    'timestamp': event.get('timestamp'),
                })

            # Track training conditions (experiment 5)
            if event_type == 'training_condition_started':
                condition = event.get('condition', 'unknown')
                metrics['training_conditions'].add(condition)

        metrics['problems_trained'] = sorted(list(metrics['problems_trained']))
        metrics['domains_seen'] = sorted(list(metrics['domains_seen']))
        metrics['difficulties_seen'] = sorted(list(metrics['difficulties_seen']))
        metrics['training_conditions'] = sorted(list(metrics['training_conditions']))

        return metrics

    def generate_training_report(self) -> str:
        """Generate detailed text report of training."""
        report = []
        report.append("\n" + "=" * 90)
        report.append(f"EXPERIMENT {self.experiment_num}: ANALYSIS REPORT")
        report.append(f"Experiment ID: {self.experiment_id}")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 90)

        metrics = self.extract_training_metrics()

        # Timeline
        if metrics['training_started'] and metrics['training_ended']:
            report.append("\n📊 TRAINING TIMELINE:")
            report.append(f"  Start: {metrics['training_started']}")
            report.append(f"  End:   {metrics['training_ended']}")

        # Summary
        report.append("\n📈 TRAINING SUMMARY:")
        report.append(f"  Total events: {metrics['events_found']}")
        report.append(f"  Total timesteps: {metrics['total_steps']}")
        report.append(f"  Problems trained: {len(metrics['problems_trained'])}")
        if metrics['domains_seen']:
            report.append(f"  Domains seen: {', '.join(metrics['domains_seen'])}")
        if metrics['difficulties_seen']:
            report.append(f"  Difficulties seen: {', '.join(metrics['difficulties_seen'])}")
        if metrics['training_conditions']:
            report.append(f"  Training conditions: {', '.join(metrics['training_conditions'])}")

        # Checkpointing
        if metrics['checkpoints_saved']:
            report.append(f"\n💾 CHECKPOINTING:")
            report.append(f"  Total checkpoints: {len(metrics['checkpoints_saved'])}")
            for i, cp in enumerate(metrics['checkpoints_saved'][:10], 1):  # Show first 10
                difficulty = cp['difficulty'] if cp['difficulty'] != 'unknown' else cp['domain']
                report.append(f"    [{i}] Step {cp['step']:,} ({difficulty}): reward={cp['reward']:.4f}")
            if len(metrics['checkpoints_saved']) > 10:
                report.append(f"    ... and {len(metrics['checkpoints_saved']) - 10} more")

        # Best models
        if metrics['best_models']:
            report.append(f"\n🌟 BEST MODELS:")
            report.append(f"  Total best models found: {len(metrics['best_models'])}")
            if metrics['best_models']:
                best_reward = max([b['reward'] for b in metrics['best_models']], default=0)
                report.append(f"  Peak reward: {best_reward:.4f}")

        # Event breakdown
        report.append("\n📋 EVENT BREAKDOWN:")
        for event_type, count in sorted(metrics['event_types'].items(), key=lambda x: -x[1])[:15]:
            report.append(f"  {event_type}: {count}")

        # Detected event types
        report.append("\n🏷️  DETECTED EVENT TYPES:")
        for et in sorted(metrics['event_types_found']):
            report.append(f"  - {et}")

        # Errors
        if metrics['errors']:
            report.append("\n⚠️  ERRORS & WARNINGS:")
            for error in metrics['errors'][:10]:
                report.append(f"  [{error['type']}] {error['details']}")
            if len(metrics['errors']) > 10:
                report.append(f"  ... and {len(metrics['errors']) - 10} more")

        report.append("\n" + "=" * 90)
        report.append("✅ Analysis complete!")
        report.append("=" * 90)

        return "\n".join(report)

    def save_report(self):
        """Save analysis report to file."""
        report = self.generate_training_report()

        report_path = os.path.join(self.analysis_dir, f"analysis_{self.experiment_id}.txt")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n📊 Report saved: {report_path}")
        print(report)

    def save_metrics_json(self):
        """Save extracted metrics as JSON."""
        metrics = self.extract_training_metrics()

        # Convert sets to lists for JSON serialization
        metrics_serializable = {
            k: list(v) if isinstance(v, set) else v
            for k, v in metrics.items()
        }
        metrics_serializable['event_types'] = dict(metrics_serializable['event_types'])

        metrics_path = os.path.join(self.analysis_dir, f"metrics_{self.experiment_id}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2, default=str)

        print(f"📋 Metrics saved: {metrics_path}")

    def plot_training_progress(self):
        """Generate plots of training and checkpointing."""
        if not HAS_MATPLOTLIB:
            print("⚠️  Matplotlib not available - skipping plots")
            return

        metrics = self.extract_training_metrics()

        # Extract checkpoint data
        checkpoint_steps = [cp['step'] for cp in metrics['checkpoints_saved']]
        checkpoint_rewards = [cp['reward'] for cp in metrics['checkpoints_saved']]

        if not checkpoint_steps:
            print("⚠️  No checkpoint data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f'Experiment {self.experiment_num}: Training Analysis ({self.experiment_id})',
            fontsize=16,
            fontweight='bold'
        )

        # Plot 1: Reward over timesteps
        ax = axes[0, 0]
        ax.plot(checkpoint_steps, checkpoint_rewards, 'b.-', linewidth=2, markersize=10, label='Checkpoint Reward')
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Reward', fontsize=11)
        ax.set_title('Model Reward Progress at Checkpoints', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Checkpoint frequency
        ax = axes[0, 1]
        if len(checkpoint_steps) > 1:
            checkpoint_intervals = np.diff(checkpoint_steps)
            ax.hist(checkpoint_intervals, bins=max(3, len(checkpoint_intervals) // 2),
                    edgecolor='black', color='green', alpha=0.7)
        ax.set_xlabel('Steps Between Checkpoints', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Checkpoint Interval Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Event type distribution
        ax = axes[1, 0]
        event_types = metrics['event_types']
        top_events = sorted(event_types.items(), key=lambda x: -x[1])[:10]
        if top_events:
            types, counts = zip(*top_events)
            bars = ax.barh(types, counts, color='purple', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Count', fontsize=11)
            ax.set_title('Top 10 Event Types', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        # Plot 4: Cumulative steps
        ax = axes[1, 1]
        if checkpoint_steps:
            ax.plot(range(len(checkpoint_steps)), checkpoint_steps, 'r.-', linewidth=2, markersize=10)
            ax.set_xlabel('Checkpoint Number', fontsize=11)
            ax.set_ylabel('Cumulative Training Step', fontsize=11)
            ax.set_title('Cumulative Training Progress', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(self.analysis_dir, f"training_progress_{self.experiment_id}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📈 Plot saved: {plot_path}")
        plt.close()

    def plot_best_model_history(self):
        """Plot best model discoveries over time."""
        if not HAS_MATPLOTLIB:
            return

        metrics = self.extract_training_metrics()
        best_models = metrics['best_models']

        if not best_models:
            print("⚠️  No best model history to plot")
            return

        # Extract rewards and indices
        rewards = [bm['reward'] for bm in best_models]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(rewards)), rewards, 'g*-', linewidth=2, markersize=15, label='Best Model Found')
        ax.set_xlabel('Best Model #', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Best Model Discovery History', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        plot_path = os.path.join(self.analysis_dir, f"best_models_{self.experiment_id}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📈 Best models plot saved: {plot_path}")
        plt.close()


def main():
    """Main entry point."""
    parser = ArgumentParser(
        description="Analyze experiment logs (universal for Experiments 1-5)",
        epilog="""
Examples:
  # Auto-detect experiment
  python analyze_logs.py 20240115_143022

  # Specify experiment explicitly
  python analyze_logs.py 20240115_143022 --experiment 5

  # Specify custom base directory
  python analyze_logs.py 20240115_143022 --base-dir misc/experiment_outputs

  # Skip plots
  python analyze_logs.py 20240115_143022 --no-plots
        """
    )
    parser.add_argument('experiment_id', type=str, help='Experiment ID to analyze')
    parser.add_argument(
        '--base-dir',
        default='misc/experiment_outputs',
        help='Base output directory (default: misc/experiment_outputs)'
    )
    parser.add_argument(
        '--experiment',
        type=int,
        default=None,
        help='Experiment number (auto-detect if not specified)'
    )
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    print("\n" + "=" * 90)
    print(f"UNIVERSAL LOG ANALYSIS - {args.experiment_id}")
    print("=" * 90 + "\n")

    try:
        analyzer = UniversalLogAnalyzer(
            args.experiment_id,
            base_dir=args.base_dir,
            experiment_num=args.experiment
        )

        # Parse log file
        if not analyzer.parse_log_file():
            return 1

        # Generate reports
        print("\n📊 Generating analysis...")
        analyzer.save_report()
        analyzer.save_metrics_json()

        if not args.no_plots:
            analyzer.plot_training_progress()
            analyzer.plot_best_model_history()

        print("\n✅ Analysis complete!")
        return 0

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())