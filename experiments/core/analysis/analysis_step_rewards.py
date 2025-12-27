#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP-WISE REWARD ANALYSIS
=========================
Analyze rewards at each merge step to understand:
- Initial merges vs later merges
- Which steps the GNN excels at
- Patterns in reward progression within episodes
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict

from experiments.core.logging import EpisodeMetrics, StepRewardLog, EpisodeStepRewardSummary


@dataclass
class StepPositionAnalysis:
    """Analysis of rewards by step position (early/mid/late)."""

    # Overall statistics
    total_episodes: int = 0
    total_steps: int = 0

    # Phase-wise statistics
    early_step_avg_reward: float = 0.0
    early_step_std_reward: float = 0.0
    mid_step_avg_reward: float = 0.0
    mid_step_std_reward: float = 0.0
    late_step_avg_reward: float = 0.0
    late_step_std_reward: float = 0.0

    # Which phase performs best?
    best_phase: str = "unknown"  # early, mid, late
    worst_phase: str = "unknown"
    phase_improvement: float = 0.0  # late_avg - early_avg

    # Quality distribution by phase
    early_excellent_rate: float = 0.0
    early_bad_rate: float = 0.0
    late_excellent_rate: float = 0.0
    late_bad_rate: float = 0.0

    # Trend detection
    improving_episodes_pct: float = 0.0
    declining_episodes_pct: float = 0.0
    stable_episodes_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepComponentAnalysis:
    """Analysis of reward components by step position."""

    # H* preservation by phase
    early_h_preservation: float = 0.0
    late_h_preservation: float = 0.0
    h_preservation_trend: str = "stable"  # improving, declining, stable

    # Transition control by phase
    early_transition_growth: float = 1.0
    late_transition_growth: float = 1.0
    transition_control_trend: str = "stable"

    # OPP by phase
    early_opp_score: float = 0.5
    late_opp_score: float = 0.5
    opp_trend: str = "stable"

    # Label combinability by phase
    early_label_comb: float = 0.5
    late_label_comb: float = 0.5
    label_comb_trend: str = "stable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_step_rewards_from_training_log(
        episode_log: List[EpisodeMetrics],
        output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Analyze step-wise rewards from training log.

    Args:
        episode_log: List of EpisodeMetrics from training
        output_dir: Optional directory to save analysis

    Returns:
        Dict with step-wise analysis results
    """
    if not episode_log:
        return {"error": "No episodes to analyze"}

    # Collect all step data
    all_step_rewards = defaultdict(list)  # step_position -> [rewards]
    all_step_components = defaultdict(lambda: defaultdict(list))  # component -> step_position -> [values]
    episode_summaries = []

    phase_rewards = {'early': [], 'mid': [], 'late': []}
    phase_h_pres = {'early': [], 'mid': [], 'late': []}
    phase_trans_growth = {'early': [], 'mid': [], 'late': []}
    phase_opp = {'early': [], 'mid': [], 'late': []}
    phase_label = {'early': [], 'mid': [], 'late': []}
    phase_quality = {'early': defaultdict(int), 'late': defaultdict(int)}

    trends = {'improving': 0, 'declining': 0, 'stable': 0}

    for metrics in episode_log:
        if metrics.error is not None:
            continue

        step_data = metrics.merge_decisions_per_step
        if not step_data:
            continue

        num_steps = len(step_data)
        if num_steps < 3:
            continue

        third = max(1, num_steps // 3)

        # Collect step-wise data
        step_rewards_list = []
        early_rewards = []
        mid_rewards = []
        late_rewards = []

        for i, step in enumerate(step_data):
            reward = step.get('immediate_reward', step.get('reward', 0.0))
            step_rewards_list.append(reward)

            h_pres = step.get('h_preservation', step.get('h_star_preservation', 1.0))
            trans_growth = step.get('transition_growth', 1.0)
            opp = step.get('opp_score', 0.5)
            label = step.get('label_combinability', 0.5)
            quality = step.get('merge_quality_category', 'neutral')

            # Categorize by phase
            if i < third:
                phase = 'early'
                early_rewards.append(reward)
            elif i < 2 * third:
                phase = 'mid'
                mid_rewards.append(reward)
            else:
                phase = 'late'
                late_rewards.append(reward)

            phase_rewards[phase].append(reward)
            phase_h_pres[phase].append(h_pres)
            phase_trans_growth[phase].append(trans_growth)
            phase_opp[phase].append(opp)
            phase_label[phase].append(label)

            if phase in ['early', 'late']:
                phase_quality[phase][quality] += 1

            # Store by absolute step position (normalized to 0-10)
            normalized_pos = int(i / num_steps * 10)
            all_step_rewards[normalized_pos].append(reward)

        # Trend detection for this episode
        early_avg = np.mean(early_rewards) if early_rewards else 0
        late_avg = np.mean(late_rewards) if late_rewards else 0

        if late_avg > early_avg + 0.1:
            trends['improving'] += 1
        elif late_avg < early_avg - 0.1:
            trends['declining'] += 1
        else:
            trends['stable'] += 1

    # Compute statistics
    total_episodes = trends['improving'] + trends['declining'] + trends['stable']

    position_analysis = StepPositionAnalysis(
        total_episodes=total_episodes,
        total_steps=sum(len(r) for r in phase_rewards.values()),
        early_step_avg_reward=float(np.mean(phase_rewards['early'])) if phase_rewards['early'] else 0.0,
        early_step_std_reward=float(np.std(phase_rewards['early'])) if phase_rewards['early'] else 0.0,
        mid_step_avg_reward=float(np.mean(phase_rewards['mid'])) if phase_rewards['mid'] else 0.0,
        mid_step_std_reward=float(np.std(phase_rewards['mid'])) if phase_rewards['mid'] else 0.0,
        late_step_avg_reward=float(np.mean(phase_rewards['late'])) if phase_rewards['late'] else 0.0,
        late_step_std_reward=float(np.std(phase_rewards['late'])) if phase_rewards['late'] else 0.0,
        improving_episodes_pct=trends['improving'] / max(1, total_episodes) * 100,
        declining_episodes_pct=trends['declining'] / max(1, total_episodes) * 100,
        stable_episodes_pct=trends['stable'] / max(1, total_episodes) * 100,
    )

    # Determine best/worst phase
    phase_avgs = {
        'early': position_analysis.early_step_avg_reward,
        'mid': position_analysis.mid_step_avg_reward,
        'late': position_analysis.late_step_avg_reward,
    }
    position_analysis.best_phase = max(phase_avgs, key=phase_avgs.get)
    position_analysis.worst_phase = min(phase_avgs, key=phase_avgs.get)
    position_analysis.phase_improvement = phase_avgs['late'] - phase_avgs['early']

    # Quality rates
    early_total = sum(phase_quality['early'].values())
    late_total = sum(phase_quality['late'].values())

    position_analysis.early_excellent_rate = phase_quality['early']['excellent'] / max(1, early_total) * 100
    position_analysis.early_bad_rate = phase_quality['early']['bad'] / max(1, early_total) * 100
    position_analysis.late_excellent_rate = phase_quality['late']['excellent'] / max(1, late_total) * 100
    position_analysis.late_bad_rate = phase_quality['late']['bad'] / max(1, late_total) * 100

    # Component analysis
    component_analysis = StepComponentAnalysis(
        early_h_preservation=float(np.mean(phase_h_pres['early'])) if phase_h_pres['early'] else 1.0,
        late_h_preservation=float(np.mean(phase_h_pres['late'])) if phase_h_pres['late'] else 1.0,
        early_transition_growth=float(np.mean(phase_trans_growth['early'])) if phase_trans_growth['early'] else 1.0,
        late_transition_growth=float(np.mean(phase_trans_growth['late'])) if phase_trans_growth['late'] else 1.0,
        early_opp_score=float(np.mean(phase_opp['early'])) if phase_opp['early'] else 0.5,
        late_opp_score=float(np.mean(phase_opp['late'])) if phase_opp['late'] else 0.5,
        early_label_comb=float(np.mean(phase_label['early'])) if phase_label['early'] else 0.5,
        late_label_comb=float(np.mean(phase_label['late'])) if phase_label['late'] else 0.5,
    )

    # Determine trends
    if component_analysis.late_h_preservation > component_analysis.early_h_preservation + 0.02:
        component_analysis.h_preservation_trend = "improving"
    elif component_analysis.late_h_preservation < component_analysis.early_h_preservation - 0.02:
        component_analysis.h_preservation_trend = "declining"

    if component_analysis.late_transition_growth < component_analysis.early_transition_growth - 0.1:
        component_analysis.transition_control_trend = "improving"
    elif component_analysis.late_transition_growth > component_analysis.early_transition_growth + 0.1:
        component_analysis.transition_control_trend = "declining"

    # Build per-position statistics
    step_position_stats = {}
    for pos in range(11):  # 0-10
        if all_step_rewards[pos]:
            step_position_stats[f"position_{pos}"] = {
                "count": len(all_step_rewards[pos]),
                "mean_reward": float(np.mean(all_step_rewards[pos])),
                "std_reward": float(np.std(all_step_rewards[pos])),
                "min_reward": float(np.min(all_step_rewards[pos])),
                "max_reward": float(np.max(all_step_rewards[pos])),
            }

    results = {
        "step_position_analysis": position_analysis.to_dict(),
        "step_component_analysis": component_analysis.to_dict(),
        "step_position_stats": step_position_stats,
        "quality_distribution": {
            "early": dict(phase_quality['early']),
            "late": dict(phase_quality['late']),
        },
        "summary": {
            "gnn_excels_in": position_analysis.best_phase + " merges",
            "gnn_struggles_in": position_analysis.worst_phase + " merges",
            "overall_trend": "IMPROVING" if position_analysis.phase_improvement > 0.05 else
            "DECLINING" if position_analysis.phase_improvement < -0.05 else "STABLE",
            "h_preservation_pattern": component_analysis.h_preservation_trend,
            "key_insight": _generate_key_insight(position_analysis, component_analysis),
        }
    }

    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "step_reward_analysis.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✓ Step reward analysis saved to: {output_dir / 'step_reward_analysis.json'}")

    return results


def _generate_key_insight(pos_analysis: StepPositionAnalysis,
                          comp_analysis: StepComponentAnalysis) -> str:
    """Generate human-readable insight from analysis."""
    insights = []

    if pos_analysis.phase_improvement > 0.1:
        insights.append("GNN improves significantly as episode progresses")
    elif pos_analysis.phase_improvement < -0.1:
        insights.append("GNN performs better on initial merges than later ones")

    if pos_analysis.late_excellent_rate > pos_analysis.early_excellent_rate + 10:
        insights.append("Quality of merges improves in later steps")
    elif pos_analysis.early_excellent_rate > pos_analysis.late_excellent_rate + 10:
        insights.append("Initial merges tend to be higher quality")

    if comp_analysis.h_preservation_trend == "declining":
        insights.append("H* preservation degrades in later steps - focus training on late-game")

    if comp_analysis.transition_control_trend == "declining":
        insights.append("Transition explosion risk increases in later steps")

    if not insights:
        insights.append("Reward distribution is relatively uniform across merge steps")

    return "; ".join(insights)


def print_step_reward_analysis(analysis: Dict[str, Any]) -> None:
    """Print human-readable step reward analysis."""

    print("\n" + "=" * 80)
    print("STEP-WISE REWARD ANALYSIS")
    print("=" * 80)

    pos = analysis.get("step_position_analysis", {})
    comp = analysis.get("step_component_analysis", {})
    summary = analysis.get("summary", {})

    print(f"\n📊 OVERVIEW")
    print(f"   Total Episodes: {pos.get('total_episodes', 0)}")
    print(f"   Total Steps: {pos.get('total_steps', 0)}")

    print(f"\n📈 PHASE-WISE PERFORMANCE")
    print(
        f"   EARLY (initial merges):  avg={pos.get('early_step_avg_reward', 0):+.4f} ± {pos.get('early_step_std_reward', 0):.4f}")
    print(
        f"   MID   (middle merges):   avg={pos.get('mid_step_avg_reward', 0):+.4f} ± {pos.get('mid_step_std_reward', 0):.4f}")
    print(
        f"   LATE  (final merges):    avg={pos.get('late_step_avg_reward', 0):+.4f} ± {pos.get('late_step_std_reward', 0):.4f}")

    print(f"\n🎯 GNN PERFORMANCE PATTERN")
    print(f"   Best Phase:  {pos.get('best_phase', 'unknown').upper()}")
    print(f"   Worst Phase: {pos.get('worst_phase', 'unknown').upper()}")
    print(f"   Phase Improvement (late-early): {pos.get('phase_improvement', 0):+.4f}")

    print(f"\n📉 EPISODE TRENDS")
    print(f"   Improving: {pos.get('improving_episodes_pct', 0):.1f}% of episodes")
    print(f"   Declining: {pos.get('declining_episodes_pct', 0):.1f}% of episodes")
    print(f"   Stable:    {pos.get('stable_episodes_pct', 0):.1f}% of episodes")

    print(f"\n⭐ MERGE QUALITY RATES")
    print(f"   Early Excellent Rate: {pos.get('early_excellent_rate', 0):.1f}%")
    print(f"   Late Excellent Rate:  {pos.get('late_excellent_rate', 0):.1f}%")
    print(f"   Early Bad Rate:       {pos.get('early_bad_rate', 0):.1f}%")
    print(f"   Late Bad Rate:        {pos.get('late_bad_rate', 0):.1f}%")

    print(f"\n🔬 COMPONENT ANALYSIS BY PHASE")
    print(
        f"   H* Preservation:     early={comp.get('early_h_preservation', 0):.3f} → late={comp.get('late_h_preservation', 0):.3f} ({comp.get('h_preservation_trend', 'stable')})")
    print(
        f"   Transition Growth:   early={comp.get('early_transition_growth', 0):.2f}x → late={comp.get('late_transition_growth', 0):.2f}x ({comp.get('transition_control_trend', 'stable')})")
    print(
        f"   OPP Score:           early={comp.get('early_opp_score', 0):.3f} → late={comp.get('late_opp_score', 0):.3f}")
    print(
        f"   Label Combinability: early={comp.get('early_label_comb', 0):.3f} → late={comp.get('late_label_comb', 0):.3f}")

    print(f"\n💡 KEY INSIGHT")
    print(f"   {summary.get('key_insight', 'No insight available')}")

    print("\n" + "=" * 80 + "\n")


def export_step_rewards_csv(
        episode_log: List[EpisodeMetrics],
        output_path: Path,
) -> Path:
    """Export step-by-step rewards to CSV for detailed analysis."""

    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'episode', 'step', 'problem_name', 'reward',
            'h_star_preservation', 'transition_growth', 'opp_score', 'label_combinability',
            'states_before', 'states_after', 'is_solvable', 'dead_end_ratio',
            'merge_quality_category', 'phase',  # phase: early/mid/late
            'component_h', 'component_trans', 'component_opp', 'component_label', 'component_bonus'
        ])

        for metrics in episode_log:
            if metrics.error is not None:
                continue

            step_data = metrics.merge_decisions_per_step
            if not step_data:
                continue

            num_steps = len(step_data)
            third = max(1, num_steps // 3)

            for i, step in enumerate(step_data):
                # Determine phase
                if i < third:
                    phase = 'early'
                elif i < 2 * third:
                    phase = 'mid'
                else:
                    phase = 'late'

                writer.writerow([
                    metrics.episode,
                    i,
                    metrics.problem_name,
                    step.get('immediate_reward', step.get('reward', 0.0)),
                    step.get('h_preservation', step.get('h_star_preservation', 1.0)),
                    step.get('transition_growth', 1.0),
                    step.get('opp_score', 0.5),
                    step.get('label_combinability', 0.5),
                    step.get('states_before', 0),
                    step.get('states_after', 0),
                    step.get('is_solvable', True),
                    step.get('dead_end_ratio', 0.0),
                    step.get('merge_quality_category', 'neutral'),
                    phase,
                    step.get('component_h', 0.0),
                    step.get('component_trans', 0.0),
                    step.get('component_opp', 0.0),
                    step.get('component_label', 0.0),
                    step.get('component_bonus', 0.0),
                ])

    print(f"✓ Step rewards exported to: {output_path}")
    return output_path