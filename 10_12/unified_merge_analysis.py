#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED MERGE & SHRINK ANALYSIS SYSTEM
=======================================
Central post-training analysis framework consolidating:
  ✅ Merge metadata collection from C++ exports
  ✅ Merge explainability (human-readable narratives)
  ✅ Merge choice pattern discovery
  ✅ GNN decision analysis
  ✅ Unified reporting and recommendations

ONE FILE - handles everything post-training!

Usage:
    # After experiment completes
    analyzer = UnifiedMergeAnalyzer(experiment_output_dir="overfit_results/")
    results = analyzer.run_complete_analysis()

    # Access individual components if needed
    merge_analysis = analyzer.merge_analyzer
    gnn_analysis = analyzer.gnn_analyzer
    patterns = analyzer.pattern_analyzer
"""

import os
import json
import logging
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

INF = 1000000000


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class MergeDecision:
    """Metadata for a single merge decision."""
    iteration: int
    ts1_id: int
    ts2_id: int
    ts1_size: int
    ts2_size: int
    expected_product_size: int
    ts1_transitions: int
    ts2_transitions: int
    ts1_density: float
    ts2_density: float
    ts1_goal_states: int
    ts2_goal_states: int
    ts1_reachable_fraction: float
    ts2_reachable_fraction: float
    ts1_variables: List[int] = field(default_factory=list)
    ts2_variables: List[int] = field(default_factory=list)
    shrunk: bool = False
    reduced: bool = False
    merged_size: int = 0
    merged_goal_states: int = 0
    merged_transitions: int = 0
    merged_density: float = 0.0
    reachable_states: int = 0
    unreachable_states: int = 0
    reachability_ratio: float = 0.0
    shrinking_ratio: float = 0.0
    ts1_f_min: int = 0
    ts1_f_max: int = 0
    ts1_f_mean: float = 0.0
    ts1_f_std: float = 0.0
    ts2_f_min: int = 0
    ts2_f_max: int = 0
    ts2_f_mean: float = 0.0
    ts2_f_std: float = 0.0
    merged_f_min: int = 0
    merged_f_max: int = 0
    merged_f_mean: float = 0.0
    merged_f_std: float = 0.0
    branching_factor: float = 1.0
    search_depth: int = 0
    solution_found: bool = False
    solution_cost: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def quality_score(self) -> float:
        """Compute quality score for this merge (0-1, higher is better).

        MERGE QUALITY FRAMEWORK:
        - Compression (35%): How well did shrinking work?
        - F-Value Stability (35%): Did heuristic quality remain stable?
        - Reachability (20%): Were states preserved?
        - Search Efficiency (10%): Did branching stay reasonable?
        """
        # Component 1: Compression efficiency
        compression = self.shrinking_ratio
        compression_score = min(compression, 1.0)

        # Component 2: F-value stability
        if self.merged_f_mean > 0 and self.ts1_f_mean > 0 and self.ts2_f_mean > 0:
            avg_before = (self.ts1_f_mean + self.ts2_f_mean) / 2.0
            f_stability = 1.0 - abs(self.merged_f_mean - avg_before) / (avg_before + 1e-6)
            f_stability_score = max(0.0, min(f_stability, 1.0))
        else:
            f_stability_score = 0.5

        # Component 3: Reachability
        reachability_score = self.reachability_ratio

        # Component 4: Search efficiency
        if self.branching_factor >= 1.0:
            bf_score = 1.0 / (1.0 + (self.branching_factor - 1.0) / 3.0)
            bf_score = max(0.0, min(bf_score, 1.0))
        else:
            bf_score = 0.5

        # Weighted combination
        overall = (
                0.35 * compression_score +
                0.35 * f_stability_score +
                0.20 * reachability_score +
                0.10 * bf_score
        )

        return max(0.0, min(overall, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# PHASE 1: MERGE METADATA COLLECTION
# ============================================================================

class MergeMetadataCollector:
    """Collects and validates merge metadata from C++ JSON exports."""

    def __init__(self, fd_output_dir: str = "downward/fd_output"):
        self.fd_output_dir = Path(fd_output_dir)
        self.merges: List[MergeDecision] = []

    def load_all_metadata(self) -> int:
        """Load all merge metadata from JSON files."""
        logger.info(f"Loading merge metadata from: {self.fd_output_dir}")

        if not self.fd_output_dir.exists():
            logger.warning(f"fd_output directory not found: {self.fd_output_dir}")
            return 0

        iteration = 0
        loaded_count = 0

        while True:
            before_file = self.fd_output_dir / f"merge_before_{iteration}.json"
            after_file = self.fd_output_dir / f"merge_after_{iteration}.json"

            if not before_file.exists() or not after_file.exists():
                break

            try:
                merge_data = self._load_merge_pair(iteration, before_file, after_file)
                if merge_data:
                    self.merges.append(merge_data)
                    loaded_count += 1
                    logger.info(f"  ✓ Loaded merge {iteration}")
            except Exception as e:
                logger.warning(f"  Failed to load merge {iteration}: {e}")

            iteration += 1

        logger.info(f"✓ Loaded {loaded_count} merge decisions")
        return loaded_count

    def _load_merge_pair(
            self,
            iteration: int,
            before_file: Path,
            after_file: Path
    ) -> Optional[MergeDecision]:
        """Load a single merge pair."""
        with open(before_file) as f:
            before_data = json.load(f)

        with open(after_file) as f:
            after_data = json.load(f)

        # Extract F-value statistics
        ts1_f_stats = before_data.get("ts1_f_stats", {})
        ts2_f_stats = before_data.get("ts2_f_stats", {})
        merged_f_stats = after_data.get("f_stats", {})

        # Count reachable states
        ts1_f = before_data.get("ts1_f_values", [])
        ts2_f = before_data.get("ts2_f_values", [])
        ts1_reachable = sum(1 for f in ts1_f if f != INF and f >= 0) / max(len(ts1_f), 1)
        ts2_reachable = sum(1 for f in ts2_f if f != INF and f >= 0) / max(len(ts2_f), 1)

        merge_data = MergeDecision(
            iteration=iteration,
            ts1_id=before_data.get("ts1_id", -1),
            ts2_id=before_data.get("ts2_id", -1),
            ts1_size=before_data.get("ts1_size", 0),
            ts2_size=before_data.get("ts2_size", 0),
            expected_product_size=before_data.get("expected_product_size", 0),
            ts1_transitions=before_data.get("ts1_transitions", 0),
            ts2_transitions=before_data.get("ts2_transitions", 0),
            ts1_density=before_data.get("ts1_density", 0.0),
            ts2_density=before_data.get("ts2_density", 0.0),
            ts1_goal_states=before_data.get("ts1_goal_states", 0),
            ts2_goal_states=before_data.get("ts2_goal_states", 0),
            ts1_reachable_fraction=ts1_reachable,
            ts2_reachable_fraction=ts2_reachable,
            ts1_variables=before_data.get("ts1_variables", []),
            ts2_variables=before_data.get("ts2_variables", []),
            shrunk=before_data.get("shrunk", False),
            reduced=before_data.get("reduced", False),
            merged_size=after_data.get("merged_size", 0),
            merged_goal_states=after_data.get("merged_goal_states", 0),
            merged_transitions=after_data.get("merged_transitions", 0),
            merged_density=after_data.get("merged_density", 0.0),
            reachable_states=after_data.get("reachable_states", 0),
            unreachable_states=after_data.get("unreachable_states", 0),
            reachability_ratio=after_data.get("reachability_ratio", 0.0),
            shrinking_ratio=after_data.get("shrinking_ratio", 0.0),
            ts1_f_min=ts1_f_stats.get("min", 0),
            ts1_f_max=ts1_f_stats.get("max", 0),
            ts1_f_mean=ts1_f_stats.get("mean", 0.0),
            ts1_f_std=ts1_f_stats.get("std", 0.0),
            ts2_f_min=ts2_f_stats.get("min", 0),
            ts2_f_max=ts2_f_stats.get("max", 0),
            ts2_f_mean=ts2_f_stats.get("mean", 0.0),
            ts2_f_std=ts2_f_stats.get("std", 0.0),
            merged_f_min=merged_f_stats.get("min", 0),
            merged_f_max=merged_f_stats.get("max", 0),
            merged_f_mean=merged_f_stats.get("mean", 0.0),
            merged_f_std=merged_f_stats.get("std", 0.0),
            branching_factor=after_data.get("search_signals", {}).get("branching_factor", 1.0),
            search_depth=after_data.get("search_signals", {}).get("search_depth", 0),
            solution_found=after_data.get("search_signals", {}).get("solution_found", False),
            solution_cost=after_data.get("search_signals", {}).get("solution_cost", 0),
        )

        return merge_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all merges."""
        if not self.merges:
            return {}

        quality_scores = [m.quality_score() for m in self.merges]
        compression_ratios = [m.shrinking_ratio for m in self.merges]
        branching_factors = [m.branching_factor for m in self.merges]

        return {
            'total_merges': len(self.merges),
            'avg_quality_score': float(np.mean(quality_scores)),
            'min_quality_score': float(np.min(quality_scores)),
            'max_quality_score': float(np.max(quality_scores)),
            'avg_compression_ratio': float(np.mean(compression_ratios)),
            'avg_branching_factor': float(np.mean(branching_factors)),
            'avg_reachability': float(np.mean([m.reachability_ratio for m in self.merges])),
            'solution_found_rate': sum(1 for m in self.merges if m.solution_found) / len(
                self.merges) if self.merges else 0,
        }


# ============================================================================
# PHASE 2: MERGE EXPLAINABILITY
# ============================================================================

@dataclass
class MergeExplanation:
    """Complete explanation for a merge decision."""
    iteration: int
    ts1_id: int
    ts2_id: int
    rationale: str
    ts1_properties: Dict[str, Any] = field(default_factory=dict)
    ts2_properties: Dict[str, Any] = field(default_factory=dict)
    relative_properties: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    success_indicators: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MergeExplainabilityAnalyzer:
    """Explains merge decisions in human-readable form."""

    def __init__(self, merges: List[MergeDecision]):
        self.merges = merges
        self.explanations: List[MergeExplanation] = []
        self._analyze_all_merges()

    def _analyze_all_merges(self) -> None:
        """Generate explanations for all merges."""
        for merge in self.merges:
            explanation = self._explain_merge(merge)
            self.explanations.append(explanation)

    def _explain_merge(self, merge: MergeDecision) -> MergeExplanation:
        """Generate explanation for a single merge."""
        ts1_props = {
            'size': merge.ts1_size,
            'transitions': merge.ts1_transitions,
            'density': merge.ts1_density,
            'goal_states': merge.ts1_goal_states,
            'variables': len(merge.ts1_variables),
            'f_value_quality': {
                'min': merge.ts1_f_min,
                'mean': merge.ts1_f_mean,
                'max': merge.ts1_f_max,
                'std': merge.ts1_f_std,
            }
        }

        ts2_props = {
            'size': merge.ts2_size,
            'transitions': merge.ts2_transitions,
            'density': merge.ts2_density,
            'goal_states': merge.ts2_goal_states,
            'variables': len(merge.ts2_variables),
            'f_value_quality': {
                'min': merge.ts2_f_min,
                'mean': merge.ts2_f_mean,
                'max': merge.ts2_f_max,
                'std': merge.ts2_f_std,
            }
        }

        relative_props = {
            'size_ratio': merge.ts2_size / max(merge.ts1_size, 1),
            'transition_ratio': merge.ts2_transitions / max(merge.ts1_transitions, 1),
            'density_similarity': 1.0 - abs(merge.ts1_density - merge.ts2_density),
        }

        outcome = {
            'merged_size': merge.merged_size,
            'expected_size': merge.expected_product_size,
            'compression_ratio': merge.shrinking_ratio,
            'reachability': merge.reachability_ratio,
            'branching_factor': merge.branching_factor,
            'solution_found': merge.solution_found,
        }

        # Generate rationale
        rationale = self._generate_rationale(merge, ts1_props, ts2_props, relative_props, outcome)

        # Identify success/risk indicators
        success, risks = self._identify_indicators(merge, outcome)

        return MergeExplanation(
            iteration=merge.iteration,
            ts1_id=merge.ts1_id,
            ts2_id=merge.ts2_id,
            rationale=rationale,
            ts1_properties=ts1_props,
            ts2_properties=ts2_props,
            relative_properties=relative_props,
            outcome=outcome,
            quality_score=merge.quality_score(),
            success_indicators=success,
            risk_indicators=risks,
        )

    def _generate_rationale(
            self,
            merge: MergeDecision,
            ts1_props: Dict,
            ts2_props: Dict,
            relative_props: Dict,
            outcome: Dict
    ) -> str:
        """Generate human-readable explanation."""
        points = []

        # Size analysis
        if merge.ts1_size > merge.ts2_size:
            points.append(
                f"Merging smaller TS{merge.ts2_id} ({merge.ts2_size} states) into larger TS{merge.ts1_id} ({merge.ts1_size} states)")
        elif merge.ts2_size > merge.ts1_size:
            points.append(
                f"Merging smaller TS{merge.ts1_id} ({merge.ts1_size} states) into larger TS{merge.ts2_id} ({merge.ts2_size} states)")
        else:
            points.append(f"Merging equally-sized transition systems ({merge.ts1_size} states each)")

        # Density analysis
        if merge.ts1_density > merge.ts2_density:
            points.append(
                f"TS{merge.ts1_id} is denser ({merge.ts1_density:.2f} edges/state) than TS{merge.ts2_id} ({merge.ts2_density:.2f})")
        else:
            points.append(
                f"TS{merge.ts2_id} is denser ({merge.ts2_density:.2f} edges/state) than TS{merge.ts1_id} ({merge.ts1_density:.2f})")

        # Outcome analysis
        if outcome['compression_ratio'] < 0.5:
            points.append(
                f"Successfully compressed to {outcome['compression_ratio'] * 100:.1f}% of theoretical maximum")
        elif outcome['compression_ratio'] < 1.0:
            points.append(
                f"Moderate compression achieved ({outcome['compression_ratio'] * 100:.1f}% of theoretical maximum)")
        else:
            points.append(
                f"WARNING: Merged size exceeds theoretical maximum")

        # Reachability
        if outcome['reachability'] > 0.9:
            points.append("Excellent reachability preserved")
        elif outcome['reachability'] > 0.7:
            points.append(f"Good reachability ({outcome['reachability'] * 100:.1f}%)")
        else:
            points.append(f"Warning: Low reachability ({outcome['reachability'] * 100:.1f}%)")

        # Search efficiency
        if outcome['branching_factor'] < 2.0:
            points.append(f"Excellent branching factor ({outcome['branching_factor']:.2f})")
        else:
            points.append(f"High branching factor ({outcome['branching_factor']:.2f})")

        return " | ".join(points)

    def _identify_indicators(self, merge: MergeDecision, outcome: Dict) -> Tuple[List[str], List[str]]:
        """Identify success and risk indicators."""
        success = []
        risks = []

        # Success indicators
        if merge.shrinking_ratio < 0.5:
            success.append("Excellent compression (< 50%)")
        if merge.reachability_ratio > 0.85:
            success.append("High reachability preserved (> 85%)")
        if merge.branching_factor < 1.5:
            success.append("Low branching factor (< 1.5)")
        if merge.solution_found:
            success.append("Solution found")
        if merge.quality_score() > 0.7:
            success.append("High quality merge")

        # Risk indicators
        if merge.shrinking_ratio > 1.0:
            risks.append("State explosion after merge")
        if merge.reachability_ratio < 0.5:
            risks.append("Low reachability (> 50% unreachable)")
        if merge.branching_factor > 5.0:
            risks.append("Very high branching factor")
        if merge.unreachable_states > merge.merged_size * 0.5:
            risks.append("Most states became unreachable")
        if merge.quality_score() < 0.3:
            risks.append("Low quality merge")

        return success, risks

    def get_best_merges(self, n: int = 5) -> List[MergeExplanation]:
        """Get top N best merges."""
        sorted_explanations = sorted(
            self.explanations,
            key=lambda e: e.quality_score,
            reverse=True
        )
        return sorted_explanations[:n]

    def get_worst_merges(self, n: int = 5) -> List[MergeExplanation]:
        """Get top N worst merges."""
        sorted_explanations = sorted(
            self.explanations,
            key=lambda e: e.quality_score
        )
        return sorted_explanations[:n]


# ============================================================================
# PHASE 3: MERGE CHOICE PATTERN DISCOVERY
# ============================================================================

class MergeChoiceAnalyzer:
    """Learns patterns from good and bad merge choices."""

    def __init__(self, merges: List[MergeDecision]):
        self.merges = merges

    def extract_features(self, merge: MergeDecision) -> Dict[str, float]:
        """Extract numerical features from a merge."""
        return {
            'ts1_size': float(merge.ts1_size),
            'ts2_size': float(merge.ts2_size),
            'size_ratio': float(merge.ts2_size / max(merge.ts1_size, 1)),
            'size_product': float(merge.ts1_size * merge.ts2_size),
            'ts1_density': float(merge.ts1_density),
            'ts2_density': float(merge.ts2_density),
            'density_diff': float(abs(merge.ts1_density - merge.ts2_density)),
            'ts1_reachable': float(merge.ts1_reachable_fraction),
            'ts2_reachable': float(merge.ts2_reachable_fraction),
            'ts1_goal_ratio': float(merge.ts1_goal_states / max(merge.ts1_size, 1)),
            'ts2_goal_ratio': float(merge.ts2_goal_states / max(merge.ts2_size, 1)),
            'ts1_f_mean': float(merge.ts1_f_mean),
            'ts2_f_mean': float(merge.ts2_f_mean),
            'f_stability_delta': float(abs(merge.ts1_f_mean - merge.ts2_f_mean)),
        }

    def analyze_good_vs_bad(self, threshold: float = 0.6) -> Dict[str, Any]:
        """Compare features of good merges vs bad merges."""
        good_merges = [m for m in self.merges if m.quality_score() >= threshold]
        bad_merges = [m for m in self.merges if m.quality_score() < threshold]

        if not good_merges or not bad_merges:
            logger.warning("Insufficient data for good/bad comparison")
            return {}

        good_features = [self.extract_features(m) for m in good_merges]
        bad_features = [self.extract_features(m) for m in bad_merges]

        analysis = {}
        all_keys = set(good_features[0].keys())

        for key in all_keys:
            good_values = np.array([f[key] for f in good_features])
            bad_values = np.array([f[key] for f in bad_features])

            analysis[key] = {
                'good': {
                    'mean': float(np.mean(good_values)),
                    'std': float(np.std(good_values)),
                    'min': float(np.min(good_values)),
                    'max': float(np.max(good_values)),
                },
                'bad': {
                    'mean': float(np.mean(bad_values)),
                    'std': float(np.std(bad_values)),
                    'min': float(np.min(bad_values)),
                    'max': float(np.max(bad_values)),
                },
                'delta': float(np.mean(good_values) - np.mean(bad_values)),
            }

        return analysis

    def find_patterns_in_good_merges(self) -> Dict[str, Any]:
        """Identify common patterns in high-quality merges."""
        good_merges = [m for m in self.merges if m.quality_score() >= 0.7]

        if not good_merges:
            return {}

        patterns = {
            'compression': [],
            'reachability': [],
            'branching': [],
            'f_stability': [],
        }

        # Compression pattern
        compressions = [m.shrinking_ratio for m in good_merges]
        patterns['compression'] = {
            'mean': float(np.mean(compressions)),
            'std': float(np.std(compressions)),
            'ideal_range': [float(np.percentile(compressions, 25)),
                            float(np.percentile(compressions, 75))],
        }

        # Reachability pattern
        reachabilities = [m.reachability_ratio for m in good_merges]
        patterns['reachability'] = {
            'mean': float(np.mean(reachabilities)),
            'std': float(np.std(reachabilities)),
            'ideal_range': [float(np.percentile(reachabilities, 25)),
                            float(np.percentile(reachabilities, 75))],
        }

        # Branching pattern
        branching = [m.branching_factor for m in good_merges]
        patterns['branching'] = {
            'mean': float(np.mean(branching)),
            'std': float(np.std(branching)),
            'ideal_range': [float(np.percentile(branching, 25)),
                            float(np.percentile(branching, 75))],
        }

        # F-stability pattern
        f_stability = [m.merged_f_std / max(m.merged_f_mean, 1) for m in good_merges]
        patterns['f_stability'] = {
            'mean': float(np.mean(f_stability)),
            'std': float(np.std(f_stability)),
            'ideal_range': [float(np.percentile(f_stability, 25)),
                            float(np.percentile(f_stability, 75))],
        }

        return patterns

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on learned patterns."""
        recommendations = []

        good_merges = [m for m in self.merges if m.quality_score() >= 0.7]
        bad_merges = [m for m in self.merges if m.quality_score() < 0.3]

        if not good_merges or not bad_merges:
            return recommendations

        # Recommendation 1: Size balance
        avg_good_ratio = np.mean([m.ts2_size / max(m.ts1_size, 1) for m in good_merges])
        avg_bad_ratio = np.mean([m.ts2_size / max(m.ts1_size, 1) for m in bad_merges])

        if avg_good_ratio < 0.3 or avg_good_ratio > 3.0:
            recommendations.append(
                f"Prefer merging similarly-sized transition systems (ratio ~1.0) rather than "
                f"highly imbalanced pairs (ratio {avg_good_ratio:.2f})"
            )

        # Recommendation 2: Density
        avg_good_density_diff = np.mean([abs(m.ts1_density - m.ts2_density) for m in good_merges])
        avg_bad_density_diff = np.mean([abs(m.ts1_density - m.ts2_density) for m in bad_merges])

        if avg_good_density_diff < avg_bad_density_diff:
            recommendations.append(
                f"Prefer merging TS with similar transition density "
                f"(good merges differ by {avg_good_density_diff:.3f}, bad by {avg_bad_density_diff:.3f})"
            )

        # Recommendation 3: Reachability
        avg_good_reachability = np.mean([m.reachability_ratio for m in good_merges])
        avg_bad_reachability = np.mean([m.reachability_ratio for m in bad_merges])

        if avg_good_reachability > avg_bad_reachability + 0.2:
            recommendations.append(
                f"Prioritize merges that preserve reachability "
                f"(good merges: {avg_good_reachability:.1%}, bad merges: {avg_bad_reachability:.1%})"
            )

        # Recommendation 4: Compression
        avg_good_compression = np.mean([m.shrinking_ratio for m in good_merges])
        avg_bad_compression = np.mean([m.shrinking_ratio for m in bad_merges])

        if avg_good_compression < 0.7:
            recommendations.append(
                f"Target merges with good compression ratios "
                f"(good: {avg_good_compression:.1%}, bad: {avg_bad_compression:.1%})"
            )

        return recommendations


# ============================================================================
# PHASE 4: GNN METADATA AGGREGATION
# ============================================================================

class GNNMetadataAggregator:
    """Aggregates GNN decision metadata from episodes."""

    def __init__(self, gnn_metadata_dir: str = "downward/gnn_metadata"):
        self.gnn_metadata_dir = Path(gnn_metadata_dir)
        self.episodes = []
        self.decisions = []
        self.problems_processed = set()

    def load_all_episodes(self) -> int:
        """Load all episode metadata files."""
        if not self.gnn_metadata_dir.exists():
            logger.warning(f"Metadata directory not found: {self.gnn_metadata_dir}")
            return 0

        logger.info(f"Loading episodes from: {self.gnn_metadata_dir}")

        count = 0
        for episode_file in sorted(self.gnn_metadata_dir.glob("episode_*.json")):
            try:
                with open(episode_file, 'r') as f:
                    episode_data = json.load(f)

                self.episodes.append(episode_data)

                # Flatten decisions
                if 'decisions' in episode_data:
                    for decision in episode_data['decisions']:
                        self.decisions.append({
                            **decision,
                            'episode_file': episode_file.name,
                        })

                    self.problems_processed.add(episode_data.get('problem', 'unknown'))

                count += 1

            except Exception as e:
                logger.warning(f"Failed to load {episode_file}: {e}")

        logger.info(f"✓ Loaded {count} episodes with {len(self.decisions)} total decisions")
        return count

    def compute_decision_statistics(self) -> Dict[str, Any]:
        """Compute statistics on GNN decisions."""
        if not self.decisions:
            logger.warning("No decisions loaded")
            return {}

        logger.info("\nComputing decision statistics...")

        rewards = [d.get('reward_received', 0) for d in self.decisions]
        deltas = [d.get('merge_info', {}).get('delta_states', 0) for d in self.decisions]
        expansions = [d.get('merge_info', {}).get('num_expansions', 0) for d in self.decisions]

        stats = {
            'total_decisions': len(self.decisions),
            'problems_processed': len(self.problems_processed),
            'problems': sorted(list(self.problems_processed)),

            'reward_statistics': {
                'mean': float(np.mean(rewards)) if rewards else 0,
                'median': float(np.median(rewards)) if rewards else 0,
                'std': float(np.std(rewards)) if rewards else 0,
                'min': float(np.min(rewards)) if rewards else 0,
                'max': float(np.max(rewards)) if rewards else 0,
                'q1': float(np.percentile(rewards, 25)) if rewards else 0,
                'q3': float(np.percentile(rewards, 75)) if rewards else 0,
            },

            'state_dynamics': {
                'mean_delta_states': float(np.mean(deltas)) if deltas else 0,
                'mean_expansions': float(np.mean(expansions)) if expansions else 0,
                'problems_with_expansion': sum(1 for d in deltas if d > 0),
                'problems_with_reduction': sum(1 for d in deltas if d < 0),
            },
        }

        return stats

    def identify_patterns(self) -> Dict[str, Any]:
        """Identify patterns in GNN decisions."""
        logger.info("\nIdentifying decision patterns...")

        high_reward_decisions = [d for d in self.decisions if d.get('reward_received', 0) > 0.1]
        low_reward_decisions = [d for d in self.decisions if d.get('reward_received', 0) < -0.5]

        patterns = {
            'high_reward_decisions': {
                'count': len(high_reward_decisions),
                'avg_delta_states': float(np.mean([
                    d.get('merge_info', {}).get('delta_states', 0)
                    for d in high_reward_decisions
                ])) if high_reward_decisions else 0,
            },

            'low_reward_decisions': {
                'count': len(low_reward_decisions),
                'avg_delta_states': float(np.mean([
                    d.get('merge_info', {}).get('delta_states', 0)
                    for d in low_reward_decisions
                ])) if low_reward_decisions else 0,
            },
        }

        return patterns


# ============================================================================
# UNIFIED SYSTEM
# ============================================================================

class UnifiedMergeAnalyzer:
    """Central unified merge analysis system for post-training evaluation."""

    def __init__(
            self,
            experiment_output_dir: str = "experiment_results/",
            fd_output_dir: str = "downward/fd_output",
            gnn_metadata_dir: str = "downward/gnn_metadata",
    ):
        self.experiment_output_dir = Path(experiment_output_dir)
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all sub-analyzers
        self.merge_collector = MergeMetadataCollector(fd_output_dir)
        self.gnn_aggregator = GNNMetadataAggregator(gnn_metadata_dir)

        self.merge_analyzer = None
        self.explainability_analyzer = None
        self.pattern_analyzer = None

        self.results = {}

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.

        Returns:
            Complete analysis results
        """
        logger.info("\n" + "=" * 90)
        logger.info("UNIFIED MERGE ANALYSIS SYSTEM - COMPLETE ANALYSIS")
        logger.info("=" * 90 + "\n")

        # ====================================================================
        # PHASE 1: Load and validate data
        # ====================================================================
        logger.info("[PHASE 1] Loading merge metadata...")
        num_merges = self.merge_collector.load_all_metadata()

        if num_merges == 0:
            logger.warning("No merge metadata found!")
            return {'error': 'No merge metadata'}

        logger.info(f"✓ Loaded {num_merges} merge decisions")

        # ====================================================================
        # PHASE 2: Explainability analysis
        # ====================================================================
        logger.info("\n[PHASE 2] Generating merge explanations...")
        self.explainability_analyzer = MergeExplainabilityAnalyzer(
            self.merge_collector.merges
        )

        best_merges = self.explainability_analyzer.get_best_merges(5)
        worst_merges = self.explainability_analyzer.get_worst_merges(5)

        logger.info(f"✓ Explained {len(self.explainability_analyzer.explanations)} merges")
        logger.info(f"  Best quality merge: {best_merges[0].quality_score:.3f}")
        logger.info(f"  Worst quality merge: {worst_merges[0].quality_score:.3f}")

        # ====================================================================
        # PHASE 3: Choice pattern discovery
        # ====================================================================
        logger.info("\n[PHASE 3] Discovering merge choice patterns...")
        self.pattern_analyzer = MergeChoiceAnalyzer(self.merge_collector.merges)

        good_vs_bad = self.pattern_analyzer.analyze_good_vs_bad()
        patterns = self.pattern_analyzer.find_patterns_in_good_merges()
        recommendations = self.pattern_analyzer.generate_recommendations()

        logger.info(f"✓ Analyzed {len(good_vs_bad)} features")
        logger.info(f"✓ Identified patterns in high-quality merges")
        logger.info(f"✓ Generated {len(recommendations)} recommendations")

        for rec in recommendations:
            logger.info(f"  • {rec}")

        # ====================================================================
        # PHASE 4: GNN metadata aggregation
        # ====================================================================
        logger.info("\n[PHASE 4] Aggregating GNN metadata...")
        num_episodes = self.gnn_aggregator.load_all_episodes()

        gnn_stats = {}
        gnn_patterns = {}
        if num_episodes > 0:
            gnn_stats = self.gnn_aggregator.compute_decision_statistics()
            gnn_patterns = self.gnn_aggregator.identify_patterns()
            logger.info(f"✓ Loaded {num_episodes} episodes")
            logger.info(f"  Avg reward: {gnn_stats.get('reward_statistics', {}).get('mean', 0):.4f}")

        # ====================================================================
        # PHASE 5: Compile results
        # ====================================================================
        logger.info("\n[PHASE 5] Compiling results...")

        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'experiment_output_dir': str(self.experiment_output_dir),
                'analysis_version': '1.0',
            },

            'merge_statistics': self.merge_collector.get_statistics(),

            'best_merges': [e.to_dict() for e in best_merges],
            'worst_merges': [e.to_dict() for e in worst_merges],

            'patterns': {
                'good_vs_bad_analysis': good_vs_bad,
                'good_merge_patterns': patterns,
                'recommendations': recommendations,
            },

            'gnn_analysis': {
                'statistics': gnn_stats,
                'patterns': gnn_patterns,
                'episodes_loaded': num_episodes,
            },

            'summary': {
                'total_merges_analyzed': num_merges,
                'avg_merge_quality': float(np.mean([m.quality_score() for m in self.merge_collector.merges])),
                'best_merge_quality': float(max([m.quality_score() for m in self.merge_collector.merges])),
                'worst_merge_quality': float(min([m.quality_score() for m in self.merge_collector.merges])),
                'gnn_episodes_analyzed': num_episodes,
            }
        }

        return self.results

    def export_results(self, output_dir: Optional[str] = None) -> None:
        """Export analysis results to files."""
        if output_dir is None:
            output_dir = str(self.experiment_output_dir / "merge_analysis_report")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = output_dir / "unified_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"✓ Saved JSON report: {json_path}")

        # Text report
        text_path = output_dir / "merge_analysis_report.txt"
        with open(text_path, 'w') as f:
            self._write_text_report(f)
        logger.info(f"✓ Saved text report: {text_path}")

        logger.info(f"\n✅ Analysis complete! Results saved to: {output_dir}/")

    def _write_text_report(self, f) -> None:
        """Write human-readable text report."""
        f.write("=" * 100 + "\n")
        f.write("UNIFIED MERGE & SHRINK ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        summary = self.results.get('summary', {})
        f.write("SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total merges analyzed: {summary.get('total_merges_analyzed', 0)}\n")
        f.write(f"Average merge quality: {summary.get('avg_merge_quality', 0):.3f}\n")
        f.write(f"Best merge quality: {summary.get('best_merge_quality', 0):.3f}\n")
        f.write(f"Worst merge quality: {summary.get('worst_merge_quality', 0):.3f}\n")
        f.write(f"GNN episodes analyzed: {summary.get('gnn_episodes_analyzed', 0)}\n\n")

        # Best merges
        f.write("=" * 100 + "\n")
        f.write("TOP 5 BEST MERGES\n")
        f.write("=" * 100 + "\n\n")

        for i, merge_dict in enumerate(self.results.get('best_merges', [])[:5], 1):
            f.write(f"[{i}] Iteration {merge_dict.get('iteration')}: "
                    f"TS{merge_dict.get('ts1_id')} + TS{merge_dict.get('ts2_id')}\n")
            f.write(f"    Quality: {merge_dict.get('quality_score', 0):.3f}\n")
            f.write(f"    {merge_dict.get('rationale', '')}\n\n")

        # Worst merges
        f.write("=" * 100 + "\n")
        f.write("TOP 5 WORST MERGES\n")
        f.write("=" * 100 + "\n\n")

        for i, merge_dict in enumerate(self.results.get('worst_merges', [])[:5], 1):
            f.write(f"[{i}] Iteration {merge_dict.get('iteration')}: "
                    f"TS{merge_dict.get('ts1_id')} + TS{merge_dict.get('ts2_id')}\n")
            f.write(f"    Quality: {merge_dict.get('quality_score', 0):.3f}\n")
            f.write(f"    {merge_dict.get('rationale', '')}\n\n")

        # Recommendations
        f.write("=" * 100 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 100 + "\n\n")

        for i, rec in enumerate(self.results.get('patterns', {}).get('recommendations', []), 1):
            f.write(f"[{i}] {rec}\n\n")


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# ============================================================================

# These allow existing code to continue working without modification

def run_merge_analysis(experiment_output_dir: str = "experiment_results/") -> bool:
    """Backward compatible entry point."""
    try:
        analyzer = UnifiedMergeAnalyzer(experiment_output_dir)
        analyzer.run_complete_analysis()
        analyzer.export_results()
        return True
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(message)s'
    )

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "experiment_results/"
    success = run_merge_analysis(output_dir)
    sys.exit(0 if success else 1)