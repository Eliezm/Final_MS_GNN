#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOG ANALYSIS UTILITIES - Extract plotting data from training logs
================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

from experiments.core.logging import (
    EpisodeMetrics,
    load_training_log,
    load_training_events,
    TrainingSummaryStats,
)


def load_experiment_data(experiment_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load all training data from an experiment directory.

    Returns:
        Dict with:
        - 'episode_metrics': List[EpisodeMetrics]
        - 'events': List[Dict]
        - 'summary': TrainingSummaryStats
        - 'dataframes': Dict[str, pd.DataFrame]
    """
    experiment_dir = Path(experiment_dir)

    result = {
        'episode_metrics': [],
        'events': [],
        'summary': None,
        'dataframes': {},
    }

    # Load episode metrics
    for log_path in [
        experiment_dir / "training" / "training_log.jsonl",
        experiment_dir / "training_log.jsonl",
    ]:
        if log_path.exists():
            result['episode_metrics'] = load_training_log(log_path)
            break

    # Load events
    for events_path in [
        experiment_dir / "logs" / "training_events.jsonl",
        experiment_dir / "training_events.jsonl",
    ]:
        if events_path.exists():
            result['events'] = load_training_events(events_path)
            break

    # Compute summary
    if result['episode_metrics']:
        result['summary'] = TrainingSummaryStats.from_episode_log(result['episode_metrics'])

    # Create dataframes for plotting
    result['dataframes'] = create_plotting_dataframes(result['episode_metrics'])

    return result


def create_plotting_dataframes(
        episode_metrics: List[EpisodeMetrics]
) -> Dict[str, pd.DataFrame]:
    """
    Create pandas DataFrames ready for plotting.

    Returns:
        Dict with:
        - 'episodes': Main episode-level DataFrame
        - 'components': Reward component DataFrame
        - 'per_problem': Per-problem aggregated DataFrame
        - 'rolling': Rolling averages DataFrame
    """
    if not episode_metrics:
        return {}

    dataframes = {}

    # Main episode DataFrame
    episode_data = []
    for m in episode_metrics:
        episode_data.append({
            'episode': m.episode,
            'problem_name': m.problem_name,
            'reward': m.reward,
            'h_star_preservation': m.h_star_preservation,
            'is_solvable': m.is_solvable,
            'eval_steps': m.eval_steps,
            'error': m.error is not None,
            'step_time_ms': m.step_time_ms,
            'peak_memory_mb': m.peak_memory_mb,
            # Component rewards
            'component_h': m.component_h_preservation,
            'component_trans': m.component_transition_control,
            'component_opp': m.component_operator_projection,
            'component_label': m.component_label_combinability,
            'component_bonus': m.component_bonus_signals,
            # Detailed signals
            'transition_growth': m.transition_growth_ratio,
            'opp_score': m.opp_score,
            'label_comb': m.label_combinability_score,
            'reachability': m.reachability_ratio,
            'dead_end_ratio': m.dead_end_ratio,
            # Penalties
            'penalty_solvability': m.penalty_solvability_loss,
            'penalty_dead_end': m.penalty_dead_end,
        })

    dataframes['episodes'] = pd.DataFrame(episode_data)

    # Component breakdown DataFrame (for component trajectory plots)
    component_data = []
    for m in episode_metrics:
        if m.error is None:
            component_data.append({
                'episode': m.episode,
                'H* Preservation': m.component_h_preservation,
                'Transition Control': m.component_transition_control,
                'Operator Projection': m.component_operator_projection,
                'Label Combinability': m.component_label_combinability,
                'Bonus Signals': m.component_bonus_signals,
            })

    if component_data:
        dataframes['components'] = pd.DataFrame(component_data)

    # Per-problem aggregated DataFrame
    if dataframes.get('episodes') is not None and len(dataframes['episodes']) > 0:
        successful = dataframes['episodes'][~dataframes['episodes']['error']]
        if len(successful) > 0:
            per_problem = successful.groupby('problem_name').agg({
                'reward': ['mean', 'std', 'count'],
                'h_star_preservation': ['mean', 'std'],
                'component_h': 'mean',
                'component_trans': 'mean',
                'transition_growth': 'mean',
            }).reset_index()
            per_problem.columns = ['_'.join(col).strip('_') for col in per_problem.columns]
            dataframes['per_problem'] = per_problem

    # Rolling averages DataFrame
    if dataframes.get('episodes') is not None and len(dataframes['episodes']) > 0:
        df = dataframes['episodes']
        window = min(50, len(df) // 10 + 1)

        rolling_data = {
            'episode': df['episode'].values,
            'reward_rolling': df['reward'].rolling(window=window, min_periods=1).mean().values,
            'h_rolling': df['h_star_preservation'].rolling(window=window, min_periods=1).mean().values,
        }
        dataframes['rolling'] = pd.DataFrame(rolling_data)

    return dataframes


def get_learning_curve_data(
        episode_metrics: List[EpisodeMetrics],
        window_size: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Extract learning curve data for plotting.

    Returns:
        Dict with arrays for:
        - episodes: Episode numbers
        - rewards: Raw rewards
        - rewards_rolling: Rolling average rewards
        - h_preservation: H* preservation values
        - h_preservation_rolling: Rolling average H*
    """
    if not episode_metrics:
        return {}

    successful = [m for m in episode_metrics if m.error is None]

    episodes = np.array([m.episode for m in successful])
    rewards = np.array([m.reward for m in successful])
    h_pres = np.array([m.h_star_preservation for m in successful])

    # Compute rolling averages
    def rolling_mean(arr, window):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window) / window, mode='valid')

    rewards_rolling = rolling_mean(rewards, window_size)
    h_rolling = rolling_mean(h_pres, window_size)

    # Align x-axis for rolling
    episodes_rolling = episodes[window_size - 1:] if len(episodes) >= window_size else episodes

    return {
        'episodes': episodes,
        'rewards': rewards,
        'rewards_rolling': rewards_rolling,
        'episodes_rolling': episodes_rolling,
        'h_preservation': h_pres,
        'h_preservation_rolling': h_rolling,
    }


def get_component_trajectory_data(
        episode_metrics: List[EpisodeMetrics],
) -> Dict[str, np.ndarray]:
    """
    Extract reward component trajectories for plotting.

    Returns:
        Dict with arrays for each component over episodes.
    """
    successful = [m for m in episode_metrics if m.error is None]

    if not successful:
        return {}

    return {
        'episodes': np.array([m.episode for m in successful]),
        'h_preservation': np.array([m.component_h_preservation for m in successful]),
        'transition_control': np.array([m.component_transition_control for m in successful]),
        'operator_projection': np.array([m.component_operator_projection for m in successful]),
        'label_combinability': np.array([m.component_label_combinability for m in successful]),
        'bonus_signals': np.array([m.component_bonus_signals for m in successful]),
        'total_reward': np.array([m.reward for m in successful]),
    }


def get_per_problem_comparison_data(
        episode_metrics: List[EpisodeMetrics],
) -> pd.DataFrame:
    """
    Get per-problem comparison data for bar charts.

    Returns:
        DataFrame with per-problem statistics.
    """
    if not episode_metrics:
        return pd.DataFrame()

    per_problem = defaultdict(lambda: {
        'rewards': [],
        'h_preservations': [],
        'failures': 0,
        'total': 0,
    })

    for m in episode_metrics:
        per_problem[m.problem_name]['total'] += 1
        if m.error is not None:
            per_problem[m.problem_name]['failures'] += 1
        else:
            per_problem[m.problem_name]['rewards'].append(m.reward)
            per_problem[m.problem_name]['h_preservations'].append(m.h_star_preservation)

    rows = []
    for problem_name, data in per_problem.items():
        rewards = data['rewards']
        h_pres = data['h_preservations']

        rows.append({
            'problem_name': problem_name,
            'num_episodes': data['total'],
            'num_failures': data['failures'],
            'failure_rate': data['failures'] / max(1, data['total']),
            'mean_reward': np.mean(rewards) if rewards else 0.0,
            'std_reward': np.std(rewards) if rewards else 0.0,
            'mean_h_preservation': np.mean(h_pres) if h_pres else 1.0,
            'std_h_preservation': np.std(h_pres) if h_pres else 0.0,
        })

    return pd.DataFrame(rows)


def get_signal_correlation_data(
        episode_metrics: List[EpisodeMetrics],
) -> pd.DataFrame:
    """
    Get data for signal correlation heatmap.

    Returns:
        DataFrame with signal values per episode.
    """
    successful = [m for m in episode_metrics if m.error is None]

    if not successful:
        return pd.DataFrame()

    data = []
    for m in successful:
        data.append({
            'reward': m.reward,
            'h_star_preservation': m.h_star_preservation,
            'transition_growth': m.transition_growth_ratio,
            'opp_score': m.opp_score,
            'label_combinability': m.label_combinability_score,
            'reachability': m.reachability_ratio,
            'component_h': m.component_h_preservation,
            'component_trans': m.component_transition_control,
        })

    return pd.DataFrame(data)


def export_for_plotting(
        episode_metrics: List[EpisodeMetrics],
        output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """
    Export all plotting data as CSV files.

    Returns:
        Dict mapping data type to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # Main episodes CSV
    dfs = create_plotting_dataframes(episode_metrics)

    if 'episodes' in dfs:
        path = output_dir / "episodes.csv"
        dfs['episodes'].to_csv(path, index=False)
        exported['episodes'] = path

    if 'components' in dfs:
        path = output_dir / "components.csv"
        dfs['components'].to_csv(path, index=False)
        exported['components'] = path

    if 'per_problem' in dfs:
        path = output_dir / "per_problem.csv"
        dfs['per_problem'].to_csv(path, index=False)
        exported['per_problem'] = path

    if 'rolling' in dfs:
        path = output_dir / "rolling.csv"
        dfs['rolling'].to_csv(path, index=False)
        exported['rolling'] = path

    # Correlation data
    corr_df = get_signal_correlation_data(episode_metrics)
    if len(corr_df) > 0:
        path = output_dir / "correlation_data.csv"
        corr_df.to_csv(path, index=False)
        exported['correlation'] = path

    # Learning curve data as JSON (for flexibility)
    lc_data = get_learning_curve_data(episode_metrics)
    if lc_data:
        path = output_dir / "learning_curves.json"
        # Convert numpy to lists for JSON
        json_data = {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in lc_data.items()}
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2)
        exported['learning_curves'] = path

    return exported