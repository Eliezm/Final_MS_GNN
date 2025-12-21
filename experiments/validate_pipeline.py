#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIPELINE VALIDATION SCRIPT
==========================
Tests each component of the experiment pipeline independently.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test all imports work."""
    print("\n[1/8] Testing imports...")

    try:
        from experiments.configs.experiment_configs import get_experiment, list_experiments
        from experiments.runners.experiment_runner import ExperimentRunner, CurriculumExperimentRunner
        from experiments.core.training import GNNTrainer, set_all_seeds
        from experiments.core.analysis import analyze_training_results
        from experiments.core.visualization import generate_all_plots
        from experiments.core.evaluation import EvaluationFramework
        from experiments.core.gnn_random_evaluation import GNNRandomEvaluationFramework
        from src.environments.thin_merge_env import ThinMergeEnv
        from src.models.gnn_policy import GNNPolicy
        from src.rewards.reward_function_enhanced import EnhancedRewardFunction
        print("   ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_configs():
    """Test experiment configurations."""
    print("\n[2/8] Testing experiment configurations...")

    try:
        from experiments.configs.experiment_configs import (
            get_experiment, list_experiments, get_paper_experiments
        )

        experiments = list_experiments()
        print(f"   Available experiments: {len(experiments)}")

        paper_exps = get_paper_experiments()
        print(f"   Paper experiments: {paper_exps}")

        for exp_name in paper_exps:
            config = get_experiment(exp_name)
            print(f"   ✓ {exp_name}: {config.num_train_episodes} episodes")

        print("   ✅ All configs valid")
        return True
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False


def test_benchmarks():
    """Test benchmark files exist."""
    print("\n[3/8] Testing benchmark files...")

    import glob
    benchmark_dir = PROJECT_ROOT / "benchmarks"

    required = [
        "blocksworld/small/domain.pddl",
        "blocksworld/small/problem_small_*.pddl",
        "blocksworld/medium/domain.pddl",
        "blocksworld/medium/problem_medium_*.pddl",
        "blocksworld/large/domain.pddl",
        "blocksworld/large/problem_large_*.pddl",
    ]

    all_found = True
    for pattern in required:
        full_pattern = str(benchmark_dir / pattern)
        matches = glob.glob(full_pattern)
        if matches:
            print(f"   ✓ {pattern}: {len(matches)} files")
        else:
            print(f"   ❌ {pattern}: NOT FOUND")
            all_found = False

    if all_found:
        print("   ✅ All benchmarks found")
    return all_found


def test_fast_downward():
    """Test Fast Downward installation."""
    print("\n[4/8] Testing Fast Downward...")

    fd_dir = PROJECT_ROOT / "downward"

    checks = [
        fd_dir / "builds" / "release" / "bin" / "downward",
        fd_dir / "builds" / "release" / "bin" / "downward.exe",
        fd_dir / "builds" / "release" / "bin" / "translate" / "translate.py",
    ]

    found = False
    for path in checks[:2]:
        if path.exists():
            print(f"   ✓ FD binary: {path}")
            found = True
            break

    if not found:
        print(f"   ❌ FD binary not found")
        return False

    if checks[2].exists():
        print(f"   ✓ Translator: {checks[2]}")
    else:
        print(f"   ❌ Translator not found")
        return False

    print("   ✅ Fast Downward ready")
    return True


def test_environment():
    """Test ThinMergeEnv creation."""
    print("\n[5/8] Testing environment...")

    try:
        from src.environments.thin_merge_env import ThinMergeEnv
        import glob

        problems = glob.glob(str(PROJECT_ROOT / "benchmarks/blocksworld/small/problem_small_*.pddl"))
        if not problems:
            print("   ❌ No test problems found")
            return False

        domain = str(PROJECT_ROOT / "benchmarks/blocksworld/small/domain.pddl")
        problem = problems[0]

        print(f"   Testing with: {Path(problem).name}")

        env = ThinMergeEnv(
            domain_file=domain,
            problem_file=problem,
            max_merges=5,
            timeout_per_step=30.0,
        )

        obs, info = env.reset()
        print(f"   ✓ Reset successful: {obs['num_nodes']} nodes, {obs['num_edges']} edges")

        if obs['num_edges'] > 0:
            action = 0
            obs2, reward, done, truncated, info2 = env.step(action)
            print(f"   ✓ Step successful: reward={reward:.4f}")

        env.close()
        print("   ✅ Environment works")
        return True

    except Exception as e:
        print(f"   ❌ Environment error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test GNN model creation."""
    print("\n[6/8] Testing GNN model...")

    try:
        import torch
        from src.models.gnn_model import GNNModel
        from src.models.gnn_policy import GNNPolicy

        model = GNNModel(input_dim=9, hidden_dim=64, edge_feature_dim=11)
        print(f"   ✓ GNNModel created")

        x = torch.randn(10, 9)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_features = torch.randn(3, 11)

        logits, embs = model(x, edge_index, edge_features)
        print(f"   ✓ Forward pass: logits={logits.shape}, embs={embs.shape}")

        print("   ✅ Model works")
        return True

    except Exception as e:
        print(f"   ❌ Model error: {e}")
        return False


def test_reward_function():
    """Test reward function."""
    print("\n[7/8] Testing reward function...")

    try:
        from src.rewards.reward_function_enhanced import EnhancedRewardFunction

        reward_fn = EnhancedRewardFunction(debug=False)

        test_obs = {
            'reward_signals': {
                'h_star_before': 10,
                'h_star_after': 10,
                'h_star_preservation': 1.0,
                'states_before': 5,
                'states_after': 8,
                'is_solvable': True,
                'dead_end_ratio': 0.1,
                'shrinkability': 0.3,
            }
        }

        reward = reward_fn.compute_reward(test_obs)
        print(f"   ✓ Reward computed: {reward:.4f}")

        breakdown = reward_fn.compute_reward_with_breakdown(test_obs)
        print(f"   ✓ Components: {list(breakdown['components'].keys())}")

        print("   ✅ Reward function works")
        return True

    except Exception as e:
        print(f"   ❌ Reward error: {e}")
        return False


def test_analysis():
    """Test analysis functions."""
    print("\n[8/8] Testing analysis...")

    try:
        from experiments.core.analysis import (
            analyze_training_results,
            analyze_component_trajectories,
            analyze_feature_reward_correlation,
        )
        from experiments.core.logging import EpisodeMetrics

        # Create dummy metrics
        dummy_log = [
            EpisodeMetrics(
                episode=i,
                problem_name=f"problem_{i % 3}",
                reward=0.5 + i * 0.01,
                h_star_preservation=0.95,
                is_solvable=True,
            )
            for i in range(10)
        ]

        # Test main analysis
        summary = analyze_training_results(
            dummy_log, [], ["p1", "p2", "p3"],
            [("d", "p1"), ("d", "p2"), ("d", "p3")],
            "test_exp"
        )
        print(f"   ✓ Main analysis: avg_reward={summary.avg_reward_over_all:.4f}")

        # Test component analysis
        comp_analysis = analyze_component_trajectories(dummy_log, None)
        print(f"   ✓ Component analysis: {len(comp_analysis)} keys")

        print("   ✅ Analysis works")
        return True

    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return False


def main():
    print("=" * 80)
    print("PIPELINE VALIDATION")
    print("=" * 80)

    results = {
        "imports": test_imports(),
        "configs": test_configs(),
        "benchmarks": test_benchmarks(),
        "fast_downward": test_fast_downward(),
        "environment": test_environment(),
        "model": test_model(),
        "reward": test_reward_function(),
        "analysis": test_analysis(),
    }

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for name, success in results.items():
        icon = "✅" if success else "❌"
        print(f"   {icon} {name}")

    print(f"\n   Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Pipeline ready!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Fix issues before running experiments")
        return 1


if __name__ == "__main__":
    sys.exit(main())