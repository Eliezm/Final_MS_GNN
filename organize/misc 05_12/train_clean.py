#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLEAN TRAINING SCRIPT - Thin Client Architecture
=================================================
Simple training loop using ThinMergeEnv.
"""

import sys
import os
import logging
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import time

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "train_clean.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


def load_benchmarks(benchmark_dir: str, difficulty: str = "small") -> List[Tuple[str, str]]:
    """Load benchmark problems from directory."""
    difficulty_dir = Path(benchmark_dir) / difficulty

    if not difficulty_dir.exists():
        logger.error(f"Benchmark directory not found: {difficulty_dir}")
        return []

    domain_file = difficulty_dir / "domain.pddl"
    if not domain_file.exists():
        logger.error(f"Domain file not found: {domain_file}")
        return []

    problems = sorted(glob.glob(str(difficulty_dir / "problem_*.pddl")))

    if not problems:
        logger.error(f"No problem files found in {difficulty_dir}")
        return []

    benchmarks = [
        (str(domain_file), str(prob))
        for prob in problems
    ]

    logger.info(f"Loaded {len(benchmarks)} {difficulty} problems")
    return benchmarks


def train_on_benchmarks(
        benchmarks: List[Tuple[str, str]],
        total_timesteps: int = 10000,
        timesteps_per_problem: int = 500,
        model_save_path: str = "models/gnn_policy.zip",
        tb_log_dir: str = "tb_logs/",
) -> Optional[str]:
    """Train GNN policy on multiple problems."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from src.environments.thin_merge_env import ThinMergeEnv
        from gnn_policy import GNNPolicy

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return None

    logger.info("=" * 80)
    logger.info("TRAINING WITH THIN CLIENT ARCHITECTURE")
    logger.info("=" * 80)
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Timesteps per problem: {timesteps_per_problem}")
    logger.info(f"Number of problems: {len(benchmarks)}")
    logger.info("")

    model = None
    total_steps = 0
    problems_trained = 0
    problem_idx = 0

    while total_steps < total_timesteps and problems_trained < len(benchmarks) * 3:
        domain_file, problem_file = benchmarks[problem_idx % len(benchmarks)]
        problem_name = os.path.basename(problem_file)

        logger.info(f"\n--- Problem {problems_trained + 1}: {problem_name} ---")

        try:
            # Create environment
            env = ThinMergeEnv(
                domain_file=domain_file,
                problem_file=problem_file,
                max_merges=50,
                timeout_per_step=120.0,
            )
            env = Monitor(env)

            # Create or update model
            if model is None:
                logger.info("Creating new PPO model...")
                model = PPO(
                    policy=GNNPolicy,
                    env=env,
                    learning_rate=3e-4,
                    n_steps=64,
                    batch_size=32,
                    ent_coef=0.01,
                    verbose=0,
                    tensorboard_log=tb_log_dir,
                    policy_kwargs={"hidden_dim": 64},
                )
            else:
                model.set_env(env)

            # Train
            steps_this_problem = min(timesteps_per_problem, total_timesteps - total_steps)

            logger.info(f"Training for {steps_this_problem} timesteps...")
            start_time = time.time()

            model.learn(
                total_timesteps=steps_this_problem,
                tb_log_name=f"thin_client_{problems_trained}",
                reset_num_timesteps=False,
            )

            elapsed = time.time() - start_time
            total_steps += steps_this_problem
            problems_trained += 1

            logger.info(f"✓ Completed in {elapsed:.1f}s (total: {total_steps}/{total_timesteps})")

            env.close()

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            break

        except Exception as e:
            logger.error(f"Problem failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

        problem_idx += 1

    # Save model
    if model is not None:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_save_path)
        logger.info(f"\n✅ Model saved: {model_save_path}")
        logger.info(f"   Total steps: {total_steps}")
        logger.info(f"   Problems trained: {problems_trained}")
        return model_save_path

    return None


def main():
    parser = argparse.ArgumentParser(description="Train GNN policy with Thin Client")
    parser.add_argument("--benchmark-dir", default="benchmarks", help="Benchmark directory")
    parser.add_argument("--difficulty", default="small", choices=["small", "medium", "large", "all"])
    parser.add_argument("--total-timesteps", type=int, default=10)
    parser.add_argument("--timesteps-per-problem", type=int, default=10)
    parser.add_argument("--output", default="models/gnn_thin_client.zip")
    parser.add_argument("--tb-log", default="tb_logs/")

    args = parser.parse_args()

    # Load benchmarks
    if args.difficulty == "all":
        benchmarks = []
        for diff in ["small", "medium", "large"]:
            benchmarks.extend(load_benchmarks(args.benchmark_dir, diff))
    else:
        benchmarks = load_benchmarks(args.benchmark_dir, args.difficulty)

    if not benchmarks:
        logger.error("No benchmarks loaded!")
        return 1

    # Train
    result = train_on_benchmarks(
        benchmarks=benchmarks,
        total_timesteps=args.total_timesteps,
        timesteps_per_problem=args.timesteps_per_problem,
        model_save_path=args.output,
        tb_log_dir=args.tb_log,
    )

    if result:
        logger.info("\n✅ Training complete!")
        return 0
    else:
        logger.error("\n❌ Training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())