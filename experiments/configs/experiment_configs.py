#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENT CONFIGURATIONS - FINAL VERSION
==========================================
Defines the exact 3 experiments for the paper:
1. Train on MEDIUM → Test S/M(seen)/M(unseen)/L
2. Train on LARGE → Test S/M/L(seen)/L(unseen)
3. Curriculum S→M→L → Test S/M/L + Logistics transfer
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from enum import Enum


class Domain(Enum):
    """Planning domains."""
    BLOCKSWORLD = "blocksworld"
    LOGISTICS = "logistics"


class ProblemSize(Enum):
    """Problem sizes."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class CurriculumPhase:
    """Single phase in curriculum learning."""
    name: str
    problem_size: ProblemSize
    problem_pattern: str
    num_problems: int
    num_episodes: int

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'problem_size': self.problem_size.value,
            'problem_pattern': self.problem_pattern,
            'num_problems': self.num_problems,
            'num_episodes': self.num_episodes,
        }


@dataclass
class TestConfig:
    """Configuration for a single test set."""
    name: str
    domain: Domain
    problem_size: ProblemSize
    problem_pattern: str
    num_problems: int
    num_runs_per_problem: int = 5
    description: str = ""
    is_seen: bool = False  # Whether these problems were in training

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'domain': self.domain.value,
            'problem_size': self.problem_size.value,
            'problem_pattern': self.problem_pattern,
            'num_problems': self.num_problems,
            'num_runs_per_problem': self.num_runs_per_problem,
            'description': self.description,
            'is_seen': self.is_seen,
        }


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    domain: Domain

    # Training configuration
    train_problem_size: ProblemSize
    train_problem_pattern: str
    train_num_problems: int

    # Test configurations
    test_configurations: List['TestConfig']

    # Training hyperparameters
    num_train_episodes: int = 100
    timesteps_per_episode: int = 128  # Was 50, now >= n_steps(64)
    max_merges: int = 128  # Increase max_merges accordingly
    timeout_per_step: float = 120.0
    checkpoint_interval: int = 500
    eval_runs_per_problem: int = 3
    seed: int = 42
    min_episodes_per_problem: int = 10

    is_curriculum: bool = False
    curriculum_phases: List['CurriculumPhase'] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['domain'] = self.domain.value
        d['train_problem_size'] = self.train_problem_size.value
        d['test_configurations'] = [tc.to_dict() for tc in self.test_configurations]
        if self.curriculum_phases:
            d['curriculum_phases'] = [cp.to_dict() for cp in self.curriculum_phases]
        return d


# ============================================================================
# PAPER EXPERIMENT 1: TRAIN ON MEDIUM PROBLEMS
# ============================================================================
# Research Question: Can GNN generalize within same size and to larger problems?

BLOCKSWORLD_EXP_1_MEDIUM_TRAIN = ExperimentConfig(
    name="blocksworld_exp_1_medium_train",
    description="[PAPER EXP 1] Train on MEDIUM → Test S/M(seen)/M(unseen)/L",
    domain=Domain.BLOCKSWORLD,
    max_merges=128,  # ← Increased from default 50
    timesteps_per_episode=128,  # ← Increased from 50
    train_problem_size=ProblemSize.MEDIUM,
    train_problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
    train_num_problems=15,
    test_configurations=[
        TestConfig(
            name="test_small_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Generalization DOWN to SMALL (untrained)",
            is_seen=False,
        ),
        TestConfig(
            name="test_medium_seen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Performance on MEDIUM training problems (seen)",
            is_seen=True,
        ),
        TestConfig(
            name="test_medium_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium_new/problem_medium_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Generalization to NEW MEDIUM problems (unseen)",
            is_seen=False,
        ),
        TestConfig(
            name="test_large_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Generalization UP to LARGE (untrained)",
            is_seen=False,
        ),
    ],
    num_train_episodes=1000,
    checkpoint_interval=300,
    seed=42,
)

# ============================================================================
# PAPER EXPERIMENT 2: TRAIN ON LARGE PROBLEMS
# ============================================================================
# Research Question: Can training on hard problems transfer to easier ones?

BLOCKSWORLD_EXP_2_LARGE_TRAIN = ExperimentConfig(
    name="blocksworld_exp_2_large_train",
    description="[PAPER EXP 2] Train on LARGE → Test S/M/L(seen)/L(unseen)",
    domain=Domain.BLOCKSWORLD,
    timesteps_per_episode=128,  # ✅ CHANGED from 50
    max_merges=128,  # ✅ CHANGED from 50
    train_problem_size=ProblemSize.LARGE,
    train_problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
    train_num_problems=15,
    test_configurations=[
        TestConfig(
            name="test_small_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Generalization DOWN to SMALL (untrained)",
            is_seen=False,
        ),
        TestConfig(
            name="test_medium_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Generalization DOWN to MEDIUM (untrained)",
            is_seen=False,
        ),
        TestConfig(
            name="test_large_seen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Performance on LARGE training problems (seen)",
            is_seen=True,
        ),
        TestConfig(
            name="test_large_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large_new/problem_large_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Generalization to NEW LARGE problems (unseen)",
            is_seen=False,
        ),
    ],
    num_train_episodes=1000,
    checkpoint_interval=300,
    timeout_per_step=180.0,  # Longer timeout for large problems
    seed=42,
)

# ============================================================================
# PAPER EXPERIMENT 3: CURRICULUM LEARNING + DOMAIN TRANSFER
# ============================================================================
# Research Question: Does curriculum help? Can GNN transfer across domains?

BLOCKSWORLD_EXP_3_CURRICULUM = ExperimentConfig(
    name="blocksworld_exp_3_curriculum",
    description="[PAPER EXP 3] Curriculum S→M→L + Logistics transfer test",
    domain=Domain.BLOCKSWORLD,
    timesteps_per_episode=128,  # ✅ CHANGED from 50
    max_merges=128,  # ✅ CHANGED from 50
    train_problem_size=ProblemSize.SMALL,  # Starts with small
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
    train_num_problems=10,
    is_curriculum=True,
    curriculum_phases=[
        CurriculumPhase(
            name="phase_1_small",
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=10,
            num_episodes=400,
        ),
        CurriculumPhase(
            name="phase_2_medium",
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=10,
            num_episodes=600,
        ),
        CurriculumPhase(
            name="phase_3_large",
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=10,
            num_episodes=500,
        ),
    ],
    test_configurations=[
        # BLOCKSWORLD tests (same domain, unseen problems)
        TestConfig(
            name="test_blocksworld_small_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Blocksworld SMALL (unseen)",
            is_seen=False,
        ),
        TestConfig(
            name="test_blocksworld_medium_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium_new/problem_medium_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Blocksworld MEDIUM (unseen)",
            is_seen=False,
        ),
        TestConfig(
            name="test_blocksworld_large_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large_new/problem_large_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="Blocksworld LARGE (unseen)",
            is_seen=False,
        ),
        # LOGISTICS tests (DOMAIN TRANSFER - completely different domain!)
        TestConfig(
            name="test_logistics_small_transfer",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/logistics/small/problem_small_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="DOMAIN TRANSFER: Logistics SMALL",
            is_seen=False,
        ),
        TestConfig(
            name="test_logistics_medium_transfer",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/medium/problem_medium_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="DOMAIN TRANSFER: Logistics MEDIUM",
            is_seen=False,
        ),
        TestConfig(
            name="test_logistics_large_transfer",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/logistics/large/problem_large_*.pddl",
            num_problems=10,
            num_runs_per_problem=3,
            description="DOMAIN TRANSFER: Logistics LARGE",
            is_seen=False,
        ),
    ],
    num_train_episodes=1500,  # Total across all phases
    checkpoint_interval=300,
    seed=42,
)

# ============================================================================
# QUICK TEST EXPERIMENTS (for validation)
# ============================================================================

QUICK_TEST_REGULAR = ExperimentConfig(
    name="quick_test_regular",
    description="[QUICK TEST] 3 problems, 50 episodes - validate pipeline",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
    train_num_problems=3,
    test_configurations=[
        TestConfig(
            name="test_quick",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=2,
            num_runs_per_problem=1,
            description="Quick validation test",
        ),
    ],
    num_train_episodes=50,
    checkpoint_interval=25,
    seed=42,
)

QUICK_TEST_CURRICULUM = ExperimentConfig(
    name="quick_test_curriculum",
    description="[QUICK TEST] Curriculum 3 phases × 10 episodes - validate pipeline",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
    train_num_problems=2,
    is_curriculum=True,
    curriculum_phases=[
        CurriculumPhase(
            name="phase_small",
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=2,
            num_episodes=10,
        ),
        CurriculumPhase(
            name="phase_medium",
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=2,
            num_episodes=10,
        ),
        CurriculumPhase(
            name="phase_large",
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=2,
            num_episodes=10,
        ),
    ],
    test_configurations=[
        TestConfig(
            name="test_quick",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=2,
            num_runs_per_problem=1,
        ),
    ],
    num_train_episodes=30,
    seed=42,
)

# ============================================================================
# MINI EXPERIMENTS (for framework validation)
# ============================================================================

MINI_REGULAR_EXP = ExperimentConfig(
    name="mini_regular_blocksworld",
    description="[MINI] Regular: 1 problem, 10 episodes",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
    train_num_problems=3,
    num_train_episodes=10,
    test_configurations=[
        TestConfig(
            name="test_mini",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=2,
            num_runs_per_problem=1,
        ),
    ],
    timeout_per_step=45.0,
    checkpoint_interval=1000,
)

MINI_CURRICULUM_EXP = ExperimentConfig(
    name="mini_curriculum_blocksworld",
    description="[MINI] Curriculum: 3 phases × 3 episodes",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_01.pddl",
    train_num_problems=1,
    is_curriculum=True,
    curriculum_phases=[
        CurriculumPhase(
            name="phase_1_small",
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_01.pddl",
            num_problems=1,
            num_episodes=3,
        ),
        CurriculumPhase(
            name="phase_2_medium",
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_01.pddl",
            num_problems=1,
            num_episodes=3,
        ),
        CurriculumPhase(
            name="phase_3_large",
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_01.pddl",
            num_problems=1,
            num_episodes=3,
        ),
    ],
    test_configurations=[
        TestConfig(
            name="test_mini",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=1,
            num_runs_per_problem=1,
        ),
    ],
    num_train_episodes=9,
    timesteps_per_episode=10,
    max_merges=10,
    timeout_per_step=30.0,
)

# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

ALL_EXPERIMENTS = {
    # PAPER EXPERIMENTS (the 3 main ones)
    "blocksworld_exp_1": BLOCKSWORLD_EXP_1_MEDIUM_TRAIN,
    "blocksworld_exp_2": BLOCKSWORLD_EXP_2_LARGE_TRAIN,
    "blocksworld_exp_3_curriculum": BLOCKSWORLD_EXP_3_CURRICULUM,

    # Quick tests
    "quick_test_regular": QUICK_TEST_REGULAR,
    "quick_test_curriculum": QUICK_TEST_CURRICULUM,

    # Mini tests (framework validation)
    "mini_regular": MINI_REGULAR_EXP,
    "mini_curriculum": MINI_CURRICULUM_EXP,
}


def get_experiment(exp_name: str) -> ExperimentConfig:
    """Get experiment config by name."""
    if exp_name not in ALL_EXPERIMENTS:
        available = list(ALL_EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {exp_name}. Available: {available}")
    return ALL_EXPERIMENTS[exp_name]


def list_experiments() -> List[str]:
    """List all available experiments."""
    return list(ALL_EXPERIMENTS.keys())


def get_paper_experiments() -> List[str]:
    """Get the 3 main paper experiments."""
    return ["blocksworld_exp_1", "blocksworld_exp_2", "blocksworld_exp_3_curriculum"]