#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENT CONFIGURATIONS
Defines all experiments and their parameters.
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
    num_episodes: int  # Episodes for THIS phase

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'problem_size': self.problem_size.value,
            'problem_pattern': self.problem_pattern,
            'num_problems': self.num_problems,
            'num_episodes': self.num_episodes,
        }



@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str  # e.g., "blocksworld_exp_1"
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
    timesteps_per_episode: int = 50
    max_merges: int = 50
    timeout_per_step: float = 120.0
    checkpoint_interval: int = 1000
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
        return d



@dataclass
class TestConfig:
    """Configuration for a single test set."""
    name: str  # e.g., "test_small_seen"
    domain: Domain
    problem_size: ProblemSize
    problem_pattern: str
    num_problems: int
    num_runs_per_problem: int = 5
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'domain': self.domain.value,
            'problem_size': self.problem_size.value,
            'problem_pattern': self.problem_pattern,
            'num_problems': self.num_problems,
            'num_runs_per_problem': self.num_runs_per_problem,
            'description': self.description,
        }

# ============================================================================
# BLOCKSWORLD EXPERIMENTS
# ============================================================================

# Test 1: Single problem, minimal training
BLOCKSWORLD_TEST_MINIMAL = ExperimentConfig(
    name="blocksworld_test_minimal",
    description="[TEST] Single small problem, 2 episodes",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_01.pddl",  # ← SINGLE
    train_num_problems=1,
    num_train_episodes=2,  # ← MINIMAL
    timesteps_per_episode=10,  # ← SHORT
    test_configurations=[
        TestConfig(
            name="test_small_seen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=1,  # ← SINGLE
            num_runs_per_problem=1,  # ← MINIMAL
            description="Quick test on same problem"
        ),
    ],
)

# Test 3: Curriculum minimal (very short phases)
BLOCKSWORLD_TEST_CURRICULUM_MINIMAL = ExperimentConfig(
    name="blocksworld_test_curriculum_minimal",
    description="[TEST] Curriculum: S(1ep) → M(1ep) → L(1ep)",
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
            num_episodes=1,  # ← 1 EPISODE
        ),
        CurriculumPhase(
            name="phase_2_medium",
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_01.pddl",
            num_problems=1,
            num_episodes=1,  # ← 1 EPISODE
        ),
        CurriculumPhase(
            name="phase_3_large",
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_01.pddl",
            num_problems=1,
            num_episodes=1,  # ← 1 EPISODE
        ),
    ],
    test_configurations=[
        TestConfig(
            name="test_small",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_01.pddl",
            num_problems=1,
            num_runs_per_problem=1,
        ),
    ],
    num_train_episodes=3,  # DUMMY (overridden by phases)
)

BLOCKSWORLD_EXP_DUMB = ExperimentConfig(
    name="blocksworld_exp_1_medium_train",
    description="Train on MEDIUM problems → Test MEDIUM (seen/unseen) and LARGE",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
    train_num_problems=1,
    test_configurations=[
        TestConfig(
            name="test_small_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=1,
            description="Test on MEDIUM problems from training set"
        ),
    ],
    num_train_episodes=3,
)

BLOCKSWORLD_CURRICULUM_DUMB = ExperimentConfig(
    name="blocksworld_curriculum_small_to_large",
    description="Curriculum: SMALL (20ep) → MEDIUM (30ep) → LARGE (50ep)",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,  # Dummy (will be overridden)
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",  # Dummy
    train_num_problems=5,  # Dummy
    is_curriculum=True,
    curriculum_phases=[
        CurriculumPhase(
            name="phase_1_small",
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=1,
            num_episodes=2,  # ← SHORT phase
        ),
        CurriculumPhase(
            name="phase_2_medium",
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=1,
            num_episodes=2,  # ← MEDIUM phase
        ),
        CurriculumPhase(
            name="phase_3_large",
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=1,
            num_episodes=2,  # ← LONG phase
        ),
    ],
    test_configurations=[
        # Test on ALL sizes after full curriculum
        TestConfig(
            name="test_small",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=1,
            description="Test final model on SMALL"
        ),
    ],
    num_train_episodes=3,  # DUMMY (will use sum of phases)
)

BLOCKSWORLD_EXP_0 = ExperimentConfig(
    name="blocksworld_exp_0_small_train",
    description="Train on SMALL problems → Test SMALL (Seen/Unseen), MEDIUM and LARGE",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
    train_num_problems=20,
    test_configurations=[
        TestConfig(
            name="test_small_seen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=10,
            description="Test on SMALL problems from training set"
        ),
        TestConfig(
            name="test_small_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=10,
            description="Test on NEW SMALL problems (unseen during training)"
        ),
        TestConfig(
            name="test_medium",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=10,
            description="Test on MEDIUM problems"
        ),
        TestConfig(
            name="test_large",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=10,
            description="Test on LARGE problems"
        ),
    ],
    num_train_episodes=1500,
)

BLOCKSWORLD_EXP_1 = ExperimentConfig(
    name="blocksworld_exp_1_medium_train",
    description="Train on MEDIUM problems → Test MEDIUM (seen/unseen) and LARGE",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.MEDIUM,
    train_problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
    train_num_problems=20,
    test_configurations=[
        TestConfig(
            name="test_medium_seen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=10,
            description="Test on MEDIUM problems from training set"
        ),
        TestConfig(
            name="test_medium_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium_new/problem_medium_*.pddl",
            num_problems=7,
            description="Test on NEW MEDIUM problems (unseen during training)"
        ),
        TestConfig(
            name="test_large",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=10,
            description="Test on LARGE problems"
        ),
    ],
    num_train_episodes=1500,
)

BLOCKSWORLD_EXP_2 = ExperimentConfig(
    name="blocksworld_exp_2_large_train",
    description="Train on LARGE problems → Test on SMALL and MEDIUM (seen/unseen)",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.LARGE,
    train_problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
    train_num_problems=20,
    test_configurations=[
        TestConfig(
            name="test_large_seen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=20,
        ),
        TestConfig(
            name="test_large_unseen",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large_new/problem_large_*.pddl",
            num_problems=10,
        ),
        TestConfig(
            name="test_medium",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=10,
        ),
        TestConfig(
            name="test_small",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=10,
        ),
    ],
    num_train_episodes=1500,
)


BLOCKSWORLD_CURRICULUM = ExperimentConfig(
    name="blocksworld_curriculum_small_to_large",
    description="Curriculum: SMALL (20ep) → MEDIUM (30ep) → LARGE (50ep)",
    domain=Domain.BLOCKSWORLD,
    train_problem_size=ProblemSize.SMALL,  # Dummy (will be overridden)
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",  # Dummy
    train_num_problems=5,  # Dummy
    is_curriculum=True,
    curriculum_phases=[
        CurriculumPhase(
            name="phase_1_small",
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
            num_problems=5,
            num_episodes=300,  # ← SHORT phase
        ),
        CurriculumPhase(
            name="phase_2_medium",
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium/problem_medium_*.pddl",
            num_problems=5,
            num_episodes=600,  # ← MEDIUM phase
        ),
        CurriculumPhase(
            name="phase_3_large",
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large/problem_large_*.pddl",
            num_problems=5,
            num_episodes=600,  # ← LONG phase
        ),
    ],
    test_configurations=[
        # Test on ALL sizes after full curriculum
        TestConfig(
            name="test_small",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=10,
            description="Test final model on SMALL"
        ),
        TestConfig(
            name="test_medium",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium_new/problem_medium_*.pddl",
            num_problems=7,
            description="Test final model on MEDIUM"
        ),
        TestConfig(
            name="test_large",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large_new/problem_large_*.pddl",
            num_problems=10,
            description="Test final model on LARGE"
        ),
        # ✅ ADD: Logistics cross-domain tests
        TestConfig(
            name="test_logistics_small",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/logistics/small_new/problem_small_*.pddl",
            num_problems=10,
            description="Transfer to LOGISTICS SMALL"
        ),
        TestConfig(
            name="test_logistics_medium",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/medium_new/problem_medium_*.pddl",
            num_problems=7,
            description="Transfer to LOGISTICS MEDIUM"
        ),
        TestConfig(
            name="test_logistics_large",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/logistics/large_new/problem_large_*.pddl",
            num_problems=10,
            description="Transfer to LOGISTICS LARGE"
        ),
    ],
    num_train_episodes=1500,  # DUMMY (will use sum of phases)
)




# ============================================================================
# LOGISTICS EXPERIMENTS
# ============================================================================

LOGISTICS_EXP_0 = ExperimentConfig(
    name="logistics_exp_0_small_train",
    description="Train on SMALL problems → Test SMALL (Seen/Unseen), MEDIUM and LARGE",
    domain=Domain.LOGISTICS,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/blocksworld/small/problem_small_*.pddl",
    train_num_problems=20,
    test_configurations=[
        TestConfig(
            name="test_small_seen",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/logistics/small/problem_small_*.pddl",
            num_problems=10,
            description="Test on SMALL problems from training set"
        ),
        TestConfig(
            name="test_small_unseen",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/logistics/small_new/problem_small_*.pddl",
            num_problems=10,
            description="Test on NEW SMALL problems (unseen during training)"
        ),
        TestConfig(
            name="test_medium",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/medium/problem_medium_*.pddl",
            num_problems=10,
            description="Test on MEDIUM problems"
        ),
        TestConfig(
            name="test_large",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/logistics/large/problem_large_*.pddl",
            num_problems=10,
            description="Test on LARGE problems"
        ),
    ],
    num_train_episodes=1500,
)

LOGISTICS_EXP_1 = ExperimentConfig(
    name="logistics_exp_1_medium_train",
    description="Train on MEDIUM problems → Test on SMALL (seen/unseen) and LARGE",
    domain=Domain.LOGISTICS,
    train_problem_size=ProblemSize.MEDIUM,
    train_problem_pattern="benchmarks/logistics/medium/*.pddl",
    train_num_problems=20,
    test_configurations=[
        TestConfig(
            name="test_small",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/logistics/small/problem_small_*.pddl",
            num_problems=10,
        ),
        TestConfig(
            name="test_medium_seen",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/medium/problem_medium_*.pddl",
            num_problems=10,
        ),
        TestConfig(
            name="test_medium_unseen",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/medium_new/problem_medium_*.pddl",
            num_problems=7,
        ),
        TestConfig(
            name="test_large",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/logistics/large/problem_large_*.pddl",
            num_problems=10,
        ),
    ],
    num_train_episodes=1500,
)

LOGISTICS_EXP_2 = ExperimentConfig(
    name="logistics_exp_2_large_train",
    description="Train on LARGE problems → Test on SMALL and MEDIUM (seen/unseen)",
    domain=Domain.LOGISTICS,
    train_problem_size=ProblemSize.LARGE,
    train_problem_pattern="benchmarks/logistics/large/problem_large_*.pddl",
    train_num_problems=20,
    test_configurations=[
        TestConfig(
            name="test_small",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/logistics/small/problem_small_*.pddl",
            num_problems=10,
        ),
        TestConfig(
            name="test_medium",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/medium/problem_medium_*.pddl",
            num_problems=10,
        ),
        TestConfig(
            name="test_large_seen",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/large/problem_large_*.pddl",
            num_problems=10,
        ),
        TestConfig(
            name="test_large_unseen",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/large_new/problem_large_*.pddl",
            num_problems=10,
        ),
    ],
    num_train_episodes=1500,
)

LOGISTICS_CURRICULUM = ExperimentConfig(
    name="logistics_curriculum_small_to_large",
    description="Curriculum: SMALL (20ep) → MEDIUM (30ep) → LARGE (50ep)",
    domain=Domain.LOGISTICS,
    train_problem_size=ProblemSize.SMALL,
    train_problem_pattern="benchmarks/logistics/small/problem_small_*.pddl",
    train_num_problems=5,
    is_curriculum=True,
    curriculum_phases=[
        CurriculumPhase(
            name="phase_1_small",
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/logistics/small/problem_small_*.pddl",
            num_problems=10,
            num_episodes=300,
        ),
        CurriculumPhase(
            name="phase_2_medium",
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/medium/problem_medium_*.pddl",
            num_problems=10,
            num_episodes=600,
        ),
        CurriculumPhase(
            name="phase_3_large",
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/logistics/large/problem_large_*.pddl",
            num_problems=10,
            num_episodes=600,
        ),
    ],
    test_configurations=[
        TestConfig(
            name="test_small",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/logistics/small_new/problem_small_*.pddl",
            num_problems=10,
        ),
        TestConfig(
            name="test_medium",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/logistics/medium_new/problem_medium_*.pddl",
            num_problems=7,
        ),
        TestConfig(
            name="test_large",
            domain=Domain.LOGISTICS,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/logistics/large_new/problem_large_*.pddl",
            num_problems=10,
        ),
        # ✅ ADD: Blocksworld cross-domain tests
        TestConfig(
            name="test_blocksworld_small",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.SMALL,
            problem_pattern="benchmarks/blocksworld/small_new/problem_small_*.pddl",
            num_problems=10,
            description="Transfer to BLOCKSWORLD SMALL"
        ),
        TestConfig(
            name="test_blocksworld_medium",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.MEDIUM,
            problem_pattern="benchmarks/blocksworld/medium_new/problem_medium_*.pddl",
            num_problems=7,
            description="Transfer to BLOCKSWORLD MEDIUM"
        ),
        TestConfig(
            name="test_blocksworld_large",
            domain=Domain.BLOCKSWORLD,
            problem_size=ProblemSize.LARGE,
            problem_pattern="benchmarks/blocksworld/large_new/problem_large_*.pddl",
            num_problems=10,
            description="Transfer to BLOCKSWORLD LARGE"
        ),
    ],
    num_train_episodes=1500,
)

# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

ALL_EXPERIMENTS = {
    "blocksworld_curriculum_dumb": BLOCKSWORLD_CURRICULUM_DUMB,
    "blocksworld_exp_dumb": BLOCKSWORLD_EXP_DUMB,
    "blocksworld_exp_0": BLOCKSWORLD_EXP_0,
    "blocksworld_exp_1": BLOCKSWORLD_EXP_1,
    "blocksworld_exp_2": BLOCKSWORLD_EXP_2,
    "blocksworld_curriculum": BLOCKSWORLD_CURRICULUM,
    "logistics_exp_0": LOGISTICS_EXP_0,
    "logistics_exp_1": LOGISTICS_EXP_1,
    "logistics_exp_2": LOGISTICS_EXP_2,
    "logistics_curriculum": LOGISTICS_CURRICULUM,
    "blocksworld_test_minimal": BLOCKSWORLD_TEST_MINIMAL,
    "blocksworld_test_curriculum_minimal": BLOCKSWORLD_TEST_CURRICULUM_MINIMAL,

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