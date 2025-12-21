#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT RESEARCH MAPPING - What research question does each plot answer?
=====================================================================
Every plot must tell a story for the research paper.
"""

PLOT_RESEARCH_QUESTIONS = {
    # =========================================================================
    # TRAINING DYNAMICS PLOTS
    # =========================================================================
    "01_learning_curves": {
        "research_question": "Does the GNN learn to improve merge decisions over training?",
        "what_it_shows": [
            "Reward trajectory showing learning progress",
            "H* preservation improvement over episodes",
            "Problem coverage ensuring generalization",
            "Failure patterns for debugging",
        ],
        "paper_section": "Training Results",
        "key_insight": "Demonstrates that RL training successfully improves merge strategy",
    },

    # =========================================================================
    # COMPONENT ANALYSIS PLOTS
    # =========================================================================
    "02_component_trajectories": {
        "research_question": "Which reward components drive learning progress?",
        "what_it_shows": [
            "Evolution of each reward component (H*, transition, OPP, label, bonus)",
            "Relative contribution changes over training",
            "Component trade-offs during optimization",
        ],
        "paper_section": "Reward Function Analysis",
        "key_insight": "H* preservation dominates; other components provide complementary signals",
    },

    "03_component_stability": {
        "research_question": "Are reward signals stable enough for reliable learning?",
        "what_it_shows": [
            "Variance of each component across training",
            "Signal-to-noise ratio per component",
        ],
        "paper_section": "Reward Function Analysis",
        "key_insight": "Low variance = reliable gradient signal for PPO",
    },

    "04_merge_quality_heatmap": {
        "research_question": "How does merge quality evolve across training phases?",
        "what_it_shows": [
            "Component quality over episodes (heatmap)",
            "Phase transitions in learning",
        ],
        "paper_section": "Training Dynamics",
        "key_insight": "Early exploration → late exploitation pattern",
    },

    # =========================================================================
    # FEATURE ANALYSIS PLOTS
    # =========================================================================
    "05_feature_importance": {
        "research_question": "Which input features does the GNN rely on most?",
        "what_it_shows": [
            "Correlation between features and GNN confidence",
            "Statistical significance of feature-decision relationships",
        ],
        "paper_section": "GNN Architecture Analysis",
        "key_insight": "Validates that GNN uses theoretically-meaningful features",
    },

    # =========================================================================
    # HEURISTIC QUALITY PLOTS
    # =========================================================================
    "06_bisimulation_preservation": {
        "research_question": "Does the learned strategy preserve heuristic quality (h*)?",
        "what_it_shows": [
            "H* preservation ratio over training",
            "Comparison to theoretical optimum (bisimulation)",
            "Phase-wise preservation rates",
        ],
        "paper_section": "Main Results - Heuristic Quality",
        "key_insight": "PRIMARY RESULT: GNN achieves >95% h* preservation",
    },

    # =========================================================================
    # SAFETY PLOTS
    # =========================================================================
    "07_dead_end_timeline": {
        "research_question": "Does the strategy avoid creating unsolvable states?",
        "what_it_shows": [
            "Dead-end creation rate over training",
            "Solvability maintenance",
            "Safety margin analysis",
        ],
        "paper_section": "Safety Analysis",
        "key_insight": "GNN learns to avoid catastrophic merges",
    },

    # =========================================================================
    # ABSTRACTION SIZE PLOTS
    # =========================================================================
    "08_label_reduction_impact": {
        "research_question": "Do merge choices enable effective label reduction?",
        "what_it_shows": [
            "Label combinability vs reward",
            "OPP score evolution",
            "Compression potential",
        ],
        "paper_section": "Abstraction Size Control",
        "key_insight": "GNN learns to create merge-friendly structures",
    },

    "09_transition_explosion": {
        "research_question": "Does the GNN learn to predict and avoid transition explosions?",
        "what_it_shows": [
            "Transition growth vs GNN confidence",
            "Explosion prediction accuracy",
            "Confidence distribution by explosion risk",
        ],
        "paper_section": "Abstraction Size Control",
        "key_insight": "GNN develops implicit explosion avoidance",
    },

    # =========================================================================
    # MERGE STRATEGY PLOTS
    # =========================================================================
    "10_causal_alignment": {
        "research_question": "Does the GNN's merge order respect problem structure?",
        "what_it_shows": [
            "H* preservation by merge sequence position",
            "Merge quality by training phase",
            "Variable distance in merge pairs",
        ],
        "paper_section": "Strategy Analysis",
        "key_insight": "GNN discovers causally-informed merge ordering",
    },

    "11_gnn_decision_quality": {
        "research_question": "Can the GNN distinguish good merges from bad merges?",
        "what_it_shows": [
            "Confusion matrix: GNN predictions vs actual quality",
            "Accuracy by merge category",
            "Confidence calibration",
        ],
        "paper_section": "GNN Performance",
        "key_insight": "GNN achieves >X% accuracy in merge classification",
    },

    "12_merge_quality_distribution": {
        "research_question": "What is the overall distribution of merge decisions?",
        "what_it_shows": [
            "Histogram of merge quality categories",
            "Proportion of excellent/good/bad merges",
        ],
        "paper_section": "Training Results",
        "key_insight": "Training shifts distribution toward good merges",
    },

    # =========================================================================
    # COMPARISON PLOTS
    # =========================================================================
    "13_three_way_comparison": {
        "research_question": "How does GNN compare to Random and FD baselines?",
        "what_it_shows": [
            "Solve rate comparison",
            "Time comparison",
            "Expansion comparison",
            "H* preservation (GNN-specific advantage)",
        ],
        "paper_section": "Main Results - Comparison",
        "key_insight": "PRIMARY RESULT: GNN outperforms random by X%, competitive with FD",
    },

    "14_per_problem_winners": {
        "research_question": "Which strategy wins on which problems?",
        "what_it_shows": [
            "Per-problem performance heatmap",
            "Winner identification per problem",
        ],
        "paper_section": "Detailed Comparison",
        "key_insight": "GNN excels on structured problems, FD on random",
    },

    "15_cumulative_solved": {
        "research_question": "How quickly do strategies solve problems?",
        "what_it_shows": [
            "Cumulative problems solved over time",
            "Anytime performance curves",
        ],
        "paper_section": "Anytime Performance",
        "key_insight": "GNN provides faster initial solutions",
    },

    "16_speedup_analysis": {
        "research_question": "What speedup does GNN achieve over baselines?",
        "what_it_shows": [
            "Speedup distribution",
            "Median/mean speedup factors",
        ],
        "paper_section": "Efficiency Analysis",
        "key_insight": "GNN achieves Xth speedup on Y% of problems",
    },

    # =========================================================================
    # LITERATURE VALIDATION
    # =========================================================================
    "17_literature_alignment": {
        "research_question": "Does implementation align with M&S theory?",
        "what_it_shows": [
            "Checklist of theoretical requirements",
            "Helmert et al. 2014 compliance",
            "Nissim et al. 2011 compliance",
        ],
        "paper_section": "Methodology Validation",
        "key_insight": "Implementation correctly captures theoretical foundations",
    },
}


def get_plot_purpose(plot_name: str) -> dict:
    """Get the research purpose of a plot."""
    return PLOT_RESEARCH_QUESTIONS.get(plot_name, {})


def print_plot_catalog():
    """Print all plots with their research questions."""
    print("\n" + "=" * 100)
    print("PLOT CATALOG - Research Questions and Purpose")
    print("=" * 100)

    for plot_name, info in PLOT_RESEARCH_QUESTIONS.items():
        print(f"\n📊 {plot_name}")
        print(f"   Research Question: {info['research_question']}")
        print(f"   Paper Section: {info['paper_section']}")
        print(f"   Key Insight: {info['key_insight']}")

    print("\n" + "=" * 100)