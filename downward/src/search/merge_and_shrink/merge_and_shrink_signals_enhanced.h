#ifndef MERGE_AND_SHRINK_SIGNALS_ENHANCED_H
#define MERGE_AND_SHRINK_SIGNALS_ENHANCED_H

#include "merge_and_shrink_signals.h"
#include "factored_transition_system.h"
#include "transition_system.h"
#include "distances.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <set>
#include <map>

namespace merge_and_shrink {

using json = nlohmann::json;

// ============================================================================
// OPERATOR PROJECTION POTENTIAL (OPP)
// ============================================================================
/**
 * Computes how many operators will have ZERO effect on remaining variables.
 * These operators can be projected away, leading to massive compression.
 *
 * From Nissim et al. (2011):
 * "Label Projection is the Maximal Conservative Label Reduction"
 * Merging variables allows projection of operators affecting only those vars.
 */
class OperatorProjectionAnalyzer {
public:
    /**
     * Estimate the "Operator Projection Potential" score [0, 1]
     *
     * High value (>0.7): Many operators can be projected → merge is GOOD
     * Low value (<0.2): Few operators can be projected → merge is RISKY
     */
    static double compute_opp_score(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Get detailed breakdown of projectable operators
     */
    static json analyze_operator_projection(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );
};

// ============================================================================
// LABEL COMBINABILITY ANALYSIS
// ============================================================================
/**
 * From Helmert et al. (2014):
 * "Labels are combinable if they are locally equivalent in all other factors"
 *
 * High label combinability → post-merge label reduction potential
 * This directly impacts compressed size of merged system.
 */
class LabelCombinaibilityAnalyzer {
public:
    /**
     * Compute label combinability score [0, 1]
     *
     * Score = (% of labels that become equivalent) after merge
     *
     * High: Merging allows aggressive label reduction
     * Low: Labels remain distinct, system stays large
     */
    static double compute_label_combinability_score(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Count how many label groups will collapse
     */
    static int count_collapsible_labels(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Detailed breakdown of label equivalence
     */
    static json analyze_label_equivalence(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );
};

// ============================================================================
// GREEDY BISIMULATION ERROR
// ============================================================================
/**
 * From Nissim et al. (2011):
 * "Greedy Bisimulation only requires bisimulation on transitions where
 *  h*(s) = h*(s') + c(l)"
 *
 * Measures h-value incompatibility between merge candidates.
 * High error → merging destroys heuristic quality.
 */
class GreedyBisimulationAnalyzer {
public:
    /**
     * Compute greedy bisimulation error [0, 1]
     *
     * 0 = Perfect: All optimal-path transitions respect h-values
     * 1 = Severe: Many h-value conflicts on optimal paths
     *
     * This is a DENSE REWARD SIGNAL for h* preservation.
     */
    static double compute_greedy_bisimulation_error(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Check if merge candidates have conflicting h-values on optimal paths
     */
    static bool violates_greedy_bisimulation(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id,
        double error_threshold = 0.2
    );

    /**
     * Detailed h-value conflict analysis
     */
    static json analyze_h_value_compatibility(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );
};

// ============================================================================
// OPERATOR COST VARIANCE SIGNAL
// ============================================================================
/**
 * From Helmert et al. (2014):
 * "Cost distribution of transitions matters for abstraction quality"
 *
 * High-cost operators need special handling:
 * - If separated: each variable sees only its subset of costs
 * - If merged: full cost information is preserved
 */
class OperatorCostAnalyzer {
public:
    /**
     * Analyze cost distribution for operators affecting merge candidates
     */
    static json analyze_operator_costs(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Score [0, 1]: How well is cost distribution captured?
     * High: Operators affecting both candidates have wide cost range
     * Low: Operators are independent in cost
     */
    static double compute_cost_preservation_score(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );
};

// ============================================================================
// CAUSAL GRAPH DISTANCE
// ============================================================================
/**
 * From Nissim et al. (2011) - M&S-gop strategy:
 * "Variables closer in causal graph should merge earlier"
 *
 * Reason: Operator projection works better when dependent variables merge
 */
class CausalGraphAnalyzer {
public:
    /**
     * Compute causal graph distance [0, 1]
     *
     * 0 = Independent (far in graph)
     * 1 = Directly dependent (adjacent in graph)
     *
     * Merging closer variables → better operator projection
     */
    static double compute_causal_proximity_score(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Check if variables are adjacent in causal graph
     */
    static bool are_causally_adjacent(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Detailed causal relationship analysis
     */
    static json analyze_causal_relationships(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );
};

// ============================================================================
// LABEL SUPPORT CORRELATION
// ============================================================================
/**
 * From papers:
 * "Operator support overlap indicates variable entanglement"
 *
 * High support overlap = variables are heavily entangled
 * Merging such variables captures global constraints better
 */
class LabelSupportAnalyzer {
public:
    /**
     * Jaccard similarity of operators affecting each variable [0, 1]
     *
     * High (>0.6): Variables share many operators → highly entangled
     * Low (<0.2): Variables mostly independent
     */
    static double compute_operator_support_overlap(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Detailed operator support analysis
     */
    static json analyze_operator_support(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );
};

// ============================================================================
// LANDMARK ACHIEVER PRESERVATION
// ============================================================================
/**
 * From papers: Merging landmark-critical variables can destroy heuristic
 *
 * Landmark achievers are transitions on optimal plans
 * We want to preserve these to maintain heuristic quality
 */
class LandmarkAnalyzer {
public:
    /**
     * Check if merge candidates achieve important landmarks
     * Returns ratio of preserved landmark achievers [0, 1]
     *
     * High: Merge preserves landmark achievement capability
     * Low: Merge might destroy important pathways
     */
    static double compute_landmark_preservation_score(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Identify critical landmark achiever transitions
     */
    static json analyze_landmark_involvement(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );
};

// ============================================================================
// TRANSITION DENSITY & EXPLOSION PREDICTION
// ============================================================================
/**
 * From papers: "Number of transitions is the real killer"
 *
 * Predicts post-merge transition explosion more accurately
 */
class TransitionExplosionPredictor {
public:
    /**
     * Estimate actual transition count in merged product
     * (not just |S1| × |S2|)
     *
     * Returns ratio of expected_transitions / (|S1| × |S2|)
     * <0.1 = good (many states unreachable)
     * >0.9 = bad (dense product)
     */
    static double predict_transition_density_ratio(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );

    /**
     * Detailed transition explosion analysis
     */
    static json predict_transition_explosion(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id
    );
};

// ============================================================================
// COMPREHENSIVE MERGE QUALITY ANALYZER
// ============================================================================
/**
 * Integration point: Computes all signals together
 */
class MergeQualityAnalyzer {
public:
    /**
     * Compute comprehensive merge quality signals
     * Returns JSON with all analyzers' results
     */
    static json compute_comprehensive_merge_signals(
        const FactoredTransitionSystem& fts,
        int ts1_id,
        int ts2_id,
        int iteration
    );
};

}  // namespace merge_and_shrink

#endif
