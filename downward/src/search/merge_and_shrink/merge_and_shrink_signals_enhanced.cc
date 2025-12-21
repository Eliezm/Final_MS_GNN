#include "merge_and_shrink_signals_enhanced.h"
#include "labels.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_set>
#include <queue>
#include <set>

namespace merge_and_shrink {

// ============================================================================
// OPERATOR PROJECTION POTENTIAL
// ============================================================================

double OperatorProjectionAnalyzer::compute_opp_score(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 0.0;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    const auto& vars1 = ts1.get_incorporated_variables();
    const auto& vars2 = ts2.get_incorporated_variables();

    // Collect variables that would be in merged system
    std::set<int> merged_vars(vars1.begin(), vars1.end());
    merged_vars.insert(vars2.begin(), vars2.end());

    // Count total labels (operators)
    int total_labels = 0;
    int projectable_labels = 0;

    // Iterate through all transition systems to find all labels
    for (int sys_idx = 0; sys_idx < fts.get_size(); ++sys_idx) {
        if (!fts.is_active(sys_idx)) continue;

        const TransitionSystem& ts = fts.get_transition_system(sys_idx);

        // Get all labels in this TS
        std::set<int> labels_in_ts;
        for (auto it = ts.begin(); it != ts.end(); ++it) {
            for (int l : (*it).get_label_group()) {
                labels_in_ts.insert(l);
            }
        }

        // For each label, check if it ONLY affects merged variables
        for (int label : labels_in_ts) {
            if (sys_idx == ts1_id || sys_idx == ts2_id) {
                // Labels in one of the merge candidates
                total_labels++;

                // Check if this label affects variables outside merged set
                bool only_merged = true;
                for (int v : vars1) {
                    if (sys_idx == ts2_id && std::find(vars2.begin(), vars2.end(), v) == vars2.end()) {
                        only_merged = false;
                        break;
                    }
                }
                for (int v : vars2) {
                    if (sys_idx == ts1_id && std::find(vars1.begin(), vars1.end(), v) == vars1.end()) {
                        only_merged = false;
                        break;
                    }
                }

                if (only_merged) {
                    projectable_labels++;
                }
            }
        }
    }

    // If no labels, return neutral
    if (total_labels == 0) {
        return 0.5;
    }

    double opp_score = static_cast<double>(projectable_labels) / total_labels;
    return std::max(0.0, std::min(1.0, opp_score));
}

json OperatorProjectionAnalyzer::analyze_operator_projection(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json analysis;
    analysis["opp_score"] = compute_opp_score(fts, ts1_id, ts2_id);
    analysis["metric_name"] = "Operator Projection Potential";
    analysis["interpretation"] = "High (>0.7) = many projectable operators = GOOD merge";
    return analysis;
}

// ============================================================================
// LABEL COMBINABILITY
// ============================================================================

double LabelCombinaibilityAnalyzer::compute_label_combinability_score(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 0.0;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    // Get labels for each TS
    std::set<int> labels1, labels2;
    for (auto it = ts1.begin(); it != ts1.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels1.insert(l);
        }
    }
    for (auto it = ts2.begin(); it != ts2.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels2.insert(l);
        }
    }

    // Labels that appear in both are "synchronizing" and will likely become equivalent
    std::set<int> shared_labels;
    std::set_intersection(labels1.begin(), labels1.end(),
                         labels2.begin(), labels2.end(),
                         std::inserter(shared_labels, shared_labels.begin()));

    // Score: ratio of shared labels
    // High = many labels already synchronized = they'll be combinable
    int total_unique = labels1.size() + labels2.size() - shared_labels.size();
    double score = (total_unique > 0) ?
        static_cast<double>(shared_labels.size()) / total_unique : 0.0;

    return std::max(0.0, std::min(1.0, score));
}

int LabelCombinaibilityAnalyzer::count_collapsible_labels(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 0;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    std::set<int> labels1, labels2;
    for (auto it = ts1.begin(); it != ts1.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels1.insert(l);
        }
    }
    for (auto it = ts2.begin(); it != ts2.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels2.insert(l);
        }
    }

    std::set<int> shared_labels;
    std::set_intersection(labels1.begin(), labels1.end(),
                         labels2.begin(), labels2.end(),
                         std::inserter(shared_labels, shared_labels.begin()));

    return static_cast<int>(shared_labels.size());
}

json LabelCombinaibilityAnalyzer::analyze_label_equivalence(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json analysis;
    analysis["combinability_score"] = compute_label_combinability_score(fts, ts1_id, ts2_id);
    analysis["collapsible_label_count"] = count_collapsible_labels(fts, ts1_id, ts2_id);
    analysis["metric_name"] = "Label Combinability";
    analysis["interpretation"] = "High score = labels will collapse post-merge = GOOD";
    return analysis;
}

// ============================================================================
// GREEDY BISIMULATION ERROR
// ============================================================================

double GreedyBisimulationAnalyzer::compute_greedy_bisimulation_error(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 1.0;  // Unknown = bad
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    const Distances& dist1 = fts.get_distances(ts1_id);
    const Distances& dist2 = fts.get_distances(ts2_id);

    const auto& h1 = dist1.get_goal_distances();
    const auto& h2 = dist2.get_goal_distances();

    // Get init states
    int init1 = ts1.get_init_state();
    int init2 = ts2.get_init_state();

    // Check h-value compatibility at init
    if (init1 < 0 || init1 >= static_cast<int>(h1.size()) ||
        init2 < 0 || init2 >= static_cast<int>(h2.size())) {
        return 1.0;
    }

    int h_init1 = h1[init1];
    int h_init2 = h2[init2];

    // If both are solvable and have similar h-values, good
    if (h_init1 != INF && h_init2 != INF && h_init1 >= 0 && h_init2 >= 0) {
        // h-value mismatch = potential bisimulation violation
        int max_h = std::max(h_init1, h_init2);
        if (max_h > 0) {
            double h_diff_ratio = static_cast<double>(std::abs(h_init1 - h_init2)) / max_h;
            // Scale to [0, 1]: 0 = perfect, 1 = severe conflict
            return std::min(1.0, h_diff_ratio);
        }
    }

    // One or both unsolvable = bad for heuristic
    if (h_init1 == INF || h_init2 == INF) {
        return 0.8;  // Not immediately fatal but risky
    }

    return 0.0;  // Good
}

bool GreedyBisimulationAnalyzer::violates_greedy_bisimulation(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id,
    double error_threshold) {

    double error = compute_greedy_bisimulation_error(fts, ts1_id, ts2_id);
    return error > error_threshold;
}

json GreedyBisimulationAnalyzer::analyze_h_value_compatibility(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json analysis;
    analysis["gb_error"] = compute_greedy_bisimulation_error(fts, ts1_id, ts2_id);
    analysis["violates_gb"] = violates_greedy_bisimulation(fts, ts1_id, ts2_id, 0.2);
    analysis["metric_name"] = "Greedy Bisimulation Error";
    analysis["interpretation"] = "Low error (0-0.2) = h-values compatible = GOOD";
    return analysis;
}

// ============================================================================
// OPERATOR COST VARIANCE
// ============================================================================

json OperatorCostAnalyzer::analyze_operator_costs(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json analysis;
    analysis["cost_variance_score"] = compute_cost_preservation_score(fts, ts1_id, ts2_id);
    analysis["metric_name"] = "Operator Cost Variance";
    return analysis;
}

double OperatorCostAnalyzer::compute_cost_preservation_score(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 0.5;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    std::set<int> labels1, labels2;
    for (auto it = ts1.begin(); it != ts1.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels1.insert(l);
        }
    }
    for (auto it = ts2.begin(); it != ts2.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels2.insert(l);
        }
    }

    // Shared operators = ones that could benefit from merge
    std::set<int> shared;
    std::set_intersection(labels1.begin(), labels1.end(),
                         labels2.begin(), labels2.end(),
                         std::inserter(shared, shared.begin()));

    if (shared.empty()) {
        return 0.3;  // No shared operators = merging loses information
    }

    // Score = proportion of shared to unique
    int total = labels1.size() + labels2.size() - shared.size();
    return std::min(1.0, static_cast<double>(shared.size()) / total);
}

// ============================================================================
// CAUSAL GRAPH DISTANCE
// ============================================================================

double CausalGraphAnalyzer::compute_causal_proximity_score(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 0.0;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    const auto& vars1 = ts1.get_incorporated_variables();
    const auto& vars2 = ts2.get_incorporated_variables();

    // Simple heuristic: variables in both merge candidates are "close"
    // In real causal graph distance, would use actual graph structure
    std::set<int> vars1_set(vars1.begin(), vars1.end());

    for (int v : vars2) {
        if (vars1_set.count(v)) {
            return 1.0;  // Direct overlap = very close
        }
    }

    // Check if they share labels (indirect interaction)
    std::set<int> labels1, labels2;
    for (auto it = ts1.begin(); it != ts1.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels1.insert(l);
        }
    }
    for (auto it = ts2.begin(); it != ts2.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels2.insert(l);
        }
    }

    // Check label overlap
    int shared_labels = 0;
    for (int l : labels1) {
        if (labels2.count(l)) {
            shared_labels++;
        }
    }

    int total_labels = labels1.size() + labels2.size();
    if (total_labels > 0) {
        double label_overlap = static_cast<double>(shared_labels) / total_labels;
        return label_overlap;  // Range [0, 1]
    }

    return 0.0;
}

bool CausalGraphAnalyzer::are_causally_adjacent(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    return compute_causal_proximity_score(fts, ts1_id, ts2_id) > 0.5;
}

json CausalGraphAnalyzer::analyze_causal_relationships(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json analysis;
    analysis["causal_proximity"] = compute_causal_proximity_score(fts, ts1_id, ts2_id);
    analysis["are_adjacent"] = are_causally_adjacent(fts, ts1_id, ts2_id);
    analysis["metric_name"] = "Causal Graph Proximity";
    analysis["interpretation"] = "Adjacent in causal graph = better operator projection";
    return analysis;
}

// ============================================================================
// LABEL SUPPORT CORRELATION
// ============================================================================

double LabelSupportAnalyzer::compute_operator_support_overlap(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 0.0;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    std::set<int> labels1, labels2;
    for (auto it = ts1.begin(); it != ts1.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels1.insert(l);
        }
    }
    for (auto it = ts2.begin(); it != ts2.end(); ++it) {
        for (int l : (*it).get_label_group()) {
            labels2.insert(l);
        }
    }

    // Jaccard similarity
    std::set<int> intersection, union_set;
    std::set_intersection(labels1.begin(), labels1.end(),
                         labels2.begin(), labels2.end(),
                         std::inserter(intersection, intersection.begin()));
    std::set_union(labels1.begin(), labels1.end(),
                   labels2.begin(), labels2.end(),
                   std::inserter(union_set, union_set.begin()));

    if (union_set.empty()) {
        return 0.0;
    }

    return static_cast<double>(intersection.size()) / union_set.size();
}

json LabelSupportAnalyzer::analyze_operator_support(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json analysis;
    analysis["support_overlap"] = compute_operator_support_overlap(fts, ts1_id, ts2_id);
    analysis["metric_name"] = "Label Support Overlap";
    analysis["interpretation"] = "High (>0.6) = variables entangled = merge preserves constraints";
    return analysis;
}

// ============================================================================
// LANDMARK PRESERVATION
// ============================================================================

double LandmarkAnalyzer::compute_landmark_preservation_score(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 0.5;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    int goals1 = 0, goals2 = 0;
    for (int i = 0; i < ts1.get_size(); ++i) {
        if (ts1.is_goal_state(i)) goals1++;
    }
    for (int i = 0; i < ts2.get_size(); ++i) {
        if (ts2.is_goal_state(i)) goals2++;
    }

    // If both have goal states, merge should preserve landmark achievement
    return (goals1 > 0 && goals2 > 0) ? 1.0 : 0.5;
}

json LandmarkAnalyzer::analyze_landmark_involvement(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json analysis;
    analysis["landmark_preservation"] = compute_landmark_preservation_score(fts, ts1_id, ts2_id);
    analysis["metric_name"] = "Landmark Preservation";
    return analysis;
}

// ============================================================================
// TRANSITION EXPLOSION PREDICTION
// ============================================================================

double TransitionExplosionPredictor::predict_transition_density_ratio(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        return 1.0;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    int ts1_size = ts1.get_size();
    int ts2_size = ts2.get_size();
    int ts1_trans = 0, ts2_trans = 0;

    for (auto it = ts1.begin(); it != ts1.end(); ++it) {
        ts1_trans += static_cast<int>((*it).get_transitions().size());
    }
    for (auto it = ts2.begin(); it != ts2.end(); ++it) {
        ts2_trans += static_cast<int>((*it).get_transitions().size());
    }

    // Estimate: product density ≈ min(density1, density2)
    // (synchronization reduces product density)
    double density1 = (ts1_size > 0) ? static_cast<double>(ts1_trans) / ts1_size : 0.0;
    double density2 = (ts2_size > 0) ? static_cast<double>(ts2_trans) / ts2_size : 0.0;

    double product_size = ts1_size * ts2_size;
    double estimated_product_trans = product_size * std::min(density1, density2);

    double ratio = (product_size > 0) ? estimated_product_trans / product_size : 1.0;
    return std::max(0.0, std::min(1.0, ratio));
}

json TransitionExplosionPredictor::predict_transition_explosion(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json prediction;
    prediction["density_ratio"] = predict_transition_density_ratio(fts, ts1_id, ts2_id);
    prediction["metric_name"] = "Transition Explosion Prediction";
    prediction["interpretation"] = "Low ratio (<0.3) = good, High (>0.7) = explosion risk";
    return prediction;
}

// ============================================================================
// COMPREHENSIVE MERGE SIGNALS
// ============================================================================

json MergeQualityAnalyzer::compute_comprehensive_merge_signals(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id,
    int iteration) {

    json comprehensive;
    comprehensive["iteration"] = iteration;
    comprehensive["ts1_id"] = ts1_id;
    comprehensive["ts2_id"] = ts2_id;

    // Run all analyzers
    comprehensive["operator_projection"] =
        OperatorProjectionAnalyzer::analyze_operator_projection(fts, ts1_id, ts2_id);

    comprehensive["label_combinability"] =
        LabelCombinaibilityAnalyzer::analyze_label_equivalence(fts, ts1_id, ts2_id);

    comprehensive["greedy_bisimulation"] =
        GreedyBisimulationAnalyzer::analyze_h_value_compatibility(fts, ts1_id, ts2_id);

    comprehensive["operator_costs"] =
        OperatorCostAnalyzer::analyze_operator_costs(fts, ts1_id, ts2_id);

    comprehensive["causal_graph"] =
        CausalGraphAnalyzer::analyze_causal_relationships(fts, ts1_id, ts2_id);

    comprehensive["label_support"] =
        LabelSupportAnalyzer::analyze_operator_support(fts, ts1_id, ts2_id);

    comprehensive["landmark_preservation"] =
        LandmarkAnalyzer::analyze_landmark_involvement(fts, ts1_id, ts2_id);

    comprehensive["transition_explosion"] =
        TransitionExplosionPredictor::predict_transition_explosion(fts, ts1_id, ts2_id);

    // Compute composite "merge quality" score
    double score = 0.0;
    score += 0.25 * comprehensive["operator_projection"]["opp_score"].get<double>();
    score += 0.20 * comprehensive["label_combinability"]["combinability_score"].get<double>();
    score += 0.20 * (1.0 - comprehensive["greedy_bisimulation"]["gb_error"].get<double>());
    score += 0.15 * comprehensive["label_support"]["support_overlap"].get<double>();
    score += 0.10 * comprehensive["landmark_preservation"]["landmark_preservation"].get<double>();
    score += 0.10 * (1.0 - comprehensive["transition_explosion"]["density_ratio"].get<double>());

    comprehensive["composite_merge_quality"] = std::max(0.0, std::min(1.0, score));

    return comprehensive;
}

}  // namespace merge_and_shrink