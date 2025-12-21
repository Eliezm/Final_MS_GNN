#include "merge_and_shrink_signals.h"
#include "merge_and_shrink_signals_enhanced.h"

#include "factored_transition_system.h"
#include "transition_system.h"
#include "distances.h"
#include "types.h"
#include <thread>

#include <filesystem>
#include <fstream>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <set>
#include <unordered_set>
#include <map>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace merge_and_shrink {

// ============================================================================
// STATIC STATE FOR H* TRACKING ACROSS ITERATIONS
// ============================================================================

// Track h* values for preservation computation
static std::map<int, int> prev_h_star_values;  // ts_id -> h* value
static int prev_combined_h_star = 0;
static int prev_iteration = -2;
static int prev_total_states = 0;
static std::vector<int> prev_f_values;

// ============================================================================
// DIRECTORY MANAGEMENT
// ============================================================================

std::string get_fd_output_directory() {
    fs::path fd_output = fs::current_path() / "fd_output";
    try {
        fs::create_directories(fd_output);
    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS] Warning: Could not create fd_output: " << e.what() << std::endl;
    }
    return fd_output.string();
}

std::string get_gnn_output_directory() {
    fs::path gnn_output = fs::current_path() / "gnn_output";
    try {
        fs::create_directories(gnn_output);
    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS] Warning: Could not create gnn_output: " << e.what() << std::endl;
    }
    return gnn_output.string();
}

// ============================================================================
// ATOMIC FILE I/O
// ============================================================================

void write_json_file_atomic(const json& data, const std::string& file_path) {
    try {
        fs::path final_path(file_path);
        fs::path dir_path = final_path.parent_path();

        fs::create_directories(dir_path);

        fs::path temp_path = final_path.string() + ".tmp";

        {
            std::ofstream temp_file(temp_path, std::ios::out | std::ios::trunc);
            if (!temp_file.is_open()) {
                throw std::runtime_error("Cannot open temp file: " + temp_path.string());
            }

            temp_file << data.dump(2);
            temp_file.flush();

            if (temp_file.fail()) {
                throw std::runtime_error("Failed to write/flush to temp file");
            }

            temp_file.close();
        }

        int rename_attempts = 0;
        const int max_rename_attempts = 10;

        while (rename_attempts < max_rename_attempts) {
            try {
                fs::rename(temp_path, final_path);
                break;
            } catch (const fs::filesystem_error& e) {
                rename_attempts++;
                if (rename_attempts >= max_rename_attempts) {
                    throw;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        std::cout << "[SIGNALS] Wrote: " << file_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS] Error writing JSON: " << e.what() << std::endl;
        throw;
    }
}

// ============================================================================
// F-VALUE STATISTICS COMPUTATION
// ============================================================================

json compute_f_statistics(const std::vector<int>& distances, int unreachable_marker) {
    std::vector<double> valid_distances;

    for (int d : distances) {
        if (d != unreachable_marker && d >= 0 && d < 1000000000) {
            valid_distances.push_back(static_cast<double>(d));
        }
    }

    json stats;

    if (valid_distances.empty()) {
        stats["min"] = unreachable_marker;
        stats["max"] = unreachable_marker;
        stats["mean"] = 0.0;
        stats["std"] = 0.0;
        stats["valid_count"] = 0;
    } else {
        double min_val = *std::min_element(valid_distances.begin(), valid_distances.end());
        stats["min"] = static_cast<int>(min_val);

        double max_val = *std::max_element(valid_distances.begin(), valid_distances.end());
        stats["max"] = static_cast<int>(max_val);

        double sum = std::accumulate(valid_distances.begin(), valid_distances.end(), 0.0);
        double mean = sum / valid_distances.size();
        stats["mean"] = mean;

        double variance = 0.0;
        for (double v : valid_distances) {
            variance += (v - mean) * (v - mean);
        }
        double std_dev = std::sqrt(variance / valid_distances.size());
        stats["std"] = std_dev;

        stats["valid_count"] = static_cast<int>(valid_distances.size());
    }

    return stats;
}

// ============================================================================
// ENHANCED 15-DIMENSIONAL NODE FEATURES
// ============================================================================

std::vector<float> compute_gnn_node_features_enhanced(
    const TransitionSystem& ts,
    const Distances& distances,
    int iteration,
    int max_state_count,
    int global_max_labels) {

    std::vector<float> features(NODE_FEATURE_DIM, 0.0f);  // NOW 9 features

    int ts_size = ts.get_size();
    const auto& goal_distances = distances.get_goal_distances();
    const auto& init_distances = distances.get_init_distances();

    // ========================================================================
    // REDUCED FEATURE SET (9 features only)
    // ========================================================================

    // Feature 0: Normalized size
    features[0] = (max_state_count > 0) ?
        static_cast<float>(ts_size) / max_state_count : 0.0f;

    // Feature 1: Mean h-value (normalized)
    int reachable_count = 0;
    double sum_h_values = 0.0;
    for (size_t i = 0; i < goal_distances.size(); ++i) {
        if (goal_distances[i] != INF && goal_distances[i] >= 0) {
            sum_h_values += goal_distances[i];
            reachable_count++;
        }
    }
    double mean_h = (reachable_count > 0) ? sum_h_values / reachable_count : 0.0;
    features[1] = std::min(1.0f, static_cast<float>(mean_h / (ts_size + 1)));

    // Feature 2: Solvability (fraction of reachable states)
    features[2] = (ts_size > 0) ?
        static_cast<float>(reachable_count) / ts_size : 0.0f;

    // Feature 3: Initial state h-value (PRIMARY SIGNAL - h* preservation!)
    int init_state = ts.get_init_state();
    int h_init = INF;
    if (init_state >= 0 && init_state < static_cast<int>(goal_distances.size())) {
        h_init = goal_distances[init_state];
    }
    features[3] = (h_init != INF && h_init >= 0) ?
        std::min(1.0f, static_cast<float>(h_init) / (ts_size + 1)) : 1.0f;

    // Feature 4: Dead-end ratio (quality indicator)
    int dead_ends = 0;
    for (size_t i = 0; i < goal_distances.size() && i < init_distances.size(); ++i) {
        if (goal_distances[i] == INF && init_distances[i] != INF) {
            dead_ends++;
        }
    }
    features[4] = (reachable_count > 0) ?
        static_cast<float>(dead_ends) / reachable_count : 0.0f;

    // Feature 5: Number of incorporated variables (normalized)
    const auto& vars = ts.get_incorporated_variables();
    features[5] = std::min(1.0f, static_cast<float>(vars.size()) / 20.0f);

    // Feature 6: Irrelevant Label Ratio (FROM HELMERT ET AL. - enables label reduction)
    std::set<int> unique_labels;
    for (auto it = ts.begin(); it != ts.end(); ++it) {
        const auto& label_group = (*it).get_label_group();
        for (int label : label_group) {
            unique_labels.insert(label);
        }
    }
    int label_count = static_cast<int>(unique_labels.size());

    int irrelevant_labels = 0;
    for (int label : unique_labels) {
        bool is_irrelevant = true;
        for (auto it = ts.begin(); it != ts.end(); ++it) {
            for (const auto& trans : (*it).get_transitions()) {
                if (trans.src != trans.target) {
                    is_irrelevant = false;
                    break;
                }
            }
            if (!is_irrelevant) break;
        }
        if (is_irrelevant) {
            irrelevant_labels++;
        }
    }
    features[6] = (label_count > 0) ?
        static_cast<float>(irrelevant_labels) / label_count : 0.0f;

    // Feature 7: Reachability from Init
    int init_reachable = 0;
    for (size_t i = 0; i < init_distances.size(); ++i) {
        if (init_distances[i] != INF && init_distances[i] >= 0) {
            init_reachable++;
        }
    }
    features[7] = (ts_size > 0) ?
        static_cast<float>(init_reachable) / ts_size : 0.0f;

    // Feature 8: Goal Reachability from Init
    int goal_reachable = 0;
    for (size_t i = 0; i < goal_distances.size(); ++i) {
        if (goal_distances[i] != INF && goal_distances[i] >= 0) {
            goal_reachable++;
        }
    }
    features[8] = (ts_size > 0) ?
        static_cast<float>(goal_reachable) / ts_size : 0.0f;

    // ========================================================================
    // CLAMP ALL FEATURES TO [0, 1]
    // ========================================================================
    for (int i = 0; i < NODE_FEATURE_DIM; ++i) {
        if (std::isnan(features[i]) || std::isinf(features[i])) {
            features[i] = 0.0f;
        }
        features[i] = std::max(0.0f, std::min(1.0f, features[i]));
    }

    return features;
}

// ============================================================================
// EDGE FEATURES COMPUTATION (C++ SIDE)
// ============================================================================

json compute_edge_features(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id) {

    json edge_feats;

    // Validate indices
    if (!fts.is_active(ts1_id) || !fts.is_active(ts2_id)) {
        edge_feats["error"] = "Invalid transition system indices";
        return edge_feats;
    }

    const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
    const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

    // ========================================================================
    // FEATURE GROUP 1: LABEL SYNCHRONIZATION (Features 0-2)
    // ========================================================================

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

    // Shared labels
    std::set<int> shared_labels;
    std::set_intersection(labels1.begin(), labels1.end(),
                          labels2.begin(), labels2.end(),
                          std::inserter(shared_labels, shared_labels.begin()));

    int total_labels = static_cast<int>(labels1.size() + labels2.size() - shared_labels.size());

    // Feature 0: Label Jaccard similarity
    double jaccard = (total_labels > 0) ?
        static_cast<double>(shared_labels.size()) / total_labels : 0.0;
    edge_feats["label_jaccard"] = jaccard;

    // Feature 1: Shared label ratio
    int min_labels = static_cast<int>(std::min(labels1.size(), labels2.size()));
    double shared_ratio = (min_labels > 0) ?
        static_cast<double>(shared_labels.size()) / min_labels : 0.0;
    edge_feats["shared_label_ratio"] = shared_ratio;

    // Feature 2: Synchronization factor estimate
    double sync_factor = 1.0 - jaccard;
    edge_feats["sync_factor"] = sync_factor;

    // ========================================================================
    // FEATURE GROUP 2: PRODUCT SIZE & STRUCTURAL (Features 3-5)
    // ========================================================================

    int ts1_size = ts1.get_size();
    int ts2_size = ts2.get_size();
    int max_product = ts1_size * ts2_size;

    // Feature 3: Product size (log normalized)
    edge_feats["product_size_log"] = std::log10(static_cast<double>(max_product) + 1);

    const auto& vars1 = ts1.get_incorporated_variables();
    const auto& vars2 = ts2.get_incorporated_variables();

    // Feature 4: Shares variables?
    std::set<int> vars1_set(vars1.begin(), vars1.end());
    bool shares_vars = false;
    for (int v : vars2) {
        if (vars1_set.count(v)) {
            shares_vars = true;
            break;
        }
    }
    edge_feats["shares_variables"] = shares_vars ? 1.0 : 0.0;

    // Feature 5: Combined variable count (normalized)
    std::set<int> all_vars(vars1.begin(), vars1.end());
    all_vars.insert(vars2.begin(), vars2.end());
    edge_feats["combined_var_count"] = static_cast<double>(all_vars.size()) / 20.0;

    // ========================================================================
    // FEATURE GROUP 3: HEURISTIC QUALITY (Features 6-8)
    // ========================================================================

    const Distances& dist1 = fts.get_distances(ts1_id);
    const Distances& dist2 = fts.get_distances(ts2_id);

    int init1 = ts1.get_init_state();
    int init2 = ts2.get_init_state();

    const auto& goal_dist1 = dist1.get_goal_distances();
    const auto& goal_dist2 = dist2.get_goal_distances();

    int h1 = (init1 >= 0 && init1 < static_cast<int>(goal_dist1.size())) ?
             goal_dist1[init1] : INF;
    int h2 = (init2 >= 0 && init2 < static_cast<int>(goal_dist2.size())) ?
             goal_dist2[init2] : INF;

    bool h1_valid = (h1 != INF && h1 >= 0);
    bool h2_valid = (h2 != INF && h2 >= 0);

    // Feature 6: Combined h* (additive for admissible heuristics)
    int combined_h = 0;
    if (h1_valid) combined_h += h1;
    if (h2_valid) combined_h += h2;
    edge_feats["combined_h_star"] = std::min(1.0, static_cast<double>(combined_h) / 1000.0);

    // Feature 7: H-value compatibility
    if (h1_valid && h2_valid && (h1 + h2) > 0) {
        edge_feats["h_compatibility"] = 1.0 - std::abs(h1 - h2) / static_cast<double>(h1 + h2);
    } else {
        edge_feats["h_compatibility"] = 0.5;
    }

    // Feature 8: Both solvable?
    edge_feats["both_solvable"] = (h1_valid && h2_valid) ? 1.0 : 0.0;

    // ========================================================================
    // FEATURE GROUP 4: STRUCTURAL COMPATIBILITY (Features 9-10)
    // ========================================================================

    // Feature 9: Size ratio (prefer balanced merges)
    double size_ratio = static_cast<double>(std::min(ts1_size, ts2_size)) /
                        std::max(ts1_size, ts2_size);
    edge_feats["size_balance"] = size_ratio;

    // Count transitions for density
    int ts1_transitions = 0;
    int ts2_transitions = 0;
    for (auto it = ts1.begin(); it != ts1.end(); ++it) {
        ts1_transitions += static_cast<int>((*it).get_transitions().size());
    }
    for (auto it = ts2.begin(); it != ts2.end(); ++it) {
        ts2_transitions += static_cast<int>((*it).get_transitions().size());
    }

    double density1 = (ts1_size > 0) ?
        static_cast<double>(ts1_transitions) / ts1_size : 0.0;
    double density2 = (ts2_size > 0) ?
        static_cast<double>(ts2_transitions) / ts2_size : 0.0;

    // Feature 10: Density ratio
    double density_ratio = (std::max(density1, density2) > 0) ?
        std::min(density1, density2) / std::max(density1, density2) : 1.0;
    edge_feats["density_ratio"] = density_ratio;

    // ========================================================================
    // FEATURE GROUP 5: CRITICAL NEW FEATURES (Features 11-14)
    // ========================================================================
    // From Nissim et al. & Helmert et al. papers

    // Feature 11: Operator Projection Score (CRITICAL!)
    double opp_score = OperatorProjectionAnalyzer::compute_opp_score(fts, ts1_id, ts2_id);
    edge_feats["operator_projection_score"] = opp_score;

    // Feature 12: Label Combinability Score (CRITICAL!)
    double label_comb = LabelCombinaibilityAnalyzer::compute_label_combinability_score(fts, ts1_id, ts2_id);
    edge_feats["label_combinability_score"] = label_comb;

    // Feature 13: Greedy Bisimulation Error
    double gb_error = GreedyBisimulationAnalyzer::compute_greedy_bisimulation_error(fts, ts1_id, ts2_id);
    edge_feats["gb_error"] = gb_error;

    // Feature 14: Causal Graph Distance / Proximity
    double causal_proximity = CausalGraphAnalyzer::compute_causal_proximity_score(fts, ts1_id, ts2_id);
    edge_feats["causal_proximity"] = causal_proximity;

    return edge_feats;
}

std::vector<double> edge_features_to_vector(const json& edge_feats) {
    std::vector<double> vec(EDGE_FEATURE_DIM, 0.0);  // NOW 11 features

    // Map: keep only these 11 features in order
    vec[0] = edge_feats.value("label_jaccard", 0.0);
    vec[1] = edge_feats.value("shared_label_ratio", 0.0);
    vec[2] = edge_feats.value("product_size_log", 0.0) / 10.0;
    vec[3] = edge_feats.value("combined_h_star", 0.0);
    vec[4] = edge_feats.value("h_compatibility", 0.5);
    vec[5] = edge_feats.value("both_solvable", 0.0);
    vec[6] = edge_feats.value("density_ratio", 0.0);
    vec[7] = edge_feats.value("operator_projection_score", 0.0);
    vec[8] = edge_feats.value("label_combinability_score", 0.0);
    vec[9] = edge_feats.value("gb_error", 0.5);
    vec[10] = edge_feats.value("causal_proximity", 0.0);

    // Clamp all to [0, 1]
    for (int i = 0; i < EDGE_FEATURE_DIM; ++i) {
        if (std::isnan(vec[i]) || std::isinf(vec[i])) {
            vec[i] = 0.0;
        }
        vec[i] = std::max(0.0, std::min(1.0, vec[i]));
    }

    return vec;
}

// ============================================================================
// COMPREHENSIVE A* SEARCH SIGNALS WITH H* PRESERVATION
// ============================================================================

json compute_comprehensive_astar_signals(
    const FactoredTransitionSystem& fts,
    const std::vector<int>& active_indices,
    int iteration) {

    json signals;

    // ========================================================================
    // AGGREGATE STATISTICS ACROSS ALL ACTIVE TRANSITION SYSTEMS
    // ========================================================================

    int total_states = 0;
    int total_reachable = 0;
    int total_transitions = 0;
    int total_goal_states = 0;
    int total_dead_ends = 0;

    // For f-value analysis
    std::vector<int> all_f_values;
    std::vector<double> all_h_values;

    // For search efficiency
    int total_nodes_expanded = 0;
    double sum_branching_factors = 0.0;
    int systems_with_branching = 0;

    // For solution quality
    int best_solution_cost = INF;
    bool any_solution_found = false;
    int total_search_depth = 0;
    int systems_with_depth = 0;

    // ========================================================================
    // H* TRACKING (THE MOST IMPORTANT SIGNAL!)
    // ========================================================================

    int current_combined_h_star = 0;
    std::map<int, int> current_h_star_values;

    // Reset tracking if this is a new episode
    if (iteration <= 0 || iteration != prev_iteration + 1) {
        prev_total_states = 0;
        prev_f_values.clear();
        prev_combined_h_star = 0;
        prev_h_star_values.clear();
        prev_iteration = -2;
    }

    // ========================================================================
    // COMPUTE SIGNALS FOR EACH ACTIVE TRANSITION SYSTEM
    // ========================================================================

    for (int idx : active_indices) {
        if (!fts.is_active(idx)) continue;

        const TransitionSystem& ts = fts.get_transition_system(idx);
        const Distances& distances = fts.get_distances(idx);

        int ts_size = ts.get_size();
        total_states += ts_size;

        // Get distances
        const auto& init_dist = distances.get_init_distances();
        const auto& goal_dist = distances.get_goal_distances();

        // ====================================================================
        // TRACK H* FOR THIS TS (init state h-value)
        // ====================================================================
        int init_state = ts.get_init_state();
        int h_star = INF;
        if (init_state >= 0 && init_state < static_cast<int>(goal_dist.size())) {
            h_star = goal_dist[init_state];
        }
        if (h_star != INF && h_star >= 0) {
            current_h_star_values[idx] = h_star;
            current_combined_h_star += h_star;
        }

        // Count transitions
        int ts_transitions = 0;
        for (auto it = ts.begin(); it != ts.end(); ++it) {
            ts_transitions += static_cast<int>((*it).get_transitions().size());
        }
        total_transitions += ts_transitions;

        // Count goal states
        for (int i = 0; i < ts_size; ++i) {
            if (ts.is_goal_state(i)) {
                total_goal_states++;
            }
        }

        // Compute f-values and analyze reachability
        int reachable_count = 0;
        int best_goal_f = INF;
        double sum_h = 0.0;
        int max_depth = 0;

        for (size_t i = 0; i < init_dist.size() && i < goal_dist.size(); ++i) {
            // Track dead-ends
            if (goal_dist[i] == INF && init_dist[i] != INF) {
                total_dead_ends++;
            }

            if (init_dist[i] != INF && goal_dist[i] != INF) {
                reachable_count++;
                total_nodes_expanded++;

                int f = init_dist[i] + goal_dist[i];
                all_f_values.push_back(f);

                if (goal_dist[i] < 1000000) {
                    all_h_values.push_back(static_cast<double>(goal_dist[i]));
                    sum_h += goal_dist[i];
                }

                if (init_dist[i] > max_depth && init_dist[i] < INF) {
                    max_depth = init_dist[i];
                }

                if (ts.is_goal_state(static_cast<int>(i))) {
                    if (f < best_goal_f) {
                        best_goal_f = f;
                    }
                    any_solution_found = true;
                }
            }
        }

        total_reachable += reachable_count;

        // Branching factor
        if (reachable_count > 0 && ts_transitions > 0) {
            double bf = static_cast<double>(ts_transitions) / static_cast<double>(reachable_count);
            if (!std::isnan(bf) && !std::isinf(bf) && bf >= 1.0) {
                sum_branching_factors += bf;
                systems_with_branching++;
            }
        }

        // Search depth
        if (reachable_count > 0) {
            total_search_depth += max_depth;
            systems_with_depth++;
        }

        // Best solution cost
        if (best_goal_f < best_solution_cost) {
            best_solution_cost = best_goal_f;
        }
    }

    // ========================================================================
    // COMPUTE H* PRESERVATION (THE KEY METRIC!)
    // ========================================================================

    double h_star_preservation = 1.0;
    int h_star_before = prev_combined_h_star;
    int h_star_after = current_combined_h_star;

    if (prev_combined_h_star > 0) {
        // h* should be preserved or improved (higher = better heuristic)
        // Ratio > 1 means h* improved, < 1 means it degraded
        h_star_preservation = static_cast<double>(h_star_after) / prev_combined_h_star;
        // Clamp to reasonable range
        h_star_preservation = std::max(0.0, std::min(2.0, h_star_preservation));
    }

    signals["h_star_before"] = h_star_before;
    signals["h_star_after"] = h_star_after;
    signals["h_star_preservation"] = h_star_preservation;

    // ========================================================================
    // COMPUTE SHRINKABILITY RATIO
    // ========================================================================

    double shrinkability = 0.0;
    int theoretical_product_size = 0;

    // Estimate theoretical product size from previous state
    if (prev_total_states > 0 && total_states > 0) {
        // Simple heuristic: if we merged, theoretical = prev_total * some factor
        theoretical_product_size = prev_total_states;  // Simplified
        if (theoretical_product_size > 0) {
            shrinkability = 1.0 - static_cast<double>(total_states) / theoretical_product_size;
            shrinkability = std::max(-1.0, std::min(1.0, shrinkability));
        }
    }

    signals["theoretical_product_size"] = theoretical_product_size;
    signals["merged_size"] = total_states;
    signals["shrinkability"] = shrinkability;

    // ========================================================================
    // COMPUTE DEAD-END RATIO
    // ========================================================================

    double dead_end_ratio = (total_reachable > 0) ?
        static_cast<double>(total_dead_ends) / total_reachable : 0.0;
    signals["dead_end_ratio"] = dead_end_ratio;
    signals["total_dead_ends"] = total_dead_ends;

    // ========================================================================
    // COMPUTE AGGREGATED SIGNALS (existing logic)
    // ========================================================================

    // --- SEARCH EFFICIENCY SIGNALS ---
    signals["nodes_expanded"] = total_nodes_expanded;

    double avg_search_depth = (systems_with_depth > 0) ?
        static_cast<double>(total_search_depth) / systems_with_depth : 0.0;
    signals["search_depth"] = avg_search_depth;

    double avg_branching_factor = (systems_with_branching > 0) ?
        sum_branching_factors / systems_with_branching : 1.0;
    signals["branching_factor"] = avg_branching_factor;

    double search_efficiency = (total_states > 0) ?
        1.0 - (static_cast<double>(total_nodes_expanded) / total_states) : 0.0;
    search_efficiency = std::max(0.0, std::min(1.0, search_efficiency));
    signals["search_efficiency_score"] = search_efficiency;

    // --- SOLUTION QUALITY SIGNALS ---
    signals["solution_found"] = any_solution_found;
    signals["solution_cost"] = any_solution_found ? best_solution_cost : 0;

    double solution_quality = 0.0;
    if (any_solution_found && best_solution_cost < INF) {
        solution_quality = 1.0 - std::min(1.0, static_cast<double>(best_solution_cost) / 1000.0);
    }
    signals["solution_quality_score"] = solution_quality;

    // --- STATE CONTROL SIGNALS ---
    signals["states_before"] = prev_total_states;
    signals["states_after"] = total_states;

    int delta_states = total_states - prev_total_states;
    signals["delta_states"] = delta_states;

    double state_explosion_penalty = 0.0;
    if (prev_total_states > 0) {
        double ratio = static_cast<double>(total_states) / prev_total_states;
        if (ratio > 1.0) {
            state_explosion_penalty = std::min(1.0, (ratio - 1.0));
        }
    }
    signals["state_explosion_penalty"] = state_explosion_penalty;

    double transition_density = (total_states > 0) ?
        static_cast<double>(total_transitions) / total_states : 0.0;
    signals["transition_density"] = transition_density;

    double reachability_ratio = (total_states > 0) ?
        static_cast<double>(total_reachable) / total_states : 0.0;
    signals["reachability_ratio"] = reachability_ratio;

    double state_control_score = reachability_ratio * (1.0 - state_explosion_penalty);
    signals["state_control_score"] = state_control_score;

    // --- F-VALUE STABILITY SIGNALS ---
    double f_value_stability = 1.0;
    int num_significant_f_changes = 0;
    double avg_f_change = 0.0;
    double max_f_change = 0.0;
    double f_preservation_score = 1.0;

    if (!prev_f_values.empty() && !all_f_values.empty()) {
        double prev_mean = 0.0, curr_mean = 0.0;
        double prev_std = 0.0, curr_std = 0.0;

        for (int f : prev_f_values) {
            if (f < INF) prev_mean += f;
        }
        prev_mean /= prev_f_values.size();

        for (int f : prev_f_values) {
            if (f < INF) {
                prev_std += (f - prev_mean) * (f - prev_mean);
            }
        }
        prev_std = std::sqrt(prev_std / prev_f_values.size());

        for (int f : all_f_values) {
            if (f < INF) curr_mean += f;
        }
        curr_mean /= all_f_values.size();

        for (int f : all_f_values) {
            if (f < INF) {
                curr_std += (f - curr_mean) * (f - curr_mean);
            }
        }
        curr_std = std::sqrt(curr_std / all_f_values.size());

        if (prev_mean > 0) {
            double mean_change = std::abs(curr_mean - prev_mean) / prev_mean;
            f_value_stability = 1.0 - std::min(1.0, mean_change);
        }

        if (prev_mean > 0 && std::abs(curr_mean - prev_mean) / prev_mean > 0.1) {
            num_significant_f_changes = 1;
        }

        avg_f_change = std::abs(curr_mean - prev_mean);
        max_f_change = std::max(std::abs(curr_mean - prev_mean),
                                std::abs(curr_std - prev_std));

        double std_ratio = (prev_std > 0) ? curr_std / prev_std : 1.0;
        f_preservation_score = f_value_stability *
            std::min(1.0, 1.0 / (1.0 + std::abs(1.0 - std_ratio)));
    }

    signals["f_value_stability"] = f_value_stability;
    signals["num_significant_f_changes"] = num_significant_f_changes;
    signals["avg_f_change"] = avg_f_change;
    signals["max_f_change"] = max_f_change;
    signals["f_preservation_score"] = f_preservation_score;

    double f_stability_score = (f_value_stability + f_preservation_score) / 2.0;
    signals["f_stability_score"] = f_stability_score;

    // --- ADDITIONAL RAW METRICS ---
    signals["total_states"] = total_states;
    signals["total_reachable"] = total_reachable;
    signals["total_transitions"] = total_transitions;
    signals["total_goal_states"] = total_goal_states;
    signals["num_active_systems"] = static_cast<int>(active_indices.size());
    signals["is_solvable"] = any_solution_found || (total_reachable > 0 && total_goal_states > 0);

    if (!all_f_values.empty()) {
        signals["f_stats"] = compute_f_statistics(all_f_values);
    }

    // ========================================================================
    // COMPUTE WEIGHTED REWARD COMPONENTS (Updated with h* focus)
    // ========================================================================

    // 1. H* Preservation (w=0.40) - THE PRIMARY SIGNAL
    double w_h_star = 0.40;
    double h_star_component = std::min(1.0, h_star_preservation);
    signals["weighted_h_preservation"] = h_star_component;

    // 2. Shrinkability (w=0.25) - How well did shrinking work?
    double w_shrink = 0.25;
    double shrink_component = std::max(0.0, shrinkability + 0.5);  // Shift to [0, 1]
    signals["weighted_shrinkability"] = shrink_component;

    // 3. State Control (w=0.20) - Avoid explosion
    double w_state_ctrl = 0.20;
    double state_ctrl_component = state_control_score;
    signals["weighted_state_control"] = state_ctrl_component;

    // 4. Solvability Maintenance (w=0.15)
    double w_solv = 0.15;
    double solv_component = signals["is_solvable"].get<bool>() ? 1.0 : 0.0;
    signals["weighted_solvability"] = solv_component;

    // Final weighted score
    double weighted_total = w_h_star * h_star_component +
                           w_shrink * shrink_component +
                           w_state_ctrl * state_ctrl_component +
                           w_solv * solv_component;
    signals["weighted_total_score"] = weighted_total;

    // Store weights for Python reference
    signals["weights"] = {
        {"h_preservation", w_h_star},
        {"shrinkability", w_shrink},
        {"state_control", w_state_ctrl},
        {"solvability", w_solv}
    };

    // ========================================================================
    // ADDITIONAL DETAILED SIGNALS FOR REWARD SHAPING
    // ========================================================================

    // Growth ratio (explicit for Python)
    double growth_ratio = 1.0;
    if (prev_total_states > 0) {
        growth_ratio = static_cast<double>(total_states) / prev_total_states;
    }
    signals["growth_ratio"] = growth_ratio;

    // Is this merge "good" by simple heuristics?
    bool is_good_merge = (h_star_preservation >= 0.95) &&
                         (growth_ratio < 5.0) &&
                         (dead_end_ratio < 0.3) &&
                         signals["is_solvable"].get<bool>();
    signals["is_good_merge"] = is_good_merge;

    // Is this merge "bad"?
    bool is_bad_merge = (h_star_preservation < 0.8) ||
                        (growth_ratio > 10.0) ||
                        (dead_end_ratio > 0.5) ||
                        !signals["is_solvable"].get<bool>();
    signals["is_bad_merge"] = is_bad_merge;

    // Merge quality score (simple heuristic: 0-1 range)
    double merge_quality = 0.0;
    merge_quality += 0.35 * std::min(1.0, h_star_preservation);  // h* component
    merge_quality += 0.25 * std::max(0.0, 1.0 - state_explosion_penalty);  // explosion
    merge_quality += 0.20 * (shrinkability + 1.0) / 2.0;  // shrinkability normalized
    merge_quality += 0.10 * (1.0 - dead_end_ratio);  // dead-end avoidance
    merge_quality += 0.10 * reachability_ratio;  // reachability
    signals["merge_quality_score"] = std::max(0.0, std::min(1.0, merge_quality));

    // Absolute h* change (useful for debugging)
    signals["h_star_delta"] = h_star_after - h_star_before;

    // Was there actual shrinking?
    signals["did_shrink"] = (total_states < prev_total_states);

    // Transition growth (another explosion indicator)
    double transition_growth = 1.0;
    // Note: would need to track prev_transitions for this
    signals["transition_growth"] = transition_growth;

    std::cout << "[SIGNALS::DEBUG]   growth_ratio=" << growth_ratio
              << ", is_good=" << is_good_merge
              << ", is_bad=" << is_bad_merge << std::endl;

    // ========================================================================
    // UPDATE TRACKING STATE FOR NEXT ITERATION
    // ========================================================================

    prev_total_states = total_states;
    prev_f_values = all_f_values;
    prev_combined_h_star = current_combined_h_star;
    prev_h_star_values = current_h_star_values;
    prev_iteration = iteration;

    // Debug logging
    std::cout << "[SIGNALS::DEBUG] iteration=" << iteration << std::endl;
    std::cout << "[SIGNALS::DEBUG]   total_states=" << total_states
              << " (prev=" << signals["states_before"].get<int>() << ")" << std::endl;
    std::cout << "[SIGNALS::DEBUG]   h_star=" << current_combined_h_star
              << " (prev=" << h_star_before << ", preservation=" << h_star_preservation << ")" << std::endl;
    std::cout << "[SIGNALS::DEBUG]   reachability=" << reachability_ratio
              << ", dead_ends=" << total_dead_ends << std::endl;
    std::cout << "[SIGNALS::DEBUG]   weighted_total=" << weighted_total << std::endl;

    return signals;
}

// ============================================================================
// INTERACTION GRAPH EDGES
// ============================================================================

std::vector<std::pair<int, int>> build_interaction_graph_edges(
    const FactoredTransitionSystem& fts) {

    std::vector<std::pair<int, int>> edges;

    std::vector<int> active_indices;
    int num_systems = fts.get_size();

    for (int i = 0; i < num_systems; ++i) {
        if (fts.is_active(i)) {
            active_indices.push_back(i);
        }
    }

    int num_active = static_cast<int>(active_indices.size());
    bool use_all_pairs = (num_active <= 50);

    if (use_all_pairs) {
        for (size_t i = 0; i < active_indices.size(); ++i) {
            for (size_t j = i + 1; j < active_indices.size(); ++j) {
                int idx_i = active_indices[i];
                int idx_j = active_indices[j];
                edges.push_back(std::make_pair(idx_i, idx_j));
            }
        }
    } else {
        // Use variable sharing for larger graphs
        for (int i = 0; i < num_systems; ++i) {
            if (!fts.is_active(i)) continue;

            const TransitionSystem& ts_i = fts.get_transition_system(i);
            const auto& vars_i = ts_i.get_incorporated_variables();
            std::set<int> vars_i_set(vars_i.begin(), vars_i.end());

            for (int j = i + 1; j < num_systems; ++j) {
                if (!fts.is_active(j)) continue;

                const TransitionSystem& ts_j = fts.get_transition_system(j);
                const auto& vars_j = ts_j.get_incorporated_variables();

                bool shares_variables = false;
                for (int var : vars_j) {
                    if (vars_i_set.count(var) > 0) {
                        shares_variables = true;
                        break;
                    }
                }

                if (shares_variables) {
                    edges.push_back(std::make_pair(i, j));
                }
            }
        }

        if (edges.empty() && num_active >= 2) {
            for (size_t i = 0; i + 1 < active_indices.size(); ++i) {
                edges.push_back(std::make_pair(active_indices[i], active_indices[i + 1]));
            }
        }
    }

    return edges;
}

// ============================================================================
// MAIN GNN OBSERVATION EXPORT (Updated with Edge Features)
// ============================================================================

json export_gnn_observation(
    const FactoredTransitionSystem& fts,
    int iteration) {

    json observation;
    observation["iteration"] = iteration;
    observation["timestamp"] = static_cast<long>(std::time(nullptr));

    try {
        // PHASE 1: Get active transition systems
        std::vector<int> active_indices;
        int num_systems = fts.get_size();

        for (int i = 0; i < num_systems; ++i) {
            if (fts.is_active(i)) {
                active_indices.push_back(i);
            }
        }

        int num_active = static_cast<int>(active_indices.size());
        observation["num_active_systems"] = num_active;

        std::cout << "[SIGNALS::GNN] Active systems: " << num_active << std::endl;

        // PHASE 2: Compute statistics for normalization
        int max_state_count = 1;
        int global_max_labels = 1;

        for (int idx : active_indices) {
            const TransitionSystem& ts = fts.get_transition_system(idx);
            int ts_size = ts.get_size();
            max_state_count = std::max(max_state_count, ts_size);

            std::set<int> unique_labels;
            for (auto it = ts.begin(); it != ts.end(); ++it) {
                const auto& label_group = (*it).get_label_group();
                for (int label : label_group) {
                    unique_labels.insert(label);
                }
            }
            global_max_labels = std::max(global_max_labels, static_cast<int>(unique_labels.size()));
        }

        // PHASE 3: Compute ENHANCED node features (15 features)
        json x_features = json::array();

        for (int idx : active_indices) {
            const TransitionSystem& ts = fts.get_transition_system(idx);
            const Distances& distances = fts.get_distances(idx);

            auto features = compute_gnn_node_features_enhanced(
                ts, distances, iteration, max_state_count, global_max_labels
            );

            json node_features = json::array();
            for (float f : features) {
                node_features.push_back(f);
            }
            x_features.push_back(node_features);
        }

        observation["x"] = x_features;
        observation["node_feature_dim"] = NODE_FEATURE_DIM;

        std::cout << "[SIGNALS::GNN] Computed " << NODE_FEATURE_DIM
                  << " features for " << num_active << " nodes" << std::endl;

        // PHASE 4: Build edges WITH FEATURES (C++ computed!)
        auto edges = build_interaction_graph_edges(fts);

        json edge_index = json::array();
        json sources = json::array();
        json targets = json::array();
        json edge_features = json::array();  // NEW!

        for (const auto& [src, tgt] : edges) {
            sources.push_back(src);
            targets.push_back(tgt);

            // Compute edge features for this merge candidate
            json ef = compute_edge_features(fts, src, tgt);

            // Convert to feature vector
            std::vector<double> edge_feat_vec = edge_features_to_vector(ef);
            edge_features.push_back(edge_feat_vec);
        }

        edge_index.push_back(sources);
        edge_index.push_back(targets);

        observation["edge_index"] = edge_index;
        observation["edge_features"] = edge_features;  // NEW!
        observation["edge_feature_dim"] = EDGE_FEATURE_DIM;
        observation["num_edges"] = static_cast<int>(edges.size());

        std::cout << "[SIGNALS::GNN] Interaction graph: " << edges.size()
                  << " edges with " << EDGE_FEATURE_DIM << " features each" << std::endl;

        // PHASE 5: Compute COMPREHENSIVE A* SEARCH SIGNALS
//        json reward_signals = compute_comprehensive_astar_signals(fts, active_indices, iteration);
//        observation["reward_signals"] = reward_signals;


        // PHASE 5: Compute COMPREHENSIVE MERGE QUALITY SIGNALS
        json reward_signals = compute_comprehensive_astar_signals(fts, active_indices, iteration);

        // After computing reward_signals, add enhanced signals
        if (active_indices.size() >= 2) {
            // Compute comprehensive merge quality analysis for the candidate merge pair
            json enhanced_signals = MergeQualityAnalyzer::compute_comprehensive_merge_signals(
                fts, active_indices[0], active_indices[1], iteration
            );

            // Merge enhanced signals into reward_signals
            for (auto& [key, value] : enhanced_signals.items()) {
                if (key != "iteration" && key != "ts1_id" && key != "ts2_id") {
                    reward_signals[key] = value;
                }
            }

            // Extract and promote top-level scores for easy access
            try {
                if (enhanced_signals.contains("operator_projection") &&
                    enhanced_signals["operator_projection"].contains("opp_score")) {
                    reward_signals["opp_score"] = enhanced_signals["operator_projection"]["opp_score"];
                }

                if (enhanced_signals.contains("label_combinability") &&
                    enhanced_signals["label_combinability"].contains("combinability_score")) {
                    reward_signals["label_combinability_score"] =
                        enhanced_signals["label_combinability"]["combinability_score"];
                }

                if (enhanced_signals.contains("greedy_bisimulation") &&
                    enhanced_signals["greedy_bisimulation"].contains("gb_error")) {
                    reward_signals["gb_error"] = enhanced_signals["greedy_bisimulation"]["gb_error"];
                }

                if (enhanced_signals.contains("causal_graph") &&
                    enhanced_signals["causal_graph"].contains("causal_proximity")) {
                    reward_signals["causal_proximity_score"] =
                        enhanced_signals["causal_graph"]["causal_proximity"];
                }

                if (enhanced_signals.contains("label_support") &&
                    enhanced_signals["label_support"].contains("support_overlap")) {
                    reward_signals["label_support_overlap"] =
                        enhanced_signals["label_support"]["support_overlap"];
                }

                if (enhanced_signals.contains("landmark_preservation") &&
                    enhanced_signals["landmark_preservation"].contains("landmark_preservation")) {
                    reward_signals["landmark_preservation"] =
                        enhanced_signals["landmark_preservation"]["landmark_preservation"];
                }

                if (enhanced_signals.contains("transition_explosion") &&
                    enhanced_signals["transition_explosion"].contains("density_ratio")) {
                    reward_signals["transition_density_ratio"] =
                        enhanced_signals["transition_explosion"]["density_ratio"];
                }

                std::cout << "[SIGNALS::GNN] Enhanced merge quality signals computed for pair ("
                          << active_indices[0] << ", " << active_indices[1] << ")" << std::endl;

            } catch (const std::exception& e) {
                std::cerr << "[SIGNALS::GNN] Warning: Failed to extract enhanced signals: "
                          << e.what() << std::endl;
            }
        }

        observation["reward_signals"] = reward_signals;


        // PHASE 6: Terminal condition
        observation["is_terminal"] = (num_active <= 1);

        std::cout << "[SIGNALS::GNN] ✅ Observation exported for iteration "
                  << iteration << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS::GNN] ERROR: " << e.what() << std::endl;
        observation["error"] = std::string(e.what());

        // Export minimal reward signals even on error
        json reward_signals;
        reward_signals["weighted_total_score"] = 0.0;
        reward_signals["h_star_preservation"] = 1.0;
        reward_signals["is_solvable"] = false;
        observation["reward_signals"] = reward_signals;
    }

    return observation;
}


// ============================================================================
// LEGACY COMPUTE_ASTAR_SIGNALS (kept for compatibility)
// ============================================================================

json compute_astar_signals(
    const FactoredTransitionSystem& fts,
    int merged_index,
    const std::vector<int>& init_distances,
    const std::vector<int>& goal_distances) {

    const TransitionSystem& ts = fts.get_transition_system(merged_index);

    int reachable_count = 0;
    int unreachable_count = 0;
    long long sum_goal_dist = 0;
    int reachable_goal_count = 0;
    int best_goal_f = INF;

    for (size_t i = 0; i < init_distances.size(); ++i) {
        if (init_distances[i] != INF && goal_distances[i] != INF) {
            reachable_count++;
            sum_goal_dist += goal_distances[i];

            if (ts.is_goal_state(i)) {
                reachable_goal_count++;
                int f = init_distances[i] + goal_distances[i];
                if (f < best_goal_f) {
                    best_goal_f = f;
                }
            }
        } else {
            unreachable_count++;
        }
    }

    int search_depth = 0;
    if (reachable_goal_count > 0) {
        search_depth = static_cast<int>(std::round(
            static_cast<double>(sum_goal_dist) / reachable_goal_count
        ));
    }

    double branching_factor = 1.0;
    int num_transitions = 0;
    for (auto it = ts.begin(); it != ts.end(); ++it) {
        num_transitions += (*it).get_transitions().size();
    }

    if (reachable_count > 0 && num_transitions > 0) {
        branching_factor = static_cast<double>(num_transitions) /
                          static_cast<double>(reachable_count);

        if (std::isnan(branching_factor) || std::isinf(branching_factor)) {
            branching_factor = 1.0;
        }
        if (branching_factor < 1.0) {
            branching_factor = 1.0;
        }
    }

    bool solution_found = (best_goal_f != INF);

    json signals;
    signals["nodes_expanded"] = reachable_count;
    signals["unreachable_states"] = unreachable_count;
    signals["search_depth"] = search_depth;
    signals["branching_factor"] = branching_factor;
    signals["solution_cost"] = solution_found ? best_goal_f : 0;
    signals["solution_found"] = solution_found;

    return signals;
}

// ============================================================================
// PRODUCT STATE MAPPING
// ============================================================================

json build_product_mapping(int ts1_size, int ts2_size) {
    json mapping;

    int counter = 0;
    for (int s = 0; s < ts1_size * ts2_size; ++s) {
        int s1 = s / ts2_size;
        int s2 = s % ts2_size;

        mapping[std::to_string(s)] = {
            {"s1", s1},
            {"s2", s2}
        };

        counter++;

        if (counter > 100000) {
            mapping["_note"] = "Product mapping truncated (too large)";
            break;
        }
    }

    return mapping;
}

// ============================================================================
// MERGE BEFORE DATA EXPORT
// ============================================================================

json export_merge_before_data(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id,
    int iteration,
    bool shrunk,
    bool reduced) {

    json before_data;
    before_data["iteration"] = iteration;
    before_data["ts1_id"] = ts1_id;
    before_data["ts2_id"] = ts2_id;

    try {
        if (!fts.is_active(ts1_id)) {
            std::cerr << "[SIGNALS] ERROR: ts1_id=" << ts1_id << " is NOT active!" << std::endl;
            before_data["error"] = "ts1_id is not active";
            before_data["ts1_active"] = false;
            return before_data;
        }
        if (!fts.is_active(ts2_id)) {
            std::cerr << "[SIGNALS] ERROR: ts2_id=" << ts2_id << " is NOT active!" << std::endl;
            before_data["error"] = "ts2_id is not active";
            before_data["ts2_active"] = false;
            return before_data;
        }

        before_data["ts1_active"] = true;
        before_data["ts2_active"] = true;

        const TransitionSystem& ts1 = fts.get_transition_system(ts1_id);
        const TransitionSystem& ts2 = fts.get_transition_system(ts2_id);

        const Distances& dist1 = fts.get_distances(ts1_id);
        const Distances& dist2 = fts.get_distances(ts2_id);

        const auto& init1 = dist1.get_init_distances();
        const auto& goal1 = dist1.get_goal_distances();
        const auto& init2 = dist2.get_init_distances();
        const auto& goal2 = dist2.get_goal_distances();

        std::vector<int> f1(init1.size());
        std::vector<int> f2(init2.size());

        for (size_t i = 0; i < f1.size(); ++i) {
            if (i < init1.size() && i < goal1.size()) {
                if (init1[i] != INF && goal1[i] != INF) {
                    f1[i] = init1[i] + goal1[i];
                } else {
                    f1[i] = INF;
                }
            } else {
                f1[i] = INF;
            }
        }

        for (size_t j = 0; j < f2.size(); ++j) {
            if (j < init2.size() && j < goal2.size()) {
                if (init2[j] != INF && goal2[j] != INF) {
                    f2[j] = init2[j] + goal2[j];
                } else {
                    f2[j] = INF;
                }
            } else {
                f2[j] = INF;
            }
        }

        int ts1_transitions = 0;
        int ts2_transitions = 0;
        for (auto it = ts1.begin(); it != ts1.end(); ++it) {
            ts1_transitions += static_cast<int>((*it).get_transitions().size());
        }
        for (auto it = ts2.begin(); it != ts2.end(); ++it) {
            ts2_transitions += static_cast<int>((*it).get_transitions().size());
        }

        int ts1_goals = 0, ts2_goals = 0;
        for (int i = 0; i < ts1.get_size(); ++i) {
            if (ts1.is_goal_state(i)) ts1_goals++;
        }
        for (int j = 0; j < ts2.get_size(); ++j) {
            if (ts2.is_goal_state(j)) ts2_goals++;
        }

        before_data["ts1_size"] = static_cast<int>(f1.size());
        before_data["ts2_size"] = static_cast<int>(f2.size());
        before_data["expected_product_size"] =
            static_cast<int>(f1.size()) * static_cast<int>(f2.size());

        before_data["ts1_transitions"] = ts1_transitions;
        before_data["ts2_transitions"] = ts2_transitions;
        before_data["ts1_density"] =
            static_cast<double>(ts1_transitions) / std::max(static_cast<int>(f1.size()), 1);
        before_data["ts2_density"] =
            static_cast<double>(ts2_transitions) / std::max(static_cast<int>(f2.size()), 1);

        before_data["ts1_goal_states"] = ts1_goals;
        before_data["ts2_goal_states"] = ts2_goals;

        before_data["ts1_f_values"] = f1;
        before_data["ts2_f_values"] = f2;

        before_data["ts1_f_stats"] = compute_f_statistics(f1);
        before_data["ts2_f_stats"] = compute_f_statistics(f2);

        before_data["ts1_variables"] = ts1.get_incorporated_variables();
        before_data["ts2_variables"] = ts2.get_incorporated_variables();

        int product_size = static_cast<int>(f1.size()) * static_cast<int>(f2.size());
        if (product_size <= 100000) {
            before_data["product_mapping"] = build_product_mapping(
                static_cast<int>(f1.size()),
                static_cast<int>(f2.size())
            );
        } else {
            before_data["product_mapping_skipped"] = true;
            before_data["product_mapping_reason"] = "Product too large";
        }

        before_data["shrunk"] = shrunk;
        before_data["reduced"] = reduced;
        before_data["timestamp"] = static_cast<long>(std::time(nullptr));

        std::cout << "[SIGNALS] export_merge_before_data: ts1_size=" << f1.size()
                  << ", ts2_size=" << f2.size() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS] ERROR in export_merge_before_data: " << e.what() << std::endl;
        before_data["error"] = std::string(e.what());
    }

    return before_data;
}

// ============================================================================
// MERGE AFTER DATA EXPORT
// ============================================================================

json export_merge_after_data(
    const FactoredTransitionSystem& fts,
    int merged_index,
    int iteration) {

    json after_data;
    after_data["iteration"] = iteration;

    try {
        const TransitionSystem& ts = fts.get_transition_system(merged_index);
        const Distances& distances = fts.get_distances(merged_index);

        const auto& init_dist = distances.get_init_distances();
        const auto& goal_dist = distances.get_goal_distances();

        std::vector<int> f_after(init_dist.size());
        for (size_t s = 0; s < f_after.size(); ++s) {
            if (init_dist[s] != INF && goal_dist[s] != INF) {
                f_after[s] = init_dist[s] + goal_dist[s];
            } else {
                f_after[s] = INF;
            }
        }

        int reachable_count = 0, unreachable_count = 0;
        for (size_t i = 0; i < f_after.size(); ++i) {
            if (init_dist[i] != INF && goal_dist[i] != INF) {
                reachable_count++;
            } else {
                unreachable_count++;
            }
        }

        int num_goals = 0;
        for (int i = 0; i < ts.get_size(); ++i) {
            if (ts.is_goal_state(i)) num_goals++;
        }

        int merged_transitions = 0;
        for (auto it = ts.begin(); it != ts.end(); ++it) {
            merged_transitions += (*it).get_transitions().size();
        }

        after_data["merged_size"] = static_cast<int>(f_after.size());
        after_data["merged_goal_states"] = num_goals;
        after_data["merged_transitions"] = merged_transitions;
        after_data["merged_density"] =
            static_cast<double>(merged_transitions) /
            std::max(static_cast<int>(f_after.size()), 1);

        after_data["reachable_states"] = reachable_count;
        after_data["unreachable_states"] = unreachable_count;
        after_data["reachability_ratio"] =
            static_cast<double>(reachable_count) /
            std::max(static_cast<int>(f_after.size()), 1);

        after_data["f_values"] = f_after;
        after_data["f_stats"] = compute_f_statistics(f_after);

        after_data["shrinking_ratio"] =
            static_cast<double>(f_after.size()) /
            std::max(1, fts.get_transition_system(merged_index).get_size());

        json astar_signals = compute_astar_signals(
            fts, merged_index, init_dist, goal_dist
        );
        after_data["search_signals"] = astar_signals;

        after_data["timestamp"] = static_cast<long>(std::time(nullptr));

    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS] Error in export_merge_after_data: " << e.what() << std::endl;
        after_data["error"] = std::string(e.what());
    }

    return after_data;
}

// ============================================================================
// TRANSITION SYSTEM DEFINITION EXPORT
// ============================================================================

json export_ts_data(
    const FactoredTransitionSystem& fts,
    int ts_index,
    int iteration) {

    json ts_json;
    ts_json["iteration"] = iteration;

    try {
        const TransitionSystem& ts = fts.get_transition_system(ts_index);

        ts_json["ts_index"] = ts_index;
        ts_json["num_states"] = ts.get_size();
        ts_json["init_state"] = ts.get_init_state();

        std::vector<int> goal_states;
        for (int i = 0; i < ts.get_size(); ++i) {
            if (ts.is_goal_state(i)) {
                goal_states.push_back(i);
            }
        }
        ts_json["goal_states"] = goal_states;

        ts_json["incorporated_variables"] = ts.get_incorporated_variables();

        std::vector<json> transitions;
        int transition_count = 0;
        const int MAX_TRANSITIONS_TO_EXPORT = 10000;

        for (auto it = ts.begin(); it != ts.end(); ++it) {
            const auto& info = *it;
            const auto& label_group = info.get_label_group();
            const auto& trans_vec = info.get_transitions();

            for (int label : label_group) {
                for (const auto& trans : trans_vec) {
                    if (transition_count < MAX_TRANSITIONS_TO_EXPORT) {
                        transitions.push_back({
                            {"src", trans.src},
                            {"target", trans.target},
                            {"label", label}
                        });
                        transition_count++;
                    }
                }
            }
        }

        ts_json["transitions"] = transitions;

        if (transition_count >= MAX_TRANSITIONS_TO_EXPORT) {
            ts_json["_transitions_note"] = "Transitions truncated (too many)";
        }

        ts_json["timestamp"] = static_cast<long>(std::time(nullptr));

    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS] Error in export_ts_data: " << e.what() << std::endl;
        ts_json["error"] = std::string(e.what());
    }

    return ts_json;
}

// ============================================================================
// FD INDEX MAPPING EXPORT
// ============================================================================

json export_fd_index_mapping_data(
    const FactoredTransitionSystem& fts,
    int iteration) {

    json index_mapping;
    index_mapping["iteration"] = iteration;
    index_mapping["timestamp"] = static_cast<long>(std::time(nullptr));

    json systems = json::array();

    try {
        for (int fd_idx = 0; fd_idx < fts.get_size(); ++fd_idx) {
            if (!fts.is_active(fd_idx)) {
                continue;
            }

            const TransitionSystem& ts = fts.get_transition_system(fd_idx);

            json system_entry;
            system_entry["fd_index"] = fd_idx;
            system_entry["num_states"] = ts.get_size();
            system_entry["incorporated_variables"] = ts.get_incorporated_variables();
            system_entry["is_active"] = true;

            systems.push_back(system_entry);
        }

    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS] Error in export_fd_index_mapping_data: "
                  << e.what() << std::endl;
        index_mapping["error"] = std::string(e.what());
    }

    index_mapping["systems"] = systems;

    return index_mapping;
}

// ============================================================================
// ERROR SIGNAL EXPORT
// ============================================================================

void export_error_signal(
    int iteration,
    const std::string& error_message,
    const std::string& fd_output_dir) {

    try {
        json error_data;
        error_data["iteration"] = iteration;
        error_data["error"] = true;
        error_data["message"] = error_message;
        error_data["timestamp"] = static_cast<long>(std::time(nullptr));

        std::string error_path = fd_output_dir + "/gnn_error_" +
                                 std::to_string(iteration) + ".json";

        write_json_file_atomic(error_data, error_path);

        std::cerr << "[SIGNALS] Error signal exported: " << error_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[SIGNALS] Failed to export error signal: " << e.what() << std::endl;
    }
}

// ============================================================================
// MAIN EXPORT FUNCTION
// ============================================================================

void export_merge_signals(
    const FactoredTransitionSystem& fts,
    int merged_index,
    int ts1_id,
    int ts2_id,
    const std::string& fd_output_dir,
    int iteration,
    bool shrunk,
    bool reduced) {

    std::cout << "\n[SIGNALS] ========================================" << std::endl;
    std::cout << "[SIGNALS] Exporting signals for iteration " << iteration << std::endl;
    std::cout << "[SIGNALS] ========================================\n" << std::endl;

    try {
        std::cout << "[SIGNALS] 1. Exporting merge_before..." << std::endl;
        json before_data = export_merge_before_data(
            fts, ts1_id, ts2_id, iteration, shrunk, reduced
        );
        std::string before_path = fd_output_dir + "/merge_before_" +
                                  std::to_string(iteration) + ".json";
        write_json_file_atomic(before_data, before_path);

        std::cout << "[SIGNALS] 2. Exporting merge_after..." << std::endl;
        json after_data = export_merge_after_data(fts, merged_index, iteration);
        std::string after_path = fd_output_dir + "/merge_after_" +
                                 std::to_string(iteration) + ".json";
        write_json_file_atomic(after_data, after_path);

        std::cout << "[SIGNALS] 3. Exporting ts definition..." << std::endl;
        json ts_data = export_ts_data(fts, merged_index, iteration);
        std::string ts_path = fd_output_dir + "/ts_" +
                              std::to_string(iteration) + ".json";
        write_json_file_atomic(ts_data, ts_path);

        std::cout << "[SIGNALS] 4. Exporting FD index mapping..." << std::endl;
        json mapping_data = export_fd_index_mapping_data(fts, iteration);
        std::string mapping_path = fd_output_dir + "/fd_index_mapping_" +
                                   std::to_string(iteration) + ".json";
        write_json_file_atomic(mapping_data, mapping_path);

        std::cout << "\n[SIGNALS] ✅ All signals exported for iteration "
                  << iteration << "\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n[SIGNALS] ❌ CRITICAL ERROR exporting signals: "
                  << e.what() << std::endl;
        export_error_signal(iteration,
                           std::string("Signal export failed: ") + e.what(),
                           fd_output_dir);
        throw;
    }
}

}  // namespace merge_and_shrink