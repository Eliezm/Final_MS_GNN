#ifndef MERGE_AND_SHRINK_SIGNALS_H
#define MERGE_AND_SHRINK_SIGNALS_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace merge_and_shrink {

// Forward declarations
class FactoredTransitionSystem;
class TransitionSystem;
class Distances;
class Labels;

// ============================================================================
// FEATURE DIMENSIONS - MUST MATCH PYTHON
// ============================================================================

constexpr int NODE_FEATURE_DIM = 15;    // Expanded from 7 to 15
constexpr int EDGE_FEATURE_DIM = 10;    // New: C++ computed edge features

// ============================================================================
// GNN OBSERVATION EXPORT - THE FEATURE ENGINE
// ============================================================================

/**
 * Export observation in GNN format with comprehensive A* signals
 * Now includes edge features computed in C++
 */
json export_gnn_observation(
    const FactoredTransitionSystem& fts,
    int iteration
);

/**
 * Compute comprehensive A* search signals for reward calculation
 * Enhanced with h* preservation tracking
 */
json compute_comprehensive_astar_signals(
    const FactoredTransitionSystem& fts,
    const std::vector<int>& active_indices,
    int iteration
);

// ============================================================================
// ENHANCED FEATURE COMPUTATION
// ============================================================================

/**
 * Compute enhanced 15-dimensional node features
 *
 * Features 0-6: Original features
 * Features 7-14: New structural and label-based features
 */
std::vector<float> compute_gnn_node_features_enhanced(
    const TransitionSystem& ts,
    const Distances& distances,
    int iteration,
    int max_state_count,
    int global_max_labels
);

/**
 * Compute edge features for a merge candidate pair
 *
 * Returns 10 features including:
 * - Label synchronization (Jaccard, shared ratio, sync factor)
 * - Product size estimates
 * - Variable relationships
 * - Heuristic quality
 * - Structural compatibility
 */
json compute_edge_features(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id
);

/**
 * Convert edge feature JSON to vector for export
 */
std::vector<double> edge_features_to_vector(const json& edge_feats);

// ============================================================================
// MERGE DATA EXPORTS
// ============================================================================

json export_merge_before_data(
    const FactoredTransitionSystem& fts,
    int ts1_id,
    int ts2_id,
    int iteration,
    bool shrunk,
    bool reduced
);

json export_merge_after_data(
    const FactoredTransitionSystem& fts,
    int merged_index,
    int iteration
);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

std::vector<std::pair<int, int>> build_interaction_graph_edges(
    const FactoredTransitionSystem& fts
);

json export_ts_data(
    const FactoredTransitionSystem& fts,
    int ts_index,
    int iteration
);

json export_fd_index_mapping_data(
    const FactoredTransitionSystem& fts,
    int iteration
);

json compute_f_statistics(
    const std::vector<int>& distances,
    int unreachable_marker = 2147483647
);

json compute_astar_signals(
    const FactoredTransitionSystem& fts,
    int merged_index,
    const std::vector<int>& init_distances,
    const std::vector<int>& goal_distances
);

json build_product_mapping(int ts1_size, int ts2_size);

void write_json_file_atomic(
    const json& data,
    const std::string& file_path
);

std::string get_fd_output_directory();
std::string get_gnn_output_directory();

void export_error_signal(
    int iteration,
    const std::string& error_message,
    const std::string& fd_output_dir
);

void export_merge_signals(
    const FactoredTransitionSystem& fts,
    int merged_index,
    int ts1_id,
    int ts2_id,
    const std::string& fd_output_dir,
    int iteration,
    bool shrunk,
    bool reduced
);

}  // namespace merge_and_shrink

#endif