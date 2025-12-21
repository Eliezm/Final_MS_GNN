#ifndef MERGE_AND_SHRINK_IO_H
#define MERGE_AND_SHRINK_IO_H

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace merge_and_shrink {

// ============================================================================
// DIRECTORY PATHS - Always relative to CWD
// ============================================================================

inline std::string get_gnn_output_dir() {
    return std::filesystem::current_path() / "gnn_output";
}

inline std::string get_fd_output_dir() {
    return std::filesystem::current_path() / "fd_output";
}

// ============================================================================
// ATOMIC FILE I/O - C++ SIDE
// ============================================================================

void write_json_atomic(const json& obj, const std::string& final_path);

// ============================================================================
// MERGE SIGNAL EXPORT
// ============================================================================

struct MergeSignalData {
    int iteration;
    int ts1_id;
    int ts2_id;
    int ts1_size;
    int ts2_size;
    std::vector<int> ts1_f_values;
    std::vector<int> ts2_f_values;
    std::map<std::string, std::map<std::string, int>> product_mapping;
};

struct MergeResultData {
    int iteration;
    int merged_id;
    int merged_size;
    std::vector<int> f_values;
    int reachable_states;
    int unreachable_states;
    double reachability_ratio;

    // A* search signals
    int nodes_expanded;
    int search_depth;
    int solution_cost;
    double branching_factor;
    bool solution_found;
};

void export_merge_before(const MergeSignalData& data, const std::string& output_dir);
void export_merge_after(const MergeResultData& data, const std::string& output_dir);

// ============================================================================
// INDEX MAPPING EXPORT
// ============================================================================

void export_fd_index_mapping(
    const FactoredTransitionSystem& fts,
    const std::string& fd_output_dir,
    int iteration
);

// ============================================================================
// ERROR REPORTING
// ============================================================================

void export_error_message(
    int iteration,
    const std::string& error_message,
    const std::string& fd_output_dir
);

}  // namespace merge_and_shrink

#endif