#include "merge_and_shrink_algorithm.h"

#include "distances.h"
#include "factored_transition_system.h"
#include "fts_factory.h"
#include "label_reduction.h"
#include "labels.h"
#include "merge_and_shrink_representation.h"
#include "merge_strategy.h"
#include "merge_strategy_factory.h"
#include "shrink_strategy.h"
#include "transition_system.h"
#include "types.h"
#include "utils.h"

#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"
#include "../utils/component_errors.h"
#include "../utils/countdown_timer.h"
#include "../utils/markup.h"
#include "../utils/math.h"
#include "../utils/system.h"
#include "../utils/timer.h"

#include "merge_and_shrink_signals.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <limits>
#include <ctime>
#include <set>

using json = nlohmann::json;
using namespace std;
namespace fs = std::filesystem;

using plugins::Bounds;
using utils::ExitCode;

namespace merge_and_shrink {


// ============================================================================
// Logging Helpers
// ============================================================================

static void log_progress(const utils::Timer &timer, const string &msg, utils::LogProxy &log) {
    log << "M&S algorithm timer: " << timer << " (" << msg << ")" << endl;
}

// ============================================================================
// MergeAndShrinkAlgorithm Implementation
// ============================================================================

MergeAndShrinkAlgorithm::MergeAndShrinkAlgorithm(
    const shared_ptr<MergeStrategyFactory> &merge_strategy,
    const shared_ptr<ShrinkStrategy> &shrink_strategy,
    const shared_ptr<LabelReduction> &label_reduction,
    bool prune_unreachable_states, bool prune_irrelevant_states,
    int max_states, int max_states_before_merge,
    int threshold_before_merge, double main_loop_max_time,
    utils::Verbosity verbosity)
    : merge_strategy_factory(merge_strategy),
      shrink_strategy(shrink_strategy),
      label_reduction(label_reduction),
      max_states(max_states),
      max_states_before_merge(max_states_before_merge),
      shrink_threshold_before_merge(threshold_before_merge),
      prune_unreachable_states(prune_unreachable_states),
      prune_irrelevant_states(prune_irrelevant_states),
      log(utils::get_log_for_verbosity(verbosity)),
      main_loop_max_time(main_loop_max_time),
      starting_peak_memory(0) {
    handle_shrink_limit_defaults();
    assert(this->max_states_before_merge >= 1);
    assert(this->max_states >= this->max_states_before_merge);
}

void MergeAndShrinkAlgorithm::handle_shrink_limit_defaults() {
    if (max_states == -1 && max_states_before_merge == -1) {
        max_states = 50000;
    }

    if (max_states_before_merge == -1) {
        max_states_before_merge = max_states;
    } else if (max_states == -1) {
        if (utils::is_product_within_limit(
                max_states_before_merge, max_states_before_merge, INF)) {
            max_states = max_states_before_merge * max_states_before_merge;
        } else {
            max_states = INF;
        }
    }

    if (max_states_before_merge > max_states) {
        max_states_before_merge = max_states;
        if (log.is_warning()) {
            log << "WARNING: "
                << "max_states_before_merge exceeds max_states, "
                << "correcting max_states_before_merge." << endl;
        }
    }

    utils::verify_argument(max_states >= 1,
                           "Transition system size must be at least 1.");

    utils::verify_argument(max_states_before_merge >= 1,
                           "Transition system size before merge must be at least 1.");

    if (shrink_threshold_before_merge == -1) {
        shrink_threshold_before_merge = max_states;
    }

    utils::verify_argument(shrink_threshold_before_merge >= 1,
                           "Threshold must be at least 1.");

    if (shrink_threshold_before_merge > max_states) {
        shrink_threshold_before_merge = max_states;
        if (log.is_warning()) {
            log << "WARNING: "
                << "threshold exceeds max_states, "
                << "correcting threshold." << endl;
        }
    }
}

void MergeAndShrinkAlgorithm::report_peak_memory_delta(bool final) const {
    if (final)
        log << "Final";
    else
        log << "Current";
    log << " peak memory increase of merge-and-shrink algorithm: "
        << utils::get_peak_memory_in_kb() - starting_peak_memory << " KB"
        << endl;
}

void MergeAndShrinkAlgorithm::dump_options() const {
    if (log.is_at_least_normal()) {
        if (merge_strategy_factory) {
            merge_strategy_factory->dump_options();
            log << endl;
        }

        log << "Options related to size limits and shrinking: " << endl;
        log << "Transition system size limit: " << max_states << endl
            << "Transition system size limit right before merge: "
            << max_states_before_merge << endl;
        log << "Threshold to trigger shrinking right before merge: "
            << shrink_threshold_before_merge << endl;
        log << endl;

        shrink_strategy->dump_options(log);
        log << endl;

        log << "Pruning unreachable states: "
            << (prune_unreachable_states ? "yes" : "no") << endl;
        log << "Pruning irrelevant states: "
            << (prune_irrelevant_states ? "yes" : "no") << endl;
        log << endl;

        if (label_reduction) {
            label_reduction->dump_options(log);
        } else {
            log << "Label reduction disabled" << endl;
        }
        log << endl;

        log << "Main loop max time in seconds: " << main_loop_max_time << endl;
        log << endl;
    }
}

void MergeAndShrinkAlgorithm::warn_on_unusual_options() const {
    string dashes(79, '=');
    if (!label_reduction) {
        if (log.is_warning()) {
            log << dashes << endl
                << "WARNING! You did not enable label reduction. " << endl
                << "This may drastically reduce the performance of merge-and-shrink!"
                << endl << dashes << endl;
        }
    } else if (label_reduction->reduce_before_merging() && label_reduction->reduce_before_shrinking()) {
        if (log.is_warning()) {
            log << dashes << endl
                << "WARNING! You set label reduction to be applied twice in each merge-and-shrink" << endl
                << "iteration, both before shrinking and merging. This double computation effort" << endl
                << "does not pay off for most configurations!"
                << endl << dashes << endl;
        }
    } else {
        if (label_reduction->reduce_before_shrinking() &&
            (shrink_strategy->get_name() == "f-preserving"
             || shrink_strategy->get_name() == "random")) {
            if (log.is_warning()) {
                log << dashes << endl
                    << "WARNING! Bucket-based shrink strategies such as f-preserving random perform" << endl
                    << "best if used with label reduction before merging, not before shrinking!"
                    << endl << dashes << endl;
            }
        }
        if (label_reduction->reduce_before_merging() &&
            shrink_strategy->get_name() == "bisimulation") {
            if (log.is_warning()) {
                log << dashes << endl
                    << "WARNING! Shrinking based on bisimulation performs best if used with label" << endl
                    << "reduction before shrinking, not before merging!"
                    << endl << dashes << endl;
            }
        }
    }

    if (!prune_unreachable_states || !prune_irrelevant_states) {
        if (log.is_warning()) {
            log << dashes << endl
                << "WARNING! Pruning is (partially) turned off!" << endl
                << "This may drastically reduce the performance of merge-and-shrink!"
                << endl << dashes << endl;
        }
    }
}

bool MergeAndShrinkAlgorithm::ran_out_of_time(
    const utils::CountdownTimer &timer) const {
    if (timer.is_expired()) {
        if (log.is_at_least_normal()) {
            log << "Ran out of time, stopping computation." << endl << endl;
        }
        return true;
    }
    return false;
}

// ============================================================================
// MAIN LOOP - COMPLETE REWRITE (SYNCHRONIZED PING-PONG)
// ============================================================================

void MergeAndShrinkAlgorithm::main_loop(
    FactoredTransitionSystem &fts,
    const TaskProxy &task_proxy) {

    utils::CountdownTimer timer(main_loop_max_time);
    if (log.is_at_least_normal()) {
        log << "Starting main loop ";
        if (main_loop_max_time == numeric_limits<double>::infinity()) {
            log << "without a time limit." << endl;
        } else {
            log << "with a time limit of " << main_loop_max_time << "s." << endl;
        }
    }

    int maximum_intermediate_size = 0;
    for (int i = 0; i < fts.get_size(); ++i) {
        int size = fts.get_transition_system(i).get_size();
        if (size > maximum_intermediate_size) {
            maximum_intermediate_size = size;
        }
    }

    if (label_reduction) {
        label_reduction->initialize(task_proxy);
    }

    unique_ptr<MergeStrategy> merge_strategy =
        merge_strategy_factory->compute_merge_strategy(task_proxy, fts);
    merge_strategy_factory = nullptr;

    auto log_main_loop_progress = [&timer, this](const string &msg) {
        log << "M&S algorithm main loop timer: "
            << timer.get_elapsed_time()
            << " (" << msg << ")" << endl;
    };

    std::string fd_output_dir = get_fd_output_directory();
    std::cout << "[M&S::MAIN] fd_output_dir: " << fd_output_dir << std::endl;

    try {
        std::filesystem::create_directories(fd_output_dir);
    } catch (const std::exception& e) {
        std::cerr << "[M&S::ERROR] Failed to create fd_output: " << e.what() << std::endl;
        throw;
    }

    int iteration = 0;

    while (fts.get_num_active_entries() > 1) {
        std::cout << "\n[M&S::MAIN] ============================================" << std::endl;
        std::cout << "[M&S::MAIN] Iteration " << iteration << " starting..." << std::endl;
        std::cout << "[M&S::MAIN] ============================================" << std::endl;

        // ✅ PHASE 1: GET NEXT MERGE PAIR FROM STRATEGY
        std::cout << "[M&S::MAIN] PHASE 1: Waiting for GNN decision (merge_"
                  << iteration << ".json)..." << std::endl;

        pair<int, int> merge_indices;
        try {
            merge_indices = merge_strategy->get_next();
        } catch (const std::exception& e) {
            std::cerr << "[M&S::ERROR] Failed to get merge decision: " << e.what() << std::endl;
            export_error_signal(iteration, std::string("get_next failed: ") + e.what(), fd_output_dir);
            throw;
        }

        std::cout << "[M&S::MAIN] PHASE 1 COMPLETE: Received decision: ("
                  << merge_indices.first << ", " << merge_indices.second << ")" << std::endl;

        if (ran_out_of_time(timer)) break;

        int merge_index1 = merge_indices.first;
        int merge_index2 = merge_indices.second;

        // ✅ VALIDATE INDICES IMMEDIATELY
        if (!fts.is_active(merge_index1)) {
            std::cerr << "[M&S::ERROR] merge_index1=" << merge_index1 << " is NOT ACTIVE!" << std::endl;
            export_error_signal(iteration, "merge_index1 is not active", fd_output_dir);
            throw std::runtime_error("Invalid merge index 1");
        }
        if (!fts.is_active(merge_index2)) {
            std::cerr << "[M&S::ERROR] merge_index2=" << merge_index2 << " is NOT ACTIVE!" << std::endl;
            export_error_signal(iteration, "merge_index2 is not active", fd_output_dir);
            throw std::runtime_error("Invalid merge index 2");
        }

        std::cout << "[M&S::MAIN] Merge indices validated: both are active" << std::endl;

        assert(merge_index1 != merge_index2);
        if (log.is_at_least_normal()) {
            log << "Next pair of indices: ("
                << merge_index1 << ", " << merge_index2 << ")" << endl;
            if (log.is_at_least_verbose()) {
                fts.statistics(merge_index1, log);
                fts.statistics(merge_index2, log);
            }
            log_main_loop_progress("after computation of next merge");
        }

        // ✅ PHASE 2: CAPTURE BEFORE DATA (for statistics only)
        std::cout << "[M&S::MAIN] PHASE 2: Capturing pre-merge data..." << std::endl;

        json before_data;
        try {
            before_data = export_merge_before_data(
                fts, merge_index1, merge_index2, iteration, false, false
            );
            std::cout << "[M&S::MAIN] PHASE 2 COMPLETE: Pre-merge data captured" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[M&S::ERROR] Failed to capture before data: " << e.what() << std::endl;
            export_error_signal(iteration, std::string("before_data failed: ") + e.what(), fd_output_dir);
            throw;
        }

        // ✅ PHASE 3: LABEL REDUCTION (BEFORE SHRINKING)
        bool reduced = false;
        if (label_reduction && label_reduction->reduce_before_shrinking()) {
            std::cout << "[M&S::MAIN] PHASE 3: Label reduction (before shrinking)..." << std::endl;
            try {
                reduced = label_reduction->reduce(merge_indices, fts, log);
                if (log.is_at_least_normal() && reduced) {
                    log_main_loop_progress("after label reduction");
                }
                std::cout << "[M&S::MAIN] PHASE 3 COMPLETE: reduced=" << reduced << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[M&S::ERROR] Label reduction failed: " << e.what() << std::endl;
                export_error_signal(iteration, std::string("label_reduction failed: ") + e.what(), fd_output_dir);
                throw;
            }
        }

        if (ran_out_of_time(timer)) break;

        // ✅ PHASE 4: SHRINKING
        std::cout << "[M&S::MAIN] PHASE 4: Shrinking..." << std::endl;
        bool shrunk = false;
        try {
            shrunk = shrink_before_merge_step(
                fts,
                merge_index1,
                merge_index2,
                max_states,
                max_states_before_merge,
                shrink_threshold_before_merge,
                *shrink_strategy,
                log);
            if (log.is_at_least_normal() && shrunk) {
                log_main_loop_progress("after shrinking");
            }
            std::cout << "[M&S::MAIN] PHASE 4 COMPLETE: shrunk=" << shrunk << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[M&S::ERROR] Shrinking failed: " << e.what() << std::endl;
            export_error_signal(iteration, std::string("shrinking failed: ") + e.what(), fd_output_dir);
            throw;
        }

        before_data["shrunk"] = shrunk;
        before_data["reduced"] = reduced;

        if (ran_out_of_time(timer)) break;

        // ✅ PHASE 5: LABEL REDUCTION (BEFORE MERGING)
        if (label_reduction && label_reduction->reduce_before_merging()) {
            std::cout << "[M&S::MAIN] PHASE 5: Label reduction (before merging)..." << std::endl;
            try {
                reduced = label_reduction->reduce(merge_indices, fts, log);
                if (log.is_at_least_normal() && reduced) {
                    log_main_loop_progress("after label reduction");
                }
                before_data["reduced"] = reduced;
                std::cout << "[M&S::MAIN] PHASE 5 COMPLETE: reduced=" << reduced << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[M&S::ERROR] Label reduction (before merge) failed: " << e.what() << std::endl;
                export_error_signal(iteration, std::string("label_reduction_before_merge failed: ") + e.what(), fd_output_dir);
                throw;
            }
        }

        if (ran_out_of_time(timer)) break;

        // ✅ PHASE 6: PERFORM ACTUAL MERGE
        std::cout << "[M&S::MAIN] PHASE 6: Performing merge..." << std::endl;
        int merged_index;
        try {
            merged_index = fts.merge(merge_index1, merge_index2, log);
            std::cout << "[M&S::MAIN] PHASE 6 COMPLETE: merged_index=" << merged_index << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[M&S::ERROR] Merge operation failed: " << e.what() << std::endl;
            export_error_signal(iteration, std::string("merge failed: ") + e.what(), fd_output_dir);
            throw;
        }

        // ✅ PHASE 7: EXPORT GNN OBSERVATION FOR NEXT ITERATION
        // ⭐ CRITICAL: This is the state AFTER the merge, so iteration number matches!
        std::cout << "[M&S::MAIN] PHASE 7: Exporting GNN observation for iteration "
                  << iteration << "..." << std::endl;
        try {
            json obs_data = export_gnn_observation(fts, iteration);  // ✅ FIX: Use iteration directly!
            std::string obs_path = fd_output_dir + "/observation_" + std::to_string(iteration) + ".json";
            write_json_file_atomic(obs_data, obs_path);
            std::cout << "[M&S::MAIN] PHASE 7 COMPLETE: " << obs_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[M&S::ERROR] GNN observation export failed: " << e.what() << std::endl;
            export_error_signal(iteration, std::string("obs export failed: ") + e.what(), fd_output_dir);
            // Don't throw - continue anyway
            std::cerr << "[M&S::WARNING] Continuing without observation export" << std::endl;
        }

        // ✅ PHASE 8: POST-MERGE OPERATIONS
        iteration++;  // Move to next iteration for next loop

        int abs_size = fts.get_transition_system(merged_index).get_size();
        if (abs_size > maximum_intermediate_size) {
            maximum_intermediate_size = abs_size;
        }

        if (log.is_at_least_normal()) {
            if (log.is_at_least_verbose()) {
                fts.statistics(merged_index, log);
            }
            log_main_loop_progress("after merging");
        }

        if (ran_out_of_time(timer)) {
            break;
        }

        // Pruning
        if (prune_unreachable_states || prune_irrelevant_states) {
            bool pruned = prune_step(
                fts,
                merged_index,
                prune_unreachable_states,
                prune_irrelevant_states,
                log);
            if (log.is_at_least_normal() && pruned) {
                if (log.is_at_least_verbose()) {
                    fts.statistics(merged_index, log);
                }
                log_main_loop_progress("after pruning");
            }
        }

        if (!fts.is_factor_solvable(merged_index)) {
            if (log.is_at_least_normal()) {
                log << "Abstract problem is unsolvable, stopping computation." << endl << endl;
            }
            break;
        }

        if (ran_out_of_time(timer)) {
            break;
        }

        if (log.is_at_least_verbose()) {
            report_peak_memory_delta();
        }
        if (log.is_at_least_normal()) {
            log << endl;
        }
    }

    log << "End of merge-and-shrink algorithm, statistics:" << endl;
    log << "Main loop runtime: " << timer.get_elapsed_time() << endl;
    log << "Maximum intermediate abstraction size: "
        << maximum_intermediate_size << endl;
    shrink_strategy = nullptr;
    label_reduction = nullptr;
}

FactoredTransitionSystem MergeAndShrinkAlgorithm::build_factored_transition_system(
    const TaskProxy &task_proxy) {
    if (starting_peak_memory) {
        cerr << "Calling build_factored_transition_system twice is not "
             << "supported!" << endl;
        utils::exit_with(utils::ExitCode::SEARCH_CRITICAL_ERROR);
    }
    starting_peak_memory = utils::get_peak_memory_in_kb();

    utils::Timer timer;
    log << "Running merge-and-shrink algorithm..." << endl;
    task_properties::verify_no_axioms(task_proxy);
    dump_options();
    warn_on_unusual_options();
    log << endl;

    const bool compute_init_distances =
        shrink_strategy->requires_init_distances() ||
        merge_strategy_factory->requires_init_distances() ||
        prune_unreachable_states;
    const bool compute_goal_distances =
        shrink_strategy->requires_goal_distances() ||
        merge_strategy_factory->requires_goal_distances() ||
        prune_irrelevant_states;
    FactoredTransitionSystem fts =
        create_factored_transition_system(
            task_proxy,
            compute_init_distances,
            compute_goal_distances,
            log);
    if (log.is_at_least_normal()) {
        log_progress(timer, "after computation of atomic factors", log);
    }

    bool pruned = false;
    bool unsolvable = false;
    for (int index = 0; index < fts.get_size(); ++index) {
        assert(fts.is_active(index));
        if (prune_unreachable_states || prune_irrelevant_states) {
            bool pruned_factor = prune_step(
                fts,
                index,
                prune_unreachable_states,
                prune_irrelevant_states,
                log);
            pruned = pruned || pruned_factor;
        }
        if (!fts.is_factor_solvable(index)) {
            log << "Atomic FTS is unsolvable, stopping computation." << endl;
            unsolvable = true;
            break;
        }
    }
    if (log.is_at_least_normal()) {
        if (pruned) {
            log_progress(timer, "after pruning atomic factors", log);
        }
        log << endl;
    }

// ✅ EXPORT INITIAL OBSERVATION (iteration -1)
    {
        std::string fd_output_dir_str = get_fd_output_directory();
        try {
            json initial_obs = export_gnn_observation(fts, -1);
            std::string initial_obs_path = fd_output_dir_str + "/observation_-1.json";
            write_json_file_atomic(initial_obs, initial_obs_path);
            std::cout << "[M&S] ✅ Exported initial GNN observation (iteration -1)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[M&S] ⚠️ Warning: Could not export initial observation: " << e.what() << std::endl;
        }
    }

    // ✅ FIX: Always run main_loop for GNN-guided M&S
    // The GNN needs to make merge decisions regardless of atomic solvability
    if (main_loop_max_time > 0 && fts.get_num_active_entries() > 1) {
        if (unsolvable) {
            std::cout << "[M&S] ⚠️ Atomic FTS detected as unsolvable, but continuing for GNN training" << std::endl;
        }
        main_loop(fts, task_proxy);
    }

    const bool final = true;
    report_peak_memory_delta(final);
    log << "Merge-and-shrink algorithm runtime: " << timer << endl;
    log << endl;
    return fts;
}

void add_merge_and_shrink_algorithm_options_to_feature(plugins::Feature &feature) {
    feature.add_option<shared_ptr<MergeStrategyFactory>>(
        "merge_strategy",
        "See detailed documentation for merge strategies. "
        "We currently recommend SCC-DFP, which can be achieved using "
        "{{{merge_strategy=merge_sccs(order_of_sccs=topological,merge_selector="
        "score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order"
        "]))}}}");

    feature.add_option<shared_ptr<ShrinkStrategy>>(
        "shrink_strategy",
        "See detailed documentation for shrink strategies. "
        "We currently recommend non-greedy shrink_bisimulation, which can be "
        "achieved using {not relevant}");

    feature.add_option<shared_ptr<LabelReduction>>(
        "label_reduction",
        "See detailed documentation for labels. There is currently only "
        "one 'option' to use label_reduction, which is {not relevant} "
        "Also note the interaction with shrink strategies.",
        plugins::ArgumentInfo::NO_DEFAULT);

    feature.add_option<bool>(
        "prune_unreachable_states",
        "If true, prune abstract states unreachable from the initial state.",
        "true");
    feature.add_option<bool>(
        "prune_irrelevant_states",
        "If true, prune abstract states from which no goal state can be "
        "reached.",
        "true");

    add_transition_system_size_limit_options_to_feature(feature);

    feature.add_option<double>(
        "main_loop_max_time",
        "A limit in seconds on the runtime of the main loop of the algorithm. "
        "If the limit is exceeded, the algorithm terminates, potentially "
        "returning a factored transition system with several factors. Also "
        "note that the time limit is only checked between transformations "
        "of the main loop, but not during, so it can be exceeded if a "
        "transformation is runtime-intense.",
        "infinity",
        Bounds("0.0", "infinity"));
}

tuple<shared_ptr<MergeStrategyFactory>, shared_ptr<ShrinkStrategy>,
      shared_ptr<LabelReduction>, bool, bool, int, int, int, double>
get_merge_and_shrink_algorithm_arguments_from_options(
    const plugins::Options &opts) {
    return tuple_cat(
        make_tuple(
            opts.get<shared_ptr<MergeStrategyFactory>>("merge_strategy"),
            opts.get<shared_ptr<ShrinkStrategy>>("shrink_strategy"),
            opts.get<shared_ptr<LabelReduction>>(
                "label_reduction", nullptr),
            opts.get<bool>("prune_unreachable_states"),
            opts.get<bool>("prune_irrelevant_states")),
        get_transition_system_size_limit_arguments_from_options(opts),
        make_tuple(opts.get<double>("main_loop_max_time"))
        );
}

void add_transition_system_size_limit_options_to_feature(plugins::Feature &feature) {
    feature.add_option<int>(
        "max_states",
        "maximum transition system size allowed at any time point.",
        "-1",
        Bounds("-1", "infinity"));
    feature.add_option<int>(
        "max_states_before_merge",
        "maximum transition system size allowed for two transition systems "
        "before being merged to form the synchronized product.",
        "-1",
        Bounds("-1", "infinity"));
    feature.add_option<int>(
        "threshold_before_merge",
        "If a transition system, before being merged, surpasses this soft "
        "transition system size limit, the shrink strategy is called to "
        "possibly shrink the transition system.",
        "-1",
        Bounds("-1", "infinity"));
}

tuple<int, int, int>
get_transition_system_size_limit_arguments_from_options(
    const plugins::Options &opts) {
    return make_tuple(
        opts.get<int>("max_states"),
        opts.get<int>("max_states_before_merge"),
        opts.get<int>("threshold_before_merge")
        );
}
}