#include "merge_strategy_gnn.h"
#include "../utils/system.h"
#include "merge_and_shrink_signals.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <thread>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

using namespace std;
using json = nlohmann::json;
using namespace merge_and_shrink;
namespace fs = std::filesystem;

MergeStrategyGNN::MergeStrategyGNN(const FactoredTransitionSystem& fts)
    : MergeStrategy(fts),
      iteration(0) {
    cout << "\n[GNN::INIT] MergeStrategyGNN initialized" << endl;
}

std::filesystem::path get_gnn_input_directory() {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path gnn_output = cwd / "gnn_output";

    try {
        std::filesystem::create_directories(gnn_output);
    } catch (const std::exception& e) {
        std::cerr << "[GNN::PATH] Warning: Could not create gnn_output: "
                  << e.what() << std::endl;
    }

    std::cout << "[GNN::PATH] Using gnn_input: " << gnn_output.string() << std::endl;
    return gnn_output;
}

std::pair<int, int> MergeStrategyGNN::get_next() {
        int idx = iteration;

    std::filesystem::path gnn_input_dir = get_gnn_input_directory();
    std::filesystem::path fd_output_dir = std::filesystem::current_path() / "fd_output";

    // Ensure directories exist
    try {
        std::filesystem::create_directories(gnn_input_dir);
        std::filesystem::create_directories(fd_output_dir);
        std::cout << "[GNN::GET_NEXT] Directories verified:" << std::endl;
        std::cout << "[GNN::GET_NEXT]   gnn_input: " << gnn_input_dir.string() << std::endl;
        std::cout << "[GNN::GET_NEXT]   fd_output: " << fd_output_dir.string() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[GNN::ERROR] Failed to create directories: " << e.what() << std::endl;
        throw;
    }

    std::filesystem::path in_path = gnn_input_dir / ("merge_" + std::to_string(idx) + ".json");
    std::filesystem::path ack_path = fd_output_dir / ("gnn_ack_" + std::to_string(idx) + ".json");

    std::cout << "\n[GNN::ITERATION " << idx << "] ========================================" << std::endl;
    std::cout << "[GNN::ITERATION " << idx << "] Waiting for merge decision at:" << std::endl;
    std::cout << "[GNN::ITERATION " << idx << "]   " << in_path.string() << std::endl;

    // Wait for file to exist
    const int sleep_ms = 50;
    const int max_wait_ms = 600000;
    int elapsed_ms = 0;
    int last_logged = 0;

    while (!std::filesystem::exists(in_path)) {
        if (elapsed_ms > last_logged + 10000) {
            std::cout << "[GNN::ITERATION " << idx << "] ... waiting ("
                      << elapsed_ms / 1000 << "s) ..." << std::endl;
            last_logged = elapsed_ms;
        }

        if (elapsed_ms > max_wait_ms) {
            std::cerr << "\n[GNN::ERROR] ❌ TIMEOUT waiting for merge decision!" << std::endl;
            export_error_signal(idx, "Timeout waiting for merge decision", fd_output_dir.string());
            throw std::runtime_error("GNN merge decision timeout");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        elapsed_ms += sleep_ms;
    }

    std::cout << "[GNN::ITERATION " << idx << "] ✅ Decision file found!" << std::endl;

    // ========================================================================
    // ✅ FIX: RETRY LOGIC FOR FILE READING (Windows file locking workaround)
    // ========================================================================
    json merge_decision;
    const int max_open_attempts = 100;  // 100 attempts * 50ms = 5 seconds max retry
    int open_attempts = 0;
    std::string last_error;

    while (open_attempts < max_open_attempts) {
        try {
            // Small delay before first attempt to let filesystem settle
            if (open_attempts == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            std::ifstream fin(in_path);
            if (!fin.is_open()) {
                last_error = "Cannot open file (may be locked)";
                open_attempts++;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            // Try to parse JSON
            fin >> merge_decision;
            fin.close();

            // Validate structure
            if (!merge_decision.contains("merge_pair") ||
                !merge_decision["merge_pair"].is_array() ||
                merge_decision["merge_pair"].size() != 2) {
                last_error = "Invalid merge_pair in JSON";
                open_attempts++;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            // Success!
            std::cout << "[GNN::ITERATION " << idx << "] ✅ JSON parsed successfully";
            if (open_attempts > 0) {
                std::cout << " (after " << open_attempts << " retries)";
            }
            std::cout << std::endl;
            break;

        } catch (const nlohmann::json::exception& e) {
            last_error = std::string("JSON parse error: ") + e.what();
            open_attempts++;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } catch (const std::exception& e) {
            last_error = std::string("Read error: ") + e.what();
            open_attempts++;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // Check if we exhausted all attempts
    if (open_attempts >= max_open_attempts) {
        std::cerr << "[GNN::ERROR] Failed to read merge decision after "
                  << max_open_attempts << " attempts: " << last_error << std::endl;
        export_error_signal(idx, "Failed after retries: " + last_error, fd_output_dir.string());
        throw std::runtime_error("Cannot read merge decision: " + last_error);
    }

    int u = merge_decision["merge_pair"][0].get<int>();
    int v = merge_decision["merge_pair"][1].get<int>();

    std::cout << "[GNN::ITERATION " << idx << "] Merge decision: (" << u << ", " << v << ")" << std::endl;



    // ✅ WRITE ACK ATOMICALLY
    try {
        std::filesystem::create_directories(fd_output_dir);

        json ack_data;
        ack_data["iteration"] = idx;
        ack_data["merge_pair"] = {u, v};
        ack_data["received"] = true;
        ack_data["timestamp"] = static_cast<long>(std::time(nullptr));

        std::filesystem::path temp_path = std::filesystem::path(ack_path.string() + ".tmp");
        {
            std::ofstream fout(temp_path);
            if (!fout.is_open()) {
                throw std::runtime_error("Cannot open temp ACK file");
            }
            fout << ack_data.dump(2);
            fout.flush();
            fout.close();

            if (fout.fail()) {
                throw std::runtime_error("Write/close failed");
            }
        }

        try {
            std::filesystem::rename(temp_path, ack_path);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "[GNN::WARNING] Rename failed, using direct write: " << e.what() << std::endl;
            std::ofstream fout(ack_path);
            if (fout.is_open()) {
                fout << ack_data.dump(2);
                fout.close();
            } else {
                throw std::runtime_error("Cannot write ACK file");
            }
        }

        std::cout << "[GNN::ITERATION " << idx << "] ✅ ACK written to:" << std::endl;
        std::cout << "[GNN::ITERATION " << idx << "]   " << ack_path.string() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[GNN::WARNING] ACK write failed (continuing): " << e.what() << std::endl;
    }

    iteration++;

    std::cout << "[GNN::ITERATION " << idx << "] ========================================" << std::endl;
    std::cout << "[GNN::ITERATION " << idx << "] ✅ Returning (" << u << ", " << v << ")\n" << std::endl;

    return {u, v};
}

void MergeStrategyGNN::set_next_merge_pair(
    const FactoredTransitionSystem&,
    const std::vector<std::shared_ptr<ShrinkStrategy>>&) {
    // Not used
}