#!/usr/bin/env python3
"""
✅ ENHANCED TEST: Verify different problems produce different results
WITH COMPREHENSIVE DIAGNOSTICS
"""

import os
import sys
import time
import hashlib
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_one_step_enhanced.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 1: PROBLEM VERIFICATION
# ============================================================================

def compute_problem_hash(problem_file: str) -> str:
    """Compute hash of problem file content to verify it's unique."""
    try:
        with open(problem_file, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception as e:
        logger.error(f"Cannot hash {problem_file}: {e}")
        return "UNKNOWN"


def verify_problems_are_different(test_problems: list) -> bool:
    """CRITICAL: Verify that test problems are actually different files."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: VERIFY PROBLEMS ARE DIFFERENT")
    logger.info("=" * 80 + "\n")

    hashes = {}
    all_different = True

    for problem_file in test_problems:
        problem_name = os.path.basename(problem_file)
        file_hash = compute_problem_hash(problem_file)
        hashes[problem_name] = file_hash

        logger.info(f"  {problem_name:<30} SHA256: {file_hash}")

        # Check file size
        try:
            size = os.path.getsize(problem_file)
            logger.info(f"  {' ':<30} Size: {size} bytes")
        except:
            pass

    # Check for duplicates
    unique_hashes = set(hashes.values())
    if len(unique_hashes) < len(hashes):
        logger.error("\n❌ CRITICAL: Some problems have IDENTICAL content!")
        logger.error("   This explains why rewards are the same.")
        all_different = False
    else:
        logger.info(f"\n✅ All {len(hashes)} problems are unique\n")

    return all_different


# ============================================================================
# PHASE 2: GRAPH DIFFERENTIATION DETECTION
# ============================================================================

def detect_graph_differences(test_problems: list) -> dict:
    """Run each problem and detect if graphs are actually different."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: DETECT GRAPH DIFFERENCES")
    logger.info("=" * 80 + "\n")

    from merge_env import MergeEnv

    results = {}
    graph_hashes = {}

    for problem_file in test_problems:
        problem_name = os.path.basename(problem_file)
        logger.info(f"\nTesting: {problem_name}")
        logger.info("-" * 60)

        # Clean old files
        logger.info("Cleaning fd_output...")
        for dir_name in ["downward/fd_output", "downward/gnn_output"]:
            if os.path.isdir(dir_name):
                for fname in os.listdir(dir_name):
                    fpath = os.path.join(dir_name, fname)
                    try:
                        os.remove(fpath)
                    except:
                        pass

        logger.info("Creating environment...")
        try:
            env = MergeEnv(
                domain_file="benchmarks/small/domain.pddl",
                problem_file=problem_file,
                max_merges=5,
                debug=False,
                reward_variant='astar_search',
                max_states=4000,
                threshold_before_merge=1
            )
            logger.info("✓ Environment created")
        except Exception as e:
            logger.error(f"✗ Failed to create env: {e}")
            continue

        # CRITICAL: Capture graph state
        logger.info("Resetting environment...")
        try:
            obs, info = env.reset()
            logger.info("✓ Reset successful")
        except Exception as e:
            logger.error(f"✗ Reset failed: {e}")
            env.close()
            continue

        # ✅ NEW: Hash the graph representation
        graph_content = {
            'num_nodes': int(obs['num_nodes']),
            'num_edges': int(obs['num_edges']),
            'node_features_sum': float(obs['x'].sum()),  # Summary stat
            'edge_index_first_row': str(obs['edge_index'][0, :5].tolist()),  # First 5 edges
        }

        graph_hash = hashlib.sha256(
            json.dumps(graph_content, sort_keys=True).encode()
        ).hexdigest()[:16]

        logger.info(f"Graph representation:")
        logger.info(f"  Nodes: {graph_content['num_nodes']}")
        logger.info(f"  Edges: {graph_content['num_edges']}")
        logger.info(f"  Hash: {graph_hash}")

        graph_hashes[problem_name] = (graph_content['num_nodes'], graph_hash)

        # Step and extract reward info
        logger.info("Performing step(0)...")
        try:
            obs, reward, done, truncated, info = env.step(0)

            results[problem_name] = {
                'initial_nodes': graph_content['num_nodes'],
                'initial_edges': graph_content['num_edges'],
                'reward': float(reward),
                'delta_states': int(info.get('delta_states', 0)),
                'merge_pair': info.get('merge_pair', None),
            }

            logger.info(f"✓ Step result:")
            logger.info(f"  Reward: {reward:.6f}")
            logger.info(f"  Delta states: {info.get('delta_states', 0)}")

        except Exception as e:
            logger.error(f"✗ Step failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            env.close()
            time.sleep(1)

    # ✅ NEW: Analyze differences
    logger.info("\n" + "-" * 60)
    logger.info("GRAPH DIFFERENTIATION ANALYSIS:")
    logger.info("-" * 60)

    unique_node_counts = set(nc for nc, _ in graph_hashes.values())
    unique_hashes = set(h for _, h in graph_hashes.values())

    logger.info(f"Unique node counts: {sorted(unique_node_counts)}")
    logger.info(f"Unique graph hashes: {len(unique_hashes)}")

    if len(unique_hashes) == 1:
        logger.error("\n❌ CRITICAL: All graphs are IDENTICAL!")
        logger.error("   All problems produced the same abstract representation.")
        logger.error("   This explains identical rewards.")
        logger.error("\nPossible causes:")
        logger.error("  1. Problems are semantically equivalent")
        logger.error("  2. FD M&S abstraction collapses them to same size")
        logger.error("  3. Files are not being reloaded (cache issue)")
    else:
        logger.info(f"\n✅ Graphs ARE different ({len(unique_hashes)} unique)")

    return results


# ============================================================================
# PHASE 3: REWARD DIFFERENTIATION ANALYSIS
# ============================================================================

def analyze_reward_differentiation(results: dict) -> bool:
    """Verify rewards vary across problems."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: REWARD DIFFERENTIATION ANALYSIS")
    logger.info("=" * 80 + "\n")

    if not results:
        logger.error("No results to analyze")
        return False

    logger.info("Results by problem:")
    for name, data in sorted(results.items()):
        logger.info(f"\n{name}:")
        logger.info(f"  Nodes: {data['initial_nodes']}")
        logger.info(f"  Edges: {data['initial_edges']}")
        logger.info(f"  Reward: {data['reward']:+.6f}")
        logger.info(f"  Delta states: {data['delta_states']:+d}")

    # Check for differentiation
    unique_rewards = set(round(r['reward'], 6) for r in results.values())
    unique_nodes = set(r['initial_nodes'] for r in results.values())
    unique_edges = set(r['initial_edges'] for r in results.values())
    unique_deltas = set(r['delta_states'] for r in results.values())

    logger.info(f"\nDifferentiation Metrics:")
    logger.info(f"  Unique rewards: {len(unique_rewards)}")
    logger.info(f"  Unique node counts: {len(unique_nodes)}")
    logger.info(f"  Unique edge counts: {len(unique_edges)}")
    logger.info(f"  Unique delta_states: {len(unique_deltas)}")

    # Verdict
    is_differentiated = (
            len(unique_rewards) > 1 and
            len(unique_nodes) > 1
    )

    if is_differentiated:
        logger.info("\n✅ SUCCESS: Problems ARE differentiated!")
        logger.info("   Rewards vary as expected")
        return True
    else:
        logger.error("\n❌ FAILURE: Problems NOT differentiated!")
        logger.error("   Rewards are identical or nodes are identical")

        if len(unique_nodes) == 1:
            logger.error("\n   ROOT CAUSE: Abstract graphs are identical")
            logger.error("   (This might be expected for these particular problems)")

        if len(unique_rewards) == 1:
            logger.error("\n   ROOT CAUSE: Reward computation is deterministic")
            logger.error("   Check if merge_info is actually different")

        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "=" * 90)
    logger.info("COMPREHENSIVE TEST: Problem Differentiation & Reward Variation")
    logger.info("=" * 90 + "\n")

    test_problems = [
        "benchmarks/small/problem_small_00.pddl",
        "benchmarks/small/problem_small_01.pddl",
        "benchmarks/small/problem_small_10.pddl",
    ]

    # Phase 1: Verify problems are different
    if not verify_problems_are_different(test_problems):
        logger.error("\n❌ Test problems are not different - fix benchmark files!")
        return 1

    # Phase 2: Detect graph differences
    results = detect_graph_differences(test_problems)

    # Phase 3: Analyze rewards
    if not analyze_reward_differentiation(results):
        logger.error("\nDiagnostics complete - see detailed output above")
        return 1

    logger.info("\n" + "=" * 90)
    logger.info("✅ TEST PASSED")
    logger.info("=" * 90)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)