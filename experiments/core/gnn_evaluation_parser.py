#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN EVALUATION PARSER - Fast Downward output parsing
=====================================================
Extracts metrics from FD logs and execution output.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FastDownwardOutputParser:
    """
    Parse Fast Downward output to extract comprehensive metrics.

    Handles:
    - Plan cost and length
    - Search statistics (expansions, generations, depth)
    - Timing breakdown
    - M&S specific metrics
    """

    @staticmethod
    def parse_fd_log_file(log_path: str) -> Dict[str, Any]:
        """
        Parse complete FD log file.
        """
        metrics = {
            'solved': False,
            'plan_cost': 0,
            'plan_length': 0,
            'nodes_expanded': 0,
            'search_time': 0.0,
        }

        try:
            if not Path(log_path).exists():
                logger.warning(f"FD log file not found: {log_path}")
                return metrics

            with open(log_path, 'r') as f:
                content = f.read()

            # ✅ FIX: More robust solution detection
            solved_indicators = [
                'Solution found',
                'Plan length:',
                'Expanded',
                'Evaluating',
                'Preferred',
            ]

            metrics['solved'] = any(indicator in content for indicator in solved_indicators)

            # Parse metrics
            parsed = FastDownwardOutputParser.parse_search_output(content)
            metrics.update(parsed)

        except Exception as e:
            logger.warning(f"Failed to parse log file {log_path}: {e}")

        return metrics

    @staticmethod
    def parse_search_output(output_text: str) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from FD search output.
        """
        metrics = {
            'solved': False,
            'plan_cost': 0,
            'plan_length': 0,
            'nodes_expanded': 0,
            'nodes_generated': 0,
            'search_depth': 0,
            'branching_factor': 1.0,
            'search_time': 0.0,
            'peak_memory_kb': 0,
            'plan_found': False,
        }

        try:
            # ✅ FIX: More comprehensive solution detection
            if any(indicator in output_text for indicator in [
                "Solution found",
                "Plan length:",
                "Evaluating",
                "Preferred operators used",
            ]):
                metrics['solved'] = True
                metrics['plan_found'] = True

            # Extract plan length/cost
            match = re.search(r'Plan length:\s*(\d+)', output_text)
            if match:
                metrics['plan_length'] = int(match.group(1))
                metrics['plan_cost'] = int(match.group(1))

            # Extract nodes expanded
            matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', output_text))
            if matches:
                metrics['nodes_expanded'] = int(matches[-1].group(1))

            # Extract nodes generated
            match = re.search(r'Generated\s+(\d+)\s+state', output_text)
            if match:
                metrics['nodes_generated'] = int(match.group(1))

            # Extract search depth
            match = re.search(r'Search depth:\s*(\d+)', output_text)
            if match:
                metrics['search_depth'] = int(match.group(1))

            # Extract branching factor
            match = re.search(r'Branching factor:\s*([\d.]+)', output_text)
            if match:
                metrics['branching_factor'] = float(match.group(1))

            # Extract search time
            match = re.search(r'Search time:\s*([\d.]+)s', output_text)
            if match:
                metrics['search_time'] = float(match.group(1))

            # Extract peak memory
            match = re.search(r'Peak memory:\s*(\d+)\s*KB', output_text)
            if match:
                metrics['peak_memory_kb'] = int(match.group(1))

        except Exception as e:
            logger.warning(f"Error parsing FD output: {e}")

        return metrics

    @staticmethod
    def extract_ms_metrics(output_text: str) -> Dict[str, Any]:
        """
        Extract Merge-and-Shrink specific metrics from output.

        Returns:
            Dict with M&S metrics
        """
        ms_metrics = {
            'max_abstraction_size': None,
            'final_abstraction_size': None,
            'num_merges': None,
            'num_shrinks': None,
            'num_label_reductions': None,
        }

        try:
            match = re.search(r'Max abstract state:\s*(\d+)', output_text)
            if match:
                ms_metrics['max_abstraction_size'] = int(match.group(1))

            match = re.search(r'Final abstraction:\s*(\d+)', output_text)
            if match:
                ms_metrics['final_abstraction_size'] = int(match.group(1))

            match = re.search(r'Total merges:\s*(\d+)', output_text)
            if match:
                ms_metrics['num_merges'] = int(match.group(1))

            match = re.search(r'Total shrinks:\s*(\d+)', output_text)
            if match:
                ms_metrics['num_shrinks'] = int(match.group(1))

            match = re.search(r'Total label reduction:\s*(\d+)', output_text)
            if match:
                ms_metrics['num_label_reductions'] = int(match.group(1))

        except Exception as e:
            logger.warning(f"Error parsing M&S metrics: {e}")

        return ms_metrics