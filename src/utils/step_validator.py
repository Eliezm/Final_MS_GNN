#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP OUTPUT VALIDATOR
=====================

Validates that Gymnasium step() returns are proper Python types, not numpy/torch scalars.
This catches the "only 0-dimensional arrays can be converted to Python scalars" error early.

Usage:
    validated_step = validate_step_output(obs, reward, done, truncated, info)
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def validate_reward(reward: Any) -> float:
    """
    Validate and convert reward to Python float.

    Args:
        reward: Value from environment step

    Returns:
        reward: Python float

    Raises:
        TypeError: If reward cannot be converted to float
    """
    try:
        # Convert to Python float
        reward_float = float(reward)

        # Check type
        if not isinstance(reward_float, float):
            raise TypeError(f"Reward is not Python float: {type(reward_float)}")

        # Check for NaN/Inf
        if np.isnan(reward_float):
            logger.warning("⚠️ Reward is NaN, setting to -1.0")
            reward_float = -1.0
        elif np.isinf(reward_float):
            logger.warning("⚠️ Reward is Inf, clamping to ±2.0")
            reward_float = 2.0 if reward_float > 0 else -2.0

        return reward_float

    except Exception as e:
        raise TypeError(f"Cannot convert reward to float: {type(reward).__name__} - {e}")


def validate_done_truncated(done: Any, truncated: Any) -> Tuple[bool, bool]:
    """
    Validate and convert done/truncated to Python booleans.

    Args:
        done: done signal from environment
        truncated: truncated signal from environment

    Returns:
        (done, truncated): Python booleans

    Raises:
        TypeError: If values cannot be converted to bool
    """
    try:
        done_bool = bool(done)
        truncated_bool = bool(truncated)

        if not isinstance(done_bool, bool):
            raise TypeError(f"done is not Python bool: {type(done_bool)}")
        if not isinstance(truncated_bool, bool):
            raise TypeError(f"truncated is not Python bool: {type(truncated_bool)}")

        return done_bool, truncated_bool

    except Exception as e:
        raise TypeError(f"Cannot convert done/truncated to bool: {e}")


def validate_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate observation dictionary structure.

    Args:
        obs: Observation from environment

    Returns:
        obs: Validated observation

    Raises:
        TypeError: If observation has invalid structure
    """
    if not isinstance(obs, dict):
        raise TypeError(f"Observation must be dict, got {type(obs).__name__}")

    # Check for required keys
    required_keys = ['x', 'edge_index', 'edge_features', 'num_nodes', 'num_edges']
    for key in required_keys:
        if key not in obs:
            raise KeyError(f"Missing required observation key: {key}")

    # Check dtypes
    if not isinstance(obs['x'], np.ndarray):
        raise TypeError(f"x must be numpy array, got {type(obs['x'])}")
    if obs['x'].dtype != np.float32:
        raise TypeError(f"x must be float32, got {obs['x'].dtype}")

    if not isinstance(obs['edge_index'], np.ndarray):
        raise TypeError(f"edge_index must be numpy array, got {type(obs['edge_index'])}")

    if not isinstance(obs['edge_features'], np.ndarray):
        raise TypeError(f"edge_features must be numpy array, got {type(obs['edge_features'])}")

    return obs


def validate_step_output(
    obs: Dict[str, Any],
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
    """
    Validate complete step() output.

    This function checks that all returns are proper Python types, not numpy/torch scalars.

    Args:
        obs: Observation dict
        reward: Reward value
        terminated: Done signal
        truncated: Truncated signal
        info: Info dict

    Returns:
        (obs, reward, terminated, truncated, info): Validated outputs

    Raises:
        TypeError: If any output has wrong type
    """
    try:
        # Validate each component
        obs = validate_observation(obs)
        reward = validate_reward(reward)
        terminated, truncated = validate_done_truncated(terminated, truncated)

        if not isinstance(info, dict):
            raise TypeError(f"info must be dict, got {type(info).__name__}")

        return obs, reward, terminated, truncated, info

    except TypeError as e:
        logger.error(f"❌ Step output validation failed: {e}")
        raise


class StepValidatorWrapper:
    """Wrapper for environment that validates step() outputs."""

    def __init__(self, env, strict: bool = True):
        """
        Initialize validator wrapper.

        Args:
            env: Gymnasium environment to wrap
            strict: If True, raise errors; if False, log warnings
        """
        self.env = env
        self.strict = strict

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Call env.step() and validate output.

        Args:
            action: Action to take

        Returns:
            (obs, reward, terminated, truncated, info): Validated step output
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        try:
            obs, reward, terminated, truncated, info = validate_step_output(
                obs, reward, terminated, truncated, info
            )
            return obs, reward, terminated, truncated, info

        except TypeError as e:
            if self.strict:
                raise
            else:
                logger.warning(f"⚠️ Step validation warning (continuing): {e}")
                # Return anyway, trying to fix what we can
                try:
                    reward = float(reward) if not isinstance(reward, float) else reward
                    terminated = bool(terminated)
                    truncated = bool(truncated)
                except Exception:
                    pass
                return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset and validate observation."""
        obs, info = self.env.reset(**kwargs)
        try:
            obs = validate_observation(obs)
        except TypeError as e:
            if self.strict:
                raise
            logger.warning(f"⚠️ Reset observation validation warning: {e}")
        return obs, info

    def __getattr__(self, name):
        """Delegate other methods to wrapped env."""
        return getattr(self.env, name)


# Utility function for easy wrapping
def wrap_with_validation(env, strict: bool = True):
    """
    Wrap an environment with step validation.

    Args:
        env: Gymnasium environment
        strict: If True, raise errors on validation failure

    Returns:
        Wrapped environment
    """
    return StepValidatorWrapper(env, strict=strict)