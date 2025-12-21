# NEW FILE: error_handler.py

"""
Comprehensive error handling for the problem generation framework.
"""

import logging
from typing import Callable, Any, Dict
import functools

logger = logging.getLogger(__name__)


class GenerationException(Exception):
    """Base exception for problem generation."""
    pass


class StateValidationException(GenerationException):
    """Exception during state validation."""
    pass


class ActionExecutionException(GenerationException):
    """Exception during action execution."""
    pass


class ProblemGenerationException(GenerationException):
    """Exception during problem generation."""
    pass


def safe_execute(func: Callable) -> Callable:
    """Decorator for safe execution with comprehensive error handling."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except StateValidationException as e:
            logger.error(f"State validation error in {func.__name__}: {e}")
            raise
        except ActionExecutionException as e:
            logger.error(f"Action execution error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise GenerationException(f"Error in {func.__name__}: {str(e)[:200]}")

    return wrapper