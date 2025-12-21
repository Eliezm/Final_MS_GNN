# FILE: logging_utils.py
import logging
import sys
from pathlib import Path


def setup_handshake_logging(log_file="handshake_debug.log"):
    """Setup detailed logging for handshake debugging."""

    # Create logger
    logger = logging.getLogger("HANDSHAKE")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_phase(logger, phase_num, name, symbol="="):
    """Log a phase marker."""
    msg = f"PHASE {phase_num}: {name}"
    logger.info(f"\n{symbol * 80}")
    logger.info(msg)
    logger.info(f"{symbol * 80}\n")


def log_file_check(logger, file_path, operation="checking"):
    """Log file existence check with details."""
    import os
    path = Path(file_path)

    logger.debug(f"[FILE CHECK] {operation}: {path}")

    if path.exists():
        size = path.stat().st_size
        logger.debug(f"  [OK] EXISTS - Size: {size} bytes")
        return True
    else:
        logger.debug(f"  [MISSING] Path: {path.absolute()}")
        return False


def log_signal_export(logger, signal_name, file_path, success=True):
    """Log signal export status."""
    status = "[OK]" if success else "[FAIL]"
    logger.info(f"{status} Signal: {signal_name:<30} -> {Path(file_path).name}")