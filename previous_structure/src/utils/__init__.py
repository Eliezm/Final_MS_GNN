from .common_utils import (
    PROJECT_ROOT, DOWNWARD_DIR, FD_OUTPUT_DIR, GNN_OUTPUT_DIR,
    ThinClientConfig,
    ensure_directories, cleanup_signal_files
)

import sys
from pathlib import Path

# Add project root to path for easy imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


__all__ = [
    "PROJECT_ROOT", "DOWNWARD_DIR", "FD_OUTPUT_DIR", "GNN_OUTPUT_DIR",
    "ThinClientConfig",
    "ensure_directories", "cleanup_signal_files"
]
