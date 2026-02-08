"""Backend implementations for the logging system.

Each backend handles a specific destination:
- MemoryBackend: In-memory metric capture for return values
- DiskBackend: Episode file persistence
- WandbBackend: W&B remote logging
"""

from .disk import DiskBackend
from .memory import MemoryBackend
from .wandb import WandbBackend

__all__ = ["MemoryBackend", "DiskBackend", "WandbBackend"]
