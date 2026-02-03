"""CcaS-CcaR gene circuit environments.

This module provides modular task wrappers for the bi-stable genetic circuit:
- control: Standard tracking task (constant or sinewave targets)
- sysid: System identification task with randomized parameters

The physics layer (Gillespie algorithm) is shared across all tasks.
"""

from .ccas_ccar.tasks import control, sysid

__all__ = ["control", "sysid"]
