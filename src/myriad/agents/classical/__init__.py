"""Classical control agents (non-learning).

This module contains traditional control theory approaches like bang-bang,
PID, and random baseline controllers.
"""

from . import bangbang, pid, random

__all__ = ["bangbang", "pid", "random"]
