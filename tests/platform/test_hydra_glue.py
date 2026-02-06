"""Tests for Hydra glue utilities."""

import sys
from unittest.mock import MagicMock

from myriad.platform.runner_utils import run_with_hydra


def test_run_with_hydra_reconstructs_argv():
    """Test that run_with_hydra correctly sets sys.argv for Hydra."""
    mock_runner = MagicMock()
    original_argv = sys.argv[:]

    try:
        run_with_hydra(mock_runner, script_name="myriad train", args=["env.name=cartpole", "run.seed=42"])

        # Verify sys.argv was reconstructed (plus hydra.job.chdir=true from setup_hydra)
        assert sys.argv[0] == "myriad train"
        assert sys.argv[1] == "env.name=cartpole"
        assert sys.argv[2] == "run.seed=42"
        assert "hydra.job.chdir=true" in sys.argv

        # Verify runner was called
        mock_runner.assert_called_once()

    finally:
        # Restore original argv
        sys.argv = original_argv


def test_run_with_hydra_default_script_name():
    """Test run_with_hydra with default script name."""
    mock_runner = MagicMock()
    original_argv = sys.argv[:]

    try:
        run_with_hydra(mock_runner, args=[])

        assert sys.argv[0] == "myriad"
        assert "hydra.job.chdir=true" in sys.argv

    finally:
        sys.argv = original_argv
