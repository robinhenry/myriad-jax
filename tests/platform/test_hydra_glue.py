"""Tests for Hydra glue utilities."""

import os
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


def test_run_with_hydra_auto_tune_sets_env_var():
    """auto_tune=True should set MYRIAD_AUTO_TUNE=1 in the environment when the runner is called."""
    env_state_during_call = {}

    def capturing_runner():
        env_state_during_call["value"] = os.environ.get("MYRIAD_AUTO_TUNE")

    original_argv = sys.argv[:]
    os.environ.pop("MYRIAD_AUTO_TUNE", None)  # ensure clean slate

    try:
        run_with_hydra(capturing_runner, args=[], auto_tune=True)
        assert env_state_during_call["value"] == "1"
    finally:
        sys.argv = original_argv
        os.environ.pop("MYRIAD_AUTO_TUNE", None)


def test_run_with_hydra_no_auto_tune_does_not_set_env_var():
    """auto_tune=False (default) must not set MYRIAD_AUTO_TUNE."""
    env_state_during_call = {}

    def capturing_runner():
        env_state_during_call["value"] = os.environ.get("MYRIAD_AUTO_TUNE")

    original_argv = sys.argv[:]
    os.environ.pop("MYRIAD_AUTO_TUNE", None)

    try:
        run_with_hydra(capturing_runner, args=[])
        assert env_state_during_call["value"] is None
    finally:
        sys.argv = original_argv
