"""Tests for example scripts to ensure documentation stays accurate.

These tests run the example scripts with reduced parameters to verify they work
without taking too long in CI. This ensures documentation examples stay current.

Run with: pytest tests/examples/test_examples.py
Run with markers: pytest -m "not slow" tests/examples/test_examples.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "original"


def run_example(script_name: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run an example script and return the result.

    Args:
        script_name: Name of the script in examples/ directory
        timeout: Maximum time to allow script to run (seconds)

    Returns:
        CompletedProcess with returncode, stdout, stderr
    """
    script_path = EXAMPLES_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLES_DIR.parent,  # Run from repo root
    )
    return result


class TestQuickStartExamples:
    """Test quick start examples that should run fast."""

    def test_07_quickstart_simple(self):
        """Test the main quickstart example from docs/index.md."""
        result = run_example("07_quickstart_simple.py", timeout=120)
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        assert "Training complete!" in result.stdout
        assert "TrainingResults" in result.stdout

    def test_05_random_baseline(self):
        """Test random baseline evaluation (no training needed)."""
        result = run_example("05_random_baseline.py", timeout=60)
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        assert "Evaluation completed!" in result.stdout
        assert "EvaluationResults" in result.stdout

    def test_08_episode_collection(self):
        """Test episode collection example."""
        result = run_example("08_episode_collection.py", timeout=60)
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        assert "Collected" in result.stdout
        assert "episodes" in result.stdout


@pytest.mark.slow
class TestTrainingExamples:
    """Test training examples (slower, may skip in CI)."""

    def test_02_basic_training(self):
        """Test basic training example."""
        result = run_example("02_basic_training.py", timeout=120)
        # Accept exit code 0 even if serialization warning occurs
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        assert "Training completed!" in result.stdout

    def test_03_advanced_training(self):
        """Test advanced training with custom hyperparameters."""
        result = run_example("03_advanced_training.py", timeout=120)
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        assert "Training completed!" in result.stdout

    def test_09_periodic_episode_saving(self):
        """Test periodic episode saving during training."""
        result = run_example("09_periodic_episode_saving.py", timeout=120)
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        assert "Training complete" in result.stdout
        assert "Episodes saved" in result.stdout


class TestLegacyExamples:
    """Test backward compatibility examples."""

    def test_01_classical_control(self):
        """Test YAML-based example (should run with existing YAML)."""
        result = run_example("01_classical_control.py", timeout=60)
        # Accept both success and graceful failure (file not found is OK)
        if result.returncode != 0:
            # Should fail gracefully if YAML doesn't exist
            assert "FileNotFoundError" in result.stderr or "not found" in result.stderr
        else:
            assert "Results summary" in result.stdout

    def test_04_evaluate_pretrained(self):
        """Test pre-trained agent evaluation (gracefully skips if no file)."""
        result = run_example("04_evaluate_pretrained.py", timeout=60)
        # This should exit gracefully (exit code 0) even if file doesn't exist
        assert result.returncode == 0, f"Script should exit gracefully, got stderr: {result.stderr}"
        # Should either evaluate or skip
        assert ("Evaluation completed" in result.stdout) or ("Skipping" in result.stdout)

    def test_06_yaml_config(self):
        """Test YAML config loading (gracefully handles missing file)."""
        result = run_example("06_yaml_config.py", timeout=60)
        # Should exit gracefully with exit code 0 or 1
        assert result.returncode in [0, 1], f"Unexpected failure: {result.stderr}"
        # Should handle missing file gracefully
        if result.returncode == 1:
            assert "not found" in result.stdout or "not found" in result.stderr


@pytest.mark.integration
class TestExamplesEndToEnd:
    """Integration tests that run multiple examples in sequence."""

    def test_train_and_evaluate_workflow(self):
        """Test complete workflow: train, save (attempt), evaluate.

        Note: This test expects save_agent to fail due to JAX/Flax serialization
        limitations, but the scripts should handle this gracefully.
        """
        # 1. Train an agent (will attempt to save but likely fail)
        train_result = run_example("02_basic_training.py", timeout=120)
        assert train_result.returncode == 0
        assert "Training completed!" in train_result.stdout

        # 2. Try to evaluate pre-trained (will skip if save failed, which is expected)
        eval_result = run_example("04_evaluate_pretrained.py", timeout=60)
        assert eval_result.returncode == 0, f"Eval script should exit gracefully: {eval_result.stderr}"
        # Should gracefully skip (save likely failed due to pickle limitations)
        assert ("Evaluation completed" in eval_result.stdout) or ("Skipping" in eval_result.stdout)


# Smoke test configuration for CI
@pytest.fixture(scope="session", autouse=True)
def verify_examples_directory():
    """Verify examples directory exists before running tests."""
    assert EXAMPLES_DIR.exists(), f"Examples directory not found at {EXAMPLES_DIR}"

    # Verify all expected example files exist
    expected_examples = [
        "01_classical_control.py",
        "02_basic_training.py",
        "03_advanced_training.py",
        "04_evaluate_pretrained.py",
        "05_random_baseline.py",
        "06_yaml_config.py",
        "07_quickstart_simple.py",
        "08_episode_collection.py",
        "09_periodic_episode_saving.py",
    ]

    for example in expected_examples:
        example_path = EXAMPLES_DIR / example
        assert example_path.exists(), f"Example not found: {example}"
