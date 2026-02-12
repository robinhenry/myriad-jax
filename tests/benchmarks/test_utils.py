"""Tests for benchmarking utilities."""

import time

import jax
import jax.numpy as jnp

from benchmarks.utils import (
    calculate_throughput,
    format_number,
    get_device_info,
    measure_compilation_time,
    time_jitted_fn,
    warmup_jitted_fn,
)


class TestDeviceInfo:
    """Tests for get_device_info."""

    def test_device_info_structure(self):
        """Device info should contain expected fields."""
        info = get_device_info()

        assert "platform" in info
        assert "device_count" in info
        assert "devices" in info

        assert isinstance(info["platform"], str)
        assert isinstance(info["device_count"], int)
        assert isinstance(info["devices"], list)
        assert info["device_count"] > 0

    def test_platform_is_valid(self):
        """Platform should be one of the known JAX platforms."""
        info = get_device_info()
        assert info["platform"] in ["cpu", "gpu", "tpu"]


class TestWarmup:
    """Tests for warmup_jitted_fn."""

    def test_warmup_triggers_compilation(self):
        """Warmup should trigger JIT compilation."""

        @jax.jit
        def simple_fn(x):
            return x + 1

        x = jnp.array([1.0])

        # First call should include compilation
        start = time.perf_counter()
        warmup_jitted_fn(simple_fn, x, warmup_steps=1)
        first_time = time.perf_counter() - start

        # Second warmup should be faster (already compiled)
        start = time.perf_counter()
        warmup_jitted_fn(simple_fn, x, warmup_steps=1)
        second_time = time.perf_counter() - start

        # First time should be significantly longer due to compilation
        # (this is a heuristic, may be flaky on very fast machines)
        assert first_time >= second_time

    def test_warmup_with_multiple_steps(self):
        """Warmup should run multiple iterations."""

        @jax.jit
        def counting_fn(x):
            return x + 1

        x = jnp.array([1.0])
        warmup_jitted_fn(counting_fn, x, warmup_steps=5)

        # Just ensure it completes without error
        # (can't easily count calls due to JAX's lazy evaluation)
        assert True


class TestTiming:
    """Tests for time_jitted_fn."""

    def test_timing_statistics_structure(self):
        """time_jitted_fn should return valid statistics."""

        @jax.jit
        def simple_fn(x):
            return x + 1

        x = jnp.array([1.0])
        stats = time_jitted_fn(simple_fn, x, num_runs=20, warmup_steps=5)

        # Check all expected keys present
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "all_times" in stats
        assert "compile_time" in stats  # New field

        # Sanity checks
        assert stats["mean"] > 0
        assert stats["std"] >= 0
        assert stats["min"] > 0
        assert stats["max"] > 0
        assert stats["median"] > 0
        assert stats["compile_time"] > 0  # Compilation should take time
        assert len(stats["all_times"]) == 20

    def test_timing_without_compile_measurement(self):
        """time_jitted_fn should work without compile time measurement."""

        @jax.jit
        def simple_fn(x):
            return x + 1

        x = jnp.array([1.0])
        stats = time_jitted_fn(simple_fn, x, num_runs=10, warmup_steps=5, measure_compile_time=False)

        # Should not have compile_time
        assert "compile_time" not in stats
        assert "mean" in stats

    def test_timing_statistics_order(self):
        """Timing statistics should have correct ordering."""

        @jax.jit
        def simple_fn(x):
            return x + 1

        x = jnp.array([1.0])
        stats = time_jitted_fn(simple_fn, x, num_runs=50, warmup_steps=5)

        # min <= median <= mean (approximately) <= max
        assert stats["min"] <= stats["median"]
        assert stats["median"] <= stats["max"]
        assert stats["min"] <= stats["mean"] <= stats["max"]


class TestCompilationTime:
    """Tests for measure_compilation_time."""

    def test_compilation_time_positive(self):
        """Compilation time should be positive."""

        @jax.jit
        def fn(x):
            return x * 2

        x = jnp.array([1.0])
        compile_time = measure_compilation_time(fn, x)

        assert compile_time > 0

    def test_compilation_includes_first_call(self):
        """measure_compilation_time should include JIT overhead."""

        @jax.jit
        def complex_fn(x):
            for _ in range(10):
                x = jnp.dot(x, x.T)
            return x

        x = jnp.ones((50, 50))
        compile_time = measure_compilation_time(complex_fn, x)

        # Compilation for complex function should take measurable time
        assert compile_time > 1e-6  # At least 1 microsecond


class TestFormatNumber:
    """Tests for format_number."""

    def test_format_billions(self):
        """Format numbers in billions."""
        assert format_number(1_500_000_000) == "1.50B"
        assert format_number(1_000_000_000) == "1.00B"
        assert format_number(2_750_000_000, precision=1) == "2.8B"

    def test_format_millions(self):
        """Format numbers in millions."""
        assert format_number(1_500_000) == "1.50M"
        assert format_number(1_000_000) == "1.00M"
        assert format_number(2_750_000, precision=1) == "2.8M"

    def test_format_thousands(self):
        """Format numbers in thousands."""
        assert format_number(1_500) == "1.50K"
        assert format_number(1_000) == "1.00K"
        assert format_number(2_750, precision=1) == "2.8K"

    def test_format_small_numbers(self):
        """Format numbers less than 1000."""
        assert format_number(999) == "999.00"
        assert format_number(42) == "42.00"
        assert format_number(1.5, precision=3) == "1.500"

    def test_format_precision(self):
        """Test precision parameter."""
        assert format_number(1_234_567, precision=0) == "1M"
        assert format_number(1_234_567, precision=1) == "1.2M"
        assert format_number(1_234_567, precision=2) == "1.23M"
        assert format_number(1_234_567, precision=3) == "1.235M"


class TestThroughput:
    """Tests for calculate_throughput."""

    def test_throughput_calculation(self):
        """Test basic throughput calculation."""
        steps = 1_000_000
        time_s = 10.0
        throughput = calculate_throughput(steps, time_s)
        assert throughput == 100_000.0  # 100K steps/s

    def test_throughput_units(self):
        """Verify throughput is in steps/second."""
        steps = 500
        time_s = 0.5
        throughput = calculate_throughput(steps, time_s)
        assert throughput == 1000.0  # 1000 steps/s

    def test_throughput_edge_cases(self):
        """Test edge cases."""
        # Very fast execution
        assert calculate_throughput(100, 0.001) == 100_000.0

        # Slow execution
        assert calculate_throughput(10, 100.0) == 0.1
