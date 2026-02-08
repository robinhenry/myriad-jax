"""Tests for memory estimation utilities."""

import jax.numpy as jnp

from myriad.utils.memory import estimate_pytree_memory_mb, get_array_memory_mb


class TestArrayMemory:
    """Tests for get_array_memory_mb."""

    def test_basic_array_memory_calculation(self):
        """Test basic array memory estimation."""
        # 1000 * 100 * 4 bytes (float32) = 400,000 bytes = 0.38147 MB
        arr = jnp.zeros((1000, 100), dtype=jnp.float32)
        mem_mb = get_array_memory_mb(arr)
        expected_mb = 400_000 / (1024 * 1024)
        assert abs(mem_mb - expected_mb) < 0.001

    def test_different_dtypes(self):
        """Test memory calculation with different data types."""
        shape = (100, 100)

        # float32: 4 bytes per element
        arr_f32 = jnp.zeros(shape, dtype=jnp.float32)
        mem_f32 = get_array_memory_mb(arr_f32)
        # Use actual dtype to calculate expected (in case JAX truncates)
        bytes_per_elem = arr_f32.dtype.itemsize
        expected_f32 = (100 * 100 * bytes_per_elem) / (1024 * 1024)
        assert abs(mem_f32 - expected_f32) < 0.001

        # int32: 4 bytes per element
        arr_i32 = jnp.zeros(shape, dtype=jnp.int32)
        mem_i32 = get_array_memory_mb(arr_i32)
        bytes_per_elem = arr_i32.dtype.itemsize
        expected_i32 = (100 * 100 * bytes_per_elem) / (1024 * 1024)
        assert abs(mem_i32 - expected_i32) < 0.001

        # int32 and float32 should use same memory (both 4 bytes)
        assert abs(mem_f32 - mem_i32) < 0.001

    def test_memory_scales_linearly(self):
        """Larger arrays should use proportionally more memory."""
        arr1 = jnp.zeros((100, 100), dtype=jnp.float32)
        arr2 = jnp.zeros((200, 200), dtype=jnp.float32)  # 4x larger

        mem1 = get_array_memory_mb(arr1)
        mem2 = get_array_memory_mb(arr2)

        assert abs(mem2 / mem1 - 4.0) < 0.01

    def test_zero_size_array(self):
        """Empty arrays should have zero or negligible memory."""
        arr = jnp.zeros((0, 0), dtype=jnp.float32)
        mem_mb = get_array_memory_mb(arr)
        assert mem_mb == 0.0


class TestPyTreeMemory:
    """Tests for estimate_pytree_memory_mb."""

    def test_simple_pytree(self):
        """Test memory estimation for simple PyTree."""
        pytree = {
            "a": jnp.zeros((100, 100), dtype=jnp.float32),  # 40KB
        }
        mem_mb = estimate_pytree_memory_mb(pytree)
        expected_mb = (100 * 100 * 4) / (1024 * 1024)
        assert abs(mem_mb - expected_mb) < 0.001

    def test_nested_pytree(self):
        """Test memory estimation for nested structures."""
        pytree = {
            "a": jnp.zeros((100, 100), dtype=jnp.float32),  # 40KB
            "b": {
                "c": jnp.zeros((50, 50), dtype=jnp.float32),  # 10KB
                "d": jnp.zeros((25, 25), dtype=jnp.float32),  # 2.5KB
            },
        }
        mem_mb = estimate_pytree_memory_mb(pytree)
        total_bytes = (100 * 100 * 4) + (50 * 50 * 4) + (25 * 25 * 4)
        expected_mb = total_bytes / (1024 * 1024)
        assert abs(mem_mb - expected_mb) < 0.001

    def test_list_pytree(self):
        """Test memory estimation for list-based PyTrees."""
        pytree = [
            jnp.zeros((100, 100), dtype=jnp.float32),
            jnp.zeros((50, 50), dtype=jnp.float32),
        ]
        mem_mb = estimate_pytree_memory_mb(pytree)
        total_bytes = (100 * 100 * 4) + (50 * 50 * 4)
        expected_mb = total_bytes / (1024 * 1024)
        assert abs(mem_mb - expected_mb) < 0.001

    def test_mixed_pytree(self):
        """Test PyTree with mixed types (should ignore non-arrays)."""
        pytree = {
            "array": jnp.zeros((100, 100), dtype=jnp.float32),
            "scalar": 42,
            "string": "ignored",
            "none": None,
        }
        mem_mb = estimate_pytree_memory_mb(pytree)
        # Should only count the array
        expected_mb = (100 * 100 * 4) / (1024 * 1024)
        assert abs(mem_mb - expected_mb) < 0.001

    def test_empty_pytree(self):
        """Empty PyTree should have zero memory."""
        pytree = {}
        mem_mb = estimate_pytree_memory_mb(pytree)
        assert mem_mb == 0.0

    def test_pytree_deterministic(self):
        """Same PyTree should always return same memory estimate."""
        pytree = {
            "a": jnp.zeros((100, 100), dtype=jnp.float32),
            "b": jnp.ones((50, 50), dtype=jnp.int32),
        }
        mem1 = estimate_pytree_memory_mb(pytree)
        mem2 = estimate_pytree_memory_mb(pytree)
        assert mem1 == mem2
