"""Performance benchmarks for Myriad.

This module contains benchmarks for validating performance claims and
tracking regressions. Benchmarks are designed to:

1. Measure throughput scaling (steps/second vs num_envs)
2. Profile memory usage (CPU and GPU)
3. Compare against other libraries (Gymnax, Gymnasium)
4. Track compilation overhead

Benchmarks are separate from user-facing scripts and include statistical
analysis and visualization code.
"""
