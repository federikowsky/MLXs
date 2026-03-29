"""Serious runtime comparison: MLXs vs mlx-lm (no Adaptive KV).

Defaults: MLXs ``compile_decode`` on, ``prefill_step`` 2048 (aligned with mlx-lm).

Run: ``PYTHONPATH=. python -m benchmarks.mlxs_vs_mlx_lm --help``
"""
