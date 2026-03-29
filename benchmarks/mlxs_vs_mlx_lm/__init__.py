"""Serious runtime comparison: MLXs vs mlx-lm (no Adaptive KV).

Named scenarios (greedy/sampling, repetition penalty, logprobs, compile, quantized KV)
are in ``scenarios.py``; CLI: ``--scenarios`` / ``--list-scenarios``.

Default MLXs ``compile_decode`` for built-in greedy/sampling scenarios is on;
``--no-compile-decode`` applies only to ``cli_default`` (empty ``--scenarios``).

Per model, each backend is loaded once per run, then all scenarios × prompts
are executed on that session (``config.load_once_per_backend`` in JSON).

Run: ``PYTHONPATH=. python -m benchmarks.mlxs_vs_mlx_lm --help``
"""
