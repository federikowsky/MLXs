from __future__ import annotations

from benchmarks.mlxs_vs_mlx_lm.backends import (
    CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD,
    _use_benchmark_prepared_step_lookahead,
)


class _Model:
    def __init__(self, model_type: str | None, *, hidden_size: int = 0) -> None:
        self.model_type = model_type
        self.args = type("_Args", (), {"hidden_size": hidden_size})()


def test_prepared_step_lookahead_uses_short_prompt_threshold_for_generic_models() -> None:
    model = _Model("llama", hidden_size=2048)

    assert _use_benchmark_prepared_step_lookahead(
        model,
        prompt_token_count=CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD,
    )
    assert not _use_benchmark_prepared_step_lookahead(
        model,
        prompt_token_count=CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD + 1,
    )


def test_prepared_step_lookahead_extends_to_qwen2_long_prompts() -> None:
    model = _Model("qwen2", hidden_size=1536)

    assert _use_benchmark_prepared_step_lookahead(
        model,
        prompt_token_count=CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD + 1,
    )


def test_prepared_step_lookahead_extends_to_llama_3b_long_prompts_only() -> None:
    llama_3b = _Model("llama", hidden_size=3072)
    llama_1b = _Model("llama", hidden_size=2048)

    assert _use_benchmark_prepared_step_lookahead(
        llama_3b,
        prompt_token_count=CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD + 1,
    )
    assert not _use_benchmark_prepared_step_lookahead(
        llama_1b,
        prompt_token_count=CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD + 1,
    )
