"""Interactive chat loop — multi-turn CLI chat with streaming and prompt cache.

Simple fallback chat (--simple-chat).  Uses the same deps as the server
(model, tokenizer, generate_fn, prompt_cache).  Streams tokens to stdout
and maintains conversation history; uses prompt_cache for KV reuse.

The new default chat experience is the Textual TUI (ui/tui/app.py).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mlxs._types import GenerateOptions
from mlxs.chat.template import (
    build_prompt_ids,
    collect_stop_token_ids,
    sanitize_assistant_text,
)
from mlxs.config.schema import GenerateConfig
from mlxs.server.deps import Dependencies

logger = logging.getLogger(__name__)


def run_chat_loop(
    deps: Dependencies,
    options: GenerateOptions | None = None,
) -> None:
    """Run the interactive multi-turn chat loop (simple/fallback mode).

    Reads user input, appends to messages, builds prompt via chat template
    (tokenize=True — no double-format), uses prompt_cache for prefix reuse,
    calls generate_fn, streams tokens to stdout, appends sanitised assistant
    reply, updates prompt cache.

    Exits on empty input, /quit, or KeyboardInterrupt.
    """
    # Discover extra stop tokens once at startup
    extra_stop_ids = collect_stop_token_ids(deps.tokenizer)

    gen_opts = options if options is not None else _options_from_config(
        deps.config.generate, extra_stop_ids=extra_stop_ids,
    )
    messages: list[dict[str, str]] = []
    model_id = _model_id_for_cache(deps)

    logger.info("Chat started (model_id=%s). Type an empty line or /quit to exit.", model_id)

    while True:
        try:
            line = _read_line()
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            break
        if line is None:
            break

        line = line.strip()
        if not line or line.lower() == "/quit":
            break

        messages.append({"role": "user", "content": line})

        # Build prompt as token ids directly (no intermediate string)
        try:
            prompt_token_ids = build_prompt_ids(deps.tokenizer, messages)
        except Exception as exc:
            print(f"Error applying chat template: {exc}", file=sys.stderr)
            messages.pop()
            continue

        if not prompt_token_ids:
            messages.pop()
            continue

        cache_state, prefix_len = deps.prompt_cache.get(model_id, tuple(prompt_token_ids))
        suffix_len = len(prompt_token_ids) - prefix_len
        prompt_for_gen: list[int]
        if cache_state is not None and prefix_len > 0 and suffix_len > 0:
            prompt_for_gen = list(prompt_token_ids[prefix_len:])
            cache_for_gen = cache_state
        else:
            prompt_for_gen = prompt_token_ids
            cache_for_gen = None

        final_cache_ref: list[Any] = []
        gen_kwargs: dict[str, Any] = {
            "prefill_step_size": deps.config.generate.prefill_step_size,
            "compile_decode": deps.config.generate.compile_decode,
            "clear_cache_interval": deps.config.generate.clear_cache_interval,
            "final_cache_out": final_cache_ref,
        }
        if cache_for_gen is not None:
            gen_kwargs["cache"] = cache_for_gen

        generated_ids: list[int] = []
        try:
            for event in deps.generate_fn(
                deps.model,
                deps.tokenizer,
                prompt_for_gen,
                gen_opts,
                **gen_kwargs,
            ):
                print(event.text, end="", flush=True)
                generated_ids.append(event.token_id)
                if event.finish_reason is not None:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            break
        except Exception as exc:
            print(f"\nGeneration error: {exc}", file=sys.stderr)
            messages.pop()
            continue

        print()

        cache_to_put = (
            cache_for_gen
            if cache_for_gen is not None
            else (final_cache_ref[0] if final_cache_ref else None)
        )
        if cache_to_put is not None:
            new_prefix = tuple(prompt_token_ids) + tuple(generated_ids)
            deps.prompt_cache.put(model_id, new_prefix, cache_to_put)

        full_response = sanitize_assistant_text(deps.tokenizer.decode(generated_ids))
        messages.append({"role": "assistant", "content": full_response})

    logger.info("Chat exiting.")


def _options_from_config(
    c: GenerateConfig,
    *,
    extra_stop_ids: tuple[int, ...] = (),
) -> GenerateOptions:
    """Build GenerateOptions from GenerateConfig, merging discovered stop ids."""
    merged_eos = tuple(sorted(set(c.extra_eos_token_ids) | set(extra_stop_ids)))
    return GenerateOptions(
        max_tokens=c.max_tokens,
        temperature=c.temperature,
        top_p=c.top_p,
        top_k=c.top_k,
        min_p=c.min_p,
        seed=c.seed,
        stop_sequences=c.stop_sequences,
        extra_eos_token_ids=merged_eos,
        repetition_penalty=c.repetition_penalty,
        logprobs=c.logprobs,
        top_logprobs=c.top_logprobs,
        stream=True,
    )


def _model_id_for_cache(deps: Dependencies) -> str:
    """Stable model id for prompt cache keys."""
    return deps.config.model.model_path or "default"

def _read_line() -> str | None:
    """Read one line from stdin. Returns None on EOF."""
    try:
        return input()
    except EOFError:
        return None
