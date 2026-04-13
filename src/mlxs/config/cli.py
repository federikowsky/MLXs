"""CLI argument parsing for config resolution (§8.1, plan-chat-cli).

Parses argv into subcommand, config_path, and cli_overrides. All config options
can be passed as explicit flags; -o KEY=VALUE remains for generic overrides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mlxs.config.loader import resolve
from mlxs.config.schema import AppConfig

_SENTINEL = object()

# More space between option and description; bool pairs (--x/--no-x) on one line.
_HELP_POSITION = 70


class _CLIHelpFormatter(argparse.HelpFormatter):
    """Wider help layout and single-line --flag / --no-flag for booleans."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("max_help_position", _HELP_POSITION)
        super().__init__(*args, **kwargs)

    def add_arguments(self, actions: list[argparse.Action]) -> None:
        """Merge consecutive store_true/store_false (same dest) into one line."""
        actions_list = list(actions)
        i = 0
        while i < len(actions_list):
            act = actions_list[i]
            next_act = actions_list[i + 1] if i + 1 < len(actions_list) else None
            if (
                next_act is not None
                and type(act).__name__ == "_StoreTrueAction"
                and type(next_act).__name__ == "_StoreFalseAction"
                and getattr(act, "dest", None) == getattr(next_act, "dest", None)
            ):
                opts = ", ".join(act.option_strings + next_act.option_strings)
                opts_plain = getattr(self, "_decolor", lambda s: s)(opts)
                inv_len = len(opts_plain) + self._current_indent
                self._action_max_length = max(self._action_max_length, inv_len)
                self._add_item(self._format_merged_bool_pair, [act, next_act])
                i += 2
            else:
                self.add_argument(act)
                i += 1

    def _format_merged_bool_pair(
        self, action_true: argparse.Action, action_false: argparse.Action
    ) -> str:
        """Single line for --flag, --no-flag with shared help."""
        help_position = min(
            self._action_max_length + 2,
            self._max_help_position,
        )
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        opts = ", ".join(action_true.option_strings + action_false.option_strings)
        opts_no_color = getattr(self, "_decolor", lambda s: s)(opts)
        if len(opts_no_color) <= action_width:
            action_header = (
                " " * self._current_indent
                + opts_no_color.ljust(action_width)
                + "  "
            )
        else:
            action_header = " " * self._current_indent + opts + "\n"
        help_text = self._expand_help(action_true)
        if help_text and help_text.strip():
            help_lines = self._split_lines(help_text, help_width)
            first_indent = 0 if len(opts_no_color) <= action_width else help_position
            parts = [
                action_header,
                " " * first_indent + help_lines[0] + "\n",
            ]
            for line in help_lines[1:]:
                parts.append(" " * help_position + line + "\n")
            return self._join_parts(parts)
        return self._join_parts([action_header, "\n"])

# (flag_name, config_key, kind, default, help_text)
# kind: str, int, float, bool, bool_false, choice, str_list, int_list
_CLI_SPECS: list[tuple[str, str, str, Any, str]] = [
    # model
    ("model", "model.model_path", "str", "", "Local path or Hugging Face model id"),
    ("adapter", "model.adapter_path", "str", None, "Optional adapter/LoRA path"),
    ("trust-remote-code", "model.trust_remote_code", "bool", False, "Allow trust_remote_code"),
    ("model-type", "model.model_type", "str", None, "Model architecture (auto-detected if unset)"),
    ("lazy-load", "model.lazy_load", "bool", True, "Load model on first use"),
    ("preload", "model.preload", "bool", False, "Load model at startup"),
    ("model-mode", "model.model_mode", "choice", "auto", "Execution mode: text, multimodal, auto"),
    (
        "weight-format",
        "model.weight_format",
        "choice",
        "auto",
        "Format: auto, safetensors, paro, awq, gptq",
    ),
    ("model-hf-revision", "model.model_hf_revision", "str", None, "Hugging Face repo revision"),
    ("model-hf-token", "model.model_hf_token", "str", None, "Hugging Face token for gated repos"),
    # generate
    ("max-tokens", "generate.max_tokens", "int", 512, "Maximum tokens to generate"),
    ("temperature", "generate.temperature", "float", 1.0, "Sampling temperature"),
    ("top-p", "generate.top_p", "float", 1.0, "Nucleus sampling threshold"),
    ("top-k", "generate.top_k", "int", 0, "Top-k sampling (0=disabled)"),
    ("min-p", "generate.min_p", "float", 0.0, "Min-p sampling (0=disabled)"),
    ("seed", "generate.seed", "int", None, "Random seed"),
    ("prefill-step-size", "generate.prefill_step_size", "int", 2048, "Prefill chunk size"),
    (
        "clear-cache-interval",
        "generate.clear_cache_interval",
        "int",
        256,
        "mx.clear_cache every N tokens (0=never)",
    ),
    ("stream", "generate.stream", "bool", True, "Stream tokens by default"),
    ("compile-decode", "generate.compile_decode", "bool", False, "Use mx.compile on decode"),
    (
        "warmup-after-load",
        "generate.warmup_after_load",
        "bool",
        False,
        "Warm graph after model load",
    ),
    (
        "stream-policy",
        "generate.stream_policy",
        "choice",
        "single",
        "Stream policy: single, overlap",
    ),
    (
        "repetition-penalty",
        "generate.repetition_penalty",
        "float",
        1.0,
        "Repetition penalty (1.0=disabled)",
    ),
    ("logprobs", "generate.logprobs", "bool", False, "Return logprobs in stream"),
    ("top-logprobs", "generate.top_logprobs", "int", 0, "Top logprobs per token (0=disabled)"),
    # memory
    ("wired-limit", "memory.wired_limit", "int", None, "MLX wired memory limit (bytes)"),
    # cache
    ("max-kv-size", "cache.max_kv_size", "int", None, "Max rotating cache size (tokens)"),
    ("kv-bits", "cache.kv_bits", "int", None, "KV cache quantization bits"),
    ("kv-group-size", "cache.kv_group_size", "int", 64, "Group size for quantized KV"),
    ("quantized-kv-start", "cache.quantized_kv_start", "int", 0, "Start quantizing after N steps"),
    (
        "kv-rotating-keep",
        "cache.kv_rotating_keep",
        "int",
        None,
        "Tokens to keep in rotating cache",
    ),
    # prompt_cache
    ("prompt-cache-enabled", "prompt_cache.enabled", "bool", True, "Enable prompt prefix caching"),
    ("prompt-cache-max-entries", "prompt_cache.max_entries", "int", 100, "Max cached prefixes"),
    ("prompt-cache-max-bytes", "prompt_cache.max_bytes", "int", None, "Max bytes for cached KV"),
    (
        "prompt-cache-trim-on-rss-gb",
        "prompt_cache.trim_on_rss_gb",
        "float",
        None,
        "Trim when RSS exceeds (GB)",
    ),
    (
        "prompt-cache-trim-on-pressure",
        "prompt_cache.trim_on_pressure",
        "int",
        None,
        "Trim on memory pressure (1,2,4)",
    ),
    (
        "prompt-cache-trim-keep-entries",
        "prompt_cache.trim_keep_entries",
        "int",
        1,
        "Min entries to keep on trim",
    ),
    ("prompt-cache-trim-step", "prompt_cache.trim_step", "int", 1, "Entries to remove per trim"),
    (
        "prompt-cache-target-rss-ratio",
        "prompt_cache.target_rss_ratio",
        "float",
        0.9,
        "Trim until RSS < max*ratio",
    ),
    (
        "prompt-cache-on-memory-ceiling",
        "prompt_cache.on_memory_ceiling",
        "choice",
        "trim_cache",
        "Policy: trim_cache, reject_only, shutdown",
    ),
    ("prompt-cache-persist", "prompt_cache.persist_path", "str", None, "Path to persist cache"),
    # batch
    (
        "batch-prefill-batch-size",
        "batch.prefill_batch_size",
        "int",
        1,
        "Max prompts per prefill batch",
    ),
    (
        "batch-completion-batch-size",
        "batch.completion_batch_size",
        "int",
        4,
        "Max concurrent decode sequences",
    ),
    (
        "batch-prefill-step-size",
        "batch.prefill_step_size",
        "int",
        2048,
        "Batched prefill chunk size",
    ),
    ("batch-padding-side", "batch.padding_side", "choice", "left", "Padding: left, right"),
    (
        "batch-time-budget-ms",
        "batch.time_budget_ms",
        "int",
        100,
        "Time budget per batch step (ms)",
    ),
    ("batch-max-batch-size", "batch.max_batch_size", "int", 8, "Hard limit on batch size"),
    # speculative
    (
        "draft-model",
        "speculative.draft_model_path",
        "str",
        None,
        "Path to draft model (disables if unset)",
    ),
    (
        "num-draft-tokens",
        "speculative.num_draft_tokens",
        "int",
        5,
        "Draft tokens per verification step",
    ),
    # tool_calling
    (
        "tool-call-parser",
        "tool_calling.tool_call_parser",
        "str",
        "generic",
        "Parser: generic, qwen, or model key",
    ),
    # server
    ("host", "server.host", "str", "127.0.0.1", "Bind address"),
    ("port", "server.port", "int", 8080, "Listen port"),
    (
        "max-concurrent-requests",
        "server.max_concurrent_requests",
        "int",
        2,
        "Max concurrent requests",
    ),
    ("max-queue-size", "server.max_queue_size", "int", 64, "Max pending requests (0=unbounded)"),
    ("request-timeout", "server.request_timeout", "float", 300.0, "Request timeout (seconds)"),
    ("workers", "server.workers", "int", 1, "Number of workers"),
    # observability
    ("log-level", "observability.log_level", "str", "INFO", "Logging level"),
    ("metrics-enabled", "observability.metrics_enabled", "bool", False, "Enable metrics"),
    ("metrics-port", "observability.metrics_port", "int", 9090, "Metrics endpoint port"),
    # root
    ("strict-validation", "strict_validation", "bool", False, "Fail on unknown config keys"),
]


def parse_argv(argv: list[str] | None = None) -> tuple[str, AppConfig, str | None]:
    """Parse argv into subcommand, resolved AppConfig, and optional chat query.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        `(subcommand, config, chat_query)` where subcommand is "serve" or "chat".
        `chat_query` is only set for `mlxs chat QUERY`.

    Raises:
        SystemExit: On parse error or invalid config (argparse or resolve).
    """
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    subcommand = args.subcommand
    config_path: Path | None = getattr(args, "config_path", None)
    cli_overrides = _collect_overrides(args)

    config = resolve(config_path=config_path, cli_overrides=cli_overrides or None)
    chat_query = getattr(args, "query", None) if subcommand == "chat" else None
    return subcommand, config, chat_query


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxs",
        description="MLXs inference server and interactive chat.",
        formatter_class=_CLIHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Subcommand")

    for cmd in ("serve", "chat"):
        sub = subparsers.add_parser(
            cmd,
            help=_subcommand_help(cmd),
            formatter_class=_CLIHelpFormatter,
        )
        if cmd == "chat":
            sub.description = (
                "Run the docked chat shell by default or pass `QUERY` for one-shot mode."
            )
        sub.add_argument(
            "--config",
            dest="config_path",
            type=Path,
            default=None,
            help="Path to YAML config file. Overrides CONFIG_PATH env.",
        )
        sub.add_argument(
            "-o",
            "--override",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Extra config override (e.g. server.port=8081). May be repeated.",
        )
        if cmd == "chat":
            sub.add_argument(
                "query",
                nargs="?",
                default=None,
                metavar="QUERY",
                help="Single query (one-shot); if omitted, run the interactive shell.",
            )
        _add_explicit_flags(sub)

    return parser


def _add_explicit_flags(sub: argparse.ArgumentParser) -> None:
    choices_map: dict[str, list[str]] = {
        "model.model_mode": ["text", "multimodal", "auto"],
        "model.weight_format": ["auto", "safetensors", "paro", "awq", "gptq"],
        "generate.stream_policy": ["single", "overlap"],
        "batch.padding_side": ["left", "right"],
        "prompt_cache.on_memory_ceiling": ["trim_cache", "reject_only", "shutdown"],
    }
    bool_dests: list[str] = []
    for flag, config_key, kind, _default, help_text in _CLI_SPECS:
        dest = config_key.replace(".", "_").replace("-", "_")
        arg_name = "--" + flag
        if kind == "str":
            sub.add_argument(arg_name, dest=dest, type=str, default=_SENTINEL, help=help_text)
        elif kind == "int":
            sub.add_argument(arg_name, dest=dest, type=int, default=_SENTINEL, help=help_text)
        elif kind == "float":
            sub.add_argument(arg_name, dest=dest, type=float, default=_SENTINEL, help=help_text)
        elif kind == "bool":
            group = sub.add_mutually_exclusive_group()
            group.add_argument(arg_name, dest=dest, action="store_true", help=help_text)
            group.add_argument(
                "--no-" + flag, dest=dest, action="store_false", help="Disable (default)"
            )
            bool_dests.append(dest)
        elif kind == "choice":
            opts = choices_map.get(config_key)
            if opts:
                sub.add_argument(
                    arg_name, dest=dest, choices=opts, default=_SENTINEL, help=help_text
                )
            else:
                sub.add_argument(arg_name, dest=dest, default=_SENTINEL, help=help_text)
        else:
            raise ValueError(f"Unknown kind: {kind}")
    if bool_dests:
        sub.set_defaults(**{d: _SENTINEL for d in bool_dests})
    # Stop sequences (append) and extra_eos_token_ids (comma-separated)
    sub.add_argument(
        "--stop",
        dest="generate_stop_sequences",
        action="append",
        default=None,
        metavar="SEQ",
        help="Stop sequence. May be repeated.",
    )
    sub.add_argument(
        "--extra-eos-token-ids",
        dest="generate_extra_eos_token_ids",
        type=str,
        default=_SENTINEL,
        metavar="ID1,ID2,...",
        help="Comma-separated EOS token ids.",
    )


def _subcommand_help(cmd: str) -> str:
    if cmd == "serve":
        return "Start the HTTP server."
    if cmd == "chat":
        return "Start the docked chat shell or run a one-shot query."
    return ""


def build_cli_arguments_md() -> str:
    """Build docs/cli-arguments.md content from _CLI_SPECS (single source of truth)."""
    sections_order = [
        ("Config file", "config", ["--config", "-o"]),
        ("Model (model.*)", "model", []),
        ("Generation (generate.*)", "generate", []),
        ("Memory (memory.*)", "memory", []),
        ("KV cache (cache.*)", "cache", []),
        ("Prompt cache (prompt_cache.*)", "prompt_cache", []),
        ("Batch (batch.*) — mainly for serve", "batch", []),
        ("Speculative (speculative.*)", "speculative", []),
        ("Tool calling (tool_calling.*)", "tool_calling", []),
        ("Server (server.*) — for `mlxs serve`", "server", []),
        ("Observability (observability.*)", "observability", []),
        ("Root", "root", []),
    ]
    header = (
        "<!-- Generated from src/mlxs/config/cli.py _CLI_SPECS. Regenerate: "
        'python -c "from pathlib import Path; from mlxs.config.cli import '
        "build_cli_arguments_md; "
        "Path('docs/cli-arguments.md').write_text(build_cli_arguments_md())\" -->\n\n"
        "# CLI arguments — full reference\n\n"
        "All configuration options can be passed via explicit flags. "
        "Flags are available for both `mlxs serve` and `mlxs chat` unless noted. "
        "Values are merged with config file and env (CLI overrides win).\n\n"
        "`mlxs chat` runs the interactive docked shell by default. "
        "Use `mlxs chat QUERY` for one-shot mode.\n"
        "In a TTY, the shell includes slash commands, completion, "
        "and `@file` attachments.\n\n"
    )
    lines = [header]
    lines.append("## Chat usage\n\n")
    lines.append("| Command | Description |\n")
    lines.append("|---------|-------------|\n")
    lines.append("| `mlxs chat` | Start the interactive docked chat shell. |\n")
    lines.append("| `mlxs chat QUERY` | Run one turn and exit. |\n\n")
    lines.append(
        "Interactive mode includes slash commands such as "
        "`/help`, `/model`, `/history`, `/export`, `/new`, `/clear`, "
        "`/retry`, `/system`, `/stats`, and `/quit`, plus `@file` attachments.\n\n",
    )
    lines.append("---\n\n")
    for title, section_key, extra_flags in sections_order:
        if section_key == "config":
            lines.append("## Config file\n\n")
            lines.append("| Flag | Type | Default | Description |\n")
            lines.append("|------|------|---------|-------------|\n")
            lines.append(
                "| `--config` | path | - | Path to YAML config file. "
                "Overrides CONFIG_PATH env. |\n\n"
            )
            lines.append(
                "Generic overrides (any key): `-o section.key=value` "
                "(e.g. `-o server.port=9090`). May be repeated.\n\n---\n\n"
            )
            continue
        items = [
            (flag, config_key, kind, default, help_text)
            for (flag, config_key, kind, default, help_text) in _CLI_SPECS
            if (config_key.split(".")[0] if "." in config_key else "root") == section_key
        ]
        if not items and not extra_flags:
            continue
        lines.append(f"## {title}\n\n")
        lines.append("| Flag | Type | Default | Description |\n")
        lines.append("|------|------|---------|-------------|\n")
        for flag, _config_key, kind, default, help_text in items:
            default_str = str(default) if default != "" else '""'
            if default is None:
                default_str = "None"
            type_str = kind
            if kind == "choice":
                type_str = "choice"
            elif kind == "bool":
                lines.append(f"| `--{flag}` | bool | {default_str} | {help_text} |\n")
                lines.append("| `--no-" + flag + "` | - | - | Disable (default). |\n")
                continue
            lines.append(f"| `--{flag}` | {type_str} | {default_str} | {help_text} |\n")
        if section_key == "generate":
            lines.append("| `--stop` | str (repeat) | - | Stop sequence. May be repeated. |\n")
            lines.append(
                "| `--extra-eos-token-ids` | str | - | Comma-separated token ids (e.g. 1,2,3). |\n"
            )
        lines.append("\n---\n\n")
    return "".join(lines).rstrip() + "\n"


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    raw: list[str] = getattr(args, "overrides", []) or []
    for s in raw:
        if "=" not in s:
            continue
        key, _, value = s.partition("=")
        key = key.strip().lower()
        if not key:
            continue
        overrides[key] = value.strip()

    for _flag, config_key, _kind, _default, _ in _CLI_SPECS:
        dest = config_key.replace(".", "_").replace("-", "_")
        val = getattr(args, dest, _SENTINEL)
        if val is _SENTINEL:
            continue
        overrides[config_key] = val

    if getattr(args, "generate_stop_sequences", None):
        overrides["generate.stop_sequences"] = tuple(args.generate_stop_sequences)

    extra_eos = getattr(args, "generate_extra_eos_token_ids", _SENTINEL)
    if extra_eos is not _SENTINEL and extra_eos is not None and str(extra_eos).strip():
        ids = [int(x.strip()) for x in str(extra_eos).split(",") if x.strip()]
        overrides["generate.extra_eos_token_ids"] = tuple(ids)

    return overrides
