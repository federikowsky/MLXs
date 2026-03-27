"""Real-repo workloads for adaptive KV evaluation (single-request, deterministic excerpts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.adaptive_kv.scenarios import ScenarioResult
from mlxs._types import GenerateOptions

WORKLOAD_IDS: tuple[str, ...] = (
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "T1",
    "T2",
    "T3",
    "T4",
)

CODING_IDS = frozenset({"C1", "C2", "C3", "C4", "C5", "C6"})
TECHNICAL_IDS = frozenset({"T1", "T2", "T3", "T4"})


def _read_lines(repo_root: Path, rel: str, start: int, end: int) -> tuple[str, dict[str, Any]]:
    """1-based inclusive line range."""
    path = repo_root / rel
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    lo = max(start - 1, 0)
    hi = min(end, len(lines))
    chunk = "\n".join(lines[lo:hi])
    meta = {"path": rel, "line_start": start, "line_end": end, "lines_read": hi - lo}
    return chunk, meta


def _wrap(snippets: list[tuple[str, str]], task: str) -> str:
    parts: list[str] = [
        "You are assisting with MLXs codebase questions. Answer concisely. "
        "Use only the excerpts below.\n\n",
        "TASK:\n",
        task,
        "\n\n",
    ]
    for title, body in snippets:
        parts.append(f"--- {title} ---\n")
        parts.append(body)
        parts.append("\n\n")
    return "".join(parts).strip() + "\n"


def build_workload(
    workload_id: str,
    tokenizer: Any,
    repo_root: Path,
    *,
    max_prompt_tokens: int = 3800,
    run_seed: int | None = None,
) -> ScenarioResult:
    """Build tokenized prompt + options + provenance meta."""
    if workload_id not in WORKLOAD_IDS:
        raise ValueError(f"unknown workload {workload_id!r}")

    snippets: list[tuple[str, str]] = []
    provenance: list[dict[str, Any]] = []

    if workload_id == "C1":
        a, m1 = _read_lines(repo_root, "src/mlxs/models/llama.py", 53, 106)
        b, m2 = _read_lines(repo_root, "src/mlxs/layers/attention.py", 295, 318)
        snippets = [("File A: llama.py (Attention)", a), ("File B: attention.py (dispatch)", b)]
        provenance = [m1, m2]
        task = (
            "In one short paragraph, trace how `cache` flows from the Llama `Attention` "
            "module into the adaptive scaled dot-product attention path and when "
            "`record_usage_from_attention` can run."
        )
    elif workload_id == "C2":
        d, m1 = _read_lines(repo_root, "src/mlxs/generate/decode.py", 135, 158)
        m, m2 = _read_lines(repo_root, "src/mlxs/adaptive_kv/manager.py", 702, 726)
        snippets = [("decode.py (adaptive hooks)", d), ("manager.py (policy window)", m)]
        provenance = [m1, m2]
        task = (
            "Explain the ordering: when does replay/recovery run relative to the model "
            "forward, and when does the adaptive policy window run after a decode step?"
        )
    elif workload_id == "C3":
        ru, m1 = _read_lines(repo_root, "src/mlxs/adaptive_kv/manager.py", 346, 366)
        su, m2 = _read_lines(repo_root, "src/mlxs/adaptive_kv/usage.py", 1, 65)
        snippets = [("record_usage_from_attention", ru), ("AdaptiveUsageCollector", su)]
        provenance = [m1, m2]
        task = (
            "What work is deferred across layers during decode vs what runs at policy "
            "window boundaries (host sync)?"
        )
    elif workload_id == "C4":
        cp, m1 = _read_lines(repo_root, "benchmarks/adaptive_kv/config_profiles.py", 1, 100)
        snippets = [("config_profiles.py", cp)]
        provenance = [m1]
        task = (
            "When CLI does not override bytes, what default soft/hard budget pair applies "
            "to adaptive_soft vs adaptive_hard under budget_profile default?"
        )
    elif workload_id == "C5":
        dag, m1 = _read_lines(repo_root, "CLAUDE.md", 12, 21)
        snippets = [("CLAUDE.md Module DAG", dag)]
        provenance = [m1]
        task = (
            "Per the DAG shown, may the `generate` module import `server` directly? "
            "Answer yes or no and quote the dependency direction."
        )
    elif workload_id == "C6":
        kv, m1 = _read_lines(repo_root, "src/mlxs/cache/kv.py", 14, 62)
        gen, m2 = _read_lines(repo_root, "src/mlxs/generate/__init__.py", 118, 149)
        att, m3 = _read_lines(repo_root, "src/mlxs/layers/attention.py", 305, 319)
        snippets = [
            ("kv.py (KVCache)", kv),
            ("generate/__init__.py (adaptive branch)", gen),
            ("attention.py (adaptive cache check)", att),
        ]
        provenance = [m1, m2, m3]
        task = (
            "Give a numbered list (3-6 steps) describing how KV flows on an adaptive "
            "decode step from cache update through attention dispatch."
        )
    elif workload_id == "T1":
        sp, m1 = _read_lines(repo_root, "docs/specs.md", 12, 52)
        snippets = [("docs/specs.md (objectives + hot path)", sp)]
        provenance = [m1]
        task = (
            "Summarize the main P0 objectives in one sentence, and name one explicit "
            "decode-loop / hot-path rule from the excerpt."
        )
    elif workload_id == "T2":
        ng, m1 = _read_lines(repo_root, "docs/specs.md", 64, 69)
        fr, m2 = _read_lines(repo_root, "docs/specs.md", 73, 82)
        combined = f"--- specs.md §3 Non-goals ---\n{ng}\n\n--- specs.md §4.1 (start) ---\n{fr}\n"
        snippets = [("specs.md (non-goals then FR table start)", combined)]
        provenance = [m1, m2]
        task = (
            "Is training in scope per §3? Name exactly one FR id from the §4.1 table "
            "shown in the excerpt."
        )
    elif workload_id == "T3":
        multiturn_doc = "docs/adaptive_kv/Adaptive KV for Multi-Turn LLM Inference.md"
        doc, m1 = _read_lines(repo_root, multiturn_doc, 1, 180)
        snippets = [("Adaptive KV design doc (excerpt)", doc)]
        provenance = [m1]
        task = (
            "In one sentence: how does this document position MLXs adaptive KV relative to "
            "multi-turn vs single-request inference scope?"
        )
    elif workload_id == "T4":
        fs = (repo_root / "docs/adaptive_kv/adaptive_kv_v1_freeze_summary.md").read_text(
            encoding="utf-8"
        )
        m1 = {"path": "docs/adaptive_kv/adaptive_kv_v1_freeze_summary.md", "full_file": True}
        snippets = [("adaptive_kv_v1_freeze_summary.md", fs)]
        provenance = [m1]
        task = (
            "List two items explicitly in-scope and two explicitly out-of-scope per the doc "
            "(short bullets)."
        )
    else:
        raise AssertionError(workload_id)

    prompt_text = _wrap(snippets, task)
    ids = tokenizer.encode(prompt_text)
    truncated = False
    if len(ids) > max_prompt_tokens:
        ids = ids[:max_prompt_tokens]
        truncated = True

    meta: dict[str, Any] = {
        "workload_id": workload_id,
        "category": "coding" if workload_id in CODING_IDS else "technical",
        "provenance": provenance,
        "prompt_chars": len(prompt_text),
        "truncated_encoder_side": truncated,
        "max_prompt_tokens_cap": max_prompt_tokens,
    }
    opts = GenerateOptions(max_tokens=96, temperature=0.0, seed=run_seed)
    return ScenarioResult(
        name=workload_id,
        prompt_token_ids=ids,
        options=opts,
        description=f"Real workload {workload_id}",
        meta=meta,
    )
