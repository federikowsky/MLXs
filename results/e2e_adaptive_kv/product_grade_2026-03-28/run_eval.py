from __future__ import annotations

import gc
import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

from benchmarks.adaptive_kv.metrics_collect import (
    estimated_full_kv_bytes_per_prompt_token,
    sanitize_debug_snapshot,
    snapshot_in_memory_metrics,
)
from mlxs._types import GenerateOptions
from mlxs.adaptive_kv import AdaptiveKVConfig, assess_generation_compatibility
from mlxs.chat.template import build_prompt_ids, sanitize_assistant_text
from mlxs.config.schema import ModelConfig
from mlxs.generate import generate
from mlxs.load.formats import load_model_and_tokenizer
from mlxs.observability.metrics import InMemoryMetrics

OUT_DIR = Path("results/e2e_adaptive_kv/product_grade_2026-03-28")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_FAMILY = "Family A"
MODEL_REASON = (
    "Stronger open instruct-tuned llama-family checkpoint that is supported by the retained "
    "Adaptive KV baseline and behaves like a usable chat model in this environment."
)

SYSTEM_PROMPT = (
    "You are a careful assistant. Reuse earlier reasoning instead of restarting from scratch, "
    "preserve user constraints unless they are explicitly changed, and answer in compact bullet "
    "points or short paragraphs. Keep responses concise but complete."
)

OPTIONS = GenerateOptions(max_tokens=96, temperature=0.0, seed=1234, stream=True)
PREFILL_STEP_SIZE = 512
CLEAR_CACHE_INTERVAL = 256


def _packet(title: str, lines: list[str], count: int) -> str:
    return f"{title}\n" + "\n".join(f"- {line}" for line in lines[:count])


TRIP_NOTES = [
    "Near Karasuma or Kawaramachi is better than Gion because flatter walking matters more than atmosphere.",
    "Breakfast must clearly label gluten-free bread, rice porridge, or other safe options.",
    "Avoid hotels that require a long uphill walk from the station or a staircase-heavy shortcut.",
    "Quiet side streets and coffee shops matter more than nightlife access.",
    "A bookstore can double as a low-energy indoor stop if rain starts.",
    "One garden stop is enough if it is easy to reach by train.",
    "Late-night buses are a poor fit; simple train hops are preferred.",
    "If rain starts, swap outdoor wandering for covered arcades, stationery shops, or crafts.",
    "Vegetarian is mandatory for one dinner, but simple tofu or noodle spots are fine.",
    "Gluten-free breakfast reliability matters more than boutique-hotel aesthetics.",
    "Stairs and steep temple approaches should be treated as avoid unless there is a lift or taxi drop.",
    "One scenic viewpoint is still wanted, but only if the access is gentle and direct.",
    "Nightly hotel total should stay below 190 dollars before taxes if possible.",
    "Do not overpack the mornings; start after a calm breakfast.",
    "Seated live music is okay once, but it must end by 10 pm.",
    "At least one rainy indoor backup should feel pleasant, not like filler.",
    "Food halls work only if vegetarian and gluten-free options are explicit rather than implied.",
    "Simple station-to-station train movement is preferred over multiple bus transfers.",
    "Two major neighborhood changes in one day is the upper limit.",
    "If an item sounds crowded, suggest the quietest version or a calmer alternative.",
    "A river walk is welcome if it is flat and close to cafes.",
    "Souvenir shopping should be near the bookstore stop, not a separate detour.",
    "Avoid long museum visits unless they are free and under 45 minutes.",
    "If a temple or garden appears, explain why the access is knee-friendly.",
    "Convenience matters more than a luxury room.",
    "Breakfast should not require early-morning queueing.",
    "If an area is known for steep side streets, deprioritize it.",
    "One afternoon should remain flexible for weather or low energy.",
    "Easy train access back to Kyoto Station is a plus on departure day.",
    "If a stop risks crowding, add the calmer time window.",
    "One tea or dessert stop is welcome if gluten-free options exist.",
    "Use calm practical language rather than aspirational travel writing.",
    "If a viewpoint is difficult, replace it instead of apologizing for it.",
    "Indoor craft, stationery, or bookstore browsing counts as a valid cultural activity.",
    "Keep walking clusters geographically tight.",
    "Mention elevator access or taxi fallback when useful.",
    "Avoid nightlife districts as the hotel base.",
    "Rain backup should not break the food constraints.",
    "Hotel breakfast reliability matters more than room size.",
    "Summarize honored constraints at the end.",
]

OPS_NOTES = [
    f"Month {i}: seats {12 + (i % 5)}, automation runs {380 + i * 23}, finance wants predictable invoices and low surprise overages."
    for i in range(1, 41)
]

PROCUREMENT_NOTES = [
    f"Shortlist note {i}: item {i} must balance return policy, desk fit, cable clutter, reliability, and whether it reduces setup friction for non-technical staff."
    for i in range(1, 41)
]

DINNER_NOTES = [
    f"Guest and logistics note {i}: keep the dinner calm, allergy-safe, budget-aware, and easy to clean in a small apartment with limited oven capacity."
    for i in range(1, 41)
]

BASE_SCENARIOS = [
    {
        "id": "A_constraint_retention_under_load",
        "title": "Constraint retention under load",
        "why_realistic": "Trip planning with evolving accessibility, dietary, budget, transit, and weather constraints plus a pasted research packet.",
        "packet_title": "Neighborhood and hotel research notes:",
        "packet_lines": TRIP_NOTES,
        "turns": [
            "Plan a 3-night Kyoto trip for me and my brother. Constraints: hotel max 190 dollars per night total, quiet walkable area, vegetarian-friendly food, one bookstore, one garden, and no bars after 10 pm.",
            "Add that my brother needs gluten-free breakfast, I have mild knee pain so steep hills and lots of stairs are bad, and we prefer simple train access over bus-heavy days.",
            None,
            "Draft a realistic 2-day plan and recommend the best hotel area.",
            "Revise it because Saturday afternoon may be rainy and we still want one pleasant indoor backup.",
            "Final answer: itinerary, hotel-area recommendation, and the constraints you are honoring.",
        ],
    },
    {
        "id": "B_reasoning_continuity",
        "title": "Reasoning continuity",
        "why_realistic": "Software-plan cost reasoning with earlier math, later volume history, and a final decision framed for a cautious operator.",
        "packet_title": "Operations history and finance notes:",
        "packet_lines": OPS_NOTES,
        "turns": [
            "Compare three project-management plans for our 14-person team. Plan A is 180 dollars including 12 seats then 18 dollars per extra seat. Plan B is 240 dollars including 20 seats plus 6 dollars per automation run above 500 each month. Plan C is 310 dollars flat with unlimited seats and automations. We usually need 14 seats and 420 automation runs. Which is cheapest? Show the math.",
            "Now assume one quarter out of four has a contractor spike and more automations. Reuse the earlier reasoning instead of restarting from scratch.",
            None,
            "Summarize the breakpoints and tell me which plan best balances predictability and cost.",
            "Final recommendation for a cautious operations manager who still cares about total spend.",
        ],
    },
    {
        "id": "C_iterative_assistant_workflow",
        "title": "Iterative assistant workflow",
        "why_realistic": "Procurement assistant flow with product shortlist notes, budget revision, inventory reuse, and tradeoff explanation.",
        "packet_title": "Product shortlist and workspace notes:",
        "packet_lines": PROCUREMENT_NOTES,
        "turns": [
            "I need a home-office upgrade plan for 3 team members with an 1800 dollar total budget. One person needs a 34-inch ultrawide, one needs an ergonomic keyboard, one needs a good noise-isolating mic, and we prefer vendors with easy returns.",
            "Add that reliability matters more than raw specs, glossy monitors are a bad idea, and we would like to keep vendor count low.",
            None,
            "Build the first purchase plan and explain the tradeoffs.",
            "Revise it because the budget drops to 1550 dollars and we already have two monitor arms and one webcam.",
            "Final answer: final cart, vendor split, and the compromises you made.",
        ],
    },
    {
        "id": "D_long_context_drift",
        "title": "Long-context drift",
        "why_realistic": "Longer dinner-planning conversation with guest constraints, logistics notes, later revisions, and a final request to honor early facts.",
        "packet_title": "Guest and logistics notes:",
        "packet_lines": DINNER_NOTES,
        "turns": [
            "Help me plan a small birthday dinner for 6 friends next Saturday. Two guests are vegan, one has a nut allergy, one does not drink alcohol, budget is 240 dollars total, and my apartment has a tiny oven.",
            "Add that I want a calm vibe, no childish games, one short toast, and one guest can only stay 90 minutes.",
            None,
            "Give me a first-draft menu and timeline.",
            "Change of plans: one guest hates mushrooms, another cannot handle spicy food, and cleanup must stay under 30 minutes because my neighbors complain about late noise.",
            "Also assume rain may make people arrive slowly, so include a simple indoor waiting flow and keep the shopping list concise.",
            "Final answer: dinner plan, menu, timeline, and explicitly list the earliest constraints you still honored.",
        ],
    },
]

REGIMES = [
    {
        "id": "comfortable",
        "packet_count": 6,
        "adaptive": {
            "enabled": True,
            "block_size_tokens": 64,
            "update_window_steps": 16,
            "soft_budget_bytes": 40_000_000,
            "hard_budget_bytes": 64_000_000,
        },
        "intent": "Conversation fits comfortably within the serving envelope; Adaptive KV should mainly preserve behavior.",
    },
    {
        "id": "borderline",
        "packet_count": 18,
        "adaptive": {
            "enabled": True,
            "block_size_tokens": 64,
            "update_window_steps": 16,
            "soft_budget_bytes": 24_000_000,
            "hard_budget_bytes": 36_000_000,
        },
        "intent": "Conversation is close to the budgeted serving envelope; pressure should appear but remain manageable.",
    },
    {
        "id": "stressed",
        "packet_count": 40,
        "adaptive": {
            "enabled": True,
            "block_size_tokens": 64,
            "update_window_steps": 16,
            "soft_budget_bytes": 12_000_000,
            "hard_budget_bytes": 18_000_000,
        },
        "intent": "Conversation significantly exceeds the budgeted serving envelope; Adaptive KV must manage repeated pressure, eviction, and replay.",
    },
]


def build_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for regime in REGIMES:
        for base in BASE_SCENARIOS:
            turns = [
                turn
                if turn is not None
                else _packet(base["packet_title"], base["packet_lines"], regime["packet_count"])
                for turn in base["turns"]
            ]
            scenarios.append(
                {
                    "regime": regime["id"],
                    "regime_intent": regime["intent"],
                    "adaptive_config": regime["adaptive"],
                    "packet_count": regime["packet_count"],
                    "id": base["id"],
                    "title": base["title"],
                    "why_realistic": base["why_realistic"],
                    "turns": turns,
                }
            )
    return scenarios


def run_one(model, tokenizer, messages: list[dict[str, str]], *, adaptive_config: AdaptiveKVConfig | None, kv_bytes_per_prompt_token: int) -> dict[str, object]:
    prompt_ids = build_prompt_ids(tokenizer, messages)
    metrics = InMemoryMetrics()
    final_adaptive: list[dict[str, object]] = []
    t0 = time.perf_counter()
    events = list(
        generate(
            model,
            tokenizer,
            prompt_ids,
            OPTIONS,
            adaptive_config=adaptive_config,
            metrics=metrics,
            final_adaptive_state_out=final_adaptive if adaptive_config is not None else None,
            prefill_step_size=PREFILL_STEP_SIZE,
            clear_cache_interval=CLEAR_CACHE_INTERVAL,
        )
    )
    elapsed = time.perf_counter() - t0
    text = sanitize_assistant_text("".join(e.text for e in events)).strip()
    output_tokens = len(events)
    metrics_dump = snapshot_in_memory_metrics(metrics)
    counters = metrics_dump["counters"]
    adaptive_snapshot = sanitize_debug_snapshot(final_adaptive[0]) if final_adaptive else None
    prompt_tokens = len(prompt_ids)
    return {
        "prompt_tokens": prompt_tokens,
        "projected_full_kv_bytes": prompt_tokens * kv_bytes_per_prompt_token,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed,
        "tok_per_s": (output_tokens / elapsed) if elapsed > 0 else None,
        "text": text,
        "metrics_counters": counters,
        "adaptive_snapshot": adaptive_snapshot,
        "evictions": counters.get("adaptive_kv_evictions_total", 0.0),
        "recomputations": counters.get("adaptive_kv_recomputations_total", 0.0),
    }


def warm_model(model, tokenizer, kv_bytes_per_prompt_token: int) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "In one sentence, say what 2 plus 2 is."},
    ]
    _ = run_one(model, tokenizer, messages, adaptive_config=None, kv_bytes_per_prompt_token=kv_bytes_per_prompt_token)
    mx.clear_cache()


def main() -> None:
    scenarios = build_scenarios()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(model_path=MODEL_REPO, preload=True, lazy_load=False)
    model, tokenizer = load_model_and_tokenizer(MODEL_REPO, model_cfg)
    compatibility = assess_generation_compatibility(
        model,
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )
    kv_bytes_per_prompt_token = estimated_full_kv_bytes_per_prompt_token(model)
    warm_model(model, tokenizer, kv_bytes_per_prompt_token)

    payload: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "repo": MODEL_REPO,
            "family": MODEL_FAMILY,
            "reason": MODEL_REASON,
            "compatibility": {
                "supported": compatibility.supported,
                "reason": compatibility.reason,
                "adapter_name": compatibility.adapter_name,
                "runtime_family": str(compatibility.runtime_family),
                "support_level": str(compatibility.support_level),
                "num_layers": compatibility.num_layers,
            },
            "estimated_full_kv_bytes_per_prompt_token": kv_bytes_per_prompt_token,
        },
        "system_prompt": SYSTEM_PROMPT,
        "options": {
            "max_tokens": OPTIONS.max_tokens,
            "temperature": OPTIONS.temperature,
            "seed": OPTIONS.seed,
            "prefill_step_size": PREFILL_STEP_SIZE,
            "clear_cache_interval": CLEAR_CACHE_INTERVAL,
        },
        "regimes": REGIMES,
        "scenarios": [],
        "family_c_probe_note": {
            "status": "omitted_from_main_matrix",
            "reason": (
                "Supported qwen3_5 checkpoints probed in this environment did not yield a usable "
                "chat-capable product baseline, so Family C was not included in the main product-grade comparison."
            ),
            "probed_checkpoints": [
                "Qwen/Qwen3.5-0.8B",
                "Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled",
                "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
            ],
        },
    }

    md_lines = ["# Adaptive KV Product-Grade E2E Conversational Evaluation", ""]

    for scenario in scenarios:
        scenario_record = {
            "regime": scenario["regime"],
            "regime_intent": scenario["regime_intent"],
            "id": scenario["id"],
            "title": scenario["title"],
            "why_realistic": scenario["why_realistic"],
            "turns": scenario["turns"],
            "modes": {},
        }
        adaptive_config = AdaptiveKVConfig(**scenario["adaptive_config"])
        hard_budget = scenario["adaptive_config"]["hard_budget_bytes"]

        for mode_name, cfg in (("baseline", None), ("adaptive", adaptive_config)):
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            turn_records = []
            for turn_index, user_text in enumerate(scenario["turns"], start=1):
                history.append({"role": "user", "content": user_text})
                result = run_one(
                    model,
                    tokenizer,
                    history,
                    adaptive_config=cfg,
                    kv_bytes_per_prompt_token=kv_bytes_per_prompt_token,
                )
                turn_records.append(
                    {
                        "turn_index": turn_index,
                        "user": user_text,
                        **result,
                    }
                )
                history.append({"role": "assistant", "content": result["text"]})
                mx.clear_cache()
            scenario_record["modes"][mode_name] = {
                "turns": turn_records,
                "summary": {
                    "total_elapsed_s": sum(t["elapsed_s"] for t in turn_records),
                    "total_output_tokens": sum(t["output_tokens"] for t in turn_records),
                    "aggregate_tok_per_s": (
                        sum(t["output_tokens"] for t in turn_records) / sum(t["elapsed_s"] for t in turn_records)
                    ),
                    "final_answer": turn_records[-1]["text"],
                    "peak_projected_full_kv_bytes": max(t["projected_full_kv_bytes"] for t in turn_records),
                    "peak_prompt_tokens": max(t["prompt_tokens"] for t in turn_records),
                    "evictions_total": sum(t["evictions"] for t in turn_records),
                    "recomputations_total": sum(t["recomputations"] for t in turn_records),
                    "pressure_progression": [
                        (t["adaptive_snapshot"] or {}).get("pressure_state") for t in turn_records
                    ],
                    "max_resident_bytes": max(
                        (t["adaptive_snapshot"] or {}).get("resident_bytes", 0) for t in turn_records
                    ),
                },
            }

        baseline_summary = scenario_record["modes"]["baseline"]["summary"]
        adaptive_summary = scenario_record["modes"]["adaptive"]["summary"]
        scenario_record["comparison"] = {
            "all_turn_text_equal": all(
                bt["text"] == at["text"]
                for bt, at in zip(
                    scenario_record["modes"]["baseline"]["turns"],
                    scenario_record["modes"]["adaptive"]["turns"],
                    strict=True,
                )
            ),
            "final_answer_equal": baseline_summary["final_answer"] == adaptive_summary["final_answer"],
            "baseline_elapsed_s": baseline_summary["total_elapsed_s"],
            "adaptive_elapsed_s": adaptive_summary["total_elapsed_s"],
            "baseline_tok_per_s": baseline_summary["aggregate_tok_per_s"],
            "adaptive_tok_per_s": adaptive_summary["aggregate_tok_per_s"],
            "adaptive_evictions_total": adaptive_summary["evictions_total"],
            "adaptive_recomputations_total": adaptive_summary["recomputations_total"],
            "adaptive_pressure_progression": adaptive_summary["pressure_progression"],
            "baseline_peak_projected_full_kv_bytes": baseline_summary["peak_projected_full_kv_bytes"],
            "adaptive_max_resident_bytes": adaptive_summary["max_resident_bytes"],
            "baseline_peak_vs_hard_budget_ratio": (
                baseline_summary["peak_projected_full_kv_bytes"] / hard_budget
            ),
        }
        payload["scenarios"].append(scenario_record)

        md_lines.append(f"## {scenario['regime']} :: {scenario['id']} — {scenario['title']}")
        md_lines.append(scenario["why_realistic"])
        md_lines.append("")
        md_lines.append(
            f"Serving envelope: hard budget {hard_budget:,} bytes; "
            f"baseline projected peak KV ratio {scenario_record['comparison']['baseline_peak_vs_hard_budget_ratio']:.2f}x"
        )
        md_lines.append("")
        for mode_name in ("baseline", "adaptive"):
            summary = scenario_record["modes"][mode_name]["summary"]
            md_lines.append(f"### {mode_name}")
            md_lines.append(
                f"elapsed={summary['total_elapsed_s']:.3f}s, tok/s={summary['aggregate_tok_per_s']:.2f}, "
                f"peak_prompt_tokens={summary['peak_prompt_tokens']}, "
                f"peak_projected_full_kv_bytes={summary['peak_projected_full_kv_bytes']:,}, "
                f"evictions={summary['evictions_total']}, recomputes={summary['recomputations_total']}, "
                f"pressure={summary['pressure_progression']}, resident_bytes_max={summary['max_resident_bytes']:,}"
            )
            md_lines.append("")
            for turn in scenario_record["modes"][mode_name]["turns"]:
                md_lines.append(f"Turn {turn['turn_index']} user: {turn['user']}")
                md_lines.append(turn["text"])
                md_lines.append(
                    f"(prompt_tokens={turn['prompt_tokens']}, projected_full_kv_bytes={turn['projected_full_kv_bytes']:,}, "
                    f"elapsed={turn['elapsed_s']:.3f}s, tok/s={turn['tok_per_s']:.2f}, "
                    f"evictions={turn['evictions']}, recomputes={turn['recomputations']}, "
                    f"pressure={(turn['adaptive_snapshot'] or {}).get('pressure_state')})"
                )
                md_lines.append("")
        md_lines.append(
            "Comparison: "
            f"all_turn_text_equal={scenario_record['comparison']['all_turn_text_equal']}, "
            f"final_answer_equal={scenario_record['comparison']['final_answer_equal']}"
        )
        md_lines.append("")

    json_path = OUT_DIR / "evaluation.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    transcript_path = OUT_DIR / "transcripts.md"
    transcript_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json_path)
    print(transcript_path)

    del model, tokenizer
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
