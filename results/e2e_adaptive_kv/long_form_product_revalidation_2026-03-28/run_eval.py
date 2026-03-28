from __future__ import annotations

import gc
import json
import statistics
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

OUT_DIR = Path("results/e2e_adaptive_kv/long_form_product_revalidation_2026-03-28")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c"
)
MODEL_DISPLAY = "unsloth/Llama-3.2-1B-Instruct"
MODEL_FAMILY = "Family A"
MODEL_REASON = (
    "Strongest realistic chat-capable supported checkpoint practically available in the current "
    "environment. A stronger 3B Llama checkpoint was not usable here because hub access was gated "
    "and the local cache was incomplete. Available Family C checkpoints remained poor product-chat baselines."
)

SYSTEM_PROMPT = (
    "You are a careful assistant. Preserve user constraints unless they are explicitly changed, "
    "reuse earlier reasoning rather than restarting from scratch, and answer in concise paragraphs "
    "or compact bullet points. When the user asks for a final synthesis, explicitly mention the "
    "earliest constraints you still honored."
)

OPTIONS = GenerateOptions(max_tokens=112, temperature=0.0, seed=1234, stream=True)
PREFILL_STEP_SIZE = 512
CLEAR_CACHE_INTERVAL = 256


def _packet(title: str, lines: list[str], count: int) -> str:
    return f"{title}\n" + "\n".join(f"- {line}" for line in lines[:count])


TRIP_NOTES = [
    "Stay near a flat train corridor rather than a scenic hill district.",
    "Breakfast must be clearly vegetarian-friendly and offer at least one substantial protein option.",
    "Avoid hotel areas that become loud after 10 pm or feel nightlife-heavy.",
    "One traveler has mild knee pain, so steep slopes and staircase shortcuts are poor fits.",
    "A bookstore or stationery stop should feel integrated into the route, not bolted on.",
    "One scenic walk is welcome, but only if it is calm and mostly flat.",
    "Simple rail movement is strongly preferred over chained bus transfers.",
    "A rainy-day backup should feel pleasant enough to replace an outdoor block, not just fill time.",
    "Crowded food halls are only acceptable if seating is easy and dietary labeling is explicit.",
    "Start the mornings slowly; do not schedule the first major move too early.",
    "One tofu-focused dinner is welcome if the place is easy to reach and not trendy for its own sake.",
    "Convenience matters more than boutique style.",
    "Hotel total should stay under 220 dollars per night before taxes.",
    "Departure day should not require a complicated luggage move before breakfast.",
    "If a viewpoint is difficult, replace it rather than apologizing for it.",
    "A calm tea or dessert stop is a plus if it fits dietary constraints.",
    "One afternoon should remain intentionally flexible for weather or energy dips.",
    "Do not send us across the city for a single small activity.",
    "Late-night live music is out; one seated low-key performance is fine if it ends early.",
    "Museum visits should be short and only if they add clear value.",
    "Temple suggestions must mention access difficulty honestly.",
    "Avoid hotel breakfasts that depend on queueing or a crowded buffet room.",
    "Rain backup should be near the main neighborhood base.",
    "If an area is famous but logistically awkward, deprioritize it.",
    "At least one river or canal walk is welcome if it is easy on the knees.",
    "A practical convenience-store breakfast backup is useful to mention.",
    "Two neighborhood changes in one day is the upper limit.",
    "Station proximity matters most on the final morning.",
    "A calm cafe for mid-afternoon recovery is useful.",
    "If a stop is likely crowded, recommend the quietest timing.",
    "Indoor browsing counts as a valid cultural activity.",
    "The final plan should explicitly summarize the honored constraints.",
    "Avoid steep shrine approaches unless there is a taxi drop or lift.",
    "One covered arcade or market is a good rainy alternative if it is not too chaotic.",
    "Do not recommend long taxi chains just to save a scenic idea.",
    "Hotel breakfast reliability matters more than room size.",
    "Keep the tone practical rather than romantic.",
    "Quiet convenience around the hotel matters more than designer interiors.",
    "One short souvenir stop near the bookstore is enough.",
    "Train simplicity matters more than squeezing in one extra landmark.",
    "If you suggest an early morning slot, explain why it is worth the effort.",
    "Use compact daily plans rather than oversized bucket lists.",
    "Avoid attractions that are mostly stairs plus a photo spot.",
    "A calm covered shopping street is acceptable as light activity.",
    "If the weather turns worse, the replacement should preserve food constraints.",
    "The final answer should still feel like a real trip, not only logistics.",
    "One garden stop is enough if it is the easiest pleasant option.",
    "Avoid recommending nightlife districts as hotel bases.",
    "A short riverside walk paired with coffee is ideal.",
    "If the plan uses a taxi, explain exactly why.",
    "Do not recommend changing hotels mid-trip.",
    "A flexible second afternoon is more valuable than adding one more shrine.",
    "Accessibility honesty matters more than aspirational travel writing.",
    "Give a short explanation for why the recommended hotel area wins.",
    "Keep the plan dense enough to be useful but light enough to feel realistic.",
    "If breakfast is uncertain, say so instead of pretending.",
    "Walking clusters should stay geographically tight.",
    "One optional evening activity is enough.",
    "A practical departure-day coffee and breakfast plan is welcome.",
    "Do not bury the budget constraint under style language.",
]

OPS_NOTES = [
    f"Month {i}: seats {13 + (i % 6)}, automation runs {420 + i * 29}, finance notes emphasize predictable invoicing and low surprise overages."
    for i in range(1, 61)
]

PROCUREMENT_NOTES = [
    f"Procurement note {i}: prefer reliable mid-market hardware, easy returns, low cable clutter, and quieter equipment over flashy peak specs."
    for i in range(1, 61)
]

EVENT_NOTES = [
    f"Event note {i}: keep the gathering calm, allergy-safe, budget-aware, and easy to clean in a small apartment with limited oven capacity and limited counter space."
    for i in range(1, 61)
]

BASE_SCENARIOS = [
    {
        "id": "A_long_constraint_retention",
        "title": "Long constraint retention",
        "why_realistic": "A long trip-planning conversation with layered accessibility, budget, food, rain, and pacing constraints plus two pasted note packets.",
        "packet_title": "Travel research packet:",
        "packet_lines": TRIP_NOTES,
        "turns": [
            "Plan a 4-night Kyoto trip for two adults. We want one calm neighborhood base, vegetarian-friendly meals, one bookstore stop, one garden, and a practical pace rather than an ambitious one.",
            "Add that one traveler has mild knee pain, we want simple train movement instead of bus-heavy days, and nightly hotel total should stay under 220 dollars before taxes.",
            "{packet1}",
            "Recommend the best hotel area and a first-draft 3-day plan.",
            "Revise it because Saturday afternoon may be rainy and we still want one pleasant indoor backup.",
            "Also add that breakfast reliability matters a lot, nightlife districts are a bad hotel fit, and departure morning should be easy.",
            "{packet2}",
            "Final answer: the recommended area, a realistic daily plan, and the earliest constraints you still honored.",
        ],
    },
    {
        "id": "B_reasoning_continuity_over_many_turns",
        "title": "Reasoning continuity over many turns",
        "why_realistic": "A multi-turn operations planning thread where later answers depend on earlier math, breakpoints, and revised assumptions.",
        "packet_title": "Operations and finance packet:",
        "packet_lines": OPS_NOTES,
        "turns": [
            "Compare three project-management plans for our operations team. Plan A is 210 dollars including 12 seats, then 19 dollars per extra seat. Plan B is 255 dollars including 18 seats plus 5 dollars per automation run above 520 each month. Plan C is 330 dollars flat with unlimited seats and automations. We usually need 15 seats and about 470 automation runs. Which is cheapest? Show the math.",
            "Now assume two months each quarter have contractor spikes and automation bursts. Reuse the earlier reasoning instead of restarting from scratch.",
            "{packet1}",
            "Summarize the breakpoints and tell me when each plan wins.",
            "Revise the recommendation for a cautious operations lead who values predictable invoices but still cares about total spend.",
            "Assume finance will accept one higher month if the annual total is clearly lower. Does that change the recommendation?",
            "{packet2}",
            "Final answer: recommended plan, the key breakpoints, and the earliest assumptions you still used.",
        ],
    },
    {
        "id": "C_iterative_real_workflow",
        "title": "Iterative real workflow",
        "why_realistic": "A practical procurement workflow with pasted shortlist notes, vendor consolidation, inventory reuse, and later budget cuts.",
        "packet_title": "Procurement shortlist packet:",
        "packet_lines": PROCUREMENT_NOTES,
        "turns": [
            "I need a home-office upgrade plan for 4 team members with a 2400 dollar total budget. One person needs a 34-inch ultrawide, one needs a quiet keyboard, one needs a reliable headset mic, and one needs a compact docking setup. Returns should be easy.",
            "Add that glossy monitors are a bad fit, reliability matters more than raw specs, and we prefer as few vendors as possible.",
            "{packet1}",
            "Build the first purchase plan and explain the tradeoffs.",
            "Revise it because the budget drops to 1950 dollars and we already have two monitor arms and one webcam.",
            "Also assume one desk is shallow, cable clutter is a big complaint, and setup time for non-technical staff should stay low.",
            "{packet2}",
            "Final answer: final cart, vendor split, compromises made, and the earliest requirements you still honored.",
        ],
    },
    {
        "id": "D_long_context_drift_recall",
        "title": "Long-context drift and recall",
        "why_realistic": "A deliberately long planning conversation where early constraints still matter after multiple revisions, pasted notes, and logistical changes.",
        "packet_title": "Guest and logistics packet:",
        "packet_lines": EVENT_NOTES,
        "turns": [
            "Help me plan a small birthday dinner for 8 people next Saturday. Two guests are vegan, one has a nut allergy, one does not drink alcohol, total budget is 280 dollars, and my apartment has a tiny oven.",
            "Add that I want a calm vibe, one short toast, no childish games, and one guest can only stay for 90 minutes.",
            "{packet1}",
            "Give me a first-draft menu and timeline.",
            "Change of plans: one guest hates mushrooms, another cannot handle spicy food, and cleanup must stay under 30 minutes because my neighbors complain about late noise.",
            "Also assume rain may make people arrive slowly, so include a simple indoor waiting flow and keep the shopping list concise.",
            "{packet2}",
            "Final answer: menu, timeline, shopping list, and explicitly list the earliest constraints you still honored.",
        ],
    },
]

REGIMES = [
    {
        "id": "comfortable",
        "packet_counts": (8, 6),
        "adaptive": {
            "enabled": True,
            "block_size_tokens": 64,
            "update_window_steps": 16,
            "soft_budget_bytes": 48_000_000,
            "hard_budget_bytes": 80_000_000,
        },
        "intent": "Baseline should fit comfortably; Adaptive KV should mainly preserve behavior.",
    },
    {
        "id": "borderline",
        "packet_counts": (24, 18),
        "adaptive": {
            "enabled": True,
            "block_size_tokens": 64,
            "update_window_steps": 16,
            "soft_budget_bytes": 24_000_000,
            "hard_budget_bytes": 36_000_000,
        },
        "intent": "Conversation should be close to the budgeted serving envelope and surface meaningful pressure.",
    },
    {
        "id": "stressed",
        "packet_counts": (48, 32),
        "adaptive": {
            "enabled": True,
            "block_size_tokens": 64,
            "update_window_steps": 16,
            "soft_budget_bytes": 12_000_000,
            "hard_budget_bytes": 18_000_000,
        },
        "intent": "Conversation should significantly exceed the budgeted serving envelope and exercise repeated pressure, eviction, and replay.",
    },
]


def build_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for regime in REGIMES:
        packet1_count, packet2_count = regime["packet_counts"]
        for base in BASE_SCENARIOS:
            turns: list[str] = []
            for turn in base["turns"]:
                if turn == "{packet1}":
                    turns.append(_packet(base["packet_title"], base["packet_lines"], packet1_count))
                elif turn == "{packet2}":
                    lines = base["packet_lines"][packet1_count : packet1_count + packet2_count]
                    turns.append(_packet(f"{base['packet_title']} follow-up:", lines, packet2_count))
                else:
                    turns.append(turn)
            scenarios.append(
                {
                    "regime": regime["id"],
                    "regime_intent": regime["intent"],
                    "adaptive_config": regime["adaptive"],
                    "packet_counts": regime["packet_counts"],
                    "id": base["id"],
                    "title": base["title"],
                    "why_realistic": base["why_realistic"],
                    "turns": turns,
                }
            )
    return scenarios


def run_one(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    adaptive_config: AdaptiveKVConfig | None,
    kv_bytes_per_prompt_token: int,
) -> dict[str, object]:
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
    _ = run_one(
        model,
        tokenizer,
        messages,
        adaptive_config=None,
        kv_bytes_per_prompt_token=kv_bytes_per_prompt_token,
    )
    mx.clear_cache()


def _scenario_summary(turn_records: list[dict[str, object]]) -> dict[str, object]:
    return {
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
        "pressure_progression": [(t["adaptive_snapshot"] or {}).get("pressure_state") for t in turn_records],
        "max_resident_bytes": max((t["adaptive_snapshot"] or {}).get("resident_bytes", 0) for t in turn_records),
    }


def main() -> None:
    scenarios = build_scenarios()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(model_path=str(MODEL_PATH), preload=True, lazy_load=False)
    model, tokenizer = load_model_and_tokenizer(str(MODEL_PATH), model_cfg)
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
            "display": MODEL_DISPLAY,
            "path": str(MODEL_PATH),
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
        "omitted_families": {
            "Family C": {
                "status": "omitted_from_main_matrix",
                "reason": (
                    "Cached supported qwen3_5 checkpoints available in this environment did not behave like "
                    "credible product-chat baselines. A quick local sanity on Qwen/Qwen3.5-0.8B still produced unusable output."
                ),
                "probed_checkpoints": [
                    "Qwen/Qwen3.5-0.8B",
                    "Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled",
                    "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
                    "aufklarer/Qwen3.5-0.8B-Chat-MLX",
                ],
            },
            "Family B": {
                "status": "omitted_from_main_matrix",
                "reason": (
                    "No practical chat-capable Ministral-family checkpoint was staged locally for a serious long-form product comparison in this environment."
                ),
            },
        },
    }

    transcripts = ["# Adaptive KV Long-Form Product E2E Revalidation Transcripts", ""]

    for scenario in scenarios:
        scenario_record = {
            "regime": scenario["regime"],
            "regime_intent": scenario["regime_intent"],
            "packet_counts": scenario["packet_counts"],
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
            turn_records: list[dict[str, object]] = []
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
                "summary": _scenario_summary(turn_records),
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

        transcripts.append(f"## {scenario['regime']} :: {scenario['id']} — {scenario['title']}")
        transcripts.append(scenario["why_realistic"])
        transcripts.append("")
        transcripts.append(
            f"Serving envelope: hard budget {hard_budget:,} bytes; "
            f"baseline projected peak KV ratio {scenario_record['comparison']['baseline_peak_vs_hard_budget_ratio']:.2f}x"
        )
        transcripts.append("")
        for mode_name in ("baseline", "adaptive"):
            summary = scenario_record["modes"][mode_name]["summary"]
            transcripts.append(f"### {mode_name}")
            transcripts.append(
                f"elapsed={summary['total_elapsed_s']:.3f}s, tok/s={summary['aggregate_tok_per_s']:.2f}, "
                f"peak_prompt_tokens={summary['peak_prompt_tokens']}, "
                f"peak_projected_full_kv_bytes={summary['peak_projected_full_kv_bytes']:,}, "
                f"evictions={summary['evictions_total']}, recomputes={summary['recomputations_total']}, "
                f"pressure={summary['pressure_progression']}, resident_bytes_max={summary['max_resident_bytes']:,}"
            )
            transcripts.append("")
            for turn in scenario_record["modes"][mode_name]["turns"]:
                transcripts.append(f"Turn {turn['turn_index']} user: {turn['user']}")
                transcripts.append(turn["text"])
                transcripts.append(
                    f"(prompt_tokens={turn['prompt_tokens']}, projected_full_kv_bytes={turn['projected_full_kv_bytes']:,}, "
                    f"elapsed={turn['elapsed_s']:.3f}s, tok/s={turn['tok_per_s']:.2f}, "
                    f"evictions={turn['evictions']}, recomputes={turn['recomputations']}, "
                    f"pressure={(turn['adaptive_snapshot'] or {}).get('pressure_state')}, "
                    f"resident_bytes={(turn['adaptive_snapshot'] or {}).get('resident_bytes', 0)})"
                )
                transcripts.append("")
        transcripts.append(
            "Comparison: "
            f"all_turn_text_equal={scenario_record['comparison']['all_turn_text_equal']}, "
            f"final_answer_equal={scenario_record['comparison']['final_answer_equal']}"
        )
        transcripts.append("")

    regime_rollup: dict[str, dict[str, float | int | None]] = {}
    for regime in {scenario["regime"] for scenario in scenarios}:
        subset = [s for s in payload["scenarios"] if s["regime"] == regime]
        baseline_elapsed = sum(s["comparison"]["baseline_elapsed_s"] for s in subset)
        adaptive_elapsed = sum(s["comparison"]["adaptive_elapsed_s"] for s in subset)
        baseline_output = sum(s["modes"]["baseline"]["summary"]["total_output_tokens"] for s in subset)
        adaptive_output = sum(s["modes"]["adaptive"]["summary"]["total_output_tokens"] for s in subset)
        ratios = [s["comparison"]["baseline_peak_vs_hard_budget_ratio"] for s in subset]
        regime_rollup[regime] = {
            "scenario_count": len(subset),
            "baseline_elapsed_s": baseline_elapsed,
            "adaptive_elapsed_s": adaptive_elapsed,
            "baseline_tok_per_s": baseline_output / baseline_elapsed,
            "adaptive_tok_per_s": adaptive_output / adaptive_elapsed,
            "adaptive_evictions_total": sum(s["comparison"]["adaptive_evictions_total"] for s in subset),
            "adaptive_recomputations_total": sum(s["comparison"]["adaptive_recomputations_total"] for s in subset),
            "baseline_peak_vs_hard_budget_ratio_mean": statistics.mean(ratios),
            "baseline_peak_vs_hard_budget_ratio_min": min(ratios),
            "baseline_peak_vs_hard_budget_ratio_max": max(ratios),
        }
    payload["regime_rollup"] = regime_rollup

    json_path = OUT_DIR / "evaluation.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    transcript_path = OUT_DIR / "transcripts.md"
    transcript_path.write_text("\n".join(transcripts), encoding="utf-8")
    print(json_path)
    print(transcript_path)

    del model, tokenizer
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
