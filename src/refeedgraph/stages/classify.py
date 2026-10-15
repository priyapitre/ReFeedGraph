#!/usr/bin/env python3
"""Step 2: synthesize judge feedback into summary, weaknesses, and actions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from refeedgraph.common import (
    DEFAULT_OUTPUT_DIR,
    call_json_llm,
    load_json,
    make_client,
    synthesis_prompt,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Step 1 JSON output.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--mock", action="store_true", help="Validate locally without LLM calls.")
    args = parser.parse_args()

    if args.output is None:
        stem = args.input.stem.replace("_step1_judge_feedback", "")
        args.output = DEFAULT_OUTPUT_DIR / f"{stem}_step2_synthesis.json"
    return args


def main() -> None:
    args = parse_args()
    step1 = load_json(args.input)
    debate_text = step1["debate_text"]
    judge_feedback = step1["judge_feedback"]

    print("Synthesizing judge feedback", flush=True)
    synthesis = {"per_judge_sentence_classification": []}
    if args.mock:
        synthesis["per_judge_sentence_classification"].append(
            {
                "judge_name": "mock_judge",
                "winner": "mock_winner",
                "loser": "mock_loser",
                "classified_sentences": [
                    {
                        "section": "feedback_to_loser.Weighing",
                        "target": "loser",
                        "sentence": "Mock sentence saying the debater should weigh a specific argument better.",
                        "category": "actionable_feedback",
                        "reason_for_category": "It tells the debater what to do next time.",
                        "debate_argument_anchor": "Mock referenced argument.",
                    }
                ],
                "summary_of_points_made_by_debaters": [
                    {
                        "section": "decision",
                        "target": "overall",
                        "sentence": "Mock neutral summary of a point made by a debater.",
                        "debate_argument_anchor": "Mock referenced argument.",
                    }
                ],
                "weaknesses": [
                    {
                        "section": "feedback_to_loser.Claim",
                        "target": "loser",
                        "sentence": "Mock sentence identifying a weakness.",
                        "debate_argument_anchor": "Mock referenced argument.",
                    }
                ],
                "actionable_feedback": [
                    {
                        "section": "feedback_to_loser.Weighing",
                        "target": "mock_target",
                        "sentence": "Mock sentence saying the debater should weigh a specific argument better.",
                        "debate_argument_anchor": "Mock referenced argument.",
                    }
                ],
            }
        )
    else:
        client = make_client()
        for index, judge in enumerate(judge_feedback, start=1):
            judge_name = judge.get("judge_name", f"judge_{index}")
            print(f"  Classifying judge {index}/{len(judge_feedback)}: {judge_name}", flush=True)
            try:
                system_prompt, user_prompt = synthesis_prompt(debate_text, [judge])
                judge_classification = call_json_llm(
                    client,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                judge_items = judge_classification.get("per_judge_sentence_classification", [])
                if judge_items:
                    synthesis["per_judge_sentence_classification"].extend(judge_items)
                else:
                    print(f"    Warning: no per-judge classification returned for {judge_name}", flush=True)
            except Exception as exc:
                print(f"    Warning: skipping judge {judge_name} after failure: {exc}", flush=True)
                continue

    output = {
        "step": "synthesis",
        "debate": step1["debate"],
        "debate_text": debate_text,
        "judge_feedback": judge_feedback,
        "synthesis": synthesis,
    }
    write_json(args.output, output)
    print(f"Saved step 2 synthesis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
