#!/usr/bin/env python3
"""Step 1: generate guide-specific judge feedback for one PanelBench debate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from refeedgraph.common import (
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    JUDGE_GUIDES_DIR,
    call_json_llm,
    format_debate_for_prompt,
    judge_feedback_prompt,
    load_judge_guides,
    load_json,
    load_panelbench_debate,
    make_client,
    write_json,
)


def mock_judge_feedback(guide: dict[str, str]) -> dict[str, object]:
    return {
        "judge_name": guide["guide_name"],
        "guide_file": guide["guide_file"],
        "feedback": {
            "winner": "mock_winner",
            "loser": "mock_loser",
            "decision": "Mock decision generated without calling an LLM.",
            "feedback_to_winner": {
                "mock_guide_section": "Mock guide-native feedback to the winner."
            },
            "feedback_to_loser": {
                "mock_guide_section": "Mock guide-native feedback to the loser."
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Normalized debate JSON; overrides dataset loading.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="debateart")
    parser.add_argument("--debate-id", default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--judge-guides-dir", type=Path, default=JUDGE_GUIDES_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--max-judges", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Validate locally without LLM calls.")
    args = parser.parse_args()

    if args.debate_id is None:
        args.debate_id = DATASETS[args.dataset]["default_id"]
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"{args.dataset}_{args.debate_id}_step1_judge_feedback.json"
    return args


def main() -> None:
    args = parse_args()
    debate = load_json(args.input) if args.input else load_panelbench_debate(args.dataset, args.debate_id)
    debate_text = format_debate_for_prompt(debate)
    guides = load_judge_guides(args.judge_guides_dir)
    if args.max_judges is not None:
        guides = guides[: args.max_judges]

    client = None if args.mock else make_client()
    judge_feedback = []

    for guide in guides:
        print(f"Running judge guide: {guide['guide_name']}", flush=True)
        if args.mock:
            judge_feedback.append(mock_judge_feedback(guide))
            continue

        system_prompt, user_prompt = judge_feedback_prompt(guide, debate_text)
        raw_feedback = call_json_llm(
            client,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        judge_feedback.append(
            {
                "judge_name": guide["guide_name"],
                "guide_file": guide["guide_file"],
                "feedback": raw_feedback,
            }
        )

    output = {
        "step": "judge_feedback",
        "debate": debate,
        "debate_text": debate_text,
        "judge_feedback": judge_feedback,
    }
    write_json(args.output, output)
    print(f"Saved step 1 judge feedback to {args.output}", flush=True)


if __name__ == "__main__":
    main()
