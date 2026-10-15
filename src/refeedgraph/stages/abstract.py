#!/usr/bin/env python3
"""Step 3: abstract deduplicated weakness/action feedback into reusable principles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from refeedgraph.common import (
    DEFAULT_OUTPUT_DIR,
    call_json_llm,
    load_json,
    make_client,
    normalize_principles,
    principles_from_deduped_feedback_prompt,
    write_json,
)


def mock_principles() -> list[dict[str, str]]:
    return [
        {
            "instance_feedback": "Mock feedback item.",
            "abstract_principle": "Feedback should be generalized into reusable practice rules.",
            "failure_pattern": "Pipeline not run with real LLM",
            "applies_when": "Testing script wiring.",
            "diagnostic_question": "Did the pipeline produce graph nodes and edges?",
            "action_template": "Run the script without --mock.",
            "principle_type": "pipeline validation",
            "source_cluster_ids": ["C_weakness_001"],
        }
    ]


def build_source_clusters(deduped_feedback: dict[str, Any]) -> list[dict[str, Any]]:
    source_clusters: list[dict[str, Any]] = []
    for category in ("weakness", "actionable_feedback"):
        for cluster in deduped_feedback.get(category, []) or []:
            if not isinstance(cluster, dict):
                continue
            source_clusters.append(
                {
                    "cluster_id": str(cluster.get("cluster_id", "")).strip(),
                    "category": category,
                    "canonical_text": str(cluster.get("canonical_text", "")).strip(),
                }
            )
    return source_clusters


def enrich_principles(
    principles: list[dict[str, str]],
    source_clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup: dict[str, list[str]] = {}
    for cluster in source_clusters:
        text = cluster.get("canonical_text", "")
        cluster_id = cluster.get("cluster_id", "")
        if text and cluster_id:
            lookup.setdefault(text, []).append(cluster_id)

    enriched: list[dict[str, Any]] = []
    for index, principle in enumerate(principles, start=1):
        instance_feedback = str(principle.get("instance_feedback", "")).strip()
        source_specific_context = str(principle.get("source_specific_context", "")).strip()
        source_argument_anchor = str(principle.get("source_argument_anchor", "")).strip()
        abstract_principle = str(principle.get("abstract_principle", "")).strip()
        source_cluster_ids = [str(value).strip() for value in principle.get("source_cluster_ids", []) if str(value).strip()]

        if not source_cluster_ids:
            if instance_feedback in lookup:
                source_cluster_ids = lookup[instance_feedback][:1]
            elif abstract_principle in lookup:
                source_cluster_ids = lookup[abstract_principle][:1]
            elif index <= len(source_clusters):
                fallback_id = source_clusters[index - 1].get("cluster_id", "")
                if fallback_id:
                    source_cluster_ids = [fallback_id]

        failure_pattern = str(principle.get("failure_pattern", "")).strip()
        trigger_pattern = str(principle.get("trigger_pattern", "")).strip()
        applies_when = str(principle.get("applies_when", "")).strip()
        diagnostic_question = str(principle.get("diagnostic_question", "")).strip()
        action_template = str(principle.get("action_template", "")).strip()
        repair_steps = str(principle.get("repair_steps", "")).strip()
        example_application = str(principle.get("example_application", "")).strip()
        principle_type = str(principle.get("principle_type", "")).strip()

        enriched.append(
            {
                "principle_id": f"P_{index:03d}",
                "instance_feedback": instance_feedback,
                "source_specific_context": source_specific_context,
                "source_argument_anchor": source_argument_anchor,
                "abstract_principle": abstract_principle,
                "principle": abstract_principle,
                "failure_pattern": failure_pattern,
                "trigger_pattern": trigger_pattern,
                "issue_label": str(principle.get("issue_label", failure_pattern)).strip() or failure_pattern,
                "applies_when": applies_when,
                "applicability": applies_when,
                "diagnostic_question": diagnostic_question,
                "action_template": action_template,
                "repair_steps": repair_steps,
                "example_application": example_application,
                "principle_type": principle_type,
                "source_cluster_ids": source_cluster_ids,
            }
        )

    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Step 2b deduplicated feedback JSON output.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--mock", action="store_true", help="Validate locally without LLM calls.")
    args = parser.parse_args()

    if args.output is None:
        stem = args.input.stem.replace("_step2b_deduplicated_feedback", "")
        args.output = DEFAULT_OUTPUT_DIR / f"{stem}_step3_principles.json"
    return args


def main() -> None:
    args = parse_args()
    deduped_feedback = load_json(args.input)
    source_clusters = build_source_clusters(deduped_feedback)

    print("Abstracting reusable principles", flush=True)
    if args.mock:
        principles = mock_principles()
    else:
        client = make_client()
        system_prompt, user_prompt = principles_from_deduped_feedback_prompt(deduped_feedback)
        abstraction = call_json_llm(
            client,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        principles = normalize_principles(abstraction)

    principles = enrich_principles(principles, source_clusters)

    output = {
        "step": "abstract_principles",
        "source_file": str(args.input),
        "deduplicated_feedback": {
            "weakness": deduped_feedback.get("weakness", []),
            "actionable_feedback": deduped_feedback.get("actionable_feedback", []),
        },
        "abstract_principles": principles,
    }
    write_json(args.output, output)
    print(f"Saved step 3 principles to {args.output}", flush=True)


if __name__ == "__main__":
    main()
