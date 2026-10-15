#!/usr/bin/env python3
"""Step 2b: deduplicate near-matching classified feedback sentences."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from refeedgraph.common import DEFAULT_OUTPUT_DIR, call_json_llm, load_json, make_client, write_json


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "unknown"


def build_side_lookup(debate: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    motion = debate.get("motion", {}) or {}
    for team in motion.get("pro_side", []) or []:
        name = str(team.get("name", "")).strip()
        if name:
            lookup[name] = "pro"
    for team in motion.get("con_side", []) or []:
        name = str(team.get("name", "")).strip()
        if name:
            lookup[name] = "con"
    return lookup


def resolve_side(target: str, winner: str, loser: str, side_lookup: dict[str, str]) -> str:
    if target == "winner":
        return side_lookup.get(winner, "unknown")
    if target == "loser":
        return side_lookup.get(loser, "unknown")
    if target in {"both", "overall", "unknown", ""}:
        return "both"
    return side_lookup.get(target, "unknown")


def compact_classified_feedback(
    classified_feedback: dict[str, Any], debate: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str, str, str, str], str]]:
    rows_by_category: dict[str, list[dict[str, Any]]] = {
        "summary": [],
        "weakness": [],
        "actionable_feedback": [],
    }
    source_lookup: dict[tuple[str, str, str, str, str], str] = {}
    seen = set()
    side_lookup = build_side_lookup(debate)
    sentence_counters: dict[str, int] = defaultdict(int)

    def add_row(category: str, judge_name: str, winner: str, loser: str, item: dict[str, Any]) -> None:
        if category not in rows_by_category:
            return
        sentence = str(item.get("sentence", "")).strip()
        if not sentence:
            return
        key = (
            category,
            judge_name,
            str(item.get("section", "")),
            str(item.get("target", "")),
            sentence,
        )
        if key in seen:
            return

        seen.add(key)
        sentence_counters[judge_name] += 1
        side = resolve_side(str(item.get("target", "")), winner, loser, side_lookup)
        sentence_id = f"F_{debate.get('debate_id', 'unknown')}_{side}_{slugify(judge_name)}_{sentence_counters[judge_name]:03d}"
        row = {
            "sentence_id": sentence_id,
            "judge_name": judge_name,
            "judge_guide": judge_name,
            "winner": winner,
            "loser": loser,
            "side": side,
            "section": str(item.get("section", "")),
            "target": str(item.get("target", "")),
            "category": category,
            "sentence": sentence,
            "debate_argument_anchor": item.get("debate_argument_anchor"),
        }
        rows_by_category[category].append(row)
        source_lookup[(category, judge_name, row["section"], row["target"], sentence)] = sentence_id

    for judge in classified_feedback.get("per_judge_sentence_classification", []):
        judge_name = str(judge.get("judge_name", "")).strip() or "unknown"
        winner = str(judge.get("winner", "")).strip()
        loser = str(judge.get("loser", "")).strip()
        for item in judge.get("classified_sentences", []):
            add_row(str(item.get("category", "")), judge_name, winner, loser, item)

        # Some step 2 outputs place filtered items here without duplicating them in
        # classified_sentences. Treat these arrays as authoritative too.
        for category, array_name in (
            ("summary", "summary_of_points_made_by_debaters"),
            ("weakness", "weaknesses"),
            ("actionable_feedback", "actionable_feedback"),
        ):
            for item in judge.get(array_name, []):
                add_row(category, judge_name, winner, loser, item)

    return rows_by_category, source_lookup


def dedupe_prompt(category: str, rows: list[dict[str, str]]) -> tuple[str, str]:
    system_prompt = """
You deduplicate one category of classified debate feedback.
Cluster semantically similar feedback across judges or sections and produce a strong
representative item for each cluster. Preserve specificity and source traceability.
Return only valid JSON.
""".strip()

    user_prompt = f"""
Deduplicate this category of classified feedback: {category}

Input shape:
- A compact list of classified feedback sentences from the same category.
- Each row has sentence_id, judge_name, section, target, category, sentence, and debate_argument_anchor.

Task:
- Remove duplicates and semantically similar items.
- Merge items that express the same underlying feedback even when wording differs.
- Merge items when they point to the same skill failure or improvement area, even if
  they mention nearby parts of the debate.
- Do not merge items that would require different fixes from the debater.
- Preserve source information for every merged item.

Return ONLY valid JSON with this shape:
{{
  "unique_items": [
    {{
      "canonical_sentence": "best concise version of the duplicated/unique sentence",
      "debate_argument_anchor": "specific argument/clash, or null",
      "merged_from": [
        {{
                    "sentence_id": "stable source sentence id",
          "judge_name": "judge name",
          "section": "source section",
          "target": "winner|loser|both|overall|unknown",
          "sentence": "original classified sentence"
        }}
      ]
    }}
  ]
}}

Deduplication rules:
- Use stronger deduplication than exact or near-exact matching.
- Merge close paraphrases and overlapping feedback.
- The canonical_sentence should be a strong representative: specific, clear, and
  more useful than any one source sentence if several source sentences overlap.
- The canonical_sentence may combine compatible details from merged sentences, but
  it must not add new feedback beyond the sources.
- For weaknesses, merge if the same debate problem is being identified, such as
  unsupported claims, failure to answer drafterman's premise/axiom argument, poor
  weighing, or hostile tone.
- For actionable_feedback, merge if the same fix is being recommended, such as
  define belief more robustly, directly answer the premise/axiom argument, compare
  the scientific-method argument against drafterman's claim, or use a more respectful
  tone.
- For summary, merge if the same debate argument is being described.
- Do not merge different failures, such as tone problems and lack of evidence.
- If a sentence has no close match, keep it as a single-item group.

CLASSIFIED_{category.upper()}_JSON:
{json.dumps(rows, indent=2, ensure_ascii=False)}
""".strip()
    return system_prompt, user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Step 2 sentence classification JSON output.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--mock", action="store_true", help="Validate locally without LLM calls.")
    args = parser.parse_args()

    if args.output is None:
        stem = args.input.stem.replace("_step2_synthesis", "").replace("_step2_sentence_classification", "")
        args.output = DEFAULT_OUTPUT_DIR / f"{stem}_step2b_deduplicated_feedback.json"
    return args


def normalize_cluster_items(
    category: str,
    unique_items: list[Any],
    source_lookup: dict[tuple[str, str, str, str, str], str],
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []

    for index, item in enumerate(unique_items, start=1):
        if not isinstance(item, dict):
            continue

        canonical_text = str(item.get("canonical_text") or item.get("canonical_sentence") or "").strip()
        merged_from_raw = item.get("merged_from", [])
        merged_from: list[dict[str, Any]] = []
        member_feedback_ids: list[str] = []

        if isinstance(merged_from_raw, list):
            for member in merged_from_raw:
                if not isinstance(member, dict):
                    continue
                judge_name = str(member.get("judge_name", "")).strip()
                section = str(member.get("section", "")).strip()
                target = str(member.get("target", "")).strip()
                sentence = str(member.get("sentence", "")).strip()
                sentence_id = str(member.get("sentence_id", "")).strip()
                if not sentence_id:
                    sentence_id = source_lookup.get((category, judge_name, section, target, sentence), "")
                if sentence_id and sentence_id not in member_feedback_ids:
                    member_feedback_ids.append(sentence_id)
                merged_from.append(
                    {
                        "sentence_id": sentence_id,
                        "judge_name": judge_name,
                        "section": section,
                        "target": target,
                        "sentence": sentence,
                    }
                )

        if not member_feedback_ids:
            member_feedback_ids = [str(value).strip() for value in item.get("member_feedback_ids", []) if str(value).strip()]

        clusters.append(
            {
                "cluster_id": f"C_{category}_{index:03d}",
                "category": category,
                "canonical_text": canonical_text,
                "member_feedback_ids": member_feedback_ids,
                "merged_from": merged_from,
                "debate_argument_anchor": item.get("debate_argument_anchor"),
            }
        )

    return clusters


def deterministic_cluster_items(
    category: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clusters = []
    for index, row in enumerate(rows, start=1):
        sentence = str(row.get("sentence", "")).strip()
        sentence_id = str(row.get("sentence_id", "")).strip()
        clusters.append(
            {
                "cluster_id": f"C_{category}_{index:03d}",
                "category": category,
                "canonical_text": sentence,
                "member_feedback_ids": [sentence_id] if sentence_id else [],
                "merged_from": [
                    {
                        "sentence_id": sentence_id,
                        "judge_name": str(row.get("judge_name", "")),
                        "section": str(row.get("section", "")),
                        "target": str(row.get("target", "")),
                        "sentence": sentence,
                    }
                ],
                "debate_argument_anchor": row.get("debate_argument_anchor"),
            }
        )
    return clusters


def merge_duplicate_clusters(category: str, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_text: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        canonical_text = str(cluster.get("canonical_text", "")).strip()
        if not canonical_text:
            continue
        key = re.sub(r"\s+", " ", canonical_text.lower())
        if key not in merged_by_text:
            merged_by_text[key] = {
                **cluster,
                "member_feedback_ids": list(cluster.get("member_feedback_ids", []) or []),
                "merged_from": list(cluster.get("merged_from", []) or []),
            }
            continue
        existing = merged_by_text[key]
        for member_id in cluster.get("member_feedback_ids", []) or []:
            if member_id not in existing["member_feedback_ids"]:
                existing["member_feedback_ids"].append(member_id)
        existing["merged_from"].extend(cluster.get("merged_from", []) or [])

    merged = list(merged_by_text.values())
    for index, cluster in enumerate(merged, start=1):
        cluster["cluster_id"] = f"C_{category}_{index:03d}"
    return merged


def dedupe_category(
    client,
    *,
    category: str,
    rows: list[dict[str, Any]],
    source_lookup: dict[tuple[str, str, str, str, str], str],
    model: str,
    max_tokens: int,
    temperature: float,
    chunk_size: int = 20,
) -> list[dict[str, Any]]:
    clusters = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        system_prompt, user_prompt = dedupe_prompt(category, chunk)
        try:
            response = call_json_llm(
                client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            unique_items = response.get("unique_items", response)
            clusters.extend(normalize_cluster_items(category, unique_items, source_lookup))
        except Exception as exc:
            print(
                f"  Warning: model dedupe failed for {category} rows "
                f"{start + 1}-{start + len(chunk)}; using local fallback: {exc}",
                flush=True,
            )
            clusters.extend(deterministic_cluster_items(category, chunk))
    return merge_duplicate_clusters(category, clusters)


def main() -> None:
    args = parse_args()
    step2 = load_json(args.input)
    debate = step2["debate"]
    classified = step2["synthesis"]
    rows_by_category, source_lookup = compact_classified_feedback(classified, debate)

    print("Deduplicating classified feedback", flush=True)
    if args.mock:
        deduped = {
            "summary": [],
            "weakness": [],
            "actionable_feedback": [],
        }
    else:
        client = make_client()
        deduped = {}
        for category in ("summary", "weakness", "actionable_feedback"):
            rows = rows_by_category[category]
            if not rows:
                deduped[category] = []
                continue
            print(f"Deduplicating {category}: {len(rows)} rows", flush=True)
            deduped[category] = dedupe_category(
                client,
                category=category,
                rows=rows,
                source_lookup=source_lookup,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

    output = {
        "step": "deduplicate_feedback",
        "source_file": str(args.input),
        "debate": step2.get("debate"),
        "debate_text": step2.get("debate_text"),
        "summary": deduped["summary"],
        "weakness": deduped["weakness"],
        "actionable_feedback": deduped["actionable_feedback"],
    }
    write_json(args.output, output)
    print(f"Saved deduplicated feedback to {args.output}", flush=True)


if __name__ == "__main__":
    main()
