#!/usr/bin/env python3
"""Step 4: build a principle evidence graph from the May21 pipeline outputs."""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from refeedgraph.common import DEFAULT_OUTPUT_DIR, load_json, write_json
from refeedgraph.stages.deduplicate import build_side_lookup, compact_classified_feedback, slugify


def normalize_issue_label(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not cleaned:
        cleaned = "general_issue"
    return f"I_{cleaned}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2", type=Path, required=True, help="Step 2 synthesis JSON output.")
    parser.add_argument("--step2b", type=Path, required=True, help="Step 2b deduplicated feedback JSON output.")
    parser.add_argument("--step3", type=Path, required=True, help="Step 3 principles JSON output.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is None:
        stem = args.step2.stem.replace("_step2_synthesis", "")
        args.output = DEFAULT_OUTPUT_DIR / f"{stem}_step4_principle_evidence_graph.json"
    return args


def collect_speech_texts(debate: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    side_lookup = build_side_lookup(debate)
    side_blocks = {"pro": [], "con": []}

    for speech in debate.get("speeches", []) or []:
        debater_name = str(speech.get("debater_name", "")).strip()
        side = side_lookup.get(debater_name)
        if side not in side_blocks:
            continue
        content = str(speech.get("content", "")).strip()
        if content:
            side_blocks[side].append(content)

    speech_texts = {side: "\n\n".join(parts) for side, parts in side_blocks.items()}
    opponent_texts = {
        "pro": speech_texts.get("con", ""),
        "con": speech_texts.get("pro", ""),
    }
    return speech_texts, opponent_texts


def determine_majority_side(step2: dict[str, Any], debate: dict[str, Any]) -> str:
    side_lookup = build_side_lookup(debate)
    votes = {"pro": 0, "con": 0}
    for judge in step2.get("synthesis", {}).get("per_judge_sentence_classification", []) or []:
        winner = str(judge.get("winner", "")).strip()
        side = side_lookup.get(winner)
        if side in votes:
            votes[side] += 1

    if votes["pro"] > votes["con"]:
        return "pro"
    if votes["con"] > votes["pro"]:
        return "con"
    return "unknown"


def build_speech_nodes(step2: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    debate = step2["debate"]
    debate_id = str(debate.get("debate_id", "unknown"))
    motion = debate.get("motion", {}) or {}
    majority_side = determine_majority_side(step2, debate)
    speech_texts, opponent_texts = collect_speech_texts(debate)

    nodes: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for side in ("pro", "con"):
        node_id = f"S_{debate_id}_{side}"
        node = {
            "id": node_id,
            "type": "SpeechInstance",
            "debate_id": debate_id,
            "motion": motion.get("motion", ""),
            "side": side,
            "speech_text": speech_texts.get(side, ""),
            "opponent_text": opponent_texts.get(side, ""),
            "outcome": "winner" if side == majority_side else ("loser" if majority_side in {"pro", "con"} else "unknown"),
        }
        nodes.append(node)
        lookup[side] = node

    return nodes, lookup


def iter_specific_feedback_rows(step2: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], str]]:
    debate = step2["debate"]
    classified = step2["synthesis"]
    rows_by_category, source_lookup = compact_classified_feedback(classified, debate)
    specific_rows = rows_by_category["weakness"] + rows_by_category["actionable_feedback"]

    tuple_lookup: dict[tuple[str, str, str, str], str] = {}
    for row in specific_rows:
        tuple_lookup[(row.get("judge_name", ""), row.get("section", ""), row.get("target", ""), row.get("sentence", ""))] = row["sentence_id"]

    return specific_rows, tuple_lookup


def build_specific_feedback_nodes(step2: dict[str, Any], speech_lookup: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    debate = step2["debate"]
    debate_id = str(debate.get("debate_id", "unknown"))
    specific_rows, _ = iter_specific_feedback_rows(step2)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for row in specific_rows:
        sentence_id = str(row.get("sentence_id", "")).strip()
        if not sentence_id:
            continue
        side = str(row.get("side", "unknown")).strip()
        judge_name = str(row.get("judge_name", "")).strip()
        category = str(row.get("category", "")).strip()
        text = str(row.get("sentence", "")).strip()
        source_sides = [side] if side in {"pro", "con"} else ["pro", "con"]
        speech_ids = [f"S_{debate_id}_{current_side}" for current_side in source_sides if current_side in speech_lookup]
        primary_speech_id = speech_ids[0] if speech_ids else f"S_{debate_id}_{side}"

        nodes.append(
            {
                "id": sentence_id,
                "type": "SpecificFeedback",
                "sentence_id": sentence_id,
                "debate_id": debate_id,
                "speech_id": primary_speech_id,
                "speech_ids": speech_ids,
                "judge_guide": judge_name,
                "judge_name": judge_name,
                "category": category,
                "text": text,
                "side": side,
                "target": row.get("target"),
                "section": row.get("section"),
                "debate_argument_anchor": row.get("debate_argument_anchor"),
            }
        )

        for speech_id in speech_ids:
            edges.append({"source": speech_id, "target": sentence_id, "type": "RECEIVED_FEEDBACK"})

    return nodes, edges


def repair_cluster_members(
    cluster: dict[str, Any],
    tuple_lookup: dict[tuple[str, str, str, str], str],
) -> list[str]:
    member_ids = [str(value).strip() for value in cluster.get("member_feedback_ids", []) or [] if str(value).strip()]
    if member_ids:
        return member_ids

    merged_from = cluster.get("merged_from", []) or []
    for member in merged_from:
        if not isinstance(member, dict):
            continue
        sentence_id = str(member.get("sentence_id", "")).strip()
        if sentence_id:
            member_ids.append(sentence_id)
            continue
        key = (
            str(member.get("judge_name", "")).strip(),
            str(member.get("section", "")).strip(),
            str(member.get("target", "")).strip(),
            str(member.get("sentence", "")).strip(),
        )
        sentence_id = tuple_lookup.get(key, "")
        if sentence_id:
            member_ids.append(sentence_id)

    seen = set()
    deduped_member_ids = []
    for member_id in member_ids:
        if member_id not in seen:
            seen.add(member_id)
            deduped_member_ids.append(member_id)
    return deduped_member_ids


def build_cluster_nodes(
    step2b: dict[str, Any],
    tuple_lookup: dict[tuple[str, str, str, str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    cluster_lookup: dict[str, dict[str, Any]] = {}

    for category in ("weakness", "actionable_feedback"):
        for index, cluster in enumerate(step2b.get(category, []) or [], start=1):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get("cluster_id", "")).strip() or f"C_{category}_{index:03d}"
            canonical_text = str(cluster.get("canonical_text", "")).strip()
            member_feedback_ids = repair_cluster_members(cluster, tuple_lookup)
            node = {
                "id": cluster_id,
                "type": "FeedbackCluster",
                "cluster_id": cluster_id,
                "category": category,
                "canonical_text": canonical_text,
                "member_feedback_ids": member_feedback_ids,
            }
            nodes.append(node)
            cluster_lookup[cluster_id] = node
            for member_id in member_feedback_ids:
                edges.append({"source": member_id, "target": cluster_id, "type": "PART_OF_CLUSTER"})

    return nodes, edges, cluster_lookup


def build_principle_nodes(
    step3: dict[str, Any],
    cluster_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]], dict[str, dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    issue_to_principles: dict[str, list[str]] = {}
    issue_nodes: dict[str, dict[str, Any]] = {}

    principles = step3.get("abstract_principles", []) or []
    for index, principle in enumerate(principles, start=1):
        if not isinstance(principle, dict):
            continue

        principle_id = str(principle.get("principle_id", "")).strip() or f"P_{index:03d}"
        instance_feedback = str(principle.get("instance_feedback", "")).strip()
        source_specific_context = str(principle.get("source_specific_context", "")).strip()
        source_argument_anchor = str(principle.get("source_argument_anchor", "")).strip()
        principle_text = str(principle.get("principle", principle.get("abstract_principle", ""))).strip()
        failure_pattern = str(principle.get("failure_pattern", "")).strip()
        trigger_pattern = str(principle.get("trigger_pattern", "")).strip()
        applicability = str(principle.get("applicability", principle.get("applies_when", ""))).strip()
        diagnostic_question = str(principle.get("diagnostic_question", "")).strip()
        action_template = str(principle.get("action_template", "")).strip()
        repair_steps = str(principle.get("repair_steps", "")).strip()
        example_application = str(principle.get("example_application", "")).strip()
        source_cluster_ids = [str(value).strip() for value in principle.get("source_cluster_ids", []) or [] if str(value).strip()]
        issue_text = str(principle.get("issue_label", failure_pattern or principle.get("principle_type", ""))).strip()
        issue_id = normalize_issue_label(issue_text)

        nodes.append(
            {
                "id": principle_id,
                "type": "Principle",
                "principle_id": principle_id,
                "instance_feedback": instance_feedback,
                "source_specific_context": source_specific_context,
                "source_argument_anchor": source_argument_anchor,
                "principle": principle_text,
                "failure_pattern": failure_pattern,
                "trigger_pattern": trigger_pattern,
                "issue_label": issue_text,
                "applicability": applicability,
                "diagnostic_question": diagnostic_question,
                "action_template": action_template,
                "repair_steps": repair_steps,
                "example_application": example_application,
                "source_cluster_ids": source_cluster_ids,
                "principle_type": principle.get("principle_type", ""),
            }
        )

        issue_nodes.setdefault(
            issue_id,
            {
                "id": issue_id,
                "type": "IssueLabel",
                "label": issue_text or issue_id,
                "description": failure_pattern or issue_text,
            },
        )

        edges.append({"source": principle_id, "target": issue_id, "type": "ADDRESSES_ISSUE"})
        issue_to_principles.setdefault(issue_id, []).append(principle_id)

        for cluster_id in source_cluster_ids:
            if cluster_id in cluster_lookup:
                edges.append({"source": cluster_id, "target": principle_id, "type": "ABSTRACTED_TO"})
                edges.append({"source": cluster_id, "target": issue_id, "type": "INDICATES_ISSUE"})

    for issue_id, principle_ids in issue_to_principles.items():
        for left, right in combinations(sorted(set(principle_ids)), 2):
            edges.append(
                {
                    "source": left,
                    "target": right,
                    "type": "RELATED_TO",
                    "relation": "same_issue",
                }
            )

    return nodes, edges, issue_to_principles, issue_nodes


def dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if node_id and node_id not in deduped:
            deduped[node_id] = node
    return list(deduped.values())


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for edge in edges:
        key = (
            str(edge.get("source", "")).strip(),
            str(edge.get("target", "")).strip(),
            str(edge.get("type", "")).strip(),
            str(edge.get("relation", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def main() -> None:
    args = parse_args()
    step2 = load_json(args.step2)
    step2b = load_json(args.step2b)
    step3 = load_json(args.step3)

    debate = step2["debate"]
    debate_id = str(debate.get("debate_id", "unknown"))

    speech_nodes, speech_lookup = build_speech_nodes(step2)
    specific_nodes, specific_edges = build_specific_feedback_nodes(step2, speech_lookup)
    specific_rows, tuple_lookup = iter_specific_feedback_rows(step2)
    cluster_nodes, cluster_edges, cluster_lookup = build_cluster_nodes(step2b, tuple_lookup)
    principle_nodes, principle_edges, _, issue_nodes = build_principle_nodes(step3, cluster_lookup)

    nodes = dedupe_nodes([
        *speech_nodes,
        *specific_nodes,
        *cluster_nodes,
        *principle_nodes,
        *issue_nodes.values(),
    ])
    edges = dedupe_edges([
        *specific_edges,
        *cluster_edges,
        *principle_edges,
    ])

    graph = {
        "metadata": {
            "debate_id": debate_id,
            "motion": debate.get("motion", {}).get("motion", ""),
            "speech_instance_count": len(speech_nodes),
            "specific_feedback_count": len(specific_nodes),
            "feedback_cluster_count": len(cluster_nodes),
            "principle_count": len(principle_nodes),
            "issue_label_count": len(issue_nodes),
            "edge_count": len(edges),
            "source_files": {
                "step2": str(args.step2),
                "step2b": str(args.step2b),
                "step3": str(args.step3),
            },
        },
        "nodes": nodes,
        "edges": edges,
    }

    write_json(args.output, graph)
    print(f"Saved step 4 principle evidence graph to {args.output}", flush=True)


if __name__ == "__main__":
    main()
