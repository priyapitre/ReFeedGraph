#!/usr/bin/env python3
"""Combine multiple Step 4 principle-evidence graphs into one graph."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from refeedgraph.common import DEFAULT_OUTPUT_DIR, load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="One or more Step 4 graph JSON files.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / "combined_principle_evidence_graph.json"
    return args


def remap_node_id(node: dict[str, Any], debate_id: str) -> str:
    node_id = str(node.get("id", "")).strip()
    node_type = str(node.get("type", "")).strip()
    if node_type in {"FeedbackCluster", "Principle"}:
        return f"{debate_id}_{node_id}"
    return node_id


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


def drop_dangling_edges(edges: list[dict[str, Any]], node_ids: set[str]) -> list[dict[str, Any]]:
    valid_edges: list[dict[str, Any]] = []
    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source in node_ids and target in node_ids:
            valid_edges.append(edge)
    return valid_edges


def main() -> None:
    args = parse_args()

    combined_nodes: dict[str, dict[str, Any]] = {}
    combined_edges: list[dict[str, Any]] = []
    issue_to_principles: dict[str, list[str]] = defaultdict(list)
    issue_nodes: dict[str, dict[str, Any]] = {}
    debate_ids: list[str] = []

    for input_path in args.inputs:
        graph = load_json(input_path)
        metadata = graph.get("metadata", {}) or {}
        debate_id = str(metadata.get("debate_id", input_path.stem)).strip()
        if debate_id and debate_id not in debate_ids:
            debate_ids.append(debate_id)

        id_map: dict[str, str] = {}
        local_issue_to_principles: dict[str, list[str]] = defaultdict(list)

        for node in graph.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            original_id = str(node.get("id", "")).strip()
            if not original_id:
                continue
            remapped_id = remap_node_id(node, debate_id)
            id_map[original_id] = remapped_id

            node_copy = dict(node)
            node_copy["id"] = remapped_id
            if node_copy.get("type") in {"FeedbackCluster", "Principle"}:
                node_copy["debate_id"] = debate_id
                if node_copy.get("type") == "FeedbackCluster":
                    node_copy["member_feedback_ids"] = [
                        feedback_id
                        for feedback_id in node_copy.get("member_feedback_ids", []) or []
                        if str(feedback_id).strip() in id_map
                    ]
            elif node_copy.get("type") == "IssueLabel":
                existing = issue_nodes.get(remapped_id)
                if existing:
                    existing_debates = existing.setdefault("debate_ids", [])
                    if debate_id not in existing_debates:
                        existing_debates.append(debate_id)
                    existing_source_clusters = existing.setdefault("source_cluster_ids", [])
                    for cluster_id in node_copy.get("source_cluster_ids", []) or []:
                        if cluster_id not in existing_source_clusters:
                            existing_source_clusters.append(cluster_id)
                    continue
                node_copy.setdefault("debate_ids", [debate_id])
                node_copy.setdefault("source_cluster_ids", [])
                issue_nodes[remapped_id] = node_copy
                continue

            combined_nodes[remapped_id] = node_copy

        for edge in graph.get("edges", []) or []:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            remapped_edge = dict(edge)
            if source in id_map:
                remapped_edge["source"] = id_map[source]
            if target in id_map:
                remapped_edge["target"] = id_map[target]
            combined_edges.append(remapped_edge)

            if remapped_edge.get("type") == "ADDRESSES_ISSUE":
                issue_id = str(remapped_edge["target"]).strip()
                principle_id = str(remapped_edge["source"]).strip()
                local_issue_to_principles[issue_id].append(principle_id)
                issue_to_principles[issue_id].append(principle_id)

    # Add cross-debate RELATED_TO edges for any principles sharing the same issue label.
    for issue_id, principle_ids in issue_to_principles.items():
        unique_principles = sorted({pid for pid in principle_ids if pid})
        for i, left in enumerate(unique_principles):
            for right in unique_principles[i + 1 :]:
                combined_edges.append(
                    {
                        "source": left,
                        "target": right,
                        "type": "RELATED_TO",
                        "relation": "same_issue",
                        "issue_id": issue_id,
                    }
                )

    combined_nodes.update(issue_nodes)
    combined_edges = drop_dangling_edges(combined_edges, set(combined_nodes))
    combined_edges = dedupe_edges(combined_edges)

    output = {
        "metadata": {
            "debate_ids": debate_ids,
            "debate_count": len(debate_ids),
            "node_count": len(combined_nodes),
            "edge_count": len(combined_edges),
            "source_files": [str(path) for path in args.inputs],
        },
        "nodes": list(combined_nodes.values()),
        "edges": combined_edges,
    }

    write_json(args.output, output)
    print(f"Saved combined graph to {args.output}", flush=True)


if __name__ == "__main__":
    main()
