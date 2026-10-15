"""Method functions extracted without algorithm or prompt changes from the research runner."""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from .common import call_json_llm, load_json

def speech_for_side(debate: dict[str, Any], side_name: str) -> str:
    speeches = [
        f"[Speech {i}: {speech.get('debater_name')}]\n{speech.get('content', '').strip()}"
        for i, speech in enumerate(debate.get("speeches", []), start=1)
        if speech.get("debater_name") == side_name
    ]
    if not speeches:
        raise ValueError(f"No speeches found for side/debater: {side_name}")
    return "\n\n".join(speeches)

def generate_generic_issues(
    client,
    *,
    model: str,
    debate_text: str,
    losing_side: str,
    temperature: float,
) -> list[dict[str, str]]:
    system_prompt = "You identify concrete issues in debate speeches. Return only valid JSON."
    user_prompt = f"""
List concrete issues with the losing side/debater: {losing_side}.
Focus on specific argument failures, missing responses, weak evidence, poor weighing,
tone problems, and unclear definitions.

Return JSON:
{{
  "issues": [
    {{
      "issue": "specific issue",
      "debate_argument_anchor": "specific argument/clash this issue concerns"
    }}
  ]
}}

DEBATE:
{debate_text}
""".strip()
    result = call_json_llm(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=3000,
        temperature=temperature,
    )
    return normalize_issues(result.get("issues", []))

def normalize_issues(raw_issues: Any) -> list[dict[str, str]]:
    if not isinstance(raw_issues, list):
        raw_issues = [raw_issues]
    issues = []
    for item in raw_issues:
        if isinstance(item, dict):
            issue = str(item.get("issue", "") or item.get("problem", "") or item.get("text", "")).strip()
            anchor = str(item.get("debate_argument_anchor", "") or item.get("anchor", "")).strip()
        else:
            issue = str(item).strip()
            anchor = ""
        if issue:
            issues.append({"issue": issue, "debate_argument_anchor": anchor})
    return issues

def principle_text(principle: dict[str, Any]) -> str:
    if not isinstance(principle, dict):
        return str(principle)
    return " ".join(
        str(principle.get(key, ""))
        for key in (
            "instance_feedback",
            "source_specific_context",
            "source_argument_anchor",
            "abstract_principle",
            "principle",
            "failure_pattern",
            "trigger_pattern",
            "applies_when",
            "applicability",
            "diagnostic_question",
            "action_template",
            "repair_steps",
            "example_application",
            "principle_type",
        )
    )

def graph_node_text(node: dict[str, Any]) -> str:
    if not isinstance(node, dict):
        return str(node)
    node_type = str(node.get("type", "")).strip()
    if node_type == "IssueLabel":
        return " ".join(
            str(node.get(key, ""))
            for key in ("label", "description")
        ).strip()
    if node_type == "Principle":
        return " ".join(
            str(node.get(key, ""))
            for key in (
                "principle",
                "instance_feedback",
                "source_specific_context",
                "source_argument_anchor",
                "failure_pattern",
                "trigger_pattern",
                "applicability",
                "diagnostic_question",
                "action_template",
                "repair_steps",
                "example_application",
                "principle_type",
            )
        ).strip()
    if node_type == "FeedbackCluster":
        return " ".join(str(node.get(key, "")) for key in ("canonical_text", "category")).strip()
    if node_type == "SpecificFeedback":
        return " ".join(str(node.get(key, "")) for key in ("text", "category", "section", "target")).strip()
    return " ".join(str(value) for value in node.values() if isinstance(value, (str, int, float))).strip()

def load_graph(path: Path) -> dict[str, Any]:
    data = load_json(path)
    nodes = {}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for node in data.get("nodes", []) or []:
        if isinstance(node, dict):
            node_id = str(node.get("id", "")).strip()
            if node_id:
                nodes[node_id] = node

    for edge in data.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source:
            outgoing[source].append(edge)
        if target:
            incoming[target].append(edge)

    return {"data": data, "nodes": nodes, "outgoing": outgoing, "incoming": incoming}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "in", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "were", "with", "you", "your",
}

def tokenize_for_retrieval(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]

def retrieval_score(query: str, document: str) -> float:
    query_tokens = tokenize_for_retrieval(query)
    document_tokens = tokenize_for_retrieval(document)
    if not query_tokens or not document_tokens:
        return 0.0

    q_counts = defaultdict(int)
    d_counts = defaultdict(int)
    for token in query_tokens:
        q_counts[token] += 1
    for token in document_tokens:
        d_counts[token] += 1

    overlap = set(q_counts) & set(d_counts)
    if not overlap:
        return 0.0

    weighted_overlap = sum(min(q_counts[token], d_counts[token]) for token in overlap)
    recall = weighted_overlap / max(1, sum(q_counts.values()))
    precision = weighted_overlap / max(1, sum(d_counts.values()))
    phrase_bonus = 0.0
    query_phrases = re.findall(r"\b[a-z0-9_]+(?:\s+[a-z0-9_]+){1,3}\b", query.lower())
    document_lower = document.lower()
    for phrase in query_phrases[:30]:
        phrase_tokens = tokenize_for_retrieval(phrase)
        if len(phrase_tokens) >= 2 and " ".join(phrase_tokens) in document_lower:
            phrase_bonus += 0.15

    return (2 * precision * recall / max(precision + recall, 1e-9)) + phrase_bonus

def dedupe_by_text(items: list[dict[str, Any]], text_key: str, *, limit: int) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            item = {text_key: str(item)}
        tokens = tokenize_for_retrieval(str(item.get(text_key, "")))
        signature = " ".join(tokens[:40])
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped

def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)

def embed_texts(client, texts: list[str], *, model: str) -> list[list[float]]:
    if not texts:
        return []
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]

def rank_items(
    query: str,
    items: list[dict[str, Any]],
    *,
    text_fn,
    retrieval_method: str,
    client=None,
    embedding_model: str = "text-embedding-3-small",
) -> list[dict[str, Any]]:
    if retrieval_method == "embedding" and client is not None and items:
        try:
            texts = [text_fn(item) for item in items]
            embeddings = embed_texts(client, [query, *texts], model=embedding_model)
            query_embedding = embeddings[0]
            item_embeddings = embeddings[1:]
            return [
                item
                for _, item in sorted(
                    zip(
                        [cosine_similarity(query_embedding, embedding) for embedding in item_embeddings],
                        items,
                    ),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ]
        except Exception as exc:
            print(f"Embedding retrieval failed; falling back to lexical retrieval: {exc}", flush=True)

    return sorted(items, key=lambda item: retrieval_score(query, text_fn(item)), reverse=True)

def retrieve_principles(
    principles: list[dict[str, Any]],
    issues: list[dict[str, str]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    issues = normalize_issues(issues)
    principles = [
        principle if isinstance(principle, dict) else {"principle": str(principle)}
        for principle in principles
    ]
    query = "\n".join(f"{x.get('issue', '')} {x.get('debate_argument_anchor', '')}" for x in issues)
    ranked = sorted(
        principles,
        key=lambda p: retrieval_score(query, principle_text(p)),
        reverse=True,
    )
    return dedupe_by_text(ranked, "abstract_principle", limit=top_k)

def retrieve_graph_context(
    graph_index: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    top_k_principles: int,
    specific_examples_per_principle: int,
    max_issue_labels: int = 8,
    max_principle_candidates: int = 80,
    max_specific_examples_total: int = 30,
    retrieval_method: str = "lexical",
    embedding_model: str = "text-embedding-3-small",
    client=None,
) -> dict[str, Any]:
    issues = normalize_issues(issues)
    nodes = graph_index["nodes"]
    outgoing = graph_index["outgoing"]
    incoming = graph_index["incoming"]
    query = "\n".join(f"{x.get('issue', '')} {x.get('debate_argument_anchor', '')}" for x in issues)

    issue_nodes = [node for node in nodes.values() if node.get("type") == "IssueLabel"]
    cluster_nodes = [node for node in nodes.values() if node.get("type") == "FeedbackCluster"]
    ranked_issues = rank_items(
        query,
        issue_nodes,
        text_fn=graph_node_text,
        retrieval_method=retrieval_method,
        client=client,
        embedding_model=embedding_model,
    )
    selected_issues = [node for node in ranked_issues if retrieval_score(query, graph_node_text(node)) > 0]
    if not selected_issues:
        selected_issues = ranked_issues
    selected_issue_ids = [
        str(node.get("id", "")).strip()
        for node in selected_issues[: max(1, min(len(selected_issues), max_issue_labels))]
    ]

    principle_ids: list[str] = []
    for issue_id in selected_issue_ids:
        for edge in incoming.get(issue_id, []):
            if str(edge.get("type", "")).strip() == "ADDRESSES_ISSUE":
                source_id = str(edge.get("source", "")).strip()
                if source_id and source_id not in principle_ids:
                    principle_ids.append(source_id)

    principle_nodes = [nodes[pid] for pid in principle_ids[:max_principle_candidates] if pid in nodes]
    ranked_principles = rank_items(
        query,
        principle_nodes,
        text_fn=graph_node_text,
        retrieval_method=retrieval_method,
        client=client,
        embedding_model=embedding_model,
    )
    ranked_principles = dedupe_by_text(ranked_principles, "principle", limit=top_k_principles)

    specific_feedback: list[dict[str, Any]] = []
    for principle in ranked_principles:
        principle_id = str(principle.get("id", "")).strip()
        cluster_ids = [
            str(cluster_id).strip()
            for cluster_id in principle.get("source_cluster_ids", []) or []
            if str(cluster_id).strip()
        ]
        if not cluster_ids:
            cluster_ids = [
                str(edge.get("source", "")).strip()
                for edge in incoming.get(principle_id, [])
                if str(edge.get("type", "")).strip() == "ABSTRACTED_TO"
            ]
        if not cluster_ids:
            ranked_clusters = rank_items(
                query + "\n" + graph_node_text(principle),
                cluster_nodes,
                text_fn=graph_node_text,
                retrieval_method=retrieval_method,
                client=client,
                embedding_model=embedding_model,
            )
            cluster_ids = [str(node.get("id", "")).strip() for node in ranked_clusters[:2] if str(node.get("id", "")).strip()]
        for cluster_id in cluster_ids:
            cluster_node = nodes.get(cluster_id)
            if not cluster_node:
                continue
            member_ids = [
                str(member_id).strip()
                for member_id in cluster_node.get("member_feedback_ids", []) or []
                if str(member_id).strip()
            ]
            if not member_ids:
                member_ids = [
                    str(edge.get("source", "")).strip()
                    for edge in incoming.get(cluster_id, [])
                    if str(edge.get("type", "")).strip() == "PART_OF_CLUSTER"
                ]
            candidate_feedback = [nodes[mid] for mid in member_ids if mid in nodes and nodes[mid].get("type") == "SpecificFeedback"]
            ranked_feedback = rank_items(
                query,
                candidate_feedback,
                text_fn=lambda node, cluster=cluster_node, current_principle=principle: (
                    graph_node_text(node) + " " + graph_node_text(cluster) + " " + graph_node_text(current_principle)
                ),
                retrieval_method=retrieval_method,
                client=client,
                embedding_model=embedding_model,
            )[:specific_examples_per_principle]
            for item in ranked_feedback:
                specific_feedback.append(
                    {
                        "specific_feedback_id": item.get("id"),
                        "cluster_id": cluster_id,
                        "principle_id": principle_id,
                        "text": item.get("text", ""),
                        "judge_guide": item.get("judge_guide", ""),
                        "category": item.get("category", ""),
                        "section": item.get("section", ""),
                        "target": item.get("target", ""),
                        "debate_argument_anchor": item.get("debate_argument_anchor"),
                    }
                )

    unique_specific_feedback = []
    seen_specific = set()
    for item in specific_feedback:
        key = (item.get("specific_feedback_id"), item.get("cluster_id"), item.get("principle_id"))
        if key in seen_specific:
            continue
        seen_specific.add(key)
        unique_specific_feedback.append(item)
        if len(unique_specific_feedback) >= max_specific_examples_total:
            break

    return {
        "issue_labels": selected_issues[:max_issue_labels],
        "retrieved_principles": ranked_principles,
        "specific_feedback": unique_specific_feedback,
    }

def specify_our_feedback(
    client,
    *,
    model: str,
    debate_text: str,
    losing_side: str,
    issues: list[dict[str, str]],
    principles: list[dict[str, Any]],
    supporting_examples: list[dict[str, Any]],
    temperature: float,
) -> str:
    system_prompt = """
You convert abstract debate-improvement principles into concrete feedback for one debate.
Return feedback only, not JSON. Be concrete, surgical, and tied to the debate text.
""".strip()
    user_prompt = f"""
Write specific feedback only to the losing side/debater: {losing_side}.

Use the generic issues and retrieved principles below, but make the feedback concrete
to THIS debate. If a principle says "add evidence" or "compare arguments", name the
specific argument/clash where that should happen.

Strategic constraints:
- First infer the losing side's strongest existing strategy from the debate transcript.
  Preserve and strengthen that strategy instead of replacing it with a different case.
- Identify 2 to 4 things the losing side already does well, especially strong evidence,
  clear structure, useful framing, or promising rebuttals. Your feedback should tell the
  rewriter to preserve and sharpen those strengths, not accidentally delete them.
- Do not ask the losing side to accept the opponent's framing, burden, theology,
  definitions, or evidentiary standard unless doing so is strategically necessary.
- Do not create new burdens for the losing side. For example, if the losing side's
  strategy is to press the opponent's burden of proof, do not require them to prove a
  full alternative theology; instead, help them explain why the opponent's evidence
  fails their burden.
- Prefer repairs that make the existing argument more judge-relevant: direct clash,
  clearer warranting, better evidence standards, sharper burden framing, and explicit
  weighing.
- Treat retrieved graph principles as candidate memories, not obligations. Use only the
  principles that help improve this debater's current speech. Ignore or down-rank a
  retrieved principle if applying it would distract from the central clash, duplicate
  another fix, impose a new burden, or reduce relevance.
- Rank feedback by likely effect on the judge's decision. Put the highest-impact
  clash or burden issue first, and lower-priority tone/style issues later.
- Prefer feedback that is likely to improve evidence, specificity, direct refutation,
  and relevance in the revised speech. If a diagnosis is true but would not improve the
  rewrite much, make it lower priority or omit it.

Output requirements:
- Give 4 to 7 numbered feedback items.
- Each item should be a substantive paragraph, not a short bullet.
- Every item must name a concrete section, argument, claim, source, definition, or clash
  from the losing side's speech.
- Every item must name the opponent-side claim or judge concern it needs to answer when
  the issue is about clash, refutation, weighing, evidence, or definitions.
- Every item must include:
  - Problem in this debate: the exact weakness in the losing side's argument.
  - Why it matters: how it affects judging, clash, evidence, burden, or weighing.
  - Preserve/strengthen: the existing argument, evidence, structure, or framing that
    should be kept while revising.
  - Rewrite move: exactly what to add, delete, reframe, compare, or explain.
- Use retrieved principles as repair rules, not as generic advice. If a retrieved principle
  includes source_specific_context or repair_steps, apply those steps to the specific
  debate material and preserve the level of specificity.
- Do not say only "add evidence", "be clearer", "engage more", or "improve structure".
  Specify what evidence would need to establish, what term must be defined, what argument
  must be answered, or what comparison must be weighed.
- For each Rewrite move, make clear how the revision strengthens the losing side's
  existing route to winning the debate.
- Avoid copying old supporting examples as if they happened in this debate; use them only
  to infer the kind of repair that is needed.

GENERIC ISSUES:
{json.dumps(issues, indent=2, ensure_ascii=False)}

RETRIEVED PRINCIPLES:
{json.dumps(principles, indent=2, ensure_ascii=False)}

SUPPORTING SPECIFIC FEEDBACK EXAMPLES:
{json.dumps(supporting_examples, indent=2, ensure_ascii=False)}

DEBATE:
{debate_text}
""".strip()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=3500,
    )
    return response.choices[0].message.content.strip()

def revise_debate_speech(
    client,
    *,
    model: str,
    title: str,
    debate_text: str,
    losing_side: str,
    original_speech: str,
    feedback: str,
    temperature: float,
    max_tokens: int,
) -> str:
    system_prompt = """
You are revising one side's debate speech in response to feedback.
Use the full debate transcript as context, but rewrite only the losing side's speech.
Your goal is to produce a stronger revised version of that side's speech, not a judge
explanation and not feedback.
You must be conservative: revise only the parts of the original speech targeted by
the feedback, and preserve all other content.
""".strip()

    user_prompt = f"""
Debate motion:
{title}

Losing side/debater to revise:
{losing_side}

Task:
- Rewrite only {losing_side}'s speech.
- Use the full debate transcript to understand the opponent's claims, definitions,
  evidence, burdens, concessions, and clashes. Use that context only to apply the
  feedback to the original speech, not to invent a new case.
- Apply every substantive feedback item, but only by editing the relevant part of the
  original speech.
- Addressability gate: only make a substantive revision when the feedback gives a
  concrete target or concrete edit, such as a named section, argument, claim, source,
  definition, opponent response, tone problem, or exact reasoning gap.
- If feedback is vague, generic, or only says things like "add evidence", "be clearer",
  "improve structure", "engage more", or "be more persuasive" without naming what to
  change, do not invent the missing instruction. Make at most a minimal local clarity
  edit, or leave that part unchanged.
- Do not turn vague feedback into new paragraphs, new examples, new responses, or new
  strategic framing. The quality of the rewrite should reflect how specific and
  addressable the feedback was.
- Do not add new arguments, new contentions, new sources, new statistics, new examples,
  new burdens, or new strategic framing unless the feedback explicitly asks for that
  addition and it is grounded in the original speech or full debate transcript.
- Do not create a new paragraph unless the feedback explicitly identifies the argument
  or opponent claim that paragraph should address.
- Keep unchanged sections unchanged when the feedback does not target them.
- Preserve the losing side's original position, organization, evidence, citations,
  examples, and voice as much as possible.
- When feedback says to answer an opponent claim, add the answer in the relevant
  existing section of the revised speech and name the opponent's claim, but do not
  create an unrelated new contention.
- When feedback says to define a term, add the definition and explain how it affects
  the motion or burden in the part of the original speech where that term already matters.
- When feedback says to support a claim, add reasoning from the debate record or
  explain what the cited source/argument establishes. Do not invent fake citations,
  statistics, or quotations.
- Remove or soften dismissive language when it weakens credibility.
- Do not polish the whole speech for style. Only make style changes where feedback
  specifically targets tone, clarity, or organization.
- Output only the revised speech. Do not include notes, labels, or commentary.

FULL DEBATE TRANSCRIPT:
{debate_text}

ORIGINAL SPEECH TO REVISE:
{original_speech}

FEEDBACK TO APPLY:
{feedback}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    return (response.choices[0].message.content or original_speech).strip()
