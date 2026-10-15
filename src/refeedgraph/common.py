from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only without PyYAML
    yaml = None
try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - exercised only without openai
    OpenAI = None


REPO_ROOT = Path(os.getenv("REFEEDGRAPH_DATA_ROOT", "data")).resolve()
JUDGE_GUIDES_DIR = Path(__file__).resolve().parent / "judge_guides"
DEFAULT_OUTPUT_DIR = Path("outputs")

DATASETS = {
    "bp_competition": {
        "data_dir": REPO_ROOT / "BP_data",
        "motion": "motion/bp_{id}.yml",
        "speech": "speech/bp_{id}.yml",
        "default_id": "003",
    },
    "bp_og_oo": {
        "data_dir": REPO_ROOT / "BP_data",
        "motion": "motion/bp_{id}.yml",
        "speech": "speech/bp_{id}.yml",
        "default_id": "003",
        "include_debaters": ["OG", "OO"],
    },
    "debateart": {
        "data_dir": REPO_ROOT / "DebateArt_data",
        "motion": "motion/debateart_{id}.yml",
        "speech": "speech/debateart_{id}.yml",
        "default_id": "0017",
    },
    "iq2": {
        "data_dir": REPO_ROOT / "iq2_data",
        "default_id": "001",
        "loader": "iq2_json",
    },
}

PRINCIPLE_KEYS = [
    "instance_feedback",
    "source_specific_context",
    "source_argument_anchor",
    "abstract_principle",
    "failure_pattern",
    "trigger_pattern",
    "applies_when",
    "diagnostic_question",
    "action_template",
    "repair_steps",
    "example_application",
    "principle_type",
]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        preview = path.read_text(encoding="utf-8", errors="replace")[:1000]
        raise ValueError(f"Failed to parse JSON file {path}: {exc}\nPreview:\n{preview}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _clean_scalar(value: str) -> str:
    value = value.strip()
    if value in {"|", ">"}:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.replace("''", "'").replace('\\"', '"')


def _simple_yaml_load(text: str) -> Any:
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if nonempty and nonempty[0].lstrip().startswith("- "):
        return _simple_yaml_load_speech(lines)
    return _simple_yaml_load_motion(lines)


def _simple_yaml_load_motion(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith(" "):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            result[key] = _clean_scalar(raw_value)
            i += 1
            continue

        child_lines: list[str] = []
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" ")):
            child_lines.append(lines[i])
            i += 1

        stripped_children = [child.strip() for child in child_lines if child.strip()]
        if not stripped_children:
            result[key] = ""
        elif all(child.startswith("- name:") for child in stripped_children):
            result[key] = [
                {"name": _clean_scalar(child.split(":", 1)[1])}
                for child in stripped_children
            ]
        elif all(child.startswith("- ") for child in stripped_children):
            result[key] = [_clean_scalar(child[2:]) for child in stripped_children]
        else:
            result[key] = "\n".join(stripped_children)
    return result


def _simple_yaml_load_speech(lines: list[str]) -> list[dict[str, str]]:
    speeches: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("- debater_name:"):
            if current is not None:
                speeches.append(current)
            current = {"debater_name": _clean_scalar(stripped.split(":", 1)[1]), "content": ""}
            i += 1
            continue

        if current is not None and stripped.startswith("content:"):
            _, raw_value = stripped.split(":", 1)
            raw_value = raw_value.strip()
            if raw_value in {"|", ">"}:
                block: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].lstrip().startswith("- debater_name:"):
                    block.append(lines[i][4:] if lines[i].startswith("    ") else lines[i].lstrip())
                    i += 1
                current["content"] = "\n".join(block).strip()
                continue
            folded = [_clean_scalar(raw_value)]
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("- debater_name:"):
                if lines[i].strip():
                    folded.append(_clean_scalar(lines[i].strip()))
                i += 1
            current["content"] = "\n".join(folded).strip()
            continue
        i += 1

    if current is not None:
        speeches.append(current)
    return speeches


def load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if yaml is None:
        return _simple_yaml_load(path.read_text(encoding="utf-8"))
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:
        preview = path.read_text(encoding="utf-8", errors="replace")[:1000]
        raise ValueError(f"Failed to parse YAML file {path}: {exc}\nPreview:\n{preview}") from exc


def load_panelbench_debate(dataset: str, debate_id: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choices: {sorted(DATASETS)}")

    config = DATASETS[dataset]
    if config.get("loader") == "iq2_json":
        json_path = config["data_dir"] / "json" / f"debate_{debate_id}.json"
        data = load_json(json_path)
        metadata = data.get("conversation_metadata", {}) or {}
        speeches = []
        speech_order = []
        for utterance in data.get("utterances", []) or []:
            utterance_metadata = utterance.get("utterance_metadata", {}) or {}
            side = str(utterance_metadata.get("speakertype", "")).strip().lower()
            if side not in {"for", "against"}:
                continue
            debater_name = "For" if side == "for" else "Against"
            speaker = utterance.get("speaker", "Unknown speaker")
            text = str(utterance.get("text", "")).strip()
            if not text:
                continue
            speeches.append(
                {
                    "debater_name": debater_name,
                    "content": f"{speaker}: {text}",
                    "speaker": speaker,
                }
            )
            speech_order.append(debater_name)

        return {
            "dataset": dataset,
            "debate_id": debate_id,
            "motion": {
                "motion": metadata.get("title", ""),
                "info_slide": metadata.get("summary", "No information."),
                "pro_side": [{"name": "For"}],
                "con_side": [{"name": "Against"}],
                "speech_order": speech_order,
                "winner": metadata.get("winner"),
            },
            "speeches": speeches,
            "source_files": {
                "json": str(json_path),
            },
        }

    data_dir = config["data_dir"]
    motion_path = data_dir / config["motion"].format(id=debate_id)
    speech_path = data_dir / config["speech"].format(id=debate_id)
    motion = load_yaml(motion_path)
    speeches = load_yaml(speech_path)

    include_debaters = config.get("include_debaters")
    if include_debaters:
        include_set = set(include_debaters)
        motion = dict(motion)
        motion["pro_side"] = [
            team for team in motion.get("pro_side", []) if team.get("name") in include_set
        ]
        motion["con_side"] = [
            team for team in motion.get("con_side", []) if team.get("name") in include_set
        ]
        motion["speech_order"] = [
            name for name in motion.get("speech_order", []) if name in include_set
        ]
        speeches = [
            speech for speech in speeches or [] if speech.get("debater_name") in include_set
        ]

    return {
        "dataset": dataset,
        "debate_id": debate_id,
        "motion": motion,
        "speeches": speeches,
        "source_files": {
            "motion": str(motion_path),
            "speech": str(speech_path),
        },
    }


def load_judge_guides(judge_guides_dir: Path = JUDGE_GUIDES_DIR) -> list[dict[str, str]]:
    guide_paths = sorted(judge_guides_dir.glob("*.txt"))
    if not guide_paths:
        raise FileNotFoundError(f"No judge guide .txt files found in {judge_guides_dir}")

    return [
        {
            "guide_name": path.stem,
            "guide_file": str(path),
            "guide_text": path.read_text(encoding="utf-8"),
        }
        for path in guide_paths
    ]


def format_debate_for_prompt(debate: dict[str, Any]) -> str:
    motion = debate["motion"]
    speeches = debate["speeches"] or []
    pro_side = ", ".join(team["name"] for team in motion.get("pro_side", []))
    con_side = ", ".join(team["name"] for team in motion.get("con_side", []))
    speech_order = ", ".join(motion.get("speech_order", []))

    parts = [
        f"Dataset: {debate['dataset']}",
        f"Debate ID: {debate['debate_id']}",
        f"Motion: {motion.get('motion', '')}",
        f"Info slide: {motion.get('info_slide', 'No information.')}",
        f"Pro side teams: {pro_side}",
        f"Con side teams: {con_side}",
        f"Speech order: {speech_order}",
        "",
        "Debate speeches:",
    ]

    for i, speech in enumerate(speeches, start=1):
        debater = speech.get("debater_name", f"speech_{i}")
        content = (speech.get("content") or "").strip()
        parts.append(f"\n--- Speech {i}: {debater} ---\n{content}")

    return "\n".join(parts)


def make_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' Python package is not installed in this Python environment. "
            "Install it with: python3 -m pip install openai"
        )
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPINFRA_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY, DEEPINFRA_API_KEY, or GROQ_API_KEY before running.")

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPINFRA_BASE_URL") or os.getenv("GROQ_BASE_URL")
    if not base_url and os.getenv("GROQ_API_KEY"):
        base_url = "https://api.groq.com/openai/v1"
    if not base_url and os.getenv("DEEPINFRA_API_KEY"):
        base_url = "https://api.deepinfra.com/v1/openai"
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    first_arr = text.find("[")
    last_arr = text.rfind("]")

    candidates = []
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(text[first_obj : last_obj + 1])
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(text[first_arr : last_arr + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from model response:\n{text[:1000]}")


def _response_preview(text: str, limit: int = 1000) -> str:
    compact = text.replace("\r", " ").replace("\n", " ")
    return compact[:limit]


def _is_json_validation_error(exc: Exception) -> bool:
    text = str(exc)
    return "json_validate_failed" in text or "Failed to validate JSON" in text


def _repair_json_response(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
    max_tokens: int,
) -> Any:
    repair_system_prompt = "You repair malformed JSON model responses. Return only valid JSON and preserve the original schema and meaning."
    repair_user_prompt = f"""
The model response below is not valid JSON or could not be parsed.

Original system prompt:
{system_prompt}

Original user prompt:
{user_prompt}

Model response to repair:
{raw_response}

Return only valid JSON. Do not add markdown or commentary.
""".strip()

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": repair_system_prompt},
            {"role": "user", "content": repair_user_prompt},
        ],
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
    }
    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if not _is_json_validation_error(exc):
            raise
        print("JSON-mode repair failed validation; retrying repair without JSON mode.", flush=True)
        response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    return extract_json(content)


def call_json_llm(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    retries: int = 3,
) -> Any:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            }
            try:
                response = client.chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                if not _is_json_validation_error(exc):
                    raise
                print(
                    f"JSON-mode LLM call failed validation on attempt {attempt}/{retries}; retrying without JSON mode.",
                    flush=True,
                )
                response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            try:
                return extract_json(content)
            except Exception as parse_error:
                print(
                    f"LLM response was not valid JSON on attempt {attempt}/{retries}; trying repair. Preview: {_response_preview(content)}",
                    flush=True,
                )
                try:
                    return _repair_json_response(
                        client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        raw_response=content,
                        max_tokens=max_tokens,
                    )
                except Exception as repair_error:
                    last_error = repair_error
                    print(f"JSON repair failed on attempt {attempt}/{retries}: {repair_error}", flush=True)
                    if attempt < retries:
                        time.sleep(3 * attempt)
                    continue
        except Exception as exc:
            last_error = exc
            print(f"LLM call failed on attempt {attempt}/{retries}: {exc}", flush=True)
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def judge_feedback_prompt(guide: dict[str, str], debate_text: str) -> tuple[str, str]:
    system_prompt = f"""
You are a debate judge. Use the judge guide below as your judging philosophy.
Only evaluate the debate material provided by the user. Do not invent facts.

JUDGE GUIDE:
{guide["guide_text"]}
""".strip()

    user_prompt = f"""
Evaluate the following debate using your assigned guide.

Return ONLY valid JSON with this shape:
{{
  "winner": "the side/team/debater you think won",
  "loser": "the side/team/debater you think lost",
  "decision": "detailed explanation of why the winner beat the loser under this judge guide",
  "feedback_to_winner": {{
    "use_the_judge_guides_recommended_section_headings": "give feedback to the winner using this guide's own output format"
  }},
  "feedback_to_loser": {{
    "use_the_judge_guides_recommended_section_headings": "give feedback to the loser using this guide's own output format"
  }}
}}

Requirements:
- Read the entire debate before judging.
- Follow the judge guide's own values, priorities, and recommended output format.
- Use the section headings from the guide's "Recommended Output Format" when one is present.
- If the guide does not contain a Recommended Output Format, use the guide's own explanation criteria as section headings.
- Do not make all judges sound the same: an NSDA judge should emphasize contentions/evidence/clash/weighing, a Toulmin judge should emphasize claim/grounds/warrant/backing/qualifier/rebuttal, a Rogerian judge should emphasize fairness/common ground/constructive resolution, a Walton judge should emphasize schemes/critical questions, and a UADC judge should emphasize burdens, clashes, persuasiveness, responses, and weighing as an ordinary intelligent voter.
- The top-level JSON keys are only a container. The actual feedback_to_winner and feedback_to_loser objects must be guide-specific, not generic.
- Compare the central arguments made by both sides through that judge guide's framework, not a generic debate rubric.
- Discuss specific arguments, clashes, concessions, missing responses, or weighing failures that matter under that guide.
- Refer to concrete debate material, such as a speaker/team's claim, example, rebuttal, or framing.
- Explain why the winning side's arguments were stronger than the losing side's best arguments according to this guide.
- Give separate feedback to the winner and loser. Do not merge them into one overall evaluation.
- Make both feedback sections useful for future improvement, not just a verdict.
- Do not give one-line feedback.
- The decision explanation should usually be 250 to 500 words.
- Each major subsection inside feedback_to_winner and feedback_to_loser should be 80 to 180 words.
- Every major subsection should mention at least one specific argument, example, quote fragment, rebuttal, or clash from the debate.
- The decision should explicitly compare at least three concrete argument-level clashes from the debate.
- The feedback_to_winner section should explain what the winner did well and what they should improve, using specific arguments from their speeches.
- The feedback_to_loser section should explain why the loser lost and what they should improve, using specific arguments from their speeches and the winner's responses.
- Avoid generic phrases like "be clearer", "add more evidence", or "improve weighing" unless you immediately explain which exact argument needed that improvement and how.
- If you cannot point to a specific argument in the debate, do not include that feedback.

DEBATE:
{debate_text}
""".strip()
    return system_prompt, user_prompt


def synthesis_prompt(debate_text: str, judge_feedback: list[dict[str, Any]]) -> tuple[str, str]:
    system_prompt = """
You classify judge feedback sentences.
The input feedback is detailed and guide-specific. Do not synthesize, merge, rewrite,
abstract, or improve it yet. Preserve the judge feedback at sentence level and simply
classify each substantive sentence as summary, weakness, or actionable_feedback.
Return only valid JSON.
""".strip()

    user_prompt = f"""
Given the debate and guide-specific judge feedback from step 1, go judge by judge.
Take every substantive sentence present in each judge's feedback, including sentences
inside sections like Contentions, Evidence, Clash, Warrant, Common ground, Critical
questions, Weighing, feedback_to_winner, feedback_to_loser, and decision.

Simply divide/classify those sentences into three categories:

1. summary_of_points_made_by_debaters:
   Neutral descriptions of points, arguments, rebuttals, examples, or clashes made by
   the debaters. These should summarize what a debater argued, not judge whether it
   was good or bad.

2. weaknesses:
   Evaluative statements about flaws, missing reasoning, weak evidence, poor clash,
   unanswered objections, tone problems, dropped arguments, bad warrants, or other
   failures identified by the judges.

3. actionable_feedback:
   Concrete advice about what a side/debater should do differently next time.

This step is intentionally mechanical. Do not merge duplicate ideas. Do not create
new advice. Do not make generalized principles. Do not compress several sentences
into one item.

Return ONLY valid JSON with this shape:
{{
  "per_judge_sentence_classification": [
    {{
      "judge_name": "judge_name",
      "winner": "winner chosen by this judge",
      "loser": "loser chosen by this judge",
      "classified_sentences": [
        {{
          "section": "where the sentence came from, e.g. decision, feedback_to_loser.Weighing and Clarity",
          "target": "winner|loser|both|overall|unknown",
          "sentence": "one substantive sentence copied or minimally cleaned from the judge feedback",
          "category": "summary|weakness|actionable_feedback",
          "reason_for_category": "brief reason this sentence belongs in that category",
          "debate_argument_anchor": "specific argument/clash mentioned by the sentence; infer the named argument, section, claim, source, definition, or opponent clash when the sentence clearly points to one"
        }}
      ],
      "summary_of_points_made_by_debaters": [
        {{
          "section": "where the sentence came from",
          "target": "winner|loser|both|overall|unknown",
          "sentence": "same sentence classified as summary",
          "debate_argument_anchor": "specific argument/clash mentioned by the sentence; infer the named argument, section, claim, source, definition, or opponent clash when clear"
        }}
      ],
      "weaknesses": [
        {{
          "section": "where the sentence came from",
          "target": "winner|loser|both|overall|unknown",
          "sentence": "same sentence classified as weakness",
          "debate_argument_anchor": "specific argument/clash mentioned by the sentence; infer the named argument, section, claim, source, definition, or opponent clash when clear"
        }}
      ],
      "actionable_feedback": [
        {{
          "section": "where the sentence came from",
          "target": "winner|loser|both|overall|unknown",
          "sentence": "same sentence classified as actionable_feedback",
          "debate_argument_anchor": "specific argument/clash mentioned by the sentence; infer the named argument, section, claim, source, definition, or opponent clash when clear"
        }}
      ]
    }}
  ]
}}

Rules:
- Process every judge separately.
- Preserve section provenance. If the sentence came from feedback_to_loser.Weighing and Clarity, say that.
- Classify every substantive feedback sentence. Skip only empty fragments, section titles, or pure metadata like winner/loser labels.
- A sentence can be lightly cleaned for readability, but do not rewrite its meaning.
- Preserve specificity. If the judge sentence names a debater, contention, example,
  source, definition, burden, clash, dropped argument, or response, keep that detail
  in the sentence and in debate_argument_anchor.
- Do not set debate_argument_anchor to null merely because the judge did not use a
  section title. Infer a concise anchor from the sentence when it points to a concrete
  argument or clash.
- Use exactly one category per sentence:
  - summary: describes what a debater argued or what happened in a clash.
  - weakness: evaluates something as flawed, missing, weak, unclear, unsupported, dismissive, unresponsive, poorly weighed, etc.
  - actionable_feedback: tells the debater what to do next time, often signaled by "should", "need to", "focus on", "ensure that", "consider", "in future debates", "could improve by".
- If a single original sentence contains both weakness and advice, split it into two sentence items only when the split is natural; otherwise choose the dominant category and preserve the full sentence.
- The category arrays must be exact filtered copies of classified_sentences. Do not add items to category arrays that are not in classified_sentences.
- Treat judge language like "in future debates", "ensure that", "focus on", "you should", "you need to", "consider", or "could improve by" as actionable feedback when it tells the debater what to do differently.

DEBATE:
{debate_text}

JUDGE_FEEDBACK_JSON:
{json.dumps(judge_feedback, indent=2)}
""".strip()
    return system_prompt, user_prompt


def abstraction_prompt(synthesis: dict[str, Any]) -> tuple[str, str]:
    system_prompt = """
You turn concrete debate feedback into reusable debate-improvement principles.
For each item, explain why it matters and generalize it to other contexts.
Return only valid JSON.
""".strip()

    example = {
        "instance_feedback": "The debater claims B but does not explain why B is stronger than C.",
        "abstract_principle": (
            "Debaters should compare their chosen answer against the strongest plausible "
            "alternative when alternatives remain unresolved."
        ),
        "failure_pattern": "Unresolved alternative / missing comparative justification",
        "applies_when": (
            "The debate contains multiple plausible answers, interpretations, policies, "
            "or explanations."
        ),
        "diagnostic_question": (
            "Did the debater explain why their preferred position is stronger than the "
            "best competing position?"
        ),
        "action_template": (
            "Compare your answer directly against the strongest alternative and explain "
            "the deciding difference."
        ),
        "principle_type": "comparative weighing",
    }

    user_prompt = f"""
Abstract the synthesized debate feedback into transferable principles.

Return ONLY valid JSON with this shape:
{{
  "principles": [
    {json.dumps(example, indent=6)}
  ]
}}

Rules:
- Create one principle for each major actionable feedback item.
- Keep instance_feedback concrete and tied to this debate.
- Keep abstract_principle reusable beyond this debate.
- Use exactly these keys for every principle: {PRINCIPLE_KEYS}

SYNTHESIZED_FEEDBACK_JSON:
{json.dumps(synthesis, indent=2)}
""".strip()
    return system_prompt, user_prompt


def principles_from_deduped_feedback_prompt(deduped_feedback: dict[str, Any]) -> tuple[str, str]:
    system_prompt = """
You convert concrete debate feedback into reusable debate-improvement principles.
Each weakness or actionable-feedback item should become one transferable, operational principle.
The principles must be general enough to apply to future debates, but specific enough
to tell a debater exactly what to look for and how to repair it.
Do not use summary-only items.
Return only valid JSON.
""".strip()

    example = {
        "instance_feedback": (
            "The debater asserts that their policy prevents harm, but never compares that "
            "mechanism against the opponent's alternative safety mechanism."
        ),
        "source_specific_context": (
            "In the source debate, the judge was reacting to a speech that repeated its "
            "own harm-prevention mechanism while leaving the opponent's named safety "
            "mechanism mostly unanswered. The concrete failure was not simply 'weak "
            "weighing'; it was the absence of a paragraph that put the two mechanisms "
            "side by side and told the judge which one controlled the round."
        ),
        "source_argument_anchor": "Own harm-prevention mechanism versus opponent's alternative safety mechanism",
        "abstract_principle": (
            "When a debate has competing mechanisms, the debater must identify the exact "
            "point of comparison and explain why their mechanism is more likely, faster, "
            "larger in impact, or less vulnerable to the opponent's objection."
        ),
        "failure_pattern": "Unresolved alternative / missing comparative justification",
        "trigger_pattern": (
            "Use this when the speech gives a standalone reason for its own side but leaves "
            "the opponent's named alternative, counter-mechanism, or competing explanation "
            "unanswered or only dismisses it in passing."
        ),
        "applies_when": (
            "The debate contains two or more plausible mechanisms, definitions, causal "
            "stories, interpretations, policies, or explanations that cannot both decide "
            "the round unless one is comparatively better."
        ),
        "diagnostic_question": (
            "Can you point to the sentence where the debater names the opponent's best "
            "alternative and gives a concrete reason their own mechanism beats it?"
        ),
        "action_template": (
            "Add a direct comparison paragraph: name the opponent's alternative, state the "
            "shared criterion for deciding between the two positions, then explain the "
            "deciding difference using debate-specific facts."
        ),
        "repair_steps": (
            "1. Quote or paraphrase the opponent's strongest alternative in neutral terms. "
            "2. State the comparison axis, such as likelihood, timeframe, magnitude, "
            "scope, reversibility, burden, or evidentiary reliability. 3. Explain why your "
            "argument wins on that axis using material already in the debate or one "
            "specific piece of evidence. 4. Close by saying how this comparison changes "
            "the judge's decision."
        ),
        "example_application": (
            "If the opponent says public education solves misinformation while you say "
            "platform regulation solves it, do not merely repeat that regulation helps. "
            "Write: 'Education may improve individual judgment over time, but regulation "
            "acts at the distribution point where the false claim spreads; because the harm "
            "occurs before most users verify it, the faster intervention should weigh more.'"
        ),
        "principle_type": "comparative weighing",
    }

    source_items = {
        "weakness": deduped_feedback.get("weakness", []),
        "actionable_feedback": deduped_feedback.get("actionable_feedback", []),
    }

    user_prompt = f"""
Convert every item in weakness and actionable_feedback into a reusable principle.
Ignore summary items.

Return ONLY valid JSON with this shape:
{{
  "principles": [
    {json.dumps(example, indent=6)}
  ]
}}

Rules:
- Create one principle for every input item in "weakness" and "actionable_feedback".
- Preserve the source specificity before generalizing. The output should read like a
  reusable memory card, not a slogan.
- instance_feedback must be concrete and tied to the original debate. Use the item's
  canonical_sentence, merged_from sentences, section labels, target, and debate_argument_anchor
  when present. It should usually be 1 to 3 sentences, not a short phrase.
- source_specific_context must explain the original debate situation in enough detail
  that a future prompt can learn from it. Include named debaters, argument labels,
  source/evidence types, opponent claims, judging section, or clash names when present.
  This field should usually be 2 to 5 sentences.
- source_argument_anchor should be the most specific reusable anchor available, such as
  "animal-study evidence applied to human MSG harms", "drafterman's premise/axiom
  argument", "FDA corruption claim without substantiation", or "dismissive tone in
  opponent engagement". Use null only if no anchor can be inferred.
- Generalize beyond this debate, but do not become vague. Bad: "use more evidence."
  Good: "when the evidence is about a proxy population, explain why the proxy transfers
  to the debated population and name the limit of that transfer."
- abstract_principle should be a full conditional rule: "When X appears, do Y because Z."
  It should usually be 2 to 4 sentences. It must retain the specific failure type,
  such as proxy evidence, unsupported institutional corruption claims, dropped premise
  objections, undefined key terms, missing impact comparison, or hostile tone replacing clash.
- failure_pattern should be a short label for the recurring mistake.
- trigger_pattern should describe observable signals in a new debate that should cause
  this principle to be retrieved. Name concrete signals: "uses animal studies for a
  human-health motion", "attacks source institution without warrant", "defines a key
  term only by denying opponent definition", "answers a contention with ridicule".
- applies_when should describe the class of future debate contexts where the principle is relevant.
- diagnostic_question should be concrete enough that a debater can answer yes/no by
  pointing to a sentence, paragraph, source, clash, or comparison in their speech.
- action_template should be a concise reusable instruction.
- repair_steps should be a long, operational recipe for how to apply the principle.
  Include 4 to 6 steps. Mention what to name, compare, explain, add, remove, or rewrite.
  The steps should be detailed enough that a feedback generator can turn them into a
  paragraph-level rewrite instruction for a new debate.
- example_application should show how the principle would sound in a different debate.
  Keep it hypothetical and reusable; do not invent facts about the source debate.
- principle_type should be a short category such as "comparative weighing",
  "warranting", "evidence use", "direct refutation", "tone/professionalism",
  "definition framing", "argument structure", or another concise type.
- Use exactly these keys for every principle: {PRINCIPLE_KEYS}
- Do not create principles from summary-only material.
- Do not invent new debate facts.

DEDUPED_WEAKNESS_AND_ACTIONABLE_FEEDBACK_JSON:
{json.dumps(source_items, indent=2, ensure_ascii=False)}
""".strip()
    return system_prompt, user_prompt


def normalize_principles(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        principles = raw.get("principles", [])
    elif isinstance(raw, list):
        principles = raw
    else:
        raise ValueError("Principle abstraction must be a JSON object or list.")

    normalized = []
    for principle in principles:
        if not isinstance(principle, dict):
            continue
        normalized.append({key: str(principle.get(key, "")).strip() for key in PRINCIPLE_KEYS})
    return normalized


def build_graph(
    debate: dict[str, Any],
    judge_feedback: list[dict[str, Any]],
    synthesis: dict[str, Any],
    principles: list[dict[str, str]],
) -> dict[str, Any]:
    debate_node_id = f"debate:{debate['dataset']}:{debate['debate_id']}"
    synthesis_node_id = "synthesis:feedback"

    nodes = [
        {
            "id": debate_node_id,
            "type": "debate",
            "dataset": debate["dataset"],
            "debate_id": debate["debate_id"],
            "motion": debate["motion"].get("motion", ""),
            "source_files": debate["source_files"],
        },
        {
            "id": synthesis_node_id,
            "type": "synthesis",
            "data": synthesis,
        },
    ]
    edges = []

    for judge in judge_feedback:
        judge_id = f"judge:{judge.get('judge_name', 'unknown')}"
        nodes.append(
            {
                "id": judge_id,
                "type": "judge_feedback",
                "judge_name": judge.get("judge_name"),
                "guide_file": judge.get("guide_file"),
                "feedback": judge.get("feedback", judge),
            }
        )
        edges.append({"source": debate_node_id, "target": judge_id, "relation": "evaluated_by"})
        edges.append({"source": judge_id, "target": synthesis_node_id, "relation": "synthesized_into"})

    for i, principle in enumerate(principles, start=1):
        principle_id = f"principle:{i:03d}"
        nodes.append({"id": principle_id, "type": "abstract_principle", **principle})
        edges.append({"source": synthesis_node_id, "target": principle_id, "relation": "abstracted_into"})

    return {
        "metadata": {
            "dataset": debate["dataset"],
            "debate_id": debate["debate_id"],
            "judge_count": len(judge_feedback),
            "principle_count": len(principles),
        },
        "nodes": nodes,
        "edges": edges,
    }
