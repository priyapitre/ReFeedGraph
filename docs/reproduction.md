# Reproducing the method

This release supports rebuilding the feedback memory and generating feedback.
It does not claim complete numerical reproduction of the paper's tables. Baselines,
meta-judging, evaluation scores, ablations, and aggregation remain outside this
method-focused release.

1. Install the package and run the offline example/tests in the README.
2. Obtain the processed data layout documented in `data/README.md`, or supply your
   own normalized debate JSON.
3. Export the `train` partition from the selected `data/splits/*.json`.
4. Build the graph on those training debates only.
5. Export the `test` partition. Use its `losing_side` values as the generation target.
6. Run `generate` with the training graph, `--variant principles` or `specific`.
7. Optionally run `rewrite` with the same saved feedback and debate.

## Settings

The preserved generation defaults are temperature 0, eight retrieved principles,
two source examples per principle, at most eight issue labels, at most 80 principle
candidates, and at most 30 examples total. Chat defaults to `gpt-4o-mini` and
embedding retrieval to `text-embedding-3-small`. The CLI also supports lexical
retrieval. `configs/method.json` documents these defaults; it is a reference file,
not a configuration file consumed automatically by the CLI.

Model aliases can change remotely. Use an explicit available snapshot ID if you
need to pin a new run, and retain that run's settings. Temperature 0 does not
promise byte-identical hosted model outputs. Offline tests validate software flow,
not scientific performance or numerical equivalence to prior model calls.

## Graph stages and prompts

| Stage | Implementation and prompt location |
|---|---|
| Five framework judges | `stages/judge.py`; `common.judge_feedback_prompt`; `judge_guides/` |
| Sentence classification | `stages/classify.py`; `common.synthesis_prompt` |
| Deduplication | `stages/deduplicate.py`; `dedupe_prompt` |
| Principle abstraction | `stages/abstract.py`; `common.principles_from_deduped_feedback_prompt` |
| Per-debate graph | `stages/build.py` |
| Merge training graphs | `stages/combine.py` |
| Target diagnosis | `method.generate_generic_issues` |
| Retrieval | `method.retrieve_graph_context` |
| Feedback adaptation | `method.specify_our_feedback` |
| Optional rewrite | `method.revise_debate_speech` |

Paths in this table are inside `src/refeedgraph/`. Prompts remain beside the
functions that use them to preserve the research source and avoid competing copies.
No trained weights or PROF installation are needed. Full trained graphs are not
included because they preserve source debate excerpts; build them from your local
training data. The synthetic graph is not a replacement for those artifacts.
