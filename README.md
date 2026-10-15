# ReFeedGraph

Implementation of **ReFeedGraph: Graph-Grounded Reusable Feedback for Debate Improvement**
by Priya Pitre, Omkar Joshi, Timon Ziegenbein, and Henning Wachsmuth.

ReFeedGraph turns prior debate critiques into reusable feedback principles. Five
framework-specific judges produce critiques; classification, deduplication, and
abstraction convert them into a graph. For a new debate, the method diagnoses the
target side's weaknesses, retrieves relevant principles, and adapts them into
concrete feedback. The `specific` variant also retrieves linked source examples.

```text
Training debates → Five judges → Classify → Deduplicate → Abstract → Graph
New debate → Diagnose target side → Retrieve graph evidence → Generate feedback
```

This is a **method-focused release**. It includes graph construction, lexical and
embedding retrieval, both feedback variants, and optional speech rewriting. It does
not include comparative baselines, scoring experiments, or scripts to reproduce all
paper tables. PROF is not a software dependency. See [scope and provenance](docs/provenance.md).

## Installation

Use Python 3.10 or newer. From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The versions used in the clean installation test are recorded in
`constraints-tested.txt`; use `python -m pip install -c constraints-tested.txt -e .`
to request those dependency versions.

On Windows, activate with `.venv\Scripts\activate` instead. Dependencies are OpenAI's
Python client and PyYAML. No local model weights or GPU are required.

## Quick start: offline retrieval

The included examples are synthetic, hand-authored fixtures, not paper results.
This command makes no API calls:

```bash
refeedgraph retrieve \
  --graph examples/graph.json \
  --issues examples/issues.json \
  --retrieval lexical \
  --output outputs/retrieved.json
```

The output contains a relevant principle and its linked supporting feedback.
Compare it with [expected retrieval](examples/expected_retrieval.json).

## Generate feedback

Set your API key in your shell. `.env.example` documents optional settings;
`.env` is not loaded automatically. Live commands make billable model API calls.

```bash
export OPENAI_API_KEY='your-key'
refeedgraph generate \
  --debate examples/debate.json --target Pro \
  --graph examples/graph.json --variant specific \
  --retrieval embedding --model gpt-4o-mini \
  --output outputs/feedback.json
```

Use `--variant principles` to omit source examples from feedback generation.
Use `--retrieval lexical` to avoid embedding API calls; diagnosis and feedback
still call the model. The supplied graph is only a tiny demonstration. Build a
training graph for substantive use.

`--target` is the exact `debater_name` to receive feedback. The caller selects the
target side; the command does not infer the dataset's gold loser automatically.
The output records feedback, diagnosis, retrieved evidence, and model settings.

For another OpenAI-compatible chat provider, set `OPENAI_BASE_URL` and use that
provider's key and model ID. Embeddings use `EMBEDDING_API_KEY` (falling back to
`OPENAI_API_KEY`) and `EMBEDDING_BASE_URL` (default: OpenAI). Set a separate embedding
key when the chat key belongs to another provider. Provider compatibility varies;
these interfaces have been tested offline, not against every hosted model.

## Build a training graph

Supply one or more normalized debate JSON files in the same format as
[the example](examples/debate.json). Use a new output directory for each run:

```bash
refeedgraph build \
  --inputs examples/debate.json \
  --model gpt-4o-mini \
  --output-dir outputs/training
```

This runs all six stages, saves intermediate JSON, and writes
`outputs/training/graph.json`. Use that graph with `generate --graph`.
Training and test debates should be separate. This one-file example illustrates
execution only; it is not a meaningful evaluation.

To exercise the construction pipeline without API calls:

```bash
refeedgraph build --inputs examples/debate.json \
  --mock --output-dir outputs/mock-training
```

Mock mode uses placeholder model responses and is for checking file flow only.
Individual stages are available as `python -m refeedgraph.stages.judge --help`
(and `classify`, `deduplicate`, `abstract`, `build`, `combine`).

## Use the paper's datasets

The exact main-experiment ID lists are in `data/splits/`:

| Dataset | Training | Reported test cases |
|---|---:|---:|
| DebateArt | 20 | 74 |
| IQ2 | 20 | 70 |
| BP Opening Government / Opening Opposition | 7 | 7 |

See [data setup](data/README.md) and [method reproduction](docs/reproduction.md).
Dataset transcripts and trained graphs containing source excerpts are not bundled.

## Optional speech revision

```bash
refeedgraph rewrite \
  --debate examples/debate.json --feedback outputs/feedback.json \
  --model gpt-4o-mini --output outputs/rewrite.json
```

This applies feedback to the target side. It does not score the revision or
reproduce the paper's evaluation. Commands refuse to overwrite their output files.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover graph evidence retrieval, graph validation, both feedback variants with
stubbed model responses, and the complete mock construction path. They need no keys.

## Citation and attribution

Use [CITATION.cff](CITATION.cff) for the paper title and authors. Publication URL,
venue, and DOI will be added when confirmed. PROF (Nair et al., 2024) motivates the
revision-based view of feedback; this method does not install or run PROF.
See [third-party notices](THIRD_PARTY_NOTICES.md) for source acknowledgments.

## License

The code is available under the [MIT License](LICENSE). Dataset and third-party
material terms remain separate; see `THIRD_PARTY_NOTICES.md`.
