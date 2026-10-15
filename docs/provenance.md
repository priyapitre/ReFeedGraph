# Scope and source provenance

This repository was extracted from the local ReFeedGraph research workspace.
`source_manifest.json` records SHA-256 hashes of the source files used.

- `common.py` comes from `debate_feedback_common_finalMay21.py`, with package-relative
  guide resources and configurable local data/output paths.
- `stages/` comes from the six `step*_finalMay21.py` construction scripts. Imports
  were updated; the judge stage also accepts normalized JSON input.
- `method.py` contains selected diagnosis, retrieval, adaptation, and rewriting
  functions from `run_feedback_method_experiment.py`. Their function bodies and
  prompts are preserved. Baseline and scoring functions were not included.
- `cli.py`, packaging, tests, and synthetic examples provide a standalone interface.
- `data/splits/` comes from the main saved experiment splits, with the IQ2 test list
  restricted to IDs represented in completed main experiment files.

Excluded: legacy PROF directories, saved PROF feedback, comparative baselines,
experiment evaluations, intermediate results, dataset transcripts, trained graphs,
private paths, credentials, and the newer GPT-4.1/Claude rerun pipeline.

Source hashes establish extraction provenance, not a legal authorship audit.
The release documentation credits conceptual influences and judging frameworks.
