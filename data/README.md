# Data

This release contains ID lists, not the source debate transcripts. Obtain the
PanelBench DebateArt/BP and Intelligence Squared data cited by the paper, subject
to their original terms. The local research project's processed layout is:

```text
data/raw/
├── DebateArt_data/
│   ├── motion/debateart_0017.yml
│   └── speech/debateart_0017.yml
├── BP_data/
│   ├── motion/bp_003.yml
│   └── speech/bp_003.yml
└── iq2_data/
    └── json/debate_001.json
```

The filenames above are examples. Preserve zero-padding in IDs. YAML motion files
contain `motion`, `pro_side`, `con_side`, and `speech_order`; speech files contain a
list of objects with `debater_name` and `content`. IQ2 JSON contains
`conversation_metadata` and `utterances`; each utterance uses
`utterance_metadata.speakertype` (`for` or `against`), `speaker`, and `text`.
Only the two debating sides are retained. BP export filters to OG and OO; the
provided split determines eligible debates.

```bash
export REFEEDGRAPH_DATA_ROOT=data/raw
refeedgraph export --dataset debateart --split data/splits/debateart.json \
  --partition train --output-dir outputs/debateart_train
refeedgraph build --inputs outputs/debateart_train/*.json \
  --model gpt-4o-mini --output-dir outputs/debateart_graph
```

The loader consumes the processed research format above; it is not a downloader
or a converter from every upstream dataset release. A public download/conversion
recipe for these exact processed data has not yet been validated. You can use your
own normalized debate JSON without these datasets; see `examples/debate.json`.

Split files preserve training order and the completed main-experiment test
population. The original IQ2 candidate split had 85 test entries; this release
contains the 70 IDs with saved main results, not the first 70 arbitrary entries.
No development set or new random split has been invented.
