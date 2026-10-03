# Zero-shot EUNIS text-ranking pilot

This pilot checks whether a pinned multilingual sentence-embedding model ranks
the existing polygon-level EUNIS code among candidate habitat definitions for
100 Description sentences. It runs only after the chosen sample and candidate
text are frozen. It does not train or fine-tune a model, relabel the source, or
recompute EUNIS geometry.

## Inputs

| Input | Pinned value |
| --- | --- |
| Overlap dataset | [`NoeFlandre/georeset-text-label-benchmark`](https://huggingface.co/datasets/NoeFlandre/georeset-text-label-benchmark) |
| Dataset revision | `2f7e2436e583e9c3b3c0cd0dca23b1d190e8fbf9` |
| Selected file | `overlap.parquet`, SHA-256 `a01168f14f519fab57b28b91c2ccd2b92a5944641465f1e36c24d68fc484721c` |
| Pre-existing overlap rows | 224,789 |
| Observed EUNIS codes | 158 |
| Candidate table | `pilot_data/eunis_candidate_labels.csv`, SHA-256 `2e6cc694da7baa8edf9debeed08a0adda719518be74e659a0533b3ae9e56afa8` |
| Model | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) |
| Model revision | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Model license | MIT |

The overlap dataset contains positive Description sentence / polygon EUNIS
overlap records. It is not the entire relevance-label table and has no negative
or missing-overlap examples. At the pinned revision it has 137,464 unique
sentence hashes and 200,941 unique polygon keys. The pilot samples from this
already-computed overlap only.

Candidate texts are the exact EEA English EUNIS name, a newline, and the exact
EEA English description. The 158 candidate names agree with the overlap names;
all candidate descriptions are populated. The CSV records the applicable EEA
release, factsheet URL, archive SHA-256, and CC-BY-4.0 attribution per code. See
[provenance](provenance.md) for the EEA sources and OpenStreetMap data terms.
The model weights and tokenizer are MIT-licensed at the pinned model revision.

## Frozen sampling protocol

Each overlap occurrence receives a stable ID from its source PBF, OSM type and
ID, Description identity, tag key, zero-based sentence index, and sentence
SHA-256. The sampler sorts these IDs by `SHA256("42:" + occurrence_id)` and
walks that order, retaining the first 100 records with both an unseen exact
sentence hash and an unseen `(source_pbf, osm_type, osm_id)` polygon key. This
deterministically avoids counting duplicated sentence text or polygons more
than once in this small pilot. The selected rows, IDs checksum, input hashes,
coverage counts, and exact candidate CSV are written before model loading.

The frozen sample is reproducible from the pinned inputs and seed. Its manifest
records the computation commit and validation commit so the published result
can be traced to the code that was checked and run.

## Embedding and ranking method

The reference implementation uses the model card's attention-mask-aware mean
pooling and L2 normalization. Input sentences use the E5 retrieval prefix
`query: `; candidate definitions use `passage: `. Tokenization truncates at 512
tokens and inference uses CPU batches of 16. Each sentence is compared against
all 158 candidate vectors with cosine similarity. The top five are retained;
ties sort by EUNIS code. No gold code or name is included in the sentence input.

The use of query/passage prefixes treats sentence-to-definition matching as a
retrieval task. The candidates are English and source sentences can be in other
languages, so per-language results should be read as descriptive pilot results.

## Outputs and metrics

The run writes:

* `frozen_sample.json`: sample IDs, exact source rows, input checksums, and
  coverage counts;
* `candidate_labels.csv`: the frozen classification text and row-level source
  citations;
* `predictions.parquet`: one row per sampled sentence, the gold polygon code,
  top-1 code and cosine score, and top-five codes, names, and scores;
* `metrics.json`: overall metrics, per-code support and hit rates, per-language
  top-1/top-5 accuracy, and the most frequent top-1 confusions;
* `manifest.json`: source/model revisions, model-file hashes, run settings,
  code commits, runtime, timings, output SHA-256 values, and limitations.

Overall top-1 and top-5 accuracy are the fraction of rows whose existing
polygon-level EUNIS code appears at rank one or among the first five. Macro-F1
is the unweighted mean over all 158 candidates; a class with no pilot support
and no predictions contributes zero. The report includes every candidate, so
the denominator does not shrink to the classes that happen to occur in the
sample. Per-language rows use source `language_code`, with null values grouped
as `missing`.

## Reproduce

Install the optional inference dependencies and download only the pinned
overlap Parquet file:

```bash
uv sync --locked --all-groups --extra pilot
hf download NoeFlandre/georeset-text-label-benchmark overlap.parquet \
  --repo-type dataset \
  --revision 2f7e2436e583e9c3b3c0cd0dca23b1d190e8fbf9 \
  --local-dir source
```

Then freeze the rows and run the one-shot CPU inference:

```bash
COMMIT_SHA=$(git rev-parse HEAD)
uv run georeset-pilot freeze \
  --source-parquet source/overlap.parquet \
  --output-dir artifacts/e5-small-100-seed42 \
  --sample-size 100 --seed 42
uv run georeset-pilot run \
  --run-dir artifacts/e5-small-100-seed42 \
  --model-cache .cache/model-e5-small \
  --computation-commit "$COMMIT_SHA" \
  --validation-commit "$COMMIT_SHA"
```

On first run the official Hub client fetches the five pinned model and
tokenizer files (about 471 MB) into the specified local cache. Re-running into
an existing output directory is rejected; use a new directory for a separately
identified run.

## Interpretation limit

The gold code is the existing EUNIS assignment on the polygon that contains or
overlaps the sentence's OSM feature. It is polygon context, not a sentence-level
habitat judgment. Scientific validation of the EUNIS reference assignments is
unconfirmed. Results only describe this 100-row sample from positive overlap;
they do not estimate performance on negatives, missing labels, or a held-out
population and should not be presented as a validated benchmark score.
