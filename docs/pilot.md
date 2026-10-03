# 100-sentence EUNIS label pilots

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
or missing-overlap examples. The 158 candidates cover the frozen natural-habitat
vocabulary; built and intensive-cropland classes are absent. At the pinned revision it has 137,464 unique
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
SHA-256. Sampling ranks occurrence rows by `SHA256("42:" + occurrence_id)`,
then keeps the first 100 with an unseen exact sentence hash and unseen
`(source_pbf, osm_type, osm_id)` polygon key. This means the input is
occurrence-weighted before the uniqueness filters. It deterministically avoids
counting duplicated sentence text or polygons more than once in this small
pilot. The selected rows, IDs checksum, input hashes, coverage counts, and exact
candidate CSV are written before model loading.

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

## Direct labels with LFM2.5 + DSpark

The second adapter asks [`LiquidAI/LFM2.5-2.6B`](https://huggingface.co/LiquidAI/LFM2.5-2.6B)
to choose one label directly from the same 158 EUNIS candidates, including each
candidate's pinned EEA name and definition. It reads only the sentence text and
candidate vocabulary; it does not pass the gold code/name, score embeddings, or
shortlist candidates. The command accepts only the published 100-row ID digest
(`74ab5826b51806947215b0e1635f173ce99af13577e41a431c263cd6a8e57e72`) and the
pinned candidate CSV. It never calls `freeze` or the sampler.

The target is pinned to revision
`654f9463ce32b05d0429d76fe1f580b27d4c1ac0`; its paired
[`LFM2.5-2.6B-DSpark` draft](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark)
is pinned to `458cedab07d0f7b2b05700c77e1aa463d43d6f04`. The validated
runtime is SGLang `0.5.20` with FlashInfer `0.6.18`, BF16, `DSPARK`, greedy
temperature `0`, one request at a time, and at most `4096` generated tokens.
This uses the [upstream LFM2 DSpark support](https://github.com/sgl-project/sglang/pull/31041)
and the launch settings in the [official draft card](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark).
DSpark is speculative decoding with a paired draft checkpoint; it is not a
hardware device. Under greedy decoding the target verifies draft tokens, so
DSpark accelerates the target model without changing its output.

LiquidAI documents a `131,072`-token model context. The SGLang path used by this
project has a `128,000`-token effective context; the adapter checks each rendered
prompt plus the full `4096`-token generation cap before it loads the serving
engine. The adapter uses the pinned tokenizer's chat template with
`enable_thinking=False`; that template still opens a `<think>` region. Parsing
uses only text after the final `</think>` and accepts an exact code from the
candidate set. Truncated answers, unclosed reasoning, unknown codes, and other
formats remain rows with their raw output and a parse failure reason.

The adapter writes three files into a separate LLM run directory beside the
frozen E5 directory: `dspark_predictions.parquet`, `dspark_metrics.json`, and
`dspark_manifest.json`. Existing E5 files are not overwritten. Each prediction
retains the raw generation, parsed code/name or null, parse status/error, prompt
hash, token counts, finish reason, timing, and DSpark accepted/proposed token
counts. Overall top-1 accuracy and macro-F1 use all 100 rows; invalid output
counts as an incorrect prediction and is included in a separate coverage rate.
The macro-F1 denominator remains all 158 candidates. Direct generation has one
label per sentence and no top-5 ranking; compare only top-1 accuracy and
macro-F1 with the E5 output.

### Run requirements

Use Linux with an NVIDIA CUDA GPU visible through `nvidia-smi`. The existing
validated serving gate admits GPUs with compute capability `8.0` or newer and
at least `16,384 MiB` of VRAM; this is an admission floor, not a guarantee that
every runtime allocation fits. LiquidAI's published SGLang throughput test used
one BF16 H100 with 80 GB. The DSpark extra resolves the CUDA 13 runtime and
FlashInfer pins, and the command checks that the Hub cache has at least 8 GiB
free before loading the tokenizer or weights. Do not run this adapter on the
CPU-only workspace used for unit tests.

First obtain `frozen_sample.json` and `candidate_labels.csv` from the authorized,
versioned run at
`pilot/runs/e5-small-100-seed42/`. The run must come from the published dataset
revision whose exact file hashes match the manifest. Install the runtime in a
dedicated environment:

```bash
uv sync --locked --all-groups --extra dspark
```

Then run the adapter against that frozen directory. By default it writes to
the sibling `lfm2.5-2.6b-dspark-100-seed42` directory; pass `--output-dir` to
choose a different new path. `HF_HOME` may be set to a large local cache path before the command; otherwise `--model-cache` is used.
SGLang downloads the target and draft revisions on the first actual run.

```bash
PILOT_REVISION=<commit-containing-the-frozen-pilot-files>
hf download NoeFlandre/georeset-text-label-benchmark \
  pilot/runs/e5-small-100-seed42/frozen_sample.json \
  pilot/runs/e5-small-100-seed42/candidate_labels.csv \
  --repo-type dataset --revision "$PILOT_REVISION" --local-dir artifacts/source
RUN_DIR=artifacts/source/pilot/runs/e5-small-100-seed42
COMMIT_SHA=$(git rev-parse HEAD)
uv run georeset-pilot run-dspark \
  --run-dir "$RUN_DIR" \
  --output-dir artifacts/source/pilot/runs/lfm2.5-2.6b-dspark-100-seed42 \
  --model-cache .cache/model-dspark \
  --computation-commit "$COMMIT_SHA" \
  --validation-commit "$COMMIT_SHA"
```

The resulting files belong under the distinct run path
`pilot/runs/lfm2.5-2.6b-dspark-100-seed42/` if published to the same Hub
dataset. Verify uploaded file SHA-256 values against `dspark_manifest.json`.
The model files, generated outputs, and inference costs are not downloaded or
produced by CI.

### Grid’5000 one-GPU execution

Run `scripts/run-dspark-grid5000.sh` only inside a separately allocated Linux
job with exactly one visible NVIDIA GPU and a one-hour wall-time limit. The
script does not submit jobs or contact a scheduler. The verified A100 SXM4
40-GiB node meets the adapter’s 16-GiB / compute-capability-8.0 admission gate.

Provide absolute paths to the already-published frozen input directory and a
new output directory on persistent storage. The persistent output parent must
already exist and be writable. For example, after the authorized owner has
placed the code and input files on the cluster:

```bash
scripts/run-dspark-grid5000.sh \
  /path/to/persistent/pilot/runs/e5-small-100-seed42 \
  /path/to/persistent/pilot/runs/lfm2.5-2.6b-dspark-100-seed42
```

The preflight requires at least 20 GiB free under job-local `$TMPDIR` (or
`/tmp`) and at least 1 GiB free in the persistent output filesystem. It places
the locked Python environment, Hugging Face model cache, SGLang/FlashInfer
runtime caches, and temporary home under a unique directory in that
job-local filesystem. It never uses the real home directory for model or
runtime caches. A monitor stops the run if job-local temporary use exceeds
20 GiB and deletes only the temporary directory it created. The script spends
at most 55 minutes on environment installation and inference to leave time
inside the one-hour allocation for cleanup. The persistent prediction,
metrics, and manifest directory is checked to remain below 1 GiB.

The script reads the fixed 100-row input directory and writes only the new
DSpark output directory. It does not alter the E5 run or submit, inspect,
cancel, or modify any other cluster jobs. A scheduler allocation and
authorization to use the persistent input/output paths are prerequisites; the
script intentionally leaves scheduler-specific job submission to the cluster
owner.

### Comparison limits

All 100 rows have polygon-level EUNIS assignments from the EEA probability-map
v1 (2021) source, with reported overlap percentages from `4.82%` to `100%`
(mean `86.89%`, median `100%`). They remain context rather than sentence-level
ground truth. For example, the pilot includes a vegetable-shop sentence whose
polygon code is Q11 (Raised bog), and a construction-site sentence whose code
is Q51 (Tall-helophyte bed). The result measures agreement with these existing
assignments on this positive-overlap sample; it does not establish that the
sentence itself describes the assigned habitat or predict performance on
negative examples. The EUNIS scientific validation status remains unconfirmed.
