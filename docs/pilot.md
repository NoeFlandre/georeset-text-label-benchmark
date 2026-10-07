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

Both adapters stage their outputs beside the destination. Before E5 downloads
weights, and before DSpark checks the GPU or loads weights, each adapter probes
the actual output filesystem for exclusive directory creation and hard links
that preserve inode identity and fail with `EEXIST` when the target already
exists. Publication uses those operations instead of Linux `renameat2` flags.
Linux alone does not establish that a network filesystem supports a particular
rename flag.

The adapters publish `manifest.json` or `dspark_manifest.json` last. Consumers
must treat that manifest as the complete-run marker. If a mount error interrupts
publication, the final directory may contain some output files without a
manifest, and the complete staged files remain beside it. A retry verifies the
frozen-input and staged-output hashes, then resumes only when every file
already in the destination is the same inode as its staged source. It never
removes or replaces an existing destination file. If the stage is incomplete,
ambiguous, or does not match the frozen run and commits, the command stops and
reports the stage path for inspection.

The preflight checks the filesystem at the path supplied for that run. It does
not guarantee availability after the probe, so a later storage error can still
interrupt publication. On Grid'5000, the documented `/home` and Group Storage
paths use NFS ([Grid'5000 storage documentation](https://www.grid5000.fr/w/Storage)).
This checkout has not tested the Grid'5000 mount itself. Test the exact
persistent output parent before inference:

```bash
uv run python -c 'from pathlib import Path; from georeset_text_label_benchmark.pilot.publication import ensure_publication_supported; ensure_publication_supported(Path("/path/to/persistent/output-parent"))'
```

The probe creates and removes only a private temporary directory under the
selected parent. It downloads no model and does not start inference. The
adapter repeats the check on every run.

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
runtime is SGLang `0.5.20` with FlashInfer `0.6.18`, BF16, paired `DSPARK`
speculative decoding, one request at a time, and at most `512` generated tokens.
This uses the [upstream LFM2 DSpark support](https://github.com/sgl-project/sglang/pull/31041)
and the launch settings in the [official draft card](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark).
The official [SGLang LFM2.5 guide](https://docs.sglang.ai/cookbook/autoregressive/LiquidAI/LFM2.5)
describes the supported serving path and warns that a generation cap can truncate
reasoning. This pilot intentionally keeps the target model card's 512-token cap
and gates it with a representative smoke before a full evaluation.
DSpark is speculative decoding with a paired draft checkpoint; it is not a
hardware device and does not add a second label model. The target remains the
LFM2.5 checkpoint, and the pinned sampling settings determine its output.

LiquidAI documents a `131,072`-token target context. SGLang v0.5.20 passes the
target context length to its DSpark draft worker, while this draft checkpoint
supports `128,000` tokens. The adapter therefore pins the engine's
`context_length` to `128,000`, and checks each rendered prompt plus the full
`512`-token generation cap before it loads the serving engine. The
[model-card example](https://huggingface.co/LiquidAI/LFM2.5-2.6B) uses
temperature `0.1`, top-k `50`, repetition penalty `1.1`, and a 512-token cap;
the adapter pins those settings. The pinned template ignores
`enable_thinking=False` and opens assistant generation with `<think>`, so the
adapter does not claim reasoning is disabled. SGLang v0.5.20's DSpark
acceptance path uses its engine RNG; its per-request `sampling_seed` is not
consumed there. Requests run sequentially with `random_seed=42`, matching the
frozen sample seed. The EOS token `<|im_end|>` (token ID `124900`) from the [pinned
tokenizer vocabulary](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/654f9463ce32b05d0429d76fe1f580b27d4c1ac0/tokenizer.json)
is an explicit stop token. The adapter keeps special tokens and the stop
marker in raw output using SGLang v0.5.20's `skip_special_tokens` and
`no_stop_trim` controls, alongside its `stop_token_ids`
fields ([pinned sampling API](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/sampling/sampling_params.py)).

At the pinned tokenizer revision, the actual rendered chat template opens the
assistant turn with `<think>` even when passed `enable_thinking=False`; the
template does not read that flag. The adapter therefore passes no false
thinking-control flag and leaves the model's reasoning format intact. Parsing
uses only text after the final `</think>` and accepts an exact code from the
candidate set. Truncated answers, unclosed reasoning, unknown codes, and other
formats remain rows with their raw output and a parse failure reason. A
regression test renders the exact pinned 5.4 KB template fixture without model
weights.

The completed `job4184197` used 100 frozen sentences, 158 EUNIS candidates,
temperature 0, no repetition penalty, no explicit stop token, and a 4096-token
cap. It reached that cap on 97 rows while still reasoning; 95 repeated a
12-word span, and none of those 97 outputs contained `</think>` or EOS. The
other three rows yielded only `MA223`, `MA221`, and `N1A`; none matched the gold
label. The parser rejected the unfinished reasoning and preserves those rows as
invalid. Do not recover a code mentioned inside a reasoning trace as a final
answer, and leave that run and its evidence unchanged.

The corrected sampled settings are grounded in the pinned target's model card
and generation configuration; they address the missing repetition control and
EOS stop and use a smaller explicit cap. They have not yet been validated on a
GPU and do not establish that accuracy will improve. Run the bounded smoke
below first. Only if it passes should the full 100-row retry use a new output
path such as `pilot/runs/lfm2.5-2.6b-dspark-100-seed42-sampling-v2/`. The
manifest records every generation setting and its rationale.

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

Use Linux with an NVIDIA CUDA GPU visible through `nvidia-smi`, a CUDA 13.0
toolkit, and an NVIDIA driver at least `580.65.06` (the CUDA 13.0 minimum in
[NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)).
SGLang v0.5.20's [official container pin](https://github.com/sgl-project/sglang/blob/v0.5.20/docker/Dockerfile)
uses CUDA 13.0.3 and FlashInfer 0.6.18. The serving gate admits GPUs with
compute capability `8.0` or newer and at least `16,384 MiB` of VRAM; this is an
admission floor, not a guarantee that every runtime allocation fits. LiquidAI's
published SGLang throughput test used one BF16 H100 with 80 GB.

Before loading any model weights, the adapter checks pinned SGLang and
FlashInfer package versions, compares `nvcc` with PyTorch's CUDA runtime, then
compiles and executes a tiny FlashInfer prefill kernel. Its result is recorded
in `runtime.cuda_preflight` in the output manifest. This tests the allocated
node's CUDA driver, toolkit, and JIT path without downloading target or draft
weights. The command also checks that the Hub cache has at least 8 GiB free
before loading the tokenizer or weights. Do not run this adapter on the
CPU-only workspace used for unit tests.

First obtain `frozen_sample.json`, `candidate_labels.csv`, and the E5
`manifest.json` from the authorized, versioned run at
`pilot/runs/e5-small-100-seed42/`. Use the published dataset revision
[`073a1e478bd719f7a8ddc8c9fca191cb87c12926`](https://huggingface.co/datasets/NoeFlandre/georeset-text-label-benchmark/tree/073a1e478bd719f7a8ddc8c9fca191cb87c12926/pilot/runs/e5-small-100-seed42),
which contains the exact frozen sample, candidate table, and E5 manifest. The
runner verifies the sample and candidate hashes against that manifest. Install
the runtime in a dedicated environment:

```bash
uv sync --locked --no-default-groups --extra dspark
```

Then run the adapter against that frozen directory. By default it writes to
the sibling `lfm2.5-2.6b-dspark-100-seed42` directory; pass `--output-dir` to
choose a different new path. `HF_HOME` may be set to a large local cache path
before the command; otherwise `--model-cache` is used. SGLang downloads the
target and draft revisions on the first actual run unless a trusted Hub cache
is supplied with `--model-cache-seed`.

The `--model-cache-seed` value is the read-only HF Hub cache root containing
`models--LiquidAI--LFM2.5-2.6B` and
`models--LiquidAI--LFM2.5-2.6B-DSpark`. The runner requires the exact pinned
snapshot revisions and complete, bounded Hub tree metadata. It compares the
cached file set and Hub identities with the package's source-controlled
`pinned_cache_files.json` inventory, then hashes every source blob in bounded
chunks using its pinned Git SHA-1 or LFS SHA-256 digest. It copies the tree
metadata and each unique pinned blob into the job-local cache, then verifies
the copied bytes against the pinned size and digest. Snapshot links resolve to
those local blobs, so the seeded cache does not depend on the source cache.
The source cache remains unchanged. The manifest records the source root, both
repository revisions, and seeding method.

Run the smoke and full inference commands only inside an eligible Grid’5000
GPU allocation. Set `TMPDIR` to that allocation's job-local scratch and use a
fresh cache directory for each command, as below.

Run this direct example from a clean, committed checkout. It records `HEAD` in
the run manifest and does not account for uncommitted checkout changes.

```bash
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Pilot requires a clean, committed checkout." >&2
  exit 1
fi
PILOT_REVISION=073a1e478bd719f7a8ddc8c9fca191cb87c12926
hf download NoeFlandre/georeset-text-label-benchmark \
  pilot/runs/e5-small-100-seed42/frozen_sample.json \
  pilot/runs/e5-small-100-seed42/candidate_labels.csv \
  pilot/runs/e5-small-100-seed42/manifest.json \
  --repo-type dataset --revision "$PILOT_REVISION" --local-dir artifacts/source
RUN_DIR=artifacts/source/pilot/runs/e5-small-100-seed42
COMMIT_SHA=$(git rev-parse HEAD)
PILOT_ATTEMPT_ID=$(python -c 'import uuid; print(uuid.uuid4().hex)')
HF_HUB_CACHE_SEED=/path/to/trusted/hf-hub-cache
SMOKE_OUT="artifacts/source/pilot/runs/lfm2.5-2.6b-dspark-smoke-8-seed42-$COMMIT_SHA-$PILOT_ATTEMPT_ID"
JOB_LOCAL_CACHE_ROOT="${TMPDIR:?set TMPDIR to allocation-local scratch}/georeset-dspark-$COMMIT_SHA-$PILOT_ATTEMPT_ID"
SMOKE_CACHE="$JOB_LOCAL_CACHE_ROOT/smoke"
FULL_CACHE="$JOB_LOCAL_CACHE_ROOT/full"
if ! env -u HF_HOME -u HF_HUB_CACHE uv run georeset-pilot run-dspark-smoke \
  --run-dir "$RUN_DIR" \
  --output-dir "$SMOKE_OUT" \
  --model-cache "$SMOKE_CACHE" \
  --model-cache-seed "$HF_HUB_CACHE_SEED" \
  --computation-commit "$COMMIT_SHA" \
  --validation-commit "$COMMIT_SHA"; then
  echo "Smoke command failed; do not start the full pilot." >&2
  exit 1
fi
FULL_OUT="artifacts/source/pilot/runs/lfm2.5-2.6b-dspark-100-seed42-retry-$COMMIT_SHA-$PILOT_ATTEMPT_ID"
if python -c 'import json, sys; gate=json.load(open(sys.argv[1], encoding="utf-8"))["smoke_gate"]; raise SystemExit(0 if gate["passed"] else 1)' \
  "$SMOKE_OUT/dspark_metrics.json"; then
  env -u HF_HOME -u HF_HUB_CACHE uv run georeset-pilot run-dspark \
    --run-dir "$RUN_DIR" \
    --output-dir "$FULL_OUT" \
    --model-cache "$FULL_CACHE" \
    --model-cache-seed "$HF_HUB_CACHE_SEED" \
    --computation-commit "$COMMIT_SHA" \
    --validation-commit "$COMMIT_SHA"
else
  echo "Smoke gate failed; do not start the full pilot." >&2
  exit 1
fi
```

Before GPU or CUDA preflight, the adapter verifies the SHA-256 values of the
complete frozen sample and candidate CSV against that E5 manifest, and checks
that its source, sample, candidates, and E5 model revision agree with the
frozen files. It records the E5 manifest digest in the DSpark output manifest.
The resulting files belong under the distinct run path
`pilot/runs/lfm2.5-2.6b-dspark-100-seed42/` if published to the same Hub
dataset. Verify uploaded file SHA-256 values against `dspark_manifest.json`.
The model files, generated outputs, and inference costs are not downloaded or
produced by CI.

### Grid’5000 one-GPU execution

Run `scripts/run-dspark-grid5000.sh` only inside a separately allocated Linux
job with exactly one visible NVIDIA GPU and a one-hour wall-time limit. The
script does not submit jobs or contact a scheduler. The existing Grid'5000
executor uses an A40 with CUDA 13 and driver 580; it meets the adapter's
16-GiB / compute-capability-8.0 admission gate.

Run both code blocks in the same shell so they use the same commit and attempt
ID. The attempt ID gives each smoke run a new output path, so a previous
passing metrics file cannot satisfy the current run's gate. Provide absolute
paths to the already-published frozen input directory and a new output
directory on persistent storage. The persistent output parent must already
exist and be writable. For example, after the authorized owner has placed the
code and input files on the cluster:

```bash
COMMIT_SHA=$(git rev-parse HEAD)
PILOT_ATTEMPT_ID=$(python -c 'import uuid; print(uuid.uuid4().hex)')
HF_HUB_CACHE_SEED=/path/to/trusted/hf-hub-cache
SMOKE_OUT="/path/to/persistent/pilot/runs/lfm2.5-2.6b-dspark-smoke-8-seed42-$COMMIT_SHA-$PILOT_ATTEMPT_ID"
if ! scripts/run-dspark-grid5000.sh \
  /path/to/persistent/pilot/runs/e5-small-100-seed42 \
  "$SMOKE_OUT" \
  --smoke \
  --model-cache-seed "$HF_HUB_CACHE_SEED"; then
  echo "Smoke command failed; do not start the full pilot." >&2
  exit 1
fi
```

Before the full 100-row retry, run the checked-in eight-row smoke against a
language-diverse subset of the existing frozen sample. It uses the exact same
model, template, serving, sampling, stop, and parser settings as the full run,
and writes to a distinct output directory through the same NFS-safe publication
path. The command exits unsuccessfully unless at least six rows contain an
exact candidate code after `</think>`, all six retain the pinned `<|im_end|>`
stop, and no row is truncated. Inspect the manifest and per-row output; leave
both the original failed run and frozen E5 input unchanged. The smoke does not
resample or relabel the full benchmark and its metrics are not a full pilot
result.

After that smoke passes, invoke the wrapper without `--smoke`, using a separate
new full-run path and the same cache seed. The script requires a clean,
committed checkout so both manifest commit fields identify the exact runtime
code.

```bash
: "${COMMIT_SHA:?run the smoke setup in this shell first}"
: "${PILOT_ATTEMPT_ID:?run the smoke setup in this shell first}"
: "${SMOKE_OUT:?run the smoke setup in this shell first}"
if python3 -c 'import json, sys; gate=json.load(open(sys.argv[1], encoding="utf-8"))["smoke_gate"]; raise SystemExit(0 if gate["passed"] else 1)' \
  "$SMOKE_OUT/dspark_metrics.json"; then
  scripts/run-dspark-grid5000.sh \
    /path/to/persistent/pilot/runs/e5-small-100-seed42 \
    "/path/to/persistent/pilot/runs/lfm2.5-2.6b-dspark-100-seed42-retry-$COMMIT_SHA-$PILOT_ATTEMPT_ID" \
    --model-cache-seed "$HF_HUB_CACHE_SEED"
else
  echo "Smoke gate failed; do not start the full pilot." >&2
  exit 1
fi
```

Grid'5000 documents `/home` and Group Storage as NFS mounts. The adapter checks
the chosen persistent output parent with the no-clobber probe before GPU
admission. This checkout has not run that probe on a Grid'5000 node. Run the
standalone command in the publication section against the actual output parent
before allocating GPU time. The job wrapper's local `$TMPDIR` checks do not
verify persistent storage behavior.

The preflight requires at least 20 GiB free under job-local `$TMPDIR` (or
`/tmp`) and at least 1 GiB free in the persistent output filesystem. It places
the locked Python environment, Hugging Face model cache, SGLang/FlashInfer
runtime caches, and temporary home under a unique directory in that
job-local filesystem. It never uses the real home directory for model or
runtime caches. A monitor stops the run if job-local temporary use exceeds
20 GiB and deletes only the temporary directory it created. The script spends
at most 55 minutes on environment installation and inference to leave time
inside the one-hour allocation for cleanup. The persistent prediction,
metrics, and manifest directory is checked to remain below 1 GiB. The wrapper
retries that size scan up to five times to tolerate a transient NFS error. A
size at or above 1 GiB remains a visible failure. If all scans fail, the
published directory is kept and the wrapper says to verify the manifest and
file checksums before any retry; do not submit a duplicate run or overwrite
that output path.

Before installing SGLang, the wrapper loads the Grid'5000 Lmod setup and the
Rennes modules `nvidia-driver-libs/580`, `cuda-toolkit/13.0.2`, and
`uv/0.10.12`. It checks the visible GPU's driver, compiles the CUDA version
gate, and exports `CUDA_HOME` and the toolkit library path. Set
`DS_DRIVER_MODULE`, `DS_CUDA_MODULE`, or `DS_UV_MODULE` if an allocated site
exposes one of these under another module name. The FlashInfer workspace cache
also stays under the job-local limit. During startup, before target and draft
weight loading, the Python adapter JIT-compiles a small FlashInfer operation;
the run stops if that smoke test fails.

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
