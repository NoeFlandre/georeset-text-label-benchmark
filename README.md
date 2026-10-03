---
license: other
license_name: OpenStreetMap ODbL-1.0 and EEA CC-BY-4.0
license_link: https://www.openstreetmap.org/copyright
language:
  - mul
tags:
  - geospatial
  - openstreetmap
  - eunis
  - land-use
  - sentence-classification
pretty_name: GeoReset text-label benchmark (Description/EUNIS overlap data)
---

# georeset-text-label-benchmark

**Author:** Noé Flandre

This project is a benchmark for evaluating prediction of geographic labels
from text across data sources and reference datasets. The current implementation
produces one reproducible Description sentence / EUNIS polygon overlap dataset
and a separate 100-sentence, zero-shot multilingual E5 ranking pilot. It does
not train or fine-tune a model, infer missing overlap labels, or recompute EUNIS
geometry.

The retained dataset contains one row per exact `yes` sentence occurrence
whose OpenStreetMap feature already has an EUNIS assignment.

The join contract uses `(description_identity, tag_key)` to find the sentence
list, then the zero-based `sentence_index` to resolve the exact occurrence.
Description values join to polygons by `(osm_type, osm_id)`; `source_pbf` must
also agree. Text hashes are verified for sentence integrity, never used as join
keys. Repeated identical text remains repeated when it occurs in different
places.

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. The source
snapshots are public and pinned by immutable Hub commit SHA. The reader fetches
only selected Parquet columns in bounded batches.

```bash
uv sync --locked --all-groups
COMMIT_SHA=$(git rev-parse HEAD)
uv run georeset-benchmark run \
  --output artifacts/description-eunis-overlap \
  --computation-commit "$COMMIT_SHA" \
  --validation-commit "$COMMIT_SHA"
```

The output path must be new. The run writes `overlap.parquet`, `summary.json`,
and `manifest.json` together after source checks and baseline counts pass. See
the [`docs`](docs/index.md) for source provenance, join rules, output fields,
quality checks, and data terms. The manifest includes the exact computation and
validation commits and SHA-256 checksums for the Parquet and summary artifacts.

## Zero-shot pilot

The pilot ranks all 158 pinned EUNIS English name-and-description candidates
for a deterministic sample of 100 distinct positive overlap sentences. It uses
`intfloat/multilingual-e5-small` at the immutable revision documented in
[`docs/pilot.md`](docs/pilot.md). The candidate sentence list is frozen before
model loading or inference.

```bash
uv sync --locked --all-groups --extra pilot
hf download NoeFlandre/georeset-text-label-benchmark overlap.parquet \
  --repo-type dataset \
  --revision 2f7e2436e583e9c3b3c0cd0dca23b1d190e8fbf9 \
  --local-dir source
COMMIT_SHA=$(git rev-parse HEAD)
uv run georeset-pilot freeze \
  --source-parquet source/overlap.parquet \
  --output-dir artifacts/e5-small-100-seed42
uv run georeset-pilot run \
  --run-dir artifacts/e5-small-100-seed42 \
  --model-cache .cache/model-e5-small \
  --computation-commit "$COMMIT_SHA" \
  --validation-commit "$COMMIT_SHA"
```

The inference outputs are a frozen sample, predictions Parquet, metrics JSON,
and a provenance manifest with pinned revisions, model-file and output hashes,
runtime, and timings. The model weights are fetched from the public Hub on first
run (about 471 MB). The pilot reports agreement with existing polygon-level
EUNIS labels; scientific validation of those labels is unconfirmed.

## Engineering checks

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check src tests scripts
uv run pytest --cov=georeset_text_label_benchmark --cov-report=json:coverage.json
uv run python scripts/check_crap.py coverage.json
uv run mutmut run
uv run python scripts/check_mutations.py
uv run mkdocs build --strict
```

The tests cover cardinality, exact sentence resolution, missing and unmatched
rows, negative decisions, skipped unsplit values, invalid labels, EUNIS nulls,
source provenance, projected shard reads, and artifact publication. The first
red test was run against a deliberately unimplemented `process_partition`
stub; it failed with `NotImplementedError`. The implementation then made the
same test pass, after which the contract suite was expanded.

## Data terms

Code is Apache-2.0. The input data have separate terms: OpenStreetMap data are
available under the [ODbL](https://www.openstreetmap.org/copyright), while the
seven EEA EUNIS probability-map collections referenced by the upstream EUNIS
manifest state [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [provenance and attribution](docs/provenance.md) before redistributing the
output. EUNIS is polygon context here, not sentence-level ground truth; the
final EUNIS scientific validation status is unconfirmed.

## Scope

Only the Description label source is implemented. Website and Wikidata
adapters remain future work. The small read-only source interface allows other
sources to be added later.
