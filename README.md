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
produces one reproducible Description sentence / EUNIS polygon overlap dataset, a
labelled pool of yes and no sentences with EUNIS codes, and one 100-sentence pilot
in which LFM2.5-2.6B + DSpark picks an EUNIS code per sentence. It does not train or
fine-tune a model, infer missing labels, or recompute EUNIS geometry.

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

## Geographic label pilot

The pilot samples 50 `yes` and 50 `no` English sentences with EUNIS codes. Samples are
spread over H3 resolution-3 cells, one sentence per cell. LFM2.5-2.6B + DSpark picks one
EUNIS code per sentence from the 158 pinned candidates. The yes/no label is never shown
to the model. See [`docs/pilot.md`](docs/pilot.md) for the sample rules, prompt, model
settings, metrics and limits.

```bash
uv sync --locked --all-groups --extra pilot
uv run georeset-pilot freeze \
  --labelled-parquet <run>/labelled-eunis.parquet \
  --pipeline-manifest <run>/manifest.json \
  --output-dir artifacts/pilot-100-seed42
```

Inference needs an NVIDIA GPU. It is run on Grid5000 with
`scripts/run-dspark-grid5000.sh`. The sample and the new results are not published yet.

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
