# GeoReset text-label benchmark

The long-term goal is a benchmark for evaluating prediction of geographic
labels from text across sources and reference datasets. This initial
implementation produces a reproducible, occurrence-level join between pinned
Description sentence labels and existing EUNIS habitat labels on OSM polygons.
It is a data integration and coverage report. It does not train or evaluate a
predictive model, fill missing values, or recompute geometry.

The local runner reads the public Hub snapshots by immutable revision and
streams only projected Parquet columns. It checks all three collections for
matching shard filenames, then checks their actual row keys and provenance;
filename alignment by itself is never treated as proof that rows match.

```bash
uv sync --locked --all-groups
uv run georeset-benchmark run --output artifacts/description-eunis-overlap
```

An output directory is published only after every shard passes cardinality,
label, source-version, and baseline checks. The run contains the Parquet result,
a stage summary, and a provenance manifest.

**Scientific limitation:** an EUNIS code is a polygon-level spatial assignment.
Its presence beside a `yes` sentence records overlap between two existing
labels; it does not establish that the sentence describes that habitat.
Scientific validation of the EUNIS labels is unconfirmed.

See [join contract](join-contract.md), [provenance](provenance.md),
[output definitions](outputs.md), and [validation](validation.md).
