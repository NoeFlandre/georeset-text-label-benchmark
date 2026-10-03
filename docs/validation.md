# Validation and engineering

## Data checks

Each run checks all of the following before publishing outputs:

* pinned source revisions and the embedded shared source revision;
* presence of all projected columns and matching shard filenames across all
  three collections;
* non-null stable keys, positive OSM IDs, globally unique polygon keys,
  Description tag keys, and sentence occurrence keys;
* exactly one polygon per Description row and matching `source_pbf` provenance;
* sentence count/list consistency, one label per expected sentence position,
  zero-based bounds, known decision values, and valid skipped-unsplit sentinel
  rows;
* exact label/Description language agreement when both are present, valid
  SHA-256 formatting, and sentence hash agreement against exact UTF-8 bytes;
* consistent null EUNIS assignment fields, non-empty assigned values, and
  finite overlap percentages within `[0, 100]`;
* pinned baseline counts for 906,631 polygons, 276,604 assigned polygons,
  919,126 Description tag rows, 1,057,002 labels, 978,773 sentence rows,
  725,371 `yes`, 241,475 `no`, 11,927 `failed`, and 78,229 skipped-unsplit
  sentinels.

Unmatched rows, missing sentence labels, duplicate keys, and unknown decisions
fail closed. The output path must be new. The run writes to a temporary sibling
directory and renames it only after all checks pass.

## Test-first evidence

The initial occurrence join test was run against a stub which raised
`NotImplementedError` and failed. Implementing the first join path made that
test green; later tests add coverage for missing/unmatched keys, duplicates,
negative and invalid decisions, exact hashes, language and source provenance,
EUNIS missingness, shard discovery, and output creation. Tests use tiny local
fixtures; no model or EUNIS geometry is executed.

## Static, complexity, and mutation checks

The project uses `ruff` for linting, formatting, and a maximum McCabe
complexity of five, `ty` for types, and branch coverage via pytest-cov. The CI
CRAP check computes the CRAP score for each production function and rejects
scores greater than or equal to six. Mutation testing runs over the same
production package and tests. The root project README lists the reproducible
commands.

## Storage and read behavior

The inspected pinned files total 995,348,189 bytes (about 949 MiB): roughly
100.4 MB of labels, 103.4 MB of Description values, and 791.5 MB of polygon
Parquet. The runner does not retain full input datasets or download geometry;
it accesses only projected columns through 8,192-row Parquet batches, keeps one
aligned shard in memory at a time, and writes compressed output to a temporary
directory. Before starting the complete run, the workspace had 52 GiB free.
