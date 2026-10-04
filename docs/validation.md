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
directory and renames it only after all checks pass. A handled exception writes
a sibling failure JSON with the exception, traceback, and temporary-file size
inventory before removing the incomplete staged output. If that diagnostic
cannot be written, the stage is retained and its path is added to the original
exception. A process terminated outside Python's exception handling cannot
write a failure JSON; its temporary stage remains for diagnosis.

## Test-first evidence

The initial occurrence join test was run against a stub which raised
`NotImplementedError` and failed. Implementing the first join path made that
test green; later tests add coverage for missing/unmatched keys, duplicates,
negative and invalid decisions, exact hashes, language and source provenance,
EUNIS missingness, shard discovery, and output creation. The pinned-baseline
alias regression was reproduced with a `label_rows` expected count and then
fixed; the fixture now verifies every baseline key end-to-end through the
summary and manifest. A separate mismatch test confirms the diagnostic is
saved before staging cleanup. Tests use tiny local fixtures; no model or EUNIS
geometry is executed.

For the zero-shot pilot, the first 13 new tests were run red against unimplemented
stubs (13 failures), then passed after the sampler, E5 embedding, metrics, and
runner implementation was added. Subsequent tests exercise fixed input hashes,
distinct sentence/polygon sampling, candidate provenance, model loader flags,
metrics, and a fully mocked run before any real inference. The pilot has a
separate frozen-sample step to keep the selected IDs fixed before model loading.

## Static, complexity, and mutation checks

The project uses `ruff` for linting, formatting, and a maximum McCabe
complexity of five, `ty` for types, and branch coverage via pytest-cov. The CI
CRAP check uses conventional McCabe decision points, including loops and
filters inside comprehensions, and computes the score for every function in
the production package. The check rejects scores greater than or equal to six
and fails if any owned source file is absent from the coverage report. The CI
mutation result gate rejects survivors, timeouts, unknown states, and mutants
without tests; it does not infer success from `mutmut run` returning zero. The
production package, including the quality gates themselves, is in the mutation
scope. The final run killed 5,933 of 5,994 mutants. The 61 surviving mutants are
individually recorded with their equivalence rationale in `quality/mutation.py`;
there were no timeouts, unknown states, or mutants without tests. The survivors
cover identical codec and library defaults, prevalidated aligned inputs, and
private temporary probe details. Tests exercise publication races, directory
ownership, hashes, and label ordering that would change behavior if those
invariants were weakened. Every other survivor or unexpected result fails the gate.

## Storage and read behavior

The inspected pinned files total 995,348,189 bytes (about 949 MiB): roughly
100.4 MB of labels, 103.4 MB of Description values, and 791.5 MB of polygon
Parquet. The runner does not retain full input datasets or download geometry;
it accesses only projected columns through 8,192-row Parquet batches, keeps one
aligned shard in memory at a time, and writes compressed output to a temporary
directory. Before starting the complete run, the workspace had 52 GiB free.
