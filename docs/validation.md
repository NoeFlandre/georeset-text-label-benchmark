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
scope. The final run killed 3,733 of 3,768 mutants. The other 35 are individually
listed in `quality/mutation.py`; there were no timeouts, unknown states, or mutants
without tests. The exact survivors and evidence are grouped here by equivalent
behavior:

| Exact mutant IDs | Equivalent change and evidence |
| --- | --- |
| `georeset_text_label_benchmark.join.x__validate_text_hash__mutmut_18` | Changes an exception-path `False` to `None`; both are false under the only following condition and raise the same validation error. Invalid hexadecimal hashes are covered. |
| `georeset_text_label_benchmark.join.x__verify_sentence_hash__mutmut_6`; `georeset_text_label_benchmark.pilot.runner.x__read_frozen_sample__mutmut_7`; `georeset_text_label_benchmark.pilot.runner.x__sha256_json__mutmut_17`; `georeset_text_label_benchmark.pilot.sampling.x__validate_sentence_hash__mutmut_5`; `georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_15` | Change `utf-8` to `UTF-8`. Python resolves both names to the same codec, so the bytes and SHA-256 are identical. Exact hashes and explicit UTF-8 input decoding are tested. |
| `georeset_text_label_benchmark.pilot.runner.x_sha256_file__mutmut_9` | Changes the hashlib algorithm name from `sha256` to `SHA256`; hashlib accepts case-insensitive names and produces the same digest for the same bytes. |
| `georeset_text_label_benchmark.pilot.sampling.x__rank_rows__mutmut_5` | Changes the codec name from `ascii` to `ASCII`; both encode the same deterministic priority string. |
| `georeset_text_label_benchmark.pipeline.x__build_run__mutmut_21`; `georeset_text_label_benchmark.pilot.runner.x__write_outputs__mutmut_14` | Change PyArrow's compression argument from `zstd` to `ZSTD`; both select the same codec. Parquet metadata is checked. |
| `georeset_text_label_benchmark.pilot.runner.x__sha256_json__mutmut_3`; `georeset_text_label_benchmark.pilot.runner.x__write_json_exclusive__mutmut_14`; `georeset_text_label_benchmark.pilot.cli.x_main__mutmut_36`; `georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_4` | Change `ensure_ascii=False` to `ensure_ascii=None`; JSON treats `None` as false and serializes identical Unicode text. Unicode JSON bytes and sample hashes are tested. |
| `georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_11` | Changes the JSON key/value separator in a sample identity payload. The payload is always a list, which has no key/value separator, so serialized bytes and the ID remain unchanged. |
| `georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_27`; `georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_30`; `georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_31`; `georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_20`; `georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_23`; `georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_24` | Replace strict zip behavior with `None`, `False`, or an omitted `strict` argument. Each caller first validates equal sequence lengths, so the zip always receives aligned inputs. Mismatched helper inputs are tested where the guard is not redundant. |
| `georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_21`; `georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_24`; `georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_25` | Replace strict zip behavior in candidate scoring. Shape validation first requires the number of candidate vectors to equal the number of codes, so each score row and code list have the same length. |
| `georeset_text_label_benchmark.pilot.embeddings.x_average_pool__mutmut_28` | Changes `unsqueeze(-1)` to `unsqueeze(+1)`; both identify the same trailing dimension. Per-row token counts and pooled vectors are tested. |
| `georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_46`; `georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_49`; `georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_50`; `georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_56` | Remove or alter explicit defaults for vector normalization (`p=2`, `dim=1`) and concatenation (`dim=0`). Library defaults preserve the same operation. A non-unit `[3, 4]` vector is checked against `[0.6, 0.8]`, and batch ordering is tested. |
| `georeset_text_label_benchmark.pilot.runner.x_freeze_sample__mutmut_32`; `georeset_text_label_benchmark.pilot.runner.x_freeze_sample__mutmut_34` | Change `Path.mkdir(exist_ok=False)` to `None` or omit it. `None` is false and omission defaults to false; both retain exclusive creation. A race test proves `exist_ok=True` is rejected. |
| `georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_10`; `georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_15`; `georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_20` | Change installed distribution names to uppercase. `importlib.metadata` normalizes package names case-insensitively and returns the same installed versions. |
| `georeset_text_label_benchmark.pilot.runner.x_run_pilot__mutmut_80` | Removes the explicit `top_k=5`; `rank_candidates` defaults to five and the pilot ranking is checked. |

All other mutant IDs must be killed for the gate to pass. Any newly surviving mutant
requires a new exact ID and behavior-based evidence before it can be accepted.

## Storage and read behavior

The inspected pinned files total 995,348,189 bytes (about 949 MiB): roughly
100.4 MB of labels, 103.4 MB of Description values, and 791.5 MB of polygon
Parquet. The runner does not retain full input datasets or download geometry;
it accesses only projected columns through 8,192-row Parquet batches, keeps one
aligned shard in memory at a time, and writes compressed output to a temporary
directory. Before starting the complete run, the workspace had 52 GiB free.
