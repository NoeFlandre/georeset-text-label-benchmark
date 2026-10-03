# Outputs

A successful run writes three files to one newly created output directory.

## `overlap.parquet`

One row represents one retained Description sentence occurrence that has an
existing polygon-level EUNIS assignment. Its 13 fields are:

| Field | Meaning |
| --- | --- |
| `source_pbf` | Source extract name, also checked for agreement across inputs |
| `osm_type`, `osm_id` | Stable polygon key |
| `description_identity`, `tag_key` | Description tag value key |
| `sentence_index` | Zero-based source sentence position |
| `sentence` | Exact source sentence, including whitespace and punctuation |
| `text_sha256` | SHA-256 of exact sentence UTF-8 bytes |
| `language_code` | Detected Description language, nullable |
| `eunis_code`, `eunis_name` | Existing EUNIS label and name |
| `eunis_overlap_percentage` | Existing source percentage, retained without thresholding |
| `eunis_source_version` | Version carried by the source polygon row |

Unlabeled polygons and `no`, `failed`, or `skipped_unsplit` occurrences are not
rows in the overlap file. Their counts remain in `summary.json`.

## `summary.json`

The report includes input row stages, decision counts, assigned/missing EUNIS
counts for each decision, unmatched polygon coverage, exact retained overlap
rows, and retained breakdowns by EUNIS code, source tag, and detected language.
It includes exact counts of unique polygon keys, Description tag keys, sentence
occurrences, and sentence hashes. `baseline_verification` records whether the
pinned expected counts were checked; a mismatch makes the run fail before
publication.

## `manifest.json`

The manifest records both repository revisions, shared input revision, EUNIS
manifest hash and reference version, join keys, output schema, output row
count, licenses and EEA asset factsheet links, and the polygon-context caveat.
The manifest is deterministic for a given summary and source revision (it does
not inject a wall-clock timestamp).

Output order follows lexicographically sorted source shard names and source
row order. No text or occurrence deduplication is performed.
