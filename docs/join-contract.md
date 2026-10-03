# Join contract

## Inputs and grain

The source consists of three immutable, sharded collections:

| Collection | Grain | Pinned revision | Verified rows |
| --- | --- | --- | ---: |
| `NoeFlandre/osm-polygon-description-tag-landuse`, `labels/language-v1/data` | One decision per Description sentence occurrence | `d52781febf61a1ee77e4aceb02d9b607628b37c5` | 1,057,002 |
| `NoeFlandre/osm-polygon-description-tag-eunis`, `language-v1/data` | One Description tag value with detected language and sentence list | `88eabd83fcc60cb67ab4c1768a16e3d05e25ffb3` | 919,126 |
| `NoeFlandre/osm-polygon-description-tag-eunis`, `data` | One OSM polygon with its existing EUNIS assignment, if any | `88eabd83fcc60cb67ab4c1768a16e3d05e25ffb3` | 906,631 |

Both data repositories reference source revision
`b4706eb315c66a8a57135289e7612dca8e03caf8`. The runner confirms the embedded
revision and decision totals on each complete run.

## Keys and cardinality

1. Pair labels with Description values using `(description_identity, tag_key)`.
2. Resolve each ordinary sentence label using the zero-based
   `sentence_index` in the Description row's `sentences` list.
3. Pair Description values with polygon context using `(osm_type, osm_id)`.
   OSM polygon keys must be unique across all 386 shards.
4. Require `source_pbf` to match across the Description and polygon rows.
5. Treat `(description_identity, tag_key, sentence_index)` as the globally
   unique sentence occurrence key.

The label `text_sha256` is checked against the UTF-8 bytes of the resolved
sentence. The Description `text_sha256` describes the entire original tag value
and is not a sentence key. Identical sentence text in different places remains
distinct; the output is not deduplicated by text or hash.

Every label must match exactly one Description occurrence and every
Description sentence position must have exactly one label. Empty sentence
lists are allowed only with the source's `skipped_unsplit` index-0 sentinel;
that sentinel is counted but never turned into an invented sentence.

## Decisions and retention

Only source decisions `yes`, `no`, `failed`, and `skipped_unsplit` are valid.
All are counted. Ordinary `yes`/`no`/`failed` occurrences are counted even when
their polygon lacks EUNIS. The Parquet output keeps only `yes` sentence
occurrences whose matched polygon has a non-null, internally consistent EUNIS
assignment. No tags, languages, or overlap percentages are filtered or
thresholded.

The run reports polygon rows and assigned polygons, polygons without a
Description value, Description tag rows, sentence labels by decision and
EUNIS availability, unsplit sentinels, `yes` labels without EUNIS, and final
retained overlap occurrences. It also reports unique polygon, Description tag,
sentence occurrence, and exact sentence hash counts. The hash count is
descriptive only; it never changes output cardinality.
