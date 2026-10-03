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
pretty_name: GeoReset text-label benchmark — Description/EUNIS overlap
---

# GeoReset text-label benchmark: Description/EUNIS overlap

This dataset is the current data-integration release of the GeoReset
text-to-geographic-label benchmark project. The long-term goal is to evaluate
prediction of geographic labels from text across sources and reference
datasets. This release contains **Description sentence relevance labels
joined to existing EUNIS polygon context only**. It includes no model outputs
and is not a prediction-evaluation result.

## Contents

* `overlap.parquet`: one row per exact `yes` sentence occurrence matched to an
  OSM polygon that already has an EUNIS assignment;
* `summary.json`: all source stages and negative, failed, unsplit, missing-label,
  and unmatched counts;
* `manifest.json`: pinned source revisions, join keys, schema, EUNIS manifest
  digest and version, terms, and output row count.

The long-form overlap contains the exact source sentence, its sentence index
and SHA-256, Description tag and language, OSM polygon key and source extract,
and the existing EUNIS code, name, overlap percentage, and source version. It
does not include polygon geometry.

## Join and limitations

Labels join to Description tag values by `(description_identity, tag_key)` and
sentence positions by zero-based `sentence_index`; Description values join to
polygons by `(osm_type, osm_id)`. `source_pbf` agreement is checked as source
provenance. Sentence hashes validate exact UTF-8 text, not join records.
Identical text occurrences in distinct locations remain distinct.

The output keeps only `yes` sentences with a non-null existing EUNIS
assignment. `no`, `failed`, `skipped_unsplit`, positive sentences without an
EUNIS code, and polygon coverage without a matching Description row remain in
the stage counts in `summary.json`. No model training/evaluation or EUNIS
geometry recomputation has been performed. EUNIS is polygon-level context,
not sentence-level ground truth. Scientific validation of the EUNIS assignments
is unconfirmed.

## Provenance

The land-use sentence labels are pinned to
[`NoeFlandre/osm-polygon-description-tag-landuse` at
`d52781febf61a1ee77e4aceb02d9b607628b37c5`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-landuse/tree/d52781febf61a1ee77e4aceb02d9b607628b37c5).
Description values and polygon context are pinned to
[`NoeFlandre/osm-polygon-description-tag-eunis` at
`88eabd83fcc60cb67ab4c1768a16e3d05e25ffb3`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-eunis/tree/88eabd83fcc60cb67ab4c1768a16e3d05e25ffb3).
The shared source revision recorded in the rows and EUNIS manifest is
`b4706eb315c66a8a57135289e7612dca8e03caf8`.

See the [code repository](https://github.com/NoeFlandre/georeset-text-label-benchmark)
for implementation, complete EEA factsheet links, testing, and attribution
details.

## Terms and attribution

The project code is Apache-2.0; the input data have separate terms. Both pinned
upstream dataset cards identify their OSM-derived data as `odbl`: [land-use
labels](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-landuse)
and [EUNIS](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-eunis).
The OSM source data are under the [Open Database License 1.0
(ODbL)](https://www.openstreetmap.org/copyright). The seven EEA EUNIS
probability-map assets referenced by the pinned EUNIS manifest state [CC-BY
4.0](https://creativecommons.org/licenses/by/4.0/). Attribution:

> Contains OpenStreetMap data © OpenStreetMap contributors, available under
> the Open Database License (ODbL). EUNIS habitat probability-map context:
> European Environment Agency, EUNIS version 1 (2021), CC-BY 4.0.

The [EEA legal notice](https://www.eea.europa.eu/en/legal-notice) and specific
collection notices apply. The code repository lists the seven collection
factsheets. Review the data terms against your intended redistribution.
