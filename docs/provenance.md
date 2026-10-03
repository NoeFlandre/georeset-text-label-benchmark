# Provenance and attribution

This project aims to compare text-to-geographic-label prediction across
sources and references. The current implementation contains only the pinned
Description/EUNIS overlap pipeline documented here; it is not a model benchmark
run and does not evaluate predictions.

## Pinned source snapshots

| Data | Hub revision |
| --- | --- |
| Description relevance labels | `NoeFlandre/osm-polygon-description-tag-landuse@d52781febf61a1ee77e4aceb02d9b607628b37c5` |
| Description values and EUNIS polygon rows | `NoeFlandre/osm-polygon-description-tag-eunis@88eabd83fcc60cb67ab4c1768a16e3d05e25ffb3` |
| Shared source revision recorded in the data | `b4706eb315c66a8a57135289e7612dca8e03caf8` |

The source adapter locates only `labels/language-v1/data/*.parquet`,
`language-v1/data/*.parquet`, and `data/*.parquet` at those revisions. It
requires 386 filename-aligned shards in each collection and projects only the
join and output columns. It does not materialize raster data, generated
sentence text files, geometries, or unrelated repository generations.

The upstream EUNIS manifest is read as a provenance record. Its EUNIS
reference is the EEA EUNIS habitat probability maps, version 1 (2021), with
seven assets. The run stores the manifest SHA-256 and asset count. The existing
polygon assignment fields (`eunis_code`, name, overlap percentage, and source
version) are carried through as-is.

## Terms and acknowledgement

Project code is licensed under Apache-2.0. The pinned land-use labels card and
EUNIS dataset card both mark the OSM-derived datasets `odbl`: [land-use labels
card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-landuse),
[EUNIS card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-eunis).
The EUNIS snapshot also references seven EEA source-map assets whose individual
factsheets state CC-BY 4.0. Data terms are separate:

* **OpenStreetMap source data:** Open Database License 1.0. When redistributing
  substantial OSM-derived data, preserve the applicable attribution and
  share-alike obligations. See the [OpenStreetMap copyright and license
  page](https://www.openstreetmap.org/copyright).
* **EEA EUNIS reference layers:** the seven collections used by the upstream
  manifest identify CC-BY 4.0 terms. Attribute the European Environment Agency
  and link the specific collection factsheets below. The [EEA legal
  notice](https://www.eea.europa.eu/en/legal-notice) says collection-specific
  notices take precedence over general site terms.

The official EEA catalogue factsheets consulted for the seven source assets:

1. [Collection 99498d2c](https://sdi.eea.europa.eu/catalogue/datahub/api/records/99498d2c-7350-4655-b914-92c5b9e016c5/formatters/xsl-view?approved=true&language=eng&output=pdf)
2. [Collection f425a73e](https://sdi.eea.europa.eu/catalogue/datahub/api/records/f425a73e-dc6c-40be-93d6-9d234d4bbe1b/formatters/xsl-view?approved=true&language=eng&output=pdf)
3. [Collection 443cdba1](https://sdi.eea.europa.eu/catalogue/datahub/api/records/443cdba1-1d4b-4cae-91d5-72fa99a8a758/formatters/xsl-view?approved=true&language=eng&output=pdf)
4. [Collection 0c4270c4](https://sdi.eea.europa.eu/catalogue/datahub/api/records/0c4270c4-fd7e-4099-ac2f-5af2079ddbd8/formatters/xsl-view?approved=true&language=eng&output=pdf)
5. [Collection ae2fdada](https://sdi.eea.europa.eu/catalogue/datahub/api/records/ae2fdada-93d6-4cf1-a11c-6ebccf25d286/formatters/xsl-view?approved=true&language=eng&output=pdf)
6. [Collection 7a2d78ec](https://sdi.eea.europa.eu/catalogue/datahub/api/records/7a2d78ec-8a31-4e7b-91df-45db6e64e842/formatters/xsl-view?approved=true&language=eng&output=pdf)
7. [Collection a6c48c2d](https://sdi.eea.europa.eu/catalogue/datahub/api/records/a6c48c2d-114f-406e-ba99-c9085a5f5aee/formatters/xsl-view?approved=true&language=eng&output=pdf)

Suggested acknowledgement: “Contains OpenStreetMap data © OpenStreetMap
contributors, available under the Open Database License (ODbL). EUNIS habitat
probability-map context: European Environment Agency, EUNIS version 1 (2021),
CC-BY 4.0.” Check the ODbL and collection notices against the intended mode of
distribution before release.

## Scientific scope

This repository computes an overlap. It does not establish that the EUNIS
polygons or the sentence relevance decisions are scientifically valid. The
final EUNIS scientific validation is unconfirmed. No geometry is recalculated.
