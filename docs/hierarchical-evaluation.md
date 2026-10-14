# Retrospective EUNIS hierarchy evaluation

This evaluation adds official ancestor scores to an existing prediction run. It
does not change the original EUNIS codes, parsed outputs, or fine-label metrics.
It is agreement with polygon-level EUNIS references, not a sentence-level
habitat judgment.

## Official parent mappings

The 158 candidate codes come from two reviewed EEA classifications: 149
terrestrial codes from the 2021 release updated in 2023, and nine marine codes
from the 2022 release. The corresponding EEA workbooks are linked from the
[EEA EUNIS tabular data page](https://www.eea.europa.eu/en/datahub/datahubitem-view/638330ea-90e6-4e41-81ea-e70f25ae7117).
The pinned workbook byte hashes, row-derived parent chains, code names, and
source metadata are recorded in
`artifacts/eunis-hierarchy-v1/taxonomy_mapping.json`.

The mapping uses each workbook's `Level` field and classification row order.
It does not slice code strings. Example chains verified against the workbooks
are `N → N1 → N11` and `M → MA2 → MA22 → MA221`. The terrestrial candidates
are at level 3; the marine candidates are at level 4. The 2012 crosswalk columns
in the workbooks are not used as parents.

The source workbook closure has seven level 1 labels, 28 level 2 labels, 152
level 3 labels, and nine level 4 labels. The native audit's listed terrestrial
level 2 parent counts sum to 27, which the EEA workbook confirms; adding marine
`MA2` gives 28 combined level 2 labels. It is not 29.

## Eligibility and metrics

Levels 1–3 include all sample rows. A row with no reference ancestor at the
requested level stays in `n` and is counted as unmapped. Level 4 includes only
references whose exact official taxonomy contains a level 4 target. Shallower
references are excluded from its metric denominator and counted in
`excluded_shallower_reference_count`. No level is carried forward to fill a
missing deeper reference.

Each row reports `yes`, `no`, and `overall` separately. `overall` combines the
yes and no rows at that level. Accuracy divides correct mapped ancestors by
eligible `n`; invalid outputs, truncations, and unmapped labels remain wrong in
that denominator. Valid, failed, and truncated counts partition `n`.
`unmapped_count` is an additional row count and may overlap those three status
counts. The level label universe contains every distinct official ancestor
reached from the full candidate list, qualified by the candidate source hash.
Macro-F1 is the unweighted mean across that fixed universe. Multiclass MCC uses
separate unmapped-gold and unresolved-prediction sentinels; they never match.
Valid-only accuracy is supplementary and uses rows with mapped gold and
prediction ancestors. When a group has no eligible rows, all four model metrics
(accuracy, valid-only accuracy, macro-F1, and multiclass MCC) are null. MCC is
zero only for a non-empty eligible set whose generalized denominator is zero.

The fine-label report is repeated under `original_leaf` using the existing
158-candidate protocol. The scorer checks it against the historical
`dspark_metrics.json` before writing any output. It also writes a compact
`hierarchical_dataset_card.md` table for later publication.

For the committed 100-row sample, all 50 yes and 50 no references are at level
3. Therefore the level 4 rows will have `sample_n=100`, `n=0`, and 100
shallower-reference exclusions. That run has no level 4 model score. A future
sample containing marine level 4 references can have an eligible level 4
metric.

The prior fine-label run reported three correct predictions, 81 valid wrong
predictions, and 16 truncated outputs. It is the 16-truncation run at
computation commit `ce2173476115cdb15c2f4e525bd0e6cbf8a0ffe3` and remains
separate from earlier superseded runs. The hierarchical summary must be
generated from that run's original prediction Parquet and manifest; an aggregate
issue comment alone is insufficient to reconstruct ancestor predictions.

## Scoring an existing run

The command validates the frozen sample, candidate CSV, prediction order, run
scope, historical output checksums, and official mapping source metadata. It
refuses an existing output directory and writes new files beside the historical
run without modifying it.

```bash
uv run georeset-pilot score-hierarchy \
  --predictions <historical-run>/dspark_predictions.parquet \
  --run-manifest <historical-run>/dspark_manifest.json \
  --frozen-sample artifacts/pilot-100-seed42/frozen_sample.json \
  --frozen-manifest artifacts/pilot-100-seed42/manifest.json \
  --candidate-csv artifacts/pilot-100-seed42/candidate_labels.csv \
  --taxonomy artifacts/eunis-hierarchy-v1/taxonomy_mapping.json \
  --output-dir <new-hierarchical-results-dir> \
  --scoring-commit <40-character-scoring-commit>
```

The output contains `hierarchical_metrics.json`, a flat CSV, a dataset-card
table, and `hierarchical_manifest.json`. The JSON includes the complete label
universes, row counts, metric definitions, both original leaf and ancestor
results, and checksums for every source artifact.

The 16-truncation historical run was retrieved from Hub revision
`d0b2337c98a44d58b815fd5d876d608418654ff7` and checked against its run manifest,
frozen sample, and original leaf metrics. Its derived hierarchy results are
published as a versioned additive directory under
`pilot/runs/dspark-100-seed42/hierarchy-v1/`; the original run files remain
unchanged. Do not substitute the older run with 97 truncations or combine its
metrics with the 16-truncation result.
