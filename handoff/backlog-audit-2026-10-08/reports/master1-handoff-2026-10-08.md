# Master 1 handoff: auto-close mismatches, conflicts, overlaps (2026-10-08)

Read-only. Nothing was changed. No PR description was edited, no comment posted, no PR merged or closed, no implementation dispatched, and no other agent contacted. Items below are for Master 1 to decide.

Sources: open-PR sweep of all nine repos (64 open PRs read in total: 50 non-Dependabot and 14 Dependabot), plus PR bodies and file lists. States were read at audit time and may have moved.

## A. Auto-close mismatches

A merge of these PRs would close the issue while scope remains, or would close an out-of-scope issue.

| # | PR | Closes | Why it may be wrong | Evidence | Suggested action |
|---|---|---|---|---|---|
| 1 | [worldcover #67](https://github.com/NoeFlandre/osm-worldcover/pull/67) | [#49](https://github.com/NoeFlandre/osm-worldcover/issues/49) | Multi-stage build is excluded. The issue names the toolchain in the final image. | PR body: "Not in this PR: a multi-stage build (the compiler toolchain still ships in the runtime image)." | Change "Fixes" to "Refs", or keep #49 open until the multi-stage build lands. |
| 2 | [worldcover #76](https://github.com/NoeFlandre/osm-worldcover/pull/76) | [#39](https://github.com/NoeFlandre/osm-worldcover/issues/39) | Adds a link sentence, but both duplicate doc blocks remain at the PR head. | Earlier verifier read docs/index.md and docs/technical-debt.md at a7321acc. Not re-read in this pass. | Re-check the head before merge. Use "Refs" if the blocks remain. |
| 3 | [geoparser #157](https://github.com/NoeFlandre/geoparser/pull/157) | [#150](https://github.com/NoeFlandre/geoparser/issues/150) | Issue's suggested fix asks for a built-wheel check. The test checks configuration only. | Test test_package_ships_the_pep561_typed_marker checks file existence, the classifier and the include entry. It does not build a wheel. | Add a built-wheel assertion, or record an owner waiver, before merge. Otherwise "Refs". |
| 4 | [benchmark #135](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance/pull/135) | [#124](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance/issues/124) | Issue's first symptom is the transformers image built twice. The PR says that stays. | PR body: "the transformers image is still built once in ci.yml and once in docker-smoke.yml." | Use "Refs". Keep #124 open for build-once. |
| 5 | [benchmark #134](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance/pull/134) | [#120](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance/issues/120) | Two listed scorers still need a maintainer decision to load. Pinning is a follow-up. | PR body: "Blocker for two listed scorers (needs a maintainer decision)." | Check #120's acceptance text. If it needs pins for unlisted models, use "Refs". |
| 6 | [golden #85](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/pull/85) | [#83](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/issues/83) | Earlier audit found the fsync path untested. Criterion 1 names "writer or file fsync". | Earlier audit (not re-read in this pass). PR is blocked. | Confirm the criterion wording before merge. |
| 7 | [wikidata #206](https://github.com/NoeFlandre/osm-polygon-wikidata-only/pull/206) | [#179](https://github.com/NoeFlandre/osm-polygon-wikidata-only/issues/179), [#193](https://github.com/NoeFlandre/osm-polygon-wikidata-only/issues/193) | #193 is Grid'5000 (out of scope). #179 is also claimed by #213. | Sweep closes list. | Remove #193 from the closing keywords. Keep one claim on #179. |
| 8 | [wikidata #213](https://github.com/NoeFlandre/osm-polygon-wikidata-only/pull/213) | [#179](https://github.com/NoeFlandre/osm-polygon-wikidata-only/issues/179) | Duplicate close claim with #206. | Sweep closes list. | Keep one. |

Checked, not a mismatch: worldcover #69 closes #51 (the suggested helper is excluded as a separate cleanup; the title's scope matches). EUNIS #143 closes #82 (tests use small tiles, not production size; minor). EUNIS #136 closes #119 (verified by CI only; a real tag push is still needed). Geoparser #171 closes #160 (red timing checks, not a scope gap). EUNIS #142 closes #121 (optional cache not done, per PR).

## B. Merge state

- **Dirty (conflicts):** description [#145](https://github.com/NoeFlandre/osm-polygon-description-tag/pull/145) (closes #137); website [#153](https://github.com/NoeFlandre/osm-polygon-website-tag/pull/153) (closes #120); EUNIS [#76](https://github.com/NoeFlandre/osm-polygon-eunis/pull/76) (edits grid5000.py, Grid'5000, out of scope).
- **Behind base (update, no conflict reported):** geoparser [#179](https://github.com/NoeFlandre/geoparser/pull/179), [#177](https://github.com/NoeFlandre/geoparser/pull/177), [#176](https://github.com/NoeFlandre/geoparser/pull/176), [#175](https://github.com/NoeFlandre/geoparser/pull/175), [#171](https://github.com/NoeFlandre/geoparser/pull/171), [#157](https://github.com/NoeFlandre/geoparser/pull/157), [#132](https://github.com/NoeFlandre/geoparser/pull/132); drafts [#172](https://github.com/NoeFlandre/geoparser/pull/172), [#121](https://github.com/NoeFlandre/geoparser/pull/121).
- **Blocked (reason not read):** geoparser [#181](https://github.com/NoeFlandre/geoparser/pull/181) and [#180](https://github.com/NoeFlandre/geoparser/pull/180) (both draft); golden [#95](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/pull/95) (draft); golden [#85](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/pull/85).
- **Unstable (checks not all passing):** 16 non-Dependabot PRs. Includes website #158, #157, #155 (drafts), #152, #147, #144; wikidata #216 (draft), #213, #212, #208, #206; EUNIS #144, #142, #141; WorldCover #66; benchmark #135.
- **Dependabot:** 14 PRs. Mergeable state unknown for EUNIS #100 and WorldCover #58 to #60.

## C. Overlaps (file level, open PRs)

- description #168 × #164: `dataset/text_migration.py`.
- description #166 × #164: `CHANGELOG.md`.
- website #156 × #144: `application/source_processing.py`.
- website #152 × #144: `web/web_fetch.py`, `tests/web/test_robots_policy.py`.
- wikidata #216 × #213 × #206: `docs/cli-reference.md`.
- wikidata #212 × #208: `utils/retry.py`, `hf/_uploader/operations.py`.
- WorldCover #69 × #66: `finalize.py`, `tests/unit/test_finalize_streaming.py`.
- WorldCover #67 × #66: `README.md`.
- geoparser #175 × #157: `pyproject.toml`, `tests/unit/test_quality/test_project_contract.py`.
- geoparser #175 × #172 and #157 × #172: `pyproject.toml`.
- geoparser #172 × #171: `mkdocs.yml`.
- EUNIS #144 × #76: `grid5000.py` (out of scope).
- georeset DSpark drafts: #11 × #9, #11 × #5, #9 × #5 share `dspark.py`, `dspark_runner.py`, `pilot/cli.py`, `docs/pilot.md`. #10 × #9, #10 × #5, #11 × #10 share `tests/test_dspark.py`.

Issue-level sequencing (not file overlap):
- Wikidata #182 waits for #208 (`cli/commands.py`). #173 waits for #216 (`audit_remote.py`, `trackio_snapshot.py`, `v2_trackio_snapshot.py`).
- Wikidata #163 waits for draft #159 (`test_crap_gate_remaining_coverage.py`).
- Geoparser #137 waits for #175 and #157 (`test_project_contract.py`).
- Geoparser #98: the English control is met only in draft #121.
- Geoparser #162 overlaps draft #121 (PAN-X spaCy files).
- georeset #14: its remaining requirement is in `dspark_runner.py`, which drafts #9, #5 and #11 edit.

## D. CI signals to triage

- **Website Mutation sweep (scheduled, main):** 0 of 7 scheduled runs after PR #103 succeeded (runs #13 to #19). 10 earlier scheduled runs also failed. Causes not diagnosed; no job logs were read. Related: website #75 and #91.
- **Geoparser Quality (schedule) on main e19b95b:** run 37761215713 failed 2026-10-08T10:06Z. Cause not diagnosed.
- **Wikidata:** PR bodies #206 and #208 say `tests/hf/test_coverage_map.py` also fails on main. Not checked against a main run in this pass.
- **Geoparser #171:** PR body says later hosted timing checks failed in unchanged benchmarked code, and those failures are not waived.
- **Benchmark #135:** an earlier check found a transformers Docker-smoke run cancelled after about 35 minutes, which matches its job timeout. Not re-read in this pass.

## E. Assignments and traceability

- **Geoparser #141, Master 4 slice (changelog.py, check_architecture.py):** not in the issue text. Confirm the slice boundary with Master 4 before work.
- **EUNIS #108, Master 2 "bounded assessment":** the words are not in the issue text. Master 2's scope needs definition.
- **Benchmark #92, Master 5 slice (speed rows, dataset_card):** matches the issue text. The `card_sections.py` boundary is ambiguous. Remainder not available to others.
- **Website #123:** no GitHub claim, comment, PR, branch or commit. Two independent checks agree. Confirm whether Master 1's queue includes it. The issue body asks to coordinate with #112 (no file overlap).
- **Presumed Master 1 queue (21 items, membership to confirm):** Description #124, #125, #126, #151; Website #75, #123; Wikidata #125, #161, #162, #164, #192; EUNIS #103; WorldCover #4, #7, #18, #21; geoparser #126, #143, #166 (remote branch `feat/uner-v2-recognition-166`); benchmark #100; golden #47.
- **Owner holds to respect:** geoparser #4, #9, #98; WorldCover #2 (blocked; Mac-related builder); benchmark #77 (cluster jobs in progress; sites look like Grid'5000, which is inference).
