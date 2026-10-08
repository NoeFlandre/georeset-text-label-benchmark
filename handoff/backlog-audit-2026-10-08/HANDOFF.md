# HANDOFF: NoeFlandre issue-backlog audit (read-only), 2026-10-08

This directory is an archival copy of a read-only audit. It is self-contained: every essential file is in this directory, and nothing depends on attachments or on the environment that produced it.

## 0. Read first

- **Type:** read-only audit. No code, tests, PRs, issue edits, comments, merges or branch changes on `main` were made.
- **Where this lives:** branch `handoff/backlog-audit-2026-10-08-f6w5` in `NoeFlandre/georeset-text-label-benchmark`, under `handoff/backlog-audit-2026-10-08/`. Archival only. `main` was not changed. No PR was opened.
- **This file on the branch:** https://github.com/NoeFlandre/georeset-text-label-benchmark/blob/handoff/backlog-audit-2026-10-08-f6w5/handoff/backlog-audit-2026-10-08/HANDOFF.md
- **Truth date:** all GitHub facts are as read on 2026-10-08. PR states, merge states, mergeable values and CI conclusions may have changed since.
- **Use these files:** `reports/issue-backlog-audit-v2-2026-10-08.md` (current report) and `reports/master1-handoff-2026-10-08.md` (current instructions). The first-pass report is not preserved: it was superseded and contained two errors that v2 corrects (Benchmark #123 was listed as unclaimed; Website #91 was listed as a closure candidate).

## 1. Repositories and scope

Owner: NoeFlandre. Nine in-scope repositories, 110 open issues (pull requests excluded).

| Repository | Open issues | `main` head read at wrap-up (full SHA) |
|---|---:|---|
| osm-polygon-description-tag | 11 | ea22740443e526836590c267afa8ec87db2c222d |
| osm-polygon-website-tag | 12 | d71ea1f9feba2b4d682f7cdc5699d326f3163dc7 |
| osm-polygon-wikidata-only | 18 | 5e765e367de603794682d9ca6cf34b17f1e49de0 |
| osm-polygon-eunis | 14 | 35ce838a444e787bc91432586916b9be6a205ef9 |
| osm-worldcover | 10 | c41a85e11f716b70287645c0804a43b37662841b |
| geoparser | 29 | e19b95b6107b734813e0b24b7550808501fa8e90 |
| benchmark-llms-landuse-relevance | 8 | 5e52649b1a583c6e96b77cb83e090d99feea6c6f |
| georeset-text-label-benchmark | 1 | dbaf4efdf8aebbd208215b858e11bc02cc132492 |
| landuse-sentence-relevance-golden-human-set | 7 | e9c360b403ba6aea1a91f98d983fbd7a8ac5598f |

Excluded by brief: filter-osm-datasets-llms-landuse, Grid'5000, the Mac, and the autonomous labeling agent.

## 2. Goal

Audit every open issue in scope for implementation status, ownership, closure classification, unclaimed work, and auto-close, merge and overlap risks. Read only.

## 3. Completed scope (exact)

1. Open-issue count refreshed per repository: 110.
2. Per-issue audit of all 110 issues, in 12 repository groups, by subagents running on the same model family as the coordinating agent. Model availability was checked with two probes before fan-out. Coverage 110 of 110, with 0 missing and 0 extra. Run recorded in `evidence/audit-results-2026-10-08.json` (26 agents, 0 errors).
3. Independent verification of the first pass: 12 fresh-context verifier agents re-derived each category and tested each claim. 108 of 110 categories agreed. 2 were corrected (Geoparser #150 to PARTIAL, #141 to DEFERRED). 28 individual claims were refuted; the ones that change a finding are recorded in the v2 report.
4. Correction pass, recorded in `evidence/corrections-results-2026-10-08.json` (17 agents, 0 errors):
   - Website #123 ownership, two independent checks.
   - Website #91 scheduled-run evidence, two independent checks.
   - WorldCover #19 and Geoparser #140 criteria split into mandatory, optional and unspecified, with main-branch CI evidence.
   - Comment-based ownership sweep, 22 issues.
   - Slice-scope reads, 4 issues.
   - Open-PR sweep of 64 open PRs (50 non-Dependabot, 14 Dependabot): closing keywords, merge state and changed files.
   - File-overlap computation between open PRs.
5. The v2 report and this handoff were written.
6. Wrap-up: read-only inspection of local checkouts (section 10), a sanitised archive of this work, and a restoration test (`evidence/restore-test-2026-10-08.md`).

## 4. Not done (boundaries)

- No code changes, tests, lints, mutation or CRAP runs, installs or downloads.
- No job logs were read. CI failure causes are undiagnosed.
- No Hugging Face Hub checks. No Grid'5000, Mac or labeling-agent work.
- No issue closures, PR edits, comments, merges, or pushes to `main`.
- Repository-wide code search returned incomplete results. Absence claims rest on file reads and PR or commit search.

## 5. Results (current, v2)

**Implementation status (110):** IMPLEMENTED 3 · COVERED by open PR 32 · PARTIAL 33 · NOT STARTED 10 · DEFERRED 19 · UNCERTAIN 3 · golden NOT ANALYSED (owned) 4 · OUT OF SCOPE 6.

**Ownership (110):** Assigned 3 · Partially assigned 2 · Existing owner 4 · In flight 47 · Owner hold 8 · Approval needed 1 · Blocked by PR 11 · Blocked (access) 1 · Unowned 6 · Unknown (presumed Master 1 queue) 21 · Out of scope 6.

**Closure classifications (nothing was closed):**

- **Website #91:** IMPLEMENTED in code. Not closable. The owner's condition is a green scheduled sweep on `main` after PR #103 (merged 2026-10-01T22:09:35Z). Observed: 7 scheduled Mutation sweep runs after the merge (runs #13 to #19, 2026-10-02 to 2026-10-08), 0 succeeded. Before the merge, 10 more scheduled runs also failed. Cause undiagnosed.
- **WorldCover #19:** IMPLEMENTED. The 3 mandatory criteria are met on `main`. Main CI passed on the merge commit 3332d5db (run 67). Open: the `.parquet` suffix literal is still duplicated, a criterion 1 caveat that needs a ruling. Optional items: loader privacy is conditional and unmet; the exhaustive-check test was not observed. Closing is an owner decision.
- **Geoparser #140:** IMPLEMENTED for the either/or fix (document and help). The body has no mandatory acceptance markers. Optional exit code 2 is unmet and needs an owner decision. Main head e19b95b: the Tests (push) run succeeded. The scheduled Quality run on e19b95b failed (run 37761215713); cause undiagnosed.

**Website #123 ownership:** no GitHub claim. Verified by two independent checks: no assignee, no comments, no PR, no branch, and no commit referencing it. The body asks to check and coordinate with #112 first. No file overlap was found with #112's open draft PR #157. It is a maintenance item, so it is presumed to be in Master 1's queue. **Not recommended unless Master 1 releases it.**

## 6. Remaining work, defects, dependencies, overlaps, decisions

### 6.1 Auto-close mismatches (check before any merge)

A merge would close the issue while scope remains, or would close an out-of-scope issue. Change keywords only with the PR owner.

| PR | Closes | Problem | Evidence |
|---|---|---|---|
| worldcover #67 | #49 | Multi-stage build excluded; the issue names the toolchain in the final image. | PR body: "Not in this PR: a multi-stage build." |
| worldcover #76 | #39 | Both duplicate doc blocks remain at the PR head. | First pass read at a7321acc; re-check the head. |
| geoparser #157 | #150 | The built-wheel check the issue requests is missing; the test checks configuration only. | Test file read in the first pass; PR files: py.typed, pyproject.toml, test. |
| benchmark #135 | #124 | The image is still built twice, which is the issue's first symptom. | PR body "What is not changed". |
| benchmark #134 | #120 | Two listed scorers still need a maintainer decision to load; pinning is a follow-up. | PR body "Blocker for two listed scorers". |
| golden #85 | #83 | Fsync path untested; criterion 1 names "writer or file fsync". | First-pass audit; not re-read. |
| wikidata #206 | #179, #193 | #193 is Grid'5000 (out of scope). #179 is also claimed by #213. | Sweep closes list. |
| wikidata #213 | #179 | Duplicate close claim with #206. | Sweep closes list. |

Checked, not a mismatch: worldcover #69 closes #51; EUNIS #143 closes #82 (small tiles only, minor); EUNIS #136 closes #119 (CI-only verification); geoparser #171 closes #160 (red timing checks, not scope); EUNIS #142 closes #121 (optional cache not done, stated in the PR).

### 6.2 Merge state (correction-pass sweep)

- **Dirty (conflicts):** description #145 (closes #137); website #153 (closes #120); EUNIS #76 (edits `grid5000.py`, Grid'5000, out of scope).
- **Behind base (update needed, no conflict reported):** geoparser #179, #177, #176, #175, #171 (closes #160), #157 (closes #150), #132 (closes #130); drafts #172 and #121.
- **Blocked (reason not read):** geoparser #181 and #180 (drafts); golden #95 (draft); golden #85.
- **Unstable (checks not all passing):** 16 non-Dependabot PRs, including website #158, #157, #155 (drafts), #152, #147, #144; wikidata #216 (draft), #213, #212, #208, #206; EUNIS #144, #142, #141; WorldCover #66; benchmark #135.
- **Dependabot (14):** mergeable state unknown for EUNIS #100 and WorldCover #58 to #60.

### 6.3 File overlaps between open PRs (file level)

- description #168 × #164: `dataset/text_migration.py`
- description #166 × #164: `CHANGELOG.md`
- website #156 × #144: `application/source_processing.py`
- website #152 × #144: `web/web_fetch.py`, `tests/web/test_robots_policy.py`
- wikidata #216 × #213 × #206: `docs/cli-reference.md`
- wikidata #212 × #208: `utils/retry.py`, `hf/_uploader/operations.py`
- worldcover #69 × #66: `finalize.py`, `tests/unit/test_finalize_streaming.py`
- worldcover #67 × #66: `README.md`
- geoparser #175 × #157: `pyproject.toml`, `tests/unit/test_quality/test_project_contract.py`
- geoparser #175 × #172 and #157 × #172: `pyproject.toml`
- geoparser #172 × #171: `mkdocs.yml`
- EUNIS #144 × #76: `grid5000.py` (out of scope)
- georeset DSpark drafts: #11 × #9, #11 × #5, #9 × #5 share `dspark.py`, `dspark_runner.py`, `pilot/cli.py`, `docs/pilot.md`. #10 × #9, #10 × #5 and #11 × #10 share `tests/test_dspark.py`.

### 6.4 Issue-level sequencing (not file overlap)

- Wikidata #182 waits for #208 (`cli/commands.py`). Wikidata #173 waits for draft #216 (`audit_remote.py`, `trackio_snapshot.py`, `v2_trackio_snapshot.py`).
- Wikidata #163 waits for draft #159 (`test_crap_gate_remaining_coverage.py`).
- Geoparser #137 waits for #175 and #157 (`test_project_contract.py`).
- Geoparser #98: its English control is met only in draft #121.
- Geoparser #162 overlaps draft #121 (PAN-X spaCy files).
- georeset #14: its remaining requirement is in `dspark_runner.py`, which drafts #9, #5 and #11 edit.

### 6.5 CI signals (undiagnosed; logs not read)

- Website scheduled Mutation sweep on `main`: 17 scheduled runs, none succeeded (7 after PR #103). Related: website #75 and #91.
- Geoparser Quality (schedule) on `main` e19b95b: run 37761215713 failed 2026-10-08T10:06Z.
- Wikidata PR bodies #206 and #208 say `tests/hf/test_coverage_map.py` also fails on `main`. Not checked against a `main` run.
- Geoparser #171 PR body: later hosted timing checks failed in unchanged benchmarked code; not waived.

### 6.6 Ownership questions

- **Presumed Master 1 queue, membership to confirm (21):** Description #124, #125, #126, #151; Website #75, #123; Wikidata #125, #161, #162, #164, #192; EUNIS #103; WorldCover #4, #7, #18, #21; geoparser #126, #143, #166 (remote branch `feat/uner-v2-recognition-166`); benchmark #100; golden #47.
- **Slices:** Geoparser #141 (Master 4: `changelog.py` and `check_architecture.py`) is not in the issue text. EUNIS #108 (Master 2: bounded assessment) is not in the issue text. Benchmark #92 (Master 5: speed rows and `dataset_card`) matches the text; its remainder has no owner and is not available.
- **Owner holds:** geoparser #4, #9, #98; WorldCover #2 (blocked; local builder is Mac-related); benchmark #77 (cluster jobs in progress; site names suggest Grid'5000, which is an inference).

### 6.7 Genuinely available work (none started)

Unowned, no open-PR overlap, and no external write needed to start scoping. Ranked:

1. Geoparser #99: restrict to the canonical 85 languages. Reuses PAN-X helpers on `main`.
2. Geoparser #165: UniTopRank as a non-neural baseline. Release availability not verified.
3. Geoparser #169: expand corpora with UniTopRank and TopoResolve. External releases and licences not verified.
4. Geoparser #100: DaMuEL subset. Depends on #98 and #99.
5. Geoparser #167 (MultiCoNER II) and #168 (MasakhaNER 2.0): external datasets; check licences and label policy first.

Available only after PR #171 merges: geoparser #163, #164, #170.

## 7. Tests and reviews (exact-commit status)

| Item | Status | Subject | Notes |
|---|---|---|---|
| Tests in any repository | **unrun** | none | No tests were run. |
| First-pass verifier review of audit claims | **passed** (108/110 categories agree); 2 corrected; 28 claims refuted | GitHub read state, 2026-10-08 | Not an exact-commit review. Stale for any PR state that changed. |
| Website #123 ownership, two checks | **passed** (both: no claim) | GitHub state, 2026-10-08 | Stale if a claim appears later. |
| Website #91 nightly counts, two checks | **passed** (both: 7 after merge, 0 successes) | Actions run list on `main` | Run conclusions only; logs unread. |
| CI conclusions read via API | **observed, not diagnosed** | website Mutation sweep; geoparser Tests (push) success and Quality (schedule) failure on e19b95b; WorldCover CI run 67 success on 3332d5db | Not tests run here. |
| Job logs | **skipped** | all | Outside the read set for these checks. |
| Mutation, CRAP, lint, Docker, Hugging Face Hub | **skipped** | all | Not run. |
| Restoration of this archive | **passed** (before this branch was pushed) | archive contents | See `evidence/restore-test-2026-10-08.md`. Remote verification is recorded in the handoff message that accompanies this branch. |

Stale: every PR state, merge state, mergeable value and CI conclusion in this file, as read on 2026-10-08.

## 8. Active processes and jobs

- None of this task's workers, background jobs or monitors are running.
- Both workflow runs completed: the audit run (26 agents, 0 errors) and the correction pass (17 agents, 0 errors).
- Nothing was killed. The platform's own runtime processes are not part of this task.

## 9. Contents of this directory

| Path | What it is |
|---|---|
| `HANDOFF.md` | This file. |
| `INVENTORY.md` | Every other file in this directory with full SHA-256, and the items deliberately excluded. |
| `SHA256SUMS` | SHA-256 manifest for every other file in this directory. Verify with `sha256sum -c SHA256SUMS` from this directory. |
| `reports/issue-backlog-audit-v2-2026-10-08.md` | Current audit report: per-issue status and ownership, reconciled counts, closure classifications, available work. |
| `reports/master1-handoff-2026-10-08.md` | Instructions for Master 1: auto-close mismatches, merge state, overlaps, CI signals, ownership questions. |
| `evidence/audit-results-2026-10-08.json` | Structured results of the 110-issue audit and first-pass verification. |
| `evidence/corrections-results-2026-10-08.json` | Structured results of the correction pass: ownership checks, nightly runs, criteria, comments, slice scopes, open-PR sweep. |
| `evidence/workflow-scripts/` | The two workflow scripts that produced the results: prompts, schemas and orchestration. |
| `evidence/restore-test-2026-10-08.md` | Restoration test record. |
| `data/v2_rows.json` | Derived per-issue rows used to build the v2 report. |
| `scripts/build_v2.py`, `scripts/report_v2.py` | Scripts that derive the rows and write the v2 report. They reference the scratch location where the results were first written. Edit those paths to rerun them. |

**Excluded on purpose:** subagent conversation transcripts (raw traces with session internals), the raw tool-result dumps (repository contents, reproducible from GitHub), the per-agent event journals (duplicates of the results above), the first-pass report (superseded), earlier helper scripts that only printed intermediate results, and the platform runtime state. Each is listed with its reason in `INVENTORY.md`.

**Sanitisation:** scanned for tokens, keys, passwords, email addresses, personal data and machine-specific paths. No credentials or email addresses were found. Machine paths and session identifiers were replaced with placeholders. Model identifiers were removed from the stored results and scripts, per repository policy; they appear only in the chat record of the audit. The GitHub owner handle appears only in public repository URLs.

## 10. Checkout inventory (read-only inspection)

Ten repository checkouts were present in the session workspace. Each was inspected with read-only git commands. None was modified by this task.

| Checkout | HEAD (full SHA) | Branch | Working tree | Stash | Local-only commits |
|---|---|---|---|---|---|
| benchmark-llms-landuse-relevance | 5e52649b1a583c6e96b77cb83e090d99feea6c6f | main | clean | 0 | none |
| georeset-text-label-benchmark | dbaf4efdf8aebbd208215b858e11bc02cc132492 | main | clean | 0 | none |
| osm-worldcover | c41a85e11f716b70287645c0804a43b37662841b | main | clean | 0 | none |
| osm-polygon-description-tag | ea22740443e526836590c267afa8ec87db2c222d | main | clean | 0 | none |
| osm-polygon-wikidata-only | 5e765e367de603794682d9ca6cf34b17f1e49de0 | main | clean | 0 | none |
| geoparser | e19b95b6107b734813e0b24b7550808501fa8e90 | main | clean | 0 | none |
| osm-polygon-eunis | 35ce838a444e787bc91432586916b9be6a205ef9 | main | clean | 0 | none |
| osm-polygon-website-tag | d71ea1f9feba2b4d682f7cdc5699d326f3163dc7 | main | clean | 0 | none |
| landuse-sentence-relevance-golden-human-set | e9c360b403ba6aea1a91f98d983fbd7a8ac5598f | main | clean | 0 | none |
| filter-osm-datasets-llms-landuse | 233cb30d4fa5d790ead9e1eec4fee1efb0760bdf | main | clean | 0 | none (out of scope; git state only) |

Each checkout has one local branch, `main`, tracking `origin/main` with no ahead or behind count. There were no unpushed commits, unpublished branches, staged or unstaged changes, untracked files, or scratch integration branches. The georeset checkout was used to create this archival branch; its local `main` was not changed.

## 11. Safest first steps for Master 1

1. Read `reports/master1-handoff-2026-10-08.md`.
2. Before any merge, resolve the 8 auto-close mismatches in 6.1. Ask each PR owner to change the keyword or split the scope.
3. Triage the website Mutation sweep (7 of 7 post-merge scheduled runs failed) and the geoparser Quality (schedule) failure on e19b95b. Job logs are on GitHub and were outside this audit's read set.
4. Confirm the 21 presumed queue items and Website #123.
5. Ask Master 4 and Master 2 to confirm their slices against the issue texts (Geoparser #141; EUNIS #108).
6. Resolve the dirty PRs (description #145, website #153). Update the behind-base geoparser PRs in an order that respects overlaps: #175 before #157 and #172, and #171 before #172.
7. Do not start the available research items without owner approval. EUNIS publication needs owner approval.

## 12. Pending decisions

- Website #91: close after a green scheduled sweep, or accept the code as done? Current condition: the sweep is failing.
- WorldCover #19: rule on the `.parquet` literal, then close, or keep open for the optional items.
- Geoparser #140: exit code 2 (owner). Close, or split the optional item first.
- Geoparser #150 and PR #157: add a built-wheel assertion, or record a waiver.
- Benchmark #120 and PR #134: maintainer decision on `trust_remote_code` for two scorers.
- Owner holds: geoparser #4, #9, #98; WorldCover #2; benchmark #77.
- Golden #83: confirm the criterion wording against PR #85.

## 13. Storage and retention

The branch is the preserved copy. Nothing is held only in an environment, and no attachment is needed. Keep the branch unchanged; it is archival.

## 14. Prompt for Master 1

> Read the handoff at https://github.com/NoeFlandre/georeset-text-label-benchmark/blob/handoff/backlog-audit-2026-10-08-f6w5/handoff/backlog-audit-2026-10-08/HANDOFF.md. It is archival and self-contained. Start with `reports/master1-handoff-2026-10-08.md` in the same directory. Do not change `main`. Before merging anything, resolve the eight auto-close mismatches and the sequencing in section 6.4. Do not close issues until the stated blockers are resolved.
