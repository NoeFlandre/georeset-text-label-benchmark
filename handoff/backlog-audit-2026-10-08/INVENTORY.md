# INVENTORY: handoff/backlog-audit-2026-10-08

Every file in this directory, with full SHA-256 of the exact bytes committed. Verify with `sha256sum -c SHA256SUMS` from this directory. `SHA256SUMS` also covers this file.

| Path | Bytes | SHA-256 | Role |
|---|---:|---|---|
| `HANDOFF.md` | 20163 | `2be69224fc447f70a3f234d5d983cbc6b7c274257490ef77cd68c85f12365155` | handoff (this directory's entry point) |
| `data/v2_rows.json` | 47446 | `8c3f40f6b7a9f35496ae33c48f27bac5115b4ec1ceaf41119d94155d75d6fbc0` | derived per-issue rows |
| `evidence/audit-results-2026-10-08.json` | 986277 | `f35636d5ade5718b12bac7cd2276896092d9d7c57e98a1e3ef6b0c7d3fdf07a9` | structured audit results (110 issues, first pass and verification) |
| `evidence/corrections-results-2026-10-08.json` | 178031 | `06fd035d00aa4f038da09859fd2e7d9c10e451fd6b7c839ed1bae9b250c5783b` | structured correction-pass results |
| `evidence/restore-test-2026-10-08.md` | 1477 | `f3757ec4091885d5b6d9b8f5bfb005c2b146903b386893c5129b16adf4310260` | restoration test record |
| `evidence/workflow-scripts/audit-workflow-2026-10-08.js` | 14642 | `d0b0514c118bd7b8e45e56ace2fe6eef0b62b9337f89b0becc35d2b1143b34eb` | workflow script (prompts, schemas, orchestration) |
| `evidence/workflow-scripts/corrections-workflow-2026-10-08.js` | 16641 | `2e9c22949b2ad4b2bd4dbc09a18d2cbb66df05c0ce4dda3891cd4666b5fcf883` | workflow script (prompts, schemas, orchestration) |
| `reports/issue-backlog-audit-v2-2026-10-08.md` | 56099 | `1d661bd8ae9fdcaf6fc0972408aef8d848248e84f03e4fbc4f9d1b7b10f1962b` | current audit report (v2) |
| `reports/master1-handoff-2026-10-08.md` | 9399 | `c620cc43b0492caa2f47d34af92e23432353be5a7aac6c3fdc6228ee851ea026` | Master 1 instructions |
| `scripts/build_v2.py` | 10335 | `1e0591b959f82510425b695665a7a725535d3a6131191bd1c9db803e6034c893` | derivation script for the v2 report |
| `scripts/report_v2.py` | 10902 | `23bf908ed4bc767c947b9cdc28460f6dee9b823415fa6dd93b5ff548b9b1b3fe` | derivation script for the v2 report |

## Deliberately excluded (not on this branch)

| Category | Files | Reason |
|---|---:|---|
| Subagent conversation transcripts | 43 | Raw traces with session internals. Their results are in the results files above. |
| Subagent metadata and prompt prefixes | 86 | Per-agent bookkeeping. Prompts are in the workflow scripts above. |
| Per-agent event journals | 2 | Duplicates of the results files above, with event-level noise. |
| Raw tool-result dumps (repository contents, PR listings, diffs) | 48 | Reproducible from GitHub at the dates recorded in the results. Bulky. |
| Verifier printout extracts | 2 | Duplicates of the verifier results inside the results file. |
| First-pass audit report | 1 | Superseded. Two errors corrected in v2. |
| Earlier helper scripts that only printed intermediate results | 3 | Superseded by the derivation scripts above. |
| Platform runtime state file | 1 | Not part of this task. |
| Repository checkouts | 10 | Not files in this directory. Clean and reproducible from GitHub at the SHAs in HANDOFF.md section 10. |

