# Restoration test record (2026-10-08)

## Scope of this record

This records the restoration test run before this branch was pushed. It covers an earlier packaging of the same audit files. The branch copy differs from that packaging in one way: model identifiers were removed from the stored results and scripts (policy). The remote post-push verification is reported in the handoff message that accompanies this branch.

## Test performed (earlier packaging)

1. Verified the archive against its recorded SHA-256 value.
2. Extracted the archive into a new, empty directory (clean restore target; the archive has no git history).
3. Verified every shipped file with its checksum manifest: 19 of 19 OK, 0 failed.
4. Byte-compared the restored tree with the source tree: identical, no differences.
5. Parsed every JSON and JSONL file: all valid. Audit result coverage recorded as 110 expected, 110 audited, 0 missing. Correction pass: 0 failed agents.
6. Counted the per-issue rows in the current report: 110.
7. Confirmed that the nine `main` head SHAs cited in the handoff are present in the restored handoff: all nine.
8. Scanned the restored tree for machine paths, session identifiers and credential patterns: none.

## Result

PASS on every check above, for the earlier packaging.

## What this does not cover

- It does not re-run the audit against live GitHub state.
- It does not check the GitHub branch; that check is performed after the push and reported separately.
