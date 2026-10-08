export const meta = {
  name: 'audit-issue-backlog',
  description: 'Read-only audit of 110 open issues across 9 NoeFlandre repos: subagent auditors per group, each checked by an independent subagent verifier',
  phases: [
    { title: 'Probe', detail: 'confirm subagent model is available before fan-out' },
    { title: 'Audit', detail: 'one subagent auditor per repo group' },
    { title: 'Verify', detail: 'independent subagent verifier re-checks each auditor claim' },
  ],
}

const CATEGORIES = ['IMPLEMENTED', 'COVERED', 'PARTIAL', 'UNCLAIMED_MAINTENANCE', 'DEFERRED', 'UNCERTAIN', 'OWNED', 'OUT_OF_SCOPE']
const OWNERSHIP = ['none_seen', 'claimed_by_open_pr', 'claimed_by_comment', 'reserved', 'owned', 'out_of_scope', 'unknown_unpublished_possible']

const R = {
  DT: 'osm-polygon-description-tag',
  WT: 'osm-polygon-website-tag',
  WD: 'osm-polygon-wikidata-only',
  EU: 'osm-polygon-eunis',
  WC: 'osm-worldcover',
  GP: 'geoparser',
  BM: 'benchmark-llms-landuse-relevance',
  GD: 'landuse-sentence-relevance-golden-human-set',
  GR: 'georeset-text-label-benchmark',
}

const GROUPS = [
  { id: 'A', items: [['DT',160],['DT',153],['DT',152],['DT',151],['DT',138,'RES'],['DT',137],['DT',136],['DT',127],['DT',126],['DT',125],['DT',124]] },
  { id: 'B', items: [['WT',145],['WT',136],['WT',135],['WT',133],['WT',125],['WT',123],['WT',122],['WT',120],['WT',118,'OOS'],['WT',112],['WT',91],['WT',75]] },
  { id: 'C1', items: [['WD',201],['WD',193,'OOS'],['WD',192],['WD',189],['WD',187],['WD',182],['WD',180],['WD',179],['WD',174]] },
  { id: 'C2', items: [['WD',173],['WD',165],['WD',164],['WD',163],['WD',162],['WD',161],['WD',158],['WD',125],['WD',108]] },
  { id: 'D1', items: [['EU',121],['EU',119],['EU',118,'OOS'],['EU',115],['EU',108,'RES'],['EU',105,'G5000'],['EU',103]] },
  { id: 'D2', items: [['EU',82],['EU',81,'OOS'],['EU',79,'OOS'],['EU',6],['EU',3],['EU',2],['EU',1]] },
  { id: 'E', items: [['WC',51],['WC',49],['WC',39],['WC',21],['WC',20],['WC',19],['WC',18],['WC',7],['WC',4],['WC',2]] },
  { id: 'F1', items: [['GP',170],['GP',169],['GP',168],['GP',167],['GP',166],['GP',165],['GP',164],['GP',163],['GP',162],['GP',161]] },
  { id: 'F2', items: [['GP',160],['GP',155],['GP',154],['GP',152],['GP',151],['GP',150],['GP',149],['GP',143],['GP',142],['GP',141]] },
  { id: 'F3', items: [['GP',140],['GP',137],['GP',130],['GP',126],['GP',100],['GP',99],['GP',98],['GP',9],['GP',4]] },
  { id: 'G', items: [['BM',124],['BM',123],['BM',120],['BM',100],['BM',93,'OOS'],['BM',92],['BM',91],['BM',77],['GR',14]] },
  { id: 'H', items: [['GD',47],['GD',52],['GD',83],['GD',43,'OWN'],['GD',44,'OWN'],['GD',45,'OWN'],['GD',46,'OWN']] },
]

const TAGS = `Item tags:
- OOS = out of scope (Grid'5000, the Mac, filter-osm, the autonomous labeling agent). Make NO tool calls for it. category OUT_OF_SCOPE, ownership_status out_of_scope.
- RES = Master 2's proposed scope, pending confirmation. Audit it normally, but ownership_status must be reserved and do not propose it as unclaimed work.
- OWN = already owned by an existing golden-set audit. Make NO tool calls. category OWNED, ownership_status owned.
- G5000 = partly about Grid'5000. Audit it normally. Use OUT_OF_SCOPE only if Grid'5000 is its primary scope, and say in uncertainty which part you skipped.
- No tag = audit normally.`

const SHARED_AUDIT = `You are a read-only audit subagent for the issue backlog of GitHub user NoeFlandre. Your output is data for a master agent, not a message to a human.

HARD RULES
- Read-only. Never create, edit, comment, label, close, merge, approve, request review, fork, branch, or delete anything. Never run tests, install packages, download models, or change permissions.
- Use only GitHub read tools. First load them with ToolSearch, query: select:mcp__github__issue_read,mcp__github__search_issues,mcp__github__search_pull_requests,mcp__github__list_pull_requests,mcp__github__pull_request_read,mcp__github__get_file_contents,mcp__github__search_code,mcp__github__search_commits,mcp__github__list_commits,mcp__github__list_branches
- If a tool is denied or fails, record it in access_failures and move on. Do not work around denied access.
- Keep facts (what you read directly, with URL, path, PR number or SHA) separate from inferences (your reasoning).

METHOD, for each item
1. issue_read method=get: read the FULL body. Copy its acceptance criteria or done conditions. Then issue_read method=get_comments for linked discussion.
2. Linked PRs: check closed_by_pull_requests from the get result. Search PRs for the issue number (search_pull_requests, e.g. query "repo:NoeFlandre/<repo> <number>"). Check PR bodies for closes, fixes or refs with the number. For each relevant PR use pull_request_read method=get for state, merged, draft, merged_at, base and head. Use get_files when the code matters.
3. Current code: get_file_contents with no ref (the default branch) on the files the criteria name. Use search_code with repo:NoeFlandre/<repo> for symbols. Judge each criterion against the actual code, not names or titles.
4. Commits: search_commits or list_commits when a change may have landed without a PR.
5. Duplicates: decide by comparing required scope, never by similar titles. Cross-repo overlap goes in duplicate_of as "related: <repo>#<n>" unless the scope is identical.

EVIDENCE RULES
- A closed, unmerged PR is NOT a fix.
- Never infer completion from a title, from a closed PR alone, or from one merged subtask. Check every criterion against code.
- An open or draft PR covering the full remaining scope makes the item COVERED. One covering part of it makes the item PARTIAL.
- You cannot see local branches or unpublished work. For any UNCLAIMED_MAINTENANCE or PARTIAL item, ownership_status is unknown_unpublished_possible unless you see an explicit claim.

CATEGORIES (pick exactly one)
- IMPLEMENTED: every criterion met in current default-branch code. Cite merged PR and/or code in closure_evidence.
- COVERED: an open or draft PR (or a named owner in the thread) covers the full remaining scope. List it in linked_prs.
- PARTIAL: some criteria met. Put the exact unmet requirement text in remaining_requirement.
- UNCLAIMED_MAINTENANCE: open, no PR covers it, bounded, achievable as a small change.
- DEFERRED: research, dataset processing or publishing, or a large refactor.
- UNCERTAIN: evidence or access is missing. Say what is missing in uncertainty.
- OWNED or OUT_OF_SCOPE: only for the tags OWN and OOS.

OWNERSHIP_STATUS: none_seen (no PR, branch or comment claims it), claimed_by_open_pr, claimed_by_comment, unknown_unpublished_possible, reserved (tag RES only), owned (tag OWN only), out_of_scope (tag OOS only).

Return every item in your group, one entry each, including OOS and OWN items. If you see an open issue in the repo that is not in your list, name it in coverage_note and do not audit it fully.`

const SHARED_VERIFY = `You are an independent verifier subagent. Read-only: never write, comment, merge, or run tests or installs. GitHub read tools only. First load them with ToolSearch, query: select:mcp__github__issue_read,mcp__github__search_pull_requests,mcp__github__pull_request_read,mcp__github__get_file_contents,mcp__github__search_code,mcp__github__search_commits,mcp__github__list_commits,mcp__github__list_branches

You receive one auditor's output for a group of issues. Treat its categories, PR states, and code claims as CLAIMS TO TEST, not answers.

For each item that is not OWN or OOS:
1. Re-read the issue with issue_read get. Derive your own acceptance criteria and your own category BEFORE you compare with the auditor.
2. Test each auditor claim. Confirm each PR number, state and merged flag with pull_request_read get. Confirm each code claim by reading the named file on the default branch. Confirm "no PR covers this" by searching PRs for the issue number and listing open PRs in that repo.
3. Look for evidence the auditor missed: open or draft PRs, merged PRs that closed the issue, or a comment naming an owner.
4. Set verified_category. Set agrees to true only if verified_category equals the auditor's category.

For OWN or OOS items: make no tool calls. Set verified_category equal to the auditor's category and agrees true.

Each check: quote the claim, give verdict confirmed, refuted or unverifiable, and give evidence (URL, path, or what you saw). Put any missed evidence in missed_evidence. List access_failures. Return via the structured output schema.`

const PROBE_SCHEMA = {
  type: 'object',
  properties: { model_id: { type: 'string' }, github_tool_loaded: { type: 'boolean' } },
  required: ['model_id', 'github_tool_loaded'],
}

const ITEM_SCHEMA = {
  type: 'object',
  properties: {
    repo: { type: 'string' },
    number: { type: 'integer' },
    title: { type: 'string' },
    url: { type: 'string' },
    category: { type: 'string', enum: CATEGORIES },
    ownership_status: { type: 'string', enum: OWNERSHIP },
    ownership_note: { type: 'string' },
    criteria: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          text: { type: 'string' },
          status: { type: 'string', enum: ['met', 'unmet', 'unknown', 'n/a'] },
          evidence: { type: 'string' },
        },
        required: ['text', 'status', 'evidence'],
      },
    },
    linked_prs: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          number: { type: 'integer' },
          state: { type: 'string' },
          merged: { type: 'boolean' },
          draft: { type: 'boolean' },
          relation: { type: 'string' },
        },
        required: ['number', 'state', 'merged', 'draft', 'relation'],
      },
    },
    closure_evidence: { type: 'string' },
    remaining_requirement: { type: 'string' },
    code_evidence: { type: 'array', items: { type: 'string' } },
    duplicate_of: { type: 'array', items: { type: 'string' } },
    facts: { type: 'array', items: { type: 'string' } },
    inferences: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    uncertainty: { type: 'string' },
  },
  required: ['repo', 'number', 'title', 'url', 'category', 'ownership_status', 'ownership_note', 'criteria', 'linked_prs', 'closure_evidence', 'remaining_requirement', 'code_evidence', 'duplicate_of', 'facts', 'inferences', 'confidence', 'uncertainty'],
}

const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    items: { type: 'array', items: ITEM_SCHEMA },
    access_failures: { type: 'array', items: { type: 'string' } },
    coverage_note: { type: 'string' },
  },
  required: ['items', 'access_failures', 'coverage_note'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          repo: { type: 'string' },
          number: { type: 'integer' },
          auditor_category: { type: 'string' },
          verified_category: { type: 'string', enum: CATEGORIES },
          agrees: { type: 'boolean' },
          checks: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                claim: { type: 'string' },
                verdict: { type: 'string', enum: ['confirmed', 'refuted', 'unverifiable'] },
                evidence: { type: 'string' },
              },
              required: ['claim', 'verdict', 'evidence'],
            },
          },
          missed_evidence: { type: 'string' },
          notes: { type: 'string' },
        },
        required: ['repo', 'number', 'auditor_category', 'verified_category', 'agrees', 'checks', 'missed_evidence', 'notes'],
      },
    },
    access_failures: { type: 'array', items: { type: 'string' } },
    coverage_note: { type: 'string' },
  },
  required: ['results', 'access_failures', 'coverage_note'],
}

function itemLines(g) {
  return g.items.map(it => '- ' + R[it[0]] + '#' + it[1] + (it[2] ? ' [' + it[2] + ']' : '')).join('\n')
}

function auditPrompt(g) {
  return SHARED_AUDIT + '\n\n' + TAGS + '\n\nYOUR GROUP ' + g.id + '. Items (repo#number):\n' + itemLines(g)
}

function verifyPrompt(g, audit) {
  return SHARED_VERIFY + '\n\nGROUP ' + g.id + ' items:\n' + itemLines(g) + '\n\n' + TAGS + '\n\nAUDITOR OUTPUT TO TEST (claims, not answers):\n' + JSON.stringify(audit)
}

phase('Probe')
const probePrompt = 'Model availability check. Reply with the model ID you are running as, exactly as you know it. Then run ToolSearch with query "select:mcp__github__get_me" and set github_tool_loaded true only if that returns a GitHub tool definition. Do nothing else.'
const [pFull, pAlias] = await parallel([
  () => agent(probePrompt, { label: 'probe:[model-id redacted]', model: '[model-id redacted]', schema: PROBE_SCHEMA }),
  () => agent(probePrompt, { label: 'probe:subagent-alias', model: 'subagent', schema: PROBE_SCHEMA }),
])
log('full-ID probe: ' + JSON.stringify(pFull))
log('alias probe: ' + JSON.stringify(pAlias))

let MODEL = null
if (pFull) {
  MODEL = '[model-id redacted]'
} else if (pAlias && String(pAlias.model_id).indexOf('5-5') !== -1) {
  MODEL = 'subagent'
  log('[model-id redacted] did not answer; alias subagent self-reports ' + pAlias.model_id + '. Using alias, recorded in output.')
}
if (!MODEL) {
  return { blocked: 'subagent model not confirmed; nothing audited.', probes: { full: pFull, alias: pAlias } }
}

phase('Audit')
const outcomes = await pipeline(
  GROUPS,
  g => agent(auditPrompt(g), { label: 'audit:' + g.id, phase: 'Audit', model: MODEL, schema: AUDIT_SCHEMA }),
  async (audit, g) => {
    if (!audit) {
      log('audit failed for group ' + g.id + '; no verification run')
      return { group: g.id, audit: null, verify: null }
    }
    const verify = await agent(verifyPrompt(g, audit), { label: 'verify:' + g.id, phase: 'Verify', model: MODEL, schema: VERIFY_SCHEMA })
    return { group: g.id, audit: audit, verify: verify }
  }
)

const expected = GROUPS.flatMap(g => g.items.map(it => R[it[0]] + '#' + it[1]))
const got = []
for (const o of outcomes) {
  if (o && o.audit && o.audit.items) {
    for (const it of o.audit.items) got.push(it.repo + '#' + it.number)
  }
}
const missing = expected.filter(k => got.indexOf(k) === -1)
const extra = got.filter(k => expected.indexOf(k) === -1)
log('coverage: expected ' + expected.length + ', audited ' + got.length + ', missing ' + missing.length + ', extra ' + extra.length)

return {
  model: MODEL,
  probes: { full: pFull, alias: pAlias },
  coverage: { expected: expected.length, audited: got.length, missing: missing, extra: extra },
  groups: outcomes,
}
