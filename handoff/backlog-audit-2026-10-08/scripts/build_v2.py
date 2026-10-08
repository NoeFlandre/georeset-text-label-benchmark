import json, re, collections
SP='<SESSION_DIR>/scratchpad/'
A=json.load(open('<SESSION_DIR>/tasks/wm64nhgvn.output'))['result']
B=json.load(open('<SESSION_DIR>/tasks/wudv6jov1.output'))['result']
SHORT={'osm-polygon-description-tag':'DT','osm-polygon-website-tag':'WT','osm-polygon-wikidata-only':'WD','osm-polygon-eunis':'EU','osm-worldcover':'WC','geoparser':'GP','benchmark-llms-landuse-relevance':'BM','georeset-text-label-benchmark':'GR','landuse-sentence-relevance-golden-human-set':'GD'}
ORDER=list(SHORT.keys())
ver={}; items={}
for g in A['groups']:
    if g['verify']:
        for r in g['verify']['results']: ver[(r['repo'],r['number'])]=r
    if g['audit']:
        for it in g['audit']['items']: items[(it['repo'],it['number'])]=it
openpr={}
for sw in B['prsweeps']:
    if sw: openpr[sw['repo'].split('/')[-1]]={x['number']:x for x in sw['prs']}

def cat_of(repo,n):
    v=ver.get((repo,n))
    return v['verified_category'] if v else items[(repo,n)]['category']

STATUS={'IMPLEMENTED':'IMPLEMENTED','COVERED':'COVERED by open PR','PARTIAL':'PARTIAL','UNCLAIMED_MAINTENANCE':'NOT STARTED','DEFERRED':'DEFERRED','UNCERTAIN':'UNCERTAIN','OWNED':'NOT ANALYSED (owned)','OUT_OF_SCOPE':'OUT OF SCOPE'}
STATUS_OVR={
 ('WT',91):'IMPLEMENTED; nightly evidence negative',
 ('WC',19):'IMPLEMENTED; mandatory met, optional open',
 ('GP',140):'IMPLEMENTED; optional item open',
 ('BM',123):'NOT STARTED',
 ('DT',138):'NOT STARTED',
 ('EU',108):'NOT STARTED',
 ('GP',141):'DEFERRED; partially assigned',
 ('BM',92):'PARTIAL; partially assigned',
}
# ownership: (class, detail). classes: Assigned, Partially assigned, Existing owner, In flight, Blocked, Owner hold, Out of scope, Unowned, Unknown
OWN_OVR={
 ('BM',123):('Assigned','Master 1'),
 ('DT',138):('Assigned','Master 2'),
 ('EU',108):('Assigned','Master 2 (bounded assessment only; implementation not assigned)'),
 ('GP',141):('Partially assigned','Master 4 owns changelog.py / check_architecture.py slice only. That slice is not named in the issue text. Remainder: no owner assigned; not available to other agents.'),
 ('BM',92):('Partially assigned','Master 5 owns speed-row and card-rendering slice only. Remainder (hf_publish.py split, agreement module, golden test): no owner assigned; not available to other agents.'),
 ('GD',43):('Existing owner','golden audit (not analysed here)'),
 ('GD',44):('Existing owner','golden audit (not analysed here)'),
 ('GD',45):('Existing owner','golden audit (not analysed here)'),
 ('GD',46):('Existing owner','golden audit (not analysed here)'),
 ('WT',91):('Owner hold','repo owner decision: close after a green scheduled sweep on main'),
 ('WC',19):('Owner hold','repo owner decision: closing'),
 ('GP',140):('Owner hold','repo owner decision on optional exit-code change'),
 ('GP',9):('Owner hold','owner deferred real-model run until configuration is chosen'),
 ('GP',4):('Owner hold','owner put scale-out on hold until pilot is accepted'),
 ('GP',98):('Owner hold','owner stopped partial run and will not restart unchanged loader; English control also blocked by draft PR #121'),
 ('WC',2):('Owner hold','owner blocked: active worker and release locks; local builder is Mac-related (out of scope)'),
 ('BM',77):('Owner hold','owner has queued cluster jobs in progress; sites look like Grid\'5000 (inference), so out of scope'),
 ('WD',182):('Blocked','open PR #208 edits cli/commands.py'),
 ('WD',173):('Blocked','draft PR #216 edits the same three run() entry points'),
 ('GP',137):('Blocked','open PRs #175 and #157 edit the same test file'),
 ('WD',163):('Blocked','draft PR #159 edits a file this issue would relocate'),
 ('GR',14):('Blocked','draft PRs #9, #5 and #11 edit dspark_runner.py, which the remaining requirement names'),
 ('WT',123):('Unknown','presumed Master 1 queue (maintenance); no GitHub claim, comment, PR, branch or commit found'),
 ('GP',166):('Unknown','remote branch feat/uner-v2-recognition-166 exists; author not visible'),
 ('WD',108):('Blocked (access)','osm-polygon-core not configured for this session; cannot verify'),
 ('GP',161):('In flight','draft PR #172 covers Otter arms; bi-mmBERT and other arms have no owner'),
 ('GP',99):('Unowned','no claim, comment or PR; open PR #171 is related (protocol and inventory API) but its files do not overlap'),
 ('GP',162):('Blocked','overlaps draft PR #121 (same PAN-X spaCy files)'),
 ('GP',163):('Blocked','sequenced after open PR #171 (frozen protocol; inferred dependency)'),
 ('GP',164):('Blocked','sequenced after open PR #171 (frozen protocol; inferred dependency)'),
 ('GP',170):('Blocked','depends on #163 to #169 and open PR #171 (inferred)'),
 ('EU',3):('Approval needed','publishing Hugging Face datasets is an external write; needs owner approval'),
 ('EU',1):('Blocked','depends on publication of #3 (external write)'),
 ('EU',2):('Blocked','card bytes must be final before publication of #3 (external write)'),
}
RESEARCH={('GP',n) for n in [4,9,98,99,100,161,162,163,164,165,166,167,168,169,170]}|{('EU',1),('EU',2),('EU',3),('WC',2),('BM',77),('WD',108)}

NOTE={
 ('WT',91):'Nightly: 0 of 7 scheduled Mutation sweep runs after PR #103 succeeded (runs #13 to #19, 2026-10-02 to 2026-10-08). Code verified on main.',
 ('WT',75):'Nightly mutation sweep fails on main (see #91). PR #100 (Dependabot) edits the same workflow files.',
 ('WC',19):'Mandatory criteria met on main; main CI success at merge commit 3332d5db (run 67). Open: .parquet suffix literal still duplicated (criterion 1 caveat).',
 ('GP',140):'No mandatory acceptance markers in body. Either/or fix (document and help) met on main. Optional exit 2 not done. Main head e19b95b: Tests (push) success; scheduled Quality run failed, cause not diagnosed.',
 ('WC',49):'PR #67 says "Fixes #49" and would auto-close it, but PR body excludes the multi-stage build the issue names.',
 ('WC',39):'PR #76 says "Closes #39", but both duplicate doc blocks remain at its head (verified earlier at a7321acc).',
 ('GP',150):'Verifier: PARTIAL. PR #157 says "Closes #150" but its test checks configuration, not a built wheel.',
 ('BM',124):'PR #135 says "Closes #124", but its own body says the transformers image is still built twice.',
 ('BM',120):'PR #134 says "Closes #120", but its body says two listed scorers still need a maintainer decision to load.',
 ('WD',179):'Closed-keyword claimed by two open PRs (#213 and #206).',
 ('WD',193):'Out of scope (Grid\'5000). PR #206 says "Closes #193" and would auto-close it.',
 ('GD',83):'PR #85 says "Closes #83". Audit: fsync path not tested (criterion 1 names writer or file fsync).',
 ('WD',206):'Out of scope item.',
 ('WD',213):'Also claimed by PR #206.',
 ('WT',120):'PR #153 is dirty (import-block conflict).',
 ('DT',137):'PR #145 is dirty.',
 ('GR',14):'Stale claim corrected: PR #13 merged 2026-10-08. Remaining gap is the direct full-run entry points.',
}

def prs_for(repo,n):
    op=openpr.get(repo,{})
    out=[]; seen=set()
    for pr in items[(repo,n)]['linked_prs']:
        seen.add(pr['number'])
    for num,x in op.items():
        if n in x['closes'] or n in x['refs']: seen.add(num)
    for num in sorted(seen):
        x=op.get(num)
        if x:
            kind='draft' if x['draft'] else 'open'
            rel=' closes' if n in x['closes'] else (' refs' if n in x['refs'] else '')
            mg='' if x['mergeable_state']=='clean' else ', '+x['mergeable_state']
            out.append('#%d %s%s%s'%(num,kind,rel,mg))
        else:
            pr=[p for p in items[(repo,n)]['linked_prs'] if p['number']==num][0]
            st='merged' if (pr['merged'] or 'MERGED' in pr['state'].upper()) and 'NOT MERGED' not in pr['state'].upper() else 'closed, unmerged'
            out.append('#%d %s'%(num,st))
    return out

def evidence(repo,n,cat):
    it=items[(repo,n)]
    if cat=='IMPLEMENTED': base=it['closure_evidence']
    elif cat=='COVERED': base='Close after merge.'
    elif cat in ('PARTIAL','UNCLAIMED_MAINTENANCE','DEFERRED'): base=it['remaining_requirement']
    elif cat=='UNCERTAIN': base=it['uncertainty']
    elif cat=='OWNED': base='Owned by existing golden audit; not analysed.'
    else: base='Out of scope; not analysed.'
    base=re.sub(r'\s+',' ',base or '').strip().replace('|','/')
    if len(base)>240: base=base[:239].rstrip()+'...'
    return base

def own_for(repo,n,cat):
    rk=SHORT[repo]
    if (rk,n) in OWN_OVR: return OWN_OVR[(rk,n)]
    if cat=='OUT_OF_SCOPE': return ('Out of scope','Grid\'5000 or excluded area')
    op=openpr.get(repo,{})
    inflight=[pr['number'] for pr in items[(repo,n)]['linked_prs'] if pr['number'] in op]
    if inflight and cat in ('COVERED','PARTIAL','UNCLAIMED_MAINTENANCE','DEFERRED','UNCERTAIN'):
        return ('In flight','open PR #%d'%inflight[0])
    if (rk,n) in RESEARCH: return ('Unowned','no GitHub claim, comment, PR or branch found')
    return ('Unknown','presumed Master 1 queue (maintenance); membership not confirmed')

rows={}; status_cnt=collections.OrderedDict(); own_cnt=collections.Counter(); total=0
warn=[]
for repo in ORDER:
    rk=SHORT[repo]
    rows[repo]=[]
    status_cnt[repo]=collections.Counter()
    for (r,n),it in sorted(((k,v) for k,v in items.items() if k[0]==repo), key=lambda kv: kv[0][1]):
        cat=cat_of(r,n)
        st=STATUS_OVR.get((rk,n), STATUS[cat])
        if cat=='COVERED' and not any(p in openpr.get(repo,{}) for p in [x['number'] for x in it['linked_prs']]):
            warn.append('COVERED without open linked PR: %s#%d'%(rk,n))
        cls,det=own_for(repo,n,cat)
        
        status_cnt[repo][st.split(';')[0]]+=1
        own_cnt[cls]+=1; total+=1
        note=NOTE.get((rk,n),'')
        ev=evidence(repo,n,cat)
        full=(ev+(' ' if ev and note else '')+('Note: '+note if note else '')).strip()
        title=re.sub(r'\s+',' ',it['title']).replace('|','/')
        if len(title)>90: title=title[:89].rstrip()+'...'
        prtxt=', '.join(prs_for(repo,n)) or '-'
        rows[repo].append('| [#%d](https://github.com/NoeFlandre/%s/issues/%d) | %s | %s | %s: %s | %s | %s |'%(n,repo,n,title,st,cls,det.replace('|','/'),prtxt,full))
print('TOTAL',total,'OWN',dict(own_cnt))
print('WARN',warn)
out=[]
out.append('ROWS_READY')
json.dump({'rows':rows,'own_cnt':dict(own_cnt),'status':{k:dict(v) for k,v in status_cnt.items()},'total':total,'warn':warn}, open(SP+'v2_rows.json','w'))
