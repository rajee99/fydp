"""Phase 1 tasks 01.13/01.14: recover official metadata and retry blocked scripts.

Recovered upstream bytes are identified by URL, response metadata and hash;
they are not asserted to be the unavailable original local copies.
"""
from pathlib import Path
import csv, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, time
import urllib.request

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
OUT = AUDIT/'recovery_attempt_02'
OUT.mkdir(exist_ok=True)
SANDBOX = OUT/'sandbox'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, value): (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def snapshot():
    return {str(p.relative_to(ROOT)):sha(p) for folder in ('src','scripts','configs','data','models')
            for p in (ROOT/folder).rglob('*') if p.is_file()}
before=snapshot()
for folder in ('src','scripts','configs','data/processed'):
    shutil.copytree(ROOT/folder,SANDBOX/folder,dirs_exist_ok=True)
(SANDBOX/'reports').mkdir(exist_ok=True)
raw=SANDBOX/'data/raw'; raw.mkdir(parents=True,exist_ok=True)
sources={
 'yamnet_class_map.csv':'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv',
 **{name:f'https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{name}' for name in
    ('balanced_train_segments.csv','eval_segments.csv','unbalanced_train_segments.csv')}}
downloads=[]
for name,url in sources.items():
    item={'file':name,'url':url,'retrieved_date':'2026-11-07','identity':'official_upstream_recovery; original_local_hash_unknown'}
    destination=raw/name
    try:
        if destination.exists():
            item['status']='cached_recovered_file'
        else:
            with urllib.request.urlopen(url,timeout=60) as response, destination.open('xb') as f:
                item['resolved_url']=response.url
                item['headers']=dict(response.headers)
                shutil.copyfileobj(response,f)
            item['status']='downloaded'
        item['bytes']=destination.stat().st_size; item['sha256']=sha(destination)
    except Exception as e:
        item['status']='failed'; item['error']=repr(e)
    downloads.append(item); save('download_register.json',downloads)
    print(name,item['status'],item.get('bytes',''),flush=True)
save('environment.json',{'python':sys.version,'executable':sys.executable,
     'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}})
freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
(OUT/'requirements_resolved.txt').write_text(freeze.stdout+freeze.stderr,encoding='utf-8')
jobs=['src/prepare_manifest.py','src/prepare_eval_manifest.py','scripts/prepare_test_set_v2_candidates.py',
      'src/prepare_combined_dataset.py','src/prepare_experimental_manifests.py']
results=[]
for job in jobs:
    prior={str(p.relative_to(SANDBOX)):sha(p) for folder in ('data/processed','configs','reports') for p in (SANDBOX/folder).rglob('*') if p.is_file()}
    cmd=[sys.executable,str(SANDBOX/job)]
    started=time.time()
    try:
        r=subprocess.run(cmd,cwd=SANDBOX,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180,
                         env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'))
        code=r.returncode; output=r.stdout+'\n'+r.stderr
    except subprocess.TimeoutExpired as e:
        code='timeout'; output=repr(e)
    log=Path(job).stem+'.txt'
    (OUT/log).write_text('Date: 2026-11-07 Asia/Dhaka\nCommand: '+repr(cmd)+'\n'+output,encoding='utf-8')
    changed=[]
    for folder in ('data/processed','configs','reports'):
        for p in (SANDBOX/folder).rglob('*'):
            if not p.is_file(): continue
            rel=str(p.relative_to(SANDBOX)); h=sha(p)
            if prior.get(rel)==h: continue
            record={'path':rel,'sha256':h,'baseline_exists':(ROOT/rel).exists()}
            if (ROOT/rel).exists(): record['byte_identical_to_baseline']=h==sha(ROOT/rel)
            if p.suffix=='.csv':
                with p.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
                record['rows']=len(rows)
                from collections import Counter
                record['class_counts']=dict(Counter(row.get('target_class','unknown') for row in rows))
            changed.append(record)
    # A byte-identical regenerated file still counts as an output; compare expected main outputs explicitly.
    main={'src/prepare_manifest.py':'target_manifest.csv','src/prepare_eval_manifest.py':'eval_manifest.csv',
          'scripts/prepare_test_set_v2_candidates.py':'test_set_v2_candidates_to_download.csv',
          'src/prepare_combined_dataset.py':'combined_training_manifest.csv'}
    expected=main.get(job)
    if code==0 and expected:
        p=SANDBOX/'data/processed'/expected
        if not any(x['path']==str(p.relative_to(SANDBOX)) for x in changed):
            with p.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
            from collections import Counter
            changed.append({'path':str(p.relative_to(SANDBOX)),'sha256':sha(p),'rows':len(rows),
                'class_counts':dict(Counter(row.get('target_class','unknown') for row in rows)),
                'byte_identical_to_baseline':sha(p)==sha(ROOT/'data/processed'/expected)})
    results.append({'script':job,'script_sha256':sha(ROOT/job),'command':cmd,'exit_code':code,'elapsed_seconds':round(time.time()-started,2),'log':log,'outputs':changed})
    save('execution_register.json',results)
    print(job,'exit',code,flush=True)
after=snapshot()
save('baseline_preservation.json',{'audited_files':len(before),'changed_files':[k for k,v in before.items() if after.get(k)!=v],
                                'new_baseline_files':sorted(set(after)-set(before))})
assert before==after,'Baseline changed'
