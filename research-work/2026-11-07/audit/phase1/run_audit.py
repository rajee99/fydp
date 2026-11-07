"""Read-only repository audit; execute existing preparation scripts in an isolated copy."""
from pathlib import Path
import collections, csv, hashlib, importlib.util, json, os, re, shutil, subprocess, sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(name, rows, fields):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()

files = sorted(p for folder in ('src','scripts','data','models','configs','experiments')
               for p in (ROOT / folder).rglob('*') if p.is_file())
before = {str(p.relative_to(ROOT)): digest(p) for p in files}
write_csv('file_inventory.csv', [{'path': k, 'bytes': (ROOT/k).stat().st_size, 'sha256': v}
          for k,v in before.items()], ['path','bytes','sha256'])
counts = []
for p in sorted((ROOT/'data/processed').rglob('*.csv')):
    with p.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    label = next((k for k in ('target_class','class_name','true_label') if rows and k in rows[0]), None)
    counts.append({'path': str(p.relative_to(ROOT)), 'rows': len(rows),
                   'label_column': label, 'class_counts': dict(collections.Counter(r.get(label,'') for r in rows)) if label else {}})
(OUT/'manifest_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
patterns = {'dataset_references': r'validation|evaluation|test|holdout|benchmark',
            'evaluation_training_review': r'eval_|test_|threshold|early_stopping|best_val',
            'csv_lineage': r'\.csv|read_csv|DictReader',
            'metric_producers': r'macro_f1|classification_report|confusion_matrix|false.emergency'}
for name, pattern in patterns.items():
    hits=[]
    for p in files:
        if p.suffix != '.py': continue
        for n,line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
            if re.search(pattern,line,re.I): hits.append({'path':str(p.relative_to(ROOT)), 'line':n, 'text':line.strip()})
    write_csv(name+'.csv',hits,['path','line','text'])
sandbox=OUT/'reproduction_sandbox'
sandbox.mkdir(exist_ok=True)
for folder in ('src','scripts','configs','data/processed'):
    shutil.copytree(ROOT/folder,sandbox/folder,dirs_exist_ok=True)
(sandbox/'reports').mkdir(exist_ok=True)
jobs = ['src/prepare_manifest.py','src/prepare_eval_manifest.py',
        'src/prepare_combined_dataset.py','src/prepare_experimental_manifests.py',
        'src/build_matched_e2_manifest.py','src/split_combined_dataset.py',
        'scripts/prepare_test_set_v2_candidates.py']
results=[]
env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
for job in jobs:
    args=[sys.executable, str(sandbox/job)]
    try:
        r=subprocess.run(args,cwd=sandbox,text=True,encoding='utf-8',errors='replace',capture_output=True,timeout=45,env=env)
        code=r.returncode; output=r.stdout+'\n'+r.stderr
    except subprocess.TimeoutExpired as e:
        code='timeout'; output=str(e)
    log=Path(job).stem+'.txt'
    (OUT/log).write_text('Command: '+repr(args)+'\n'+output,encoding='utf-8')
    results.append({'script':job,'exit_code':code,'log':log,'script_sha256':digest(ROOT/job)})
    print(job,code,flush=True)
comparisons=[]
for name in ('e2_matched_train_split_manifest.csv','e2_matched_val_split_manifest.csv','e2_matched_manifest.csv',
             'combined_train_split_manifest.csv','combined_val_split_manifest.csv'):
    a=ROOT/'data/processed'/name; b=sandbox/'data/processed'/name
    if a.exists() and b.exists():
        with a.open(encoding='utf-8-sig',newline='') as f: ar=list(csv.DictReader(f))
        with b.open(encoding='utf-8-sig',newline='') as f: br=list(csv.DictReader(f))
        comparisons.append({'file':name,'baseline_rows':len(ar),'reproduced_rows':len(br),'byte_identical':digest(a)==digest(b),'parsed_rows_identical':ar==br})
changed=[k for k,v in before.items() if not (ROOT/k).exists() or digest(ROOT/k)!=v]
summary={'date':'2026-11-07','timezone':'Asia/Dhaka','commit':git('rev-parse','HEAD'),
 'branch':git('branch','--show-current'),'remote':git('remote','get-url','origin'),
 'python':sys.version,'executable':sys.executable,
 'dependencies':{x:importlib.util.find_spec(x) is not None for x in ('numpy','pandas','soundfile','torch','torchaudio')},
 'raw_directory_exists':(ROOT/'data/raw').exists(), 'audio_files':sum(p.suffix.lower() in ('.wav','.mp3','.flac') for p in files),
 'weight_files':sum(p.suffix.lower() in ('.pt','.pth') for p in files),
 'inventory_files':len(files),'root_experiments_directory_exists':(ROOT/'experiments').exists(),
 'runs':results,'comparisons':comparisons,'baseline_files_changed':changed}
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
assert not changed, changed
print(json.dumps(summary,indent=2))
