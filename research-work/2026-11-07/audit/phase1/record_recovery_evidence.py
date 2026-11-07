"""Summarize current Phase 1 recovery outputs without running any generators."""
from pathlib import Path
from collections import Counter
import csv, hashlib, importlib.metadata, json, shutil, sys

AUDIT=Path(__file__).resolve().parent;ROOT=AUDIT.parents[3]
OUT=AUDIT/'recovery_attempt_02';BOX=OUT/'sandbox'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
names=['target_manifest.csv','eval_manifest.csv','test_set_v2_candidates_to_download.csv',
       'shared_val_split_manifest.csv','e1_baseline_manifest.csv','e1_train_split_manifest.csv','e1_val_split_manifest.csv',
       'e2_targeted_manifest.csv','e2_train_split_manifest.csv','e2_val_split_manifest.csv',
       'e3_general_candidate_manifest.csv','e3_general_train_candidates.csv','e3_general_val_candidates.csv','e3_candidates_to_download.csv']
paths=[BOX/'data/processed'/name for name in names]
paths+=list((BOX/'data/processed/e4_ablations').glob('*.csv'))
output=[]
for p in paths:
    if not p.exists():continue
    rel=p.relative_to(BOX); a=ROOT/rel
    rr=rows(p)
    item={'path':str(rel),'rows':len(rr),'class_counts':dict(Counter(r.get('target_class','unknown') for r in rr)),
          'sha256':sha(p),'baseline_exists':a.exists()}
    if a.exists():item.update(baseline_rows=len(rows(a)),byte_identical_to_baseline=sha(p)==sha(a),parsed_rows_identical=rr==rows(a))
    destination=OUT/'generated_manifests'/p.relative_to(BOX/'data/processed')
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
    output.append(item)
for rel in ('data/processed/experimental_manifests_summary.json','configs/e4_ablation_config.json'):
    p=BOX/rel
    if p.exists():
        dest=OUT/'generated_artifacts'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
(OUT/'generated_output_register.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
originals=rows(AUDIT/'file_inventory.csv')
changes=[r['path'] for r in originals if not (ROOT/r['path']).exists() or sha(ROOT/r['path'])!=r['sha256']]
(OUT/'baseline_preservation.json').write_text(json.dumps({'audited_files':len(originals),'changed_files':changes},indent=2),encoding='utf-8')
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,
    'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}},indent=2),encoding='utf-8')
scripts=['recover_and_retry.py','probe_audio_recovery.py','recover_required_audio.py','recover_nigens.py','compare_recovered_manifests.py','record_recovery_evidence.py']
(OUT/'audit_helper_versions.json').write_text(json.dumps({n:sha(AUDIT/n) for n in scripts},indent=2),encoding='utf-8')
assert not changes,changes
print('Saved',len(output),'output summaries; baseline changes:',len(changes))
