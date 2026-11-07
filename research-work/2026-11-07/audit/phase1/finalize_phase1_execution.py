"""Phase 1 Final Execution and Evidence Registration Script.

Executes all five preparation scripts in the isolated recovery sandbox:
1. src/prepare_manifest.py
2. src/prepare_eval_manifest.py
3. src/prepare_combined_dataset.py
4. src/prepare_experimental_manifests.py
5. scripts/prepare_test_set_v2_candidates.py

Captures execution commands, logs, exit codes, outputs, row counts,
class distributions, manifest comparisons, and validates 100% baseline preservation.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
OUT = AUDIT / 'recovery_attempt_02'
OUT.mkdir(exist_ok=True)
SANDBOX = OUT / 'sandbox'
GEN_MANIFESTS = OUT / 'generated_manifests'
GEN_ARTIFACTS = OUT / 'generated_artifacts'
GEN_MANIFESTS.mkdir(exist_ok=True)
GEN_ARTIFACTS.mkdir(exist_ok=True)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def read_csv_rows(p: Path):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

# 1. Before snapshot of baseline 949 files
inventory_file = AUDIT / 'file_inventory.csv'
inventory = read_csv_rows(inventory_file)
before_changes = []
for r in inventory:
    fp = ROOT / r['path']
    if not fp.exists() or sha256_file(fp) != r['sha256']:
        before_changes.append(r['path'])
assert not before_changes, f"Baseline dirty before run: {before_changes}"
print(f"[Check 1] Baseline verified intact: {len(inventory)} files match SHA-256.", flush=True)

# 2. Run all five scripts in sequence in sandbox
jobs = [
    ('src/prepare_manifest.py', 'prepare_manifest.txt', 'target_manifest.csv'),
    ('src/prepare_eval_manifest.py', 'prepare_eval_manifest.txt', 'eval_manifest.csv'),
    ('src/prepare_combined_dataset.py', 'prepare_combined_dataset.txt', 'combined_training_manifest.csv'),
    ('src/prepare_experimental_manifests.py', 'prepare_experimental_manifests.txt', 'e1_baseline_manifest.csv'),
    ('scripts/prepare_test_set_v2_candidates.py', 'prepare_test_set_v2_candidates.txt', 'test_set_v2_candidates_to_download.csv'),
]

execution_records = []

for rel_script, log_name, expected_main in jobs:
    script_path = SANDBOX / rel_script
    assert script_path.exists(), f"Script missing in sandbox: {script_path}"
    script_hash = sha256_file(ROOT / rel_script)
    
    cmd = [sys.executable, str(script_path)]
    print(f"\n[Executing] {rel_script} ...", flush=True)
    started = time.time()
    
    proc = subprocess.run(
        cmd,
        cwd=str(SANDBOX),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=300,
        env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    )
    elapsed = round(time.time() - started, 2)
    output_log = (
        f"Date: 2026-11-07 Asia/Dhaka\n"
        f"Command: {cmd!r}\n"
        f"Exit code: {proc.returncode}\n"
        f"Elapsed seconds: {elapsed}\n"
        f"Script SHA-256: {script_hash}\n"
        f"----------------------------------------\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    (OUT / log_name).write_text(output_log, encoding='utf-8')
    print(f"  Exited with code {proc.returncode} in {elapsed}s. Log saved to {log_name}", flush=True)
    
    if proc.returncode != 0:
        raise RuntimeError(f"Script {rel_script} failed with exit code {proc.returncode}:\n{proc.stderr}")
    
    # Analyze generated output files
    expected_p = SANDBOX / 'data/processed' / expected_main
    outputs = []
    if expected_p.exists():
        rows = read_csv_rows(expected_p)
        classes = dict(Counter(r.get('target_class', 'unknown') for r in rows))
        h = sha256_file(expected_p)
        base_p = ROOT / 'data/processed' / expected_main
        outputs.append({
            'path': str(expected_p.relative_to(SANDBOX)),
            'sha256': h,
            'rows': len(rows),
            'class_counts': classes,
            'baseline_exists': base_p.exists(),
            'byte_identical_to_baseline': (h == sha256_file(base_p)) if base_p.exists() else False
        })
    
    execution_records.append({
        'script': rel_script,
        'script_sha256': script_hash,
        'command': cmd,
        'exit_code': proc.returncode,
        'elapsed_seconds': elapsed,
        'log': log_name,
        'outputs': outputs
    })

(OUT / 'execution_register.json').write_text(json.dumps(execution_records, indent=2), encoding='utf-8')

# 3. Collect all generated manifests and copy to generated_manifests / generated_artifacts
manifest_names = [
    'target_manifest.csv',
    'eval_manifest.csv',
    'combined_training_manifest.csv',
    'test_set_v2_candidates_to_download.csv',
    'shared_val_split_manifest.csv',
    'e1_baseline_manifest.csv',
    'e1_train_split_manifest.csv',
    'e1_val_split_manifest.csv',
    'e2_targeted_manifest.csv',
    'e2_train_split_manifest.csv',
    'e2_val_split_manifest.csv',
    'e3_general_candidate_manifest.csv',
    'e3_general_train_candidates.csv',
    'e3_general_val_candidates.csv',
    'e3_candidates_to_download.csv',
]

gen_output_register = []
for m_name in manifest_names:
    p = SANDBOX / 'data/processed' / m_name
    if not p.exists():
        continue
    rel = p.relative_to(SANDBOX)
    base_file = ROOT / rel
    rows = read_csv_rows(p)
    classes = dict(Counter(r.get('target_class', 'unknown') for r in rows))
    h = sha256_file(p)
    dest = GEN_MANIFESTS / m_name
    shutil.copy2(p, dest)
    
    item = {
        'path': str(rel),
        'rows': len(rows),
        'class_counts': classes,
        'sha256': h,
        'baseline_exists': base_file.exists(),
    }
    if base_file.exists():
        b_rows = read_csv_rows(base_file)
        b_classes = dict(Counter(r.get('target_class', 'unknown') for r in b_rows))
        b_hash = sha256_file(base_file)
        item.update({
            'baseline_rows': len(b_rows),
            'baseline_class_counts': b_classes,
            'baseline_sha256': b_hash,
            'byte_identical_to_baseline': (h == b_hash),
            'parsed_rows_identical': (rows == b_rows),
        })
    gen_output_register.append(item)

# Also collect e4 ablation CSVs
e4_dir = SANDBOX / 'data/processed/e4_ablations'
if e4_dir.exists():
    for p in sorted(e4_dir.glob('*.csv')):
        rel = p.relative_to(SANDBOX)
        base_file = ROOT / rel
        rows = read_csv_rows(p)
        classes = dict(Counter(r.get('target_class', 'unknown') for r in rows))
        h = sha256_file(p)
        dest = GEN_MANIFESTS / 'e4_ablations' / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        item = {
            'path': str(rel),
            'rows': len(rows),
            'class_counts': classes,
            'sha256': h,
            'baseline_exists': base_file.exists(),
        }
        if base_file.exists():
            b_rows = read_csv_rows(base_file)
            b_hash = sha256_file(base_file)
            item.update({
                'baseline_rows': len(b_rows),
                'byte_identical_to_baseline': (h == b_hash),
                'parsed_rows_identical': (rows == b_rows),
            })
        gen_output_register.append(item)

# Copy reports and json artifacts
for art_rel in [
    'data/processed/experimental_manifests_summary.json',
    'configs/e4_ablation_config.json',
    'reports/combined_training_dataset_report.md',
]:
    sp = SANDBOX / art_rel
    if sp.exists():
        dp = GEN_ARTIFACTS / art_rel
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dp)

(OUT / 'generated_output_register.json').write_text(json.dumps(gen_output_register, indent=2), encoding='utf-8')
print(f"[Check 2] Registered {len(gen_output_register)} generated manifests.", flush=True)

# 4. Detailed comparisons against baseline manifests
comparisons = []
for name in ['target_manifest.csv', 'eval_manifest.csv', 'combined_training_manifest.csv', 'test_set_v2_candidates_to_download.csv']:
    a = ROOT / 'data/processed' / name
    b = SANDBOX / 'data/processed' / name
    if not (a.exists() and b.exists()):
        continue
    ar, br = read_csv_rows(a), read_csv_rows(b)
    
    def key_fn(r):
        return (r.get('video_id') or r.get('sample_id'), r.get('start_seconds'), r.get('end_seconds'), r.get('target_class'))
    
    aset, bset = {key_fn(r) for r in ar}, {key_fn(r) for r in br}
    ae = {key_fn(r) for r in ar if r.get('target_class') != 'other'}
    be = {key_fn(r) for r in br if r.get('target_class') != 'other'}
    
    comp = {
        'file': name,
        'baseline_rows': len(ar),
        'generated_rows': len(br),
        'baseline_class_counts': dict(Counter(r.get('target_class') for r in ar)),
        'generated_class_counts': dict(Counter(r.get('target_class') for r in br)),
        'baseline_sha256': sha256_file(a),
        'generated_sha256': sha256_file(b),
        'parsed_rows_identical': (ar == br),
        'shared_segment_class_keys': len(aset & bset),
        'baseline_only_keys': len(aset - bset),
        'generated_only_keys': len(bset - aset),
        'emergency_segment_keys_equal': (ae == be),
    }
    comparisons.append(comp)

(OUT / 'manifest_comparisons.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
print(f"[Check 3] Detailed manifest comparisons recorded for 4 key manifests.", flush=True)

# 5. After verification of all 949 baseline files
after_changes = []
for r in inventory:
    fp = ROOT / r['path']
    if not fp.exists() or sha256_file(fp) != r['sha256']:
        after_changes.append(r['path'])

preservation = {
    'audited_files': len(inventory),
    'changed_files': after_changes,
    'status': 'PASS' if not after_changes else 'FAIL'
}
(OUT / 'baseline_preservation.json').write_text(json.dumps(preservation, indent=2), encoding='utf-8')
assert not after_changes, f"Baseline altered: {after_changes}"
print(f"[Check 4] Baseline post-execution integrity verified: 0 changed files out of {len(inventory)}.", flush=True)

# 6. Record environment
env_info = {
    'python': sys.version,
    'executable': sys.executable,
    'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
}
(OUT / 'environment.json').write_text(json.dumps(env_info, indent=2), encoding='utf-8')

print("\n[SUCCESS] Final Phase 1 execution and evidence registration COMPLETE!", flush=True)

