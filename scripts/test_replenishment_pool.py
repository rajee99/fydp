import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prepare_experimental_manifests import (
    BALANCED_SEGMENTS_PATH, get_all_exclusion_mids, OTHER_SUBDOMAINS,
    PROCESSED_DIR, EVAL_MANIFEST_PATH, PILOT_MANIFEST_PATH
)

forbidden = set()
for p in [PROCESSED_DIR / 'e1_baseline_manifest.csv', PROCESSED_DIR / 'shared_val_split_manifest.csv']:
    if p.exists():
        df = pd.read_csv(p)
        forbidden.update(df['video_id'].dropna().str.strip())
if EVAL_MANIFEST_PATH.exists():
    df = pd.read_csv(EVAL_MANIFEST_PATH)
    forbidden.update(df['video_id'].dropna().str.strip())
if PILOT_MANIFEST_PATH.exists():
    df = pd.read_csv(PILOT_MANIFEST_PATH)
    forbidden.update(df['video_id'].dropna().str.strip())
exp_p = PROCESSED_DIR / 'targeted_expansion_candidates.csv'
if exp_p.exists():
    df = pd.read_csv(exp_p)
    forbidden.update(df['video_id'].dropna().str.strip())
e3_p = PROCESSED_DIR / 'e3_general_candidate_manifest.csv'
if e3_p.exists():
    df = pd.read_csv(e3_p)
    forbidden.update(df['video_id'].dropna().str.strip())

print(f'Total forbidden video IDs: {len(forbidden)}')

excl_mids = get_all_exclusion_mids()
sub_pools = {s: [] for s in OTHER_SUBDOMAINS}

with open(BALANCED_SEGMENTS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip().strip('"') for p in line.split(',')]
        if len(parts) < 4:
            continue
        vid = parts[0]
        start = float(parts[1])
        end = float(parts[2])
        labels = set([l.strip().strip('"') for l in parts[3:]])

        if vid in forbidden or labels.intersection(excl_mids):
            continue

        for sub, sub_mids in OTHER_SUBDOMAINS.items():
            matched = labels.intersection(set(sub_mids.keys()))
            if matched:
                sub_pools[sub].append(vid)

for sub, vids in sub_pools.items():
    print(f'{sub}: {len(set(vids))} available candidates')
