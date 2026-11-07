"""Sample Replacement Candidates for E3 General Background Audio.

Samples exactly 52 replacement candidates from AudioSet balanced training data
to replenish the 52-sample deficit (172 accepted vs 224 target) following content review.

Guarantees:
- Seed: 43
- Source: AudioSet balanced_train_segments.csv
- Exclusion: All emergency MIDs and acoustic confusers (via get_all_exclusion_mids()).
- Isolation: Strict exclusion of E1 baseline (422 train, 105 shared val),
  held-out evaluation (744/519), pilot (15), targeted expansion (672),
  and initial E3 candidates (224). Total forbidden video IDs > 2,100.
- Subdomain distribution: Matched to deficit to restore exact balanced targets:
  household (13), animals (10), music (10), speech (8), traffic (7), environmental (4).
"""

from datetime import datetime
from pathlib import Path
import random
import sys
from typing import Dict, List, Set

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import CLASS_TO_ID
from src.prepare_experimental_manifests import (
    BALANCED_SEGMENTS_PATH,
    OTHER_SUBDOMAINS,
    PROCESSED_DIR,
    EVAL_MANIFEST_PATH,
    PILOT_MANIFEST_PATH,
    get_all_exclusion_mids,
)

REPLACEMENT_MANIFEST_PATH = PROCESSED_DIR / "e3_replacement_candidates.csv"
DEST_AUDIO_DIR = ROOT / "data" / "raw" / "e3_replacements" / "other"


def collect_forbidden_video_ids() -> Set[str]:
    """Collect all historically reserved video IDs across all project splits."""
    forbidden = set()

    # 1. E1 Baseline & Shared Val
    for p in [PROCESSED_DIR / "e1_baseline_manifest.csv", PROCESSED_DIR / "shared_val_split_manifest.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            forbidden.update(df["video_id"].dropna().astype(str).str.strip())

    # 2. Held-out Evaluation Set
    if EVAL_MANIFEST_PATH.exists():
        df = pd.read_csv(EVAL_MANIFEST_PATH)
        forbidden.update(df["video_id"].dropna().astype(str).str.strip())

    # 3. Pilot Audio
    if PILOT_MANIFEST_PATH.exists():
        df = pd.read_csv(PILOT_MANIFEST_PATH)
        if "video_id" in df.columns:
            forbidden.update(df["video_id"].dropna().astype(str).str.strip())

    # 4. Targeted Expansion (P1, P2, P3 candidates)
    exp_p = PROCESSED_DIR / "targeted_expansion_candidates.csv"
    if exp_p.exists():
        df = pd.read_csv(exp_p)
        forbidden.update(df["video_id"].dropna().astype(str).str.strip())

    # 5. Prior E3 Candidates (all 224 original candidates)
    e3_p = PROCESSED_DIR / "e3_general_candidate_manifest.csv"
    if e3_p.exists():
        df = pd.read_csv(e3_p)
        forbidden.update(df["video_id"].dropna().astype(str).str.strip())

    return forbidden


def sample_replacements(seed: int = 43) -> pd.DataFrame:
    """Sample exactly 52 replacement candidates across the 6 subdomains."""
    forbidden = collect_forbidden_video_ids()
    print(f"Total forbidden video IDs across all historical splits: {len(forbidden)}")

    excl_mids = get_all_exclusion_mids()
    rng = random.Random(seed)

    # Subdomain deficit targets
    sub_deficits = {
        "household": 13,
        "animals": 10,
        "music": 10,
        "speech": 8,
        "traffic": 7,
        "environmental": 4,
    }
    target_count = sum(sub_deficits.values())  # 52

    sub_candidates = {s: [] for s in OTHER_SUBDOMAINS}

    with open(BALANCED_SEGMENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip().strip('"') for p in line.split(",")]
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
                    first_mid = sorted(matched)[0]
                    sub_candidates[sub].append({
                        "video_id": vid,
                        "start_seconds": start,
                        "end_seconds": end,
                        "target_class": "other",
                        "class_id": CLASS_TO_ID["other"],
                        "subdomain": sub,
                        "audio_set_mid": first_mid,
                        "matched_label": sub_mids[first_mid],
                        "source_dataset": "audioset_general_control",
                        "origin": "balanced_train_segments_seed43",
                        "candidate_priority": "E3_REPLACEMENT",
                        "split": "train",
                    })

    selected = []
    used_vids = set()

    for sub in sorted(sub_deficits.keys()):
        needed = sub_deficits[sub]
        pool = [c for c in sub_candidates[sub] if c["video_id"] not in used_vids]
        pool.sort(key=lambda x: str(x["video_id"]))
        rng.shuffle(pool)

        if len(pool) < needed:
            raise ValueError(f"Insufficient candidates for {sub}: needed {needed}, available {len(pool)}")

        chosen = pool[:needed]
        for c in chosen:
            used_vids.add(c["video_id"])
            fname = f"other_{c['video_id']}_{int(c['start_seconds'])}_{int(c['end_seconds'])}.wav"
            c["filename"] = fname
            c["relative_path"] = f"data/raw/e3_replacements/other/{fname}"
            selected.append(c)

    df = pd.DataFrame(selected)

    # Verification checks
    assert len(df) == target_count, f"Expected {target_count}, got {len(df)}"
    assert len(df["video_id"].unique()) == target_count, "Duplicate video IDs detected in replacement selection!"
    overlap = set(df["video_id"]).intersection(forbidden)
    assert len(overlap) == 0, f"Source-level leakage! {len(overlap)} IDs overlap with forbidden sets: {overlap}"

    print(f"\nSuccessfully sampled {len(df)} replacements with Seed {seed}:")
    for sub, count in df["subdomain"].value_counts().items():
        print(f"  - {sub:<16}: {count} candidates")

    cols = [
        "video_id", "start_seconds", "end_seconds", "target_class", "class_id",
        "relative_path", "filename", "subdomain", "matched_label", "audio_set_mid",
        "source_dataset", "origin", "candidate_priority", "split"
    ]
    df = df[cols]
    df.to_csv(REPLACEMENT_MANIFEST_PATH, index=False, encoding="utf-8")
    print(f"\nSaved replacement manifest to: {REPLACEMENT_MANIFEST_PATH}")
    return df


if __name__ == "__main__":
    sample_replacements(seed=43)
