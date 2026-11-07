"""Build Matched-Size Primary E2 Manifest (212 Targeted Hard-Negatives).

Creates a controlled matched-size E2 dataset containing exactly 212 targeted
hard-negative additions (paired directly with E3's 212 general background additions),
leaving the existing full E2 manifest (224 targeted additions) completely intact.

Design:
- Documented Random Seed: 42
- Excluded: exactly 12 targeted additions chosen randomly from deterministic sort.
- Train split: 422 baseline train + 212 matched targeted additions = 634 samples.
- Val split: 105 samples (exact shared validation set).
- Master manifest: 739 samples.
"""

import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
SHARED_VAL_PATH = PROCESSED_DIR / "shared_val_split_manifest.csv"
E2_FULL_TRAIN_PATH = PROCESSED_DIR / "e2_train_split_manifest.csv"
E3_TRAIN_PATH = PROCESSED_DIR / "e3_train_split_manifest.csv"

E2_MATCHED_TRAIN_PATH = PROCESSED_DIR / "e2_matched_train_split_manifest.csv"
E2_MATCHED_VAL_PATH = PROCESSED_DIR / "e2_matched_val_split_manifest.csv"
E2_MATCHED_MASTER_PATH = PROCESSED_DIR / "e2_matched_manifest.csv"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_matched_e2(seed: int = 42):
    print("=" * 80)
    print("BUILDING MATCHED-SIZE E2 DATASET MANIFEST (N = 212 ADDITIONS)")
    print("=" * 80)

    # 1. Load full E2 train manifest and shared validation manifest
    df_e2_full = pd.read_csv(E2_FULL_TRAIN_PATH)
    df_shared_val = pd.read_csv(SHARED_VAL_PATH)
    df_e3_train = pd.read_csv(E3_TRAIN_PATH)

    baseline_train = df_e2_full[df_e2_full["source_dataset"] == "audioset"].copy()
    targeted_full = df_e2_full[df_e2_full["source_dataset"] == "audioset_targeted_expansion"].copy()

    assert len(baseline_train) == 422, f"Expected 422 baseline samples, got {len(baseline_train)}"
    assert len(targeted_full) == 224, f"Expected 224 targeted additions, got {len(targeted_full)}"

    # 2. Select 212 additions reproducibly using Seed 42
    target_count = 212
    num_to_exclude = len(targeted_full) - target_count  # 12

    rng = random.Random(seed)
    targeted_indices = list(targeted_full.index)
    targeted_indices.sort()  # deterministic base ordering
    rng.shuffle(targeted_indices)

    excluded_indices = set(targeted_indices[:num_to_exclude])
    excluded_df = targeted_full.loc[targeted_full.index.isin(excluded_indices)].copy()
    matched_targeted = targeted_full.loc[~targeted_full.index.isin(excluded_indices)].copy()

    assert len(matched_targeted) == target_count, f"Expected {target_count}, got {len(matched_targeted)}"
    assert len(excluded_df) == num_to_exclude, f"Expected {num_to_exclude}, got {len(excluded_df)}"

    print(f"Random seed: {seed}")
    print(f"Targeted additions in full E2   : {len(targeted_full)}")
    print(f"Targeted additions in matched E2: {len(matched_targeted)}")
    print(f"Excluded from matched selection : {len(excluded_df)}")
    print("\nExcluded video IDs and subdomains:")
    for _, r in excluded_df.iterrows():
        print(f"  - [{r['video_id']}] subdomain={r['subdomain']} filename={r['filename']}")

    # 3. Assemble E2 Matched Train Manifest
    matched_targeted["candidate_priority"] = "E2_MATCHED_PRIMARY"
    e2_matched_train = pd.concat([baseline_train, matched_targeted], ignore_index=True)
    e2_matched_train.sort_values(by=["target_class", "video_id"], inplace=True, ignore_index=True)

    assert len(e2_matched_train) == 634, f"Expected 634 train samples, got {len(e2_matched_train)}"
    assert e2_matched_train["video_id"].nunique() == 634, "Duplicate video IDs in matched E2 train!"

    # 4. Assemble E2 Matched Val Manifest (Shared Benchmark)
    e2_matched_val = df_shared_val.copy()
    e2_matched_val["split"] = "val"
    assert len(e2_matched_val) == 105, f"Expected 105 val samples, got {len(e2_matched_val)}"

    # 5. Assemble E2 Matched Master Manifest
    e2_matched_master = pd.concat([e2_matched_train, e2_matched_val], ignore_index=True)
    assert len(e2_matched_master) == 739, f"Expected 739 total samples, got {len(e2_matched_master)}"

    # 6. Verify identical sizes and baseline samples between E2 Matched and E3
    print("\n--- Comparative Verification: E2 Matched vs E3 General ---")
    print(f"E2 Matched Train Samples: {len(e2_matched_train)}")
    print(f"E3 General Train Samples: {len(df_e3_train)}")
    assert len(e2_matched_train) == len(df_e3_train) == 634, "Size mismatch between E2 matched and E3!"

    # Verify identical 422 baseline train samples
    e2_base_vids = set(baseline_train["video_id"])
    e3_base_vids = set(df_e3_train[df_e3_train["source_dataset"] == "audioset"]["video_id"])
    assert e2_base_vids == e3_base_vids, "Baseline train samples differ between E2 matched and E3!"
    print(f"  [PASS] Both datasets share the identical 422 baseline training video IDs.")

    # Verify identical 105 shared validation samples
    assert set(e2_matched_val["video_id"]) == set(df_shared_val["video_id"]), "Validation set mismatch!"
    print(f"  [PASS] Both datasets evaluate on the identical 105-sample shared validation set.")

    # Verify zero train/val overlap
    assert len(set(e2_matched_train["video_id"]).intersection(set(e2_matched_val["video_id"]))) == 0
    print(f"  [PASS] Zero train/validation video ID overlap in E2 matched.")

    # 7. Write manifests
    e2_matched_train.to_csv(E2_MATCHED_TRAIN_PATH, index=False, encoding="utf-8")
    e2_matched_val.to_csv(E2_MATCHED_VAL_PATH, index=False, encoding="utf-8")
    e2_matched_master.to_csv(E2_MATCHED_MASTER_PATH, index=False, encoding="utf-8")

    print(f"\nSaved E2 Matched Train Manifest:  {E2_MATCHED_TRAIN_PATH.name} ({len(e2_matched_train)} rows)")
    print(f"Saved E2 Matched Val Manifest:    {E2_MATCHED_VAL_PATH.name} ({len(e2_matched_val)} rows)")
    print(f"Saved E2 Matched Master Manifest: {E2_MATCHED_MASTER_PATH.name} ({len(e2_matched_master)} rows)")

    # 8. Record audit log
    audit_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_seed": seed,
        "full_e2_targeted_additions": 224,
        "matched_e2_targeted_additions": 212,
        "e3_general_background_additions": 212,
        "baseline_train_samples": 422,
        "shared_validation_samples": 105,
        "matched_train_total": 634,
        "matched_master_total": 739,
        "excluded_from_matched": [
            {"video_id": r["video_id"], "subdomain": r["subdomain"], "filename": r["filename"]}
            for _, r in excluded_df.iterrows()
        ],
        "file_hashes": {
            "e2_matched_train_split_manifest.csv": sha256_file(E2_MATCHED_TRAIN_PATH),
            "e2_matched_val_split_manifest.csv": sha256_file(E2_MATCHED_VAL_PATH),
            "e2_matched_manifest.csv": sha256_file(E2_MATCHED_MASTER_PATH),
            "e3_train_split_manifest.csv": sha256_file(E3_TRAIN_PATH),
            "shared_val_split_manifest.csv": sha256_file(SHARED_VAL_PATH),
        }
    }

    audit_path = PROCESSED_DIR / "e2_matched_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved matched audit record to: {audit_path.name}")


if __name__ == "__main__":
    build_matched_e2(seed=42)
