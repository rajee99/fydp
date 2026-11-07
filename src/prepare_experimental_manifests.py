"""Experimental Manifests and Reproducible Partitioning Pipeline.

Implements the experimental dataset preparation for the research question:
"Does model-error-guided hard-negative data selection improve emergency audio
classification more effectively than simply increasing the amount of general
background audio?"

Shared Validation Design:
  - Shared Validation Benchmark: Exactly the 105 baseline validation samples
    (fixed and identical across E1, E2, E3, and all E4 ablations).
  - E1 (Baseline): Original verified 527 AudioSet clips (422 train / 105 val).
  - E2 (Targeted Hard-Negatives): Baseline 422 train + 224 validated hard negatives = 646 train / 105 val.
  - E3 (General Background Audio): Baseline 422 train + 224 candidate general samples = 646 train / 105 val.
  - E4 (Subdomain Ablations): Each ablation uses the shared 105 validation samples,
    with individual hard-negative subdomains removed from the 646 training samples.

Integrity and reproducibility guarantees:
  - Exclusion of NEEDS_SECOND_REVIEW recording (ja-Qxq5NAoE).
  - Final target class 'other' for all 4 RECLASSIFY recordings.
  - Zero video ID overlap between train and validation splits (source-grouped by YouTube ID).
  - Zero overlap with independent evaluation set (519 audio files / 744 manifest rows).
  - Zero overlap with pilot audio (15 clips).
  - Seed 42 for all deterministic sampling and shuffling.
  - Preservation of exact baseline split assignments for all 527 baseline samples.
"""

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import soundfile as sf

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

try:
    from class_config import (
        CLASSES,
        CLASS_TO_ID,
        ID_TO_CLASS,
        OTHER_SUBDOMAINS,
        get_all_exclusion_mids,
    )
except ImportError:
    from src.class_config import (
        CLASSES,
        CLASS_TO_ID,
        ID_TO_CLASS,
        OTHER_SUBDOMAINS,
        get_all_exclusion_mids,
    )

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
CONFIGS_DIR = ROOT / "configs"
E4_DIR = PROCESSED_DIR / "e4_ablations"

TARGET_MANIFEST_PATH = PROCESSED_DIR / "target_manifest.csv"
TRAIN_SPLIT_MANIFEST_PATH = PROCESSED_DIR / "train_split_manifest.csv"
VAL_SPLIT_MANIFEST_PATH = PROCESSED_DIR / "val_split_manifest.csv"
EVAL_MANIFEST_PATH = PROCESSED_DIR / "eval_manifest.csv"
PILOT_MANIFEST_PATH = PROCESSED_DIR / "pilot_manifest.csv"
REVIEW_RESULTS_PATH = PROCESSED_DIR / "targeted_expansion_review_results.csv"
BALANCED_SEGMENTS_PATH = RAW_DIR / "balanced_train_segments.csv"
UNBALANCED_SEGMENTS_PATH = RAW_DIR / "unbalanced_train_segments.csv"


def resolve_review_status(
    review_results_path: Path = REVIEW_RESULTS_PATH,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Resolve human review decisions and extract the 224 validated usable other recordings.

    Returns:
        usable_other_records: Exactly 224 validated records with target_class='other'.
        review_audit_metadata: Summary of decisions, exclusions, and reclassifications.
    """
    if not review_results_path.exists():
        raise FileNotFoundError(f"Review results CSV not found: {review_results_path}")

    df = pd.read_csv(review_results_path)

    # 1. Audit decisions
    verdict_counts = df["verdict"].value_counts().to_dict()

    # 2. Identify and strictly exclude NEEDS_SECOND_REVIEW
    nsr_df = df[df["verdict"] == "NEEDS_SECOND_REVIEW"]
    nsr_vids = nsr_df["video_id"].tolist()
    if "ja-Qxq5NAoE" not in nsr_vids:
        raise ValueError("Expected ja-Qxq5NAoE to be in NEEDS_SECOND_REVIEW list.")

    # 3. Identify REJECT
    reject_df = df[df["verdict"] == "REJECT"]

    # 4. Identify RECLASSIFY
    reclass_df = df[df["verdict"] == "RECLASSIFY"]
    expected_reclass_vids = {"ad-8X58aIEM", "Pdau8nsBqxs", "d4v3_z0ISrM", "I1TWWYbjvJ8"}
    found_reclass_vids = set(reclass_df["video_id"])
    if found_reclass_vids != expected_reclass_vids:
        raise ValueError(
            f"RECLASSIFY video ID mismatch. Expected {expected_reclass_vids}, found {found_reclass_vids}"
        )

    # 5. Extract usable other recordings
    # 220 directly accepted 'other'
    accepted_other = df[(df["target_class"] == "other") & (df["verdict"] == "ACCEPT")].copy()
    if len(accepted_other) != 220:
        raise ValueError(f"Expected 220 ACCEPT other records, found {len(accepted_other)}")

    # 4 reclassified recordings -> mapped to 'other'
    reclass_subdomain_map = {
        "ad-8X58aIEM": "human_shouting",
        "Pdau8nsBqxs": "human_shouting",
        "I1TWWYbjvJ8": "door_slams",
        "d4v3_z0ISrM": "other_vocal",
    }

    reclass_records = []
    for _, row in reclass_df.iterrows():
        vid = row["video_id"]
        sub = reclass_subdomain_map.get(vid, "other")
        rec = dict(row)
        rec["original_target_class"] = row["target_class"]
        rec["target_class"] = "other"
        rec["original_subdomain"] = row["subdomain"]
        rec["subdomain"] = sub
        rec["class_id"] = CLASS_TO_ID["other"]
        reclass_records.append(rec)

    accepted_records = []
    for _, row in accepted_other.iterrows():
        rec = dict(row)
        rec["original_target_class"] = "other"
        rec["target_class"] = "other"
        rec["original_subdomain"] = row["subdomain"]
        rec["class_id"] = CLASS_TO_ID["other"]
        accepted_records.append(rec)

    usable_records = accepted_records + reclass_records

    if len(usable_records) != 224:
        raise ValueError(f"Expected exactly 224 usable other records, got {len(usable_records)}")

    # Verify no screams or glass_breaking are in usable_records
    for r in usable_records:
        if r["target_class"] != "other":
            raise ValueError(f"Sample {r['video_id']} has target_class '{r['target_class']}', expected 'other'")
        if r["video_id"] in nsr_vids:
            raise ValueError(f"Sample {r['video_id']} is marked NEEDS_SECOND_REVIEW but found in usable records!")

    # Verify audio properties on disk
    for r in usable_records:
        p = Path(r["file_path"])
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Usable audio file not found on disk: {p}")
        info = sf.info(str(p))
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.duration <= 0:
            raise ValueError(f"Invalid audio format for {p}: {info}")
        r["relative_path"] = Path(r["file_path"]).as_posix()
        r["filename"] = p.name

    review_audit_metadata = {
        "total_reviewed": len(df),
        "verdict_counts": verdict_counts,
        "needs_second_review": nsr_vids,
        "reclassified": {r["video_id"]: {"from": r["original_target_class"], "to": r["target_class"], "subdomain": r["subdomain"]} for r in reclass_records},
        "rejected_count": len(reject_df),
        "usable_targeted_other_count": len(usable_records),
    }

    return usable_records, review_audit_metadata


def build_e1_baseline_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build E1 (Baseline) manifests preserving exact 527 samples and baseline splits.

    Returns:
        (e1_master_df, e1_train_df, e1_val_df)
    """
    train_df = pd.read_csv(TRAIN_SPLIT_MANIFEST_PATH)
    val_df = pd.read_csv(VAL_SPLIT_MANIFEST_PATH)

    train_df["split"] = "train"
    val_df["split"] = "val"

    # Add source dataset and subdomain
    other_sub_map = {}
    if (PROCESSED_DIR / "other_manifest.csv").exists():
        om_df = pd.read_csv(PROCESSED_DIR / "other_manifest.csv")
        for _, row in om_df.iterrows():
            other_sub_map[row["video_id"]] = row.get("subdomain", "general")

    for df in [train_df, val_df]:
        df["source_dataset"] = "audioset"
        df["origin"] = "baseline_audioset"
        subdomains = []
        for _, row in df.iterrows():
            cname = row["target_class"]
            vid = row["video_id"]
            if cname == "other":
                subdomains.append(other_sub_map.get(vid, "general"))
            else:
                subdomains.append(cname)
        df["subdomain"] = subdomains

    e1_master_df = pd.concat([train_df, val_df], ignore_index=True)

    if len(e1_master_df) != 527:
        raise ValueError(f"Expected 527 baseline samples, got {len(e1_master_df)}")
    if len(train_df) != 422:
        raise ValueError(f"Expected 422 baseline train samples, got {len(train_df)}")
    if len(val_df) != 105:
        raise ValueError(f"Expected 105 baseline val samples, got {len(val_df)}")

    return e1_master_df, train_df, val_df


def build_e2_targeted_dataset(
    e1_train_df: pd.DataFrame,
    shared_val_df: pd.DataFrame,
    usable_other_records: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build E2 (Targeted Hard-Negatives) manifests using shared validation set.

    Design:
      - Train split: 422 baseline train + 224 validated targeted hard-negatives = 646 samples.
      - Val split: 105 samples (exact shared validation set).
      - Master: 751 samples.

    Returns:
        (e2_master_df, e2_train_df, e2_val_df)
    """
    cols = ["video_id", "target_class", "class_id", "relative_path", "filename", "subdomain", "source_dataset", "origin", "split"]

    train_additions = []
    for r in usable_other_records:
        r_copy = dict(r)
        r_copy["split"] = "train"
        r_copy["source_dataset"] = "audioset_targeted_expansion"
        r_copy["origin"] = r.get("origin", "targeted_expansion")
        train_additions.append(r_copy)

    # Sort additions deterministically by video ID
    train_additions.sort(key=lambda x: str(x["video_id"]))
    df_train_add = pd.DataFrame(train_additions)[cols]

    e1_train_aligned = e1_train_df[cols].copy()
    e2_val_df = shared_val_df[cols].copy()

    e2_train_df = pd.concat([e1_train_aligned, df_train_add], ignore_index=True)
    e2_train_df.sort_values(by=["target_class", "video_id"], inplace=True, ignore_index=True)

    e2_master_df = pd.concat([e2_train_df, e2_val_df], ignore_index=True)

    if len(e2_master_df) != 751:
        raise ValueError(f"Expected 751 E2 samples (527 + 224), got {len(e2_master_df)}")
    if len(e2_train_df) != 646:
        raise ValueError(f"Expected 646 E2 train samples (422 + 224), got {len(e2_train_df)}")
    if len(e2_val_df) != 105:
        raise ValueError(f"Expected 105 E2 val samples, got {len(e2_val_df)}")

    return e2_master_df, e2_train_df, e2_val_df


def sample_e3_general_candidates(
    forbidden_video_ids: Set[str],
    target_count: int = 224,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample exactly 224 general background candidates from AudioSet balanced segments.

    Enforces:
      - 6 non-target subdomains: speech, music, animals, household, traffic, environmental.
      - Equal distribution across subdomains (37 in 4 subdomains, 38 in 2 subdomains).
      - Strict exclusion of all emergency MIDs and confusers.
      - Zero overlap with forbidden video IDs.
      - Reproducible with seed 42.
    """
    if not BALANCED_SEGMENTS_PATH.exists():
        raise FileNotFoundError(f"Balanced segments CSV not found: {BALANCED_SEGMENTS_PATH}")

    excl_mids = get_all_exclusion_mids()
    rng = random.Random(seed)

    subdomain_candidates = {sub: [] for sub in OTHER_SUBDOMAINS}

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

            if vid in forbidden_video_ids:
                continue
            if labels.intersection(excl_mids):
                continue

            for sub, sub_mids in OTHER_SUBDOMAINS.items():
                matched = labels.intersection(set(sub_mids.keys()))
                if matched:
                    first_mid = sorted(matched)[0]
                    subdomain_candidates[sub].append({
                        "video_id": vid,
                        "start_seconds": start,
                        "end_seconds": end,
                        "target_class": "other",
                        "class_id": CLASS_TO_ID["other"],
                        "subdomain": sub,
                        "matched_mid": first_mid,
                        "matched_label": sub_mids[first_mid],
                        "source_dataset": "audioset_general_control",
                        "origin": "balanced_train_segments",
                        "download_status": "PENDING_DOWNLOAD",
                    })

    # Subdomain quotas summing to 224:
    # 224 / 6 = 37 with remainder 2 -> 37, 37, 37, 37, 38, 38
    sub_quotas = {
        "speech": 37,
        "music": 37,
        "animals": 37,
        "household": 37,
        "traffic": 38,
        "environmental": 38,
    }

    selected_candidates = []
    used_vids = set()

    for sub in sorted(sub_quotas.keys()):
        quota = sub_quotas[sub]
        pool = [c for c in subdomain_candidates[sub] if c["video_id"] not in used_vids]
        pool.sort(key=lambda x: str(x["video_id"]))
        rng.shuffle(pool)

        if len(pool) < quota:
            raise ValueError(f"Insufficient general candidates for subdomain {sub}: needed {quota}, available {len(pool)}")

        chosen = pool[:quota]
        for c in chosen:
            used_vids.add(c["video_id"])
            selected_candidates.append(c)

    if len(selected_candidates) != target_count:
        raise ValueError(f"Expected {target_count} E3 candidates, got {len(selected_candidates)}")

    return selected_candidates


def build_e3_general_dataset(
    e1_train_df: pd.DataFrame,
    shared_val_df: pd.DataFrame,
    forbidden_video_ids: Set[str],
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build E3 candidate dataset using shared validation set.

    Design:
      - Train split: 422 baseline train + 224 general background candidates = 646 candidates.
      - Val split: 105 samples (exact shared validation set).
      - Master: 751 candidate samples.
      - Candidates to download: exactly the 224 candidate rows with full downloader metadata.

    Returns:
        (e3_master_df, e3_train_df, e3_val_df, e3_candidates_to_download_df)
    """
    candidates = sample_e3_general_candidates(forbidden_video_ids, target_count=224, seed=seed)

    cols = [
        "video_id", "start_seconds", "end_seconds", "target_class", "class_id",
        "relative_path", "filename", "subdomain", "matched_label", "audio_set_mid",
        "source_dataset", "origin", "candidate_priority", "split"
    ]

    train_cands = []
    for r in candidates:
        r_copy = dict(r)
        r_copy["split"] = "train"
        r_copy["relative_path"] = f"data/raw/e3_general_background/other/other_{r['video_id']}_{int(r['start_seconds'])}_{int(r['end_seconds'])}.wav"
        r_copy["filename"] = Path(r_copy["relative_path"]).name
        r_copy["candidate_priority"] = "E3_CANDIDATE"
        r_copy["audio_set_mid"] = r_copy.get("audio_set_mid") or r_copy.get("matched_mid", "")
        train_cands.append(r_copy)

    # Sort candidates deterministically
    train_cands.sort(key=lambda x: str(x["video_id"]))
    df_train_cands = pd.DataFrame(train_cands)[cols]

    # Create dedicated download manifest for 224 candidates
    e3_candidates_to_download_df = df_train_cands.copy()

    # Align baseline training columns
    e1_train_aligned = e1_train_df.copy()
    for col in cols:
        if col not in e1_train_aligned.columns:
            e1_train_aligned[col] = ""
    e1_train_aligned = e1_train_aligned[cols].copy()

    # Align shared validation columns
    shared_val_aligned = shared_val_df.copy()
    for col in cols:
        if col not in shared_val_aligned.columns:
            shared_val_aligned[col] = ""
    e3_val_df = shared_val_aligned[cols].copy()

    e3_train_df = pd.concat([e1_train_aligned, df_train_cands], ignore_index=True)
    e3_train_df.sort_values(by=["target_class", "video_id"], inplace=True, ignore_index=True)

    e3_master_df = pd.concat([e3_train_df, e3_val_df], ignore_index=True)

    if len(e3_master_df) != 751:
        raise ValueError(f"Expected 751 E3 samples, got {len(e3_master_df)}")
    if len(e3_train_df) != 646:
        raise ValueError(f"Expected 646 E3 train samples (422 + 224), got {len(e3_train_df)}")
    if len(e3_val_df) != 105:
        raise ValueError(f"Expected 105 E3 val samples, got {len(e3_val_df)}")
    if len(e3_candidates_to_download_df) != 224:
        raise ValueError(f"Expected 224 E3 candidates to download, got {len(e3_candidates_to_download_df)}")

    return e3_master_df, e3_train_df, e3_val_df, e3_candidates_to_download_df


def build_e4_ablations(
    e2_train_df: pd.DataFrame,
    shared_val_df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Build category-removal ablations and matched-size random controls using shared validation set.

    For every E4 category ablation:
      1. Category ablation: Specific hard-negative subdomain is removed from the 646 E2 training samples.
      2. Matched-size control: Same number of samples randomly removed from the 224 targeted additions (Seed 42).
      Both conditions evaluate on the identical shared 105 validation samples.

    Returns:
        ablation_summary: Dict mapping experiment name to sample counts and excluded subdomains.
    """
    E4_DIR.mkdir(parents=True, exist_ok=True)

    primary_subdomains = [
        ("human_shouting", "Scream false alarm reduction (shouting/yelling)"),
        ("metal_impacts", "Gunshot and glass shattering false alarm reduction (metal clangs/pans)"),
        ("door_slams", "Gunshot false alarm reduction (sudden impulsive door slams)"),
        ("construction_tools", "Broad mechanical/industrial false alarm suppression (jackhammers, drilling)"),
        ("car_alarms_detectors", "Siren false alarm reduction (pulsed electronic alarms)"),
        ("fireworks_firecrackers", "Gunshot false alarm reduction (explosive percussive bursts)"),
        ("whistles_high_pitch", "Siren and scream false alarm reduction (high frequency resonant tones)"),
    ]

    # Partition E2 train into baseline (422) and targeted additions (224)
    baseline_train = e2_train_df[e2_train_df["source_dataset"] == "audioset"].copy()
    targeted_train = e2_train_df[e2_train_df["source_dataset"] == "audioset_targeted_expansion"].copy()
    if len(baseline_train) != 422 or len(targeted_train) != 224:
        raise ValueError(f"Unexpected partition: baseline {len(baseline_train)}, targeted {len(targeted_train)}")

    ablation_summary = {}

    for sub, test_purpose in primary_subdomains:
        # 1. Category ablation (removes specific subdomain)
        exp_name_ablation = f"e4_minus_{sub}"
        targeted_remaining = targeted_train[targeted_train["subdomain"] != sub].copy()
        train_filtered = pd.concat([baseline_train, targeted_remaining], ignore_index=True)
        val_filtered = shared_val_df.copy()

        train_path_ablation = E4_DIR / f"{exp_name_ablation}_train.csv"
        val_path_ablation = E4_DIR / f"{exp_name_ablation}_val.csv"
        train_filtered.to_csv(train_path_ablation, index=False)
        val_filtered.to_csv(val_path_ablation, index=False)

        num_excluded = len(targeted_train) - len(targeted_remaining)

        ablation_summary[exp_name_ablation] = {
            "type": "category_ablation",
            "target_subdomain": sub,
            "research_hypothesis": test_purpose,
            "train_samples": len(train_filtered),
            "val_samples": len(val_filtered),
            "total_samples": len(train_filtered) + len(val_filtered),
            "excluded_samples": num_excluded,
            "train_manifest": str(train_path_ablation.relative_to(ROOT)).replace("\\", "/"),
            "val_manifest": str(val_path_ablation.relative_to(ROOT)).replace("\\", "/"),
        }

        # 2. Matched-size random removal control
        exp_name_control = f"e4_matched_control_{sub}"
        rng = random.Random(seed)
        targeted_indices = list(targeted_train.index)
        targeted_indices.sort()  # deterministic base order
        rng.shuffle(targeted_indices)
        excluded_indices = set(targeted_indices[:num_excluded])
        targeted_control_remaining = targeted_train.loc[~targeted_train.index.isin(excluded_indices)].copy()
        train_control_filtered = pd.concat([baseline_train, targeted_control_remaining], ignore_index=True)

        if len(train_control_filtered) != len(train_filtered):
            raise ValueError(
                f"Matched-size mismatch for {sub}: control has {len(train_control_filtered)}, ablation has {len(train_filtered)}"
            )

        train_path_control = E4_DIR / f"{exp_name_control}_train.csv"
        val_path_control = E4_DIR / f"{exp_name_control}_val.csv"
        train_control_filtered.to_csv(train_path_control, index=False)
        val_filtered.to_csv(val_path_control, index=False)

        ablation_summary[exp_name_control] = {
            "type": "matched_size_control",
            "target_subdomain": sub,
            "research_hypothesis": f"Control for sample-size effect of removing {num_excluded} hard-negative samples",
            "train_samples": len(train_control_filtered),
            "val_samples": len(val_filtered),
            "total_samples": len(train_control_filtered) + len(val_filtered),
            "excluded_samples": num_excluded,
            "random_seed": seed,
            "train_manifest": str(train_path_control.relative_to(ROOT)).replace("\\", "/"),
            "val_manifest": str(val_path_control.relative_to(ROOT)).replace("\\", "/"),
        }

    # Save e4 configuration JSON
    config_data = {
        "description": "E4 Subdomain Ablation & Matched-Size Random Controls (Shared Validation Design)",
        "baseline_samples": 527,
        "shared_validation_samples": 105,
        "full_targeted_e2_samples": 751,
        "random_seed": seed,
        "experiments": ablation_summary,
    }

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIGS_DIR / "e4_ablation_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return ablation_summary

    # Save e4 configuration JSON
    config_data = {
        "description": "E4 Hard-Negative Subdomain Leave-One-Out Ablation Configuration (Shared Validation Design)",
        "baseline_samples": 527,
        "shared_validation_samples": 105,
        "full_targeted_e2_samples": 751,
        "experiments": ablation_summary,
    }

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIGS_DIR / "e4_ablation_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return ablation_summary


def verify_all_isolation(
    e1_train: pd.DataFrame,
    shared_val: pd.DataFrame,
    e2_train: pd.DataFrame,
    e3_train: pd.DataFrame,
    eval_manifest_path: Path = EVAL_MANIFEST_PATH,
    pilot_manifest_path: Path = PILOT_MANIFEST_PATH,
) -> Dict[str, Any]:
    """Verify zero cross-split leakage and zero overlap with eval and pilot sets under shared validation."""
    eval_df = pd.read_csv(eval_manifest_path)
    eval_vids = set(eval_df["video_id"].dropna().str.strip())

    pilot_vids = set()
    if pilot_manifest_path.exists():
        pilot_df = pd.read_csv(pilot_manifest_path)
        if "video_id" in pilot_df.columns:
            pilot_vids = set(pilot_df["video_id"].dropna().str.strip())

    checks = {}

    va_vids = set(shared_val["video_id"])
    e1_tr_vids = set(e1_train["video_id"])
    e2_tr_vids = set(e2_train["video_id"])
    e3_tr_vids = set(e3_train["video_id"])

    # Train vs Val Overlaps (must all be 0)
    checks["e1_train_val_overlap"] = len(e1_tr_vids.intersection(va_vids))
    checks["e2_train_val_overlap"] = len(e2_tr_vids.intersection(va_vids))
    checks["e3_train_val_overlap"] = len(e3_tr_vids.intersection(va_vids))

    if checks["e1_train_val_overlap"] > 0:
        raise ValueError(f"E1 train/val leakage: {checks['e1_train_val_overlap']}")
    if checks["e2_train_val_overlap"] > 0:
        raise ValueError(f"E2 train/val leakage: {checks['e2_train_val_overlap']}")
    if checks["e3_train_val_overlap"] > 0:
        raise ValueError(f"E3 train/val leakage: {checks['e3_train_val_overlap']}")

    # Eval Overlaps (must all be 0)
    checks["shared_val_eval_overlap"] = len(va_vids.intersection(eval_vids))
    checks["e1_train_eval_overlap"] = len(e1_tr_vids.intersection(eval_vids))
    checks["e2_train_eval_overlap"] = len(e2_tr_vids.intersection(eval_vids))
    checks["e3_train_eval_overlap"] = len(e3_tr_vids.intersection(eval_vids))

    if checks["shared_val_eval_overlap"] > 0:
        raise ValueError(f"Shared val overlaps with eval: {checks['shared_val_eval_overlap']}")
    if checks["e1_train_eval_overlap"] > 0:
        raise ValueError(f"E1 train overlaps with eval: {checks['e1_train_eval_overlap']}")
    if checks["e2_train_eval_overlap"] > 0:
        raise ValueError(f"E2 train overlaps with eval: {checks['e2_train_eval_overlap']}")
    if checks["e3_train_eval_overlap"] > 0:
        raise ValueError(f"E3 train overlaps with eval: {checks['e3_train_eval_overlap']}")

    # Check additions vs Pilot (must be 0)
    e2_new_vids = e2_tr_vids - e1_tr_vids
    e3_new_vids = e3_tr_vids - e1_tr_vids

    checks["e2_additions_pilot_overlap"] = len(e2_new_vids.intersection(pilot_vids))
    checks["e3_additions_pilot_overlap"] = len(e3_new_vids.intersection(pilot_vids))

    if checks["e2_additions_pilot_overlap"] > 0:
        raise ValueError(f"E2 additions overlap with pilot: {checks['e2_additions_pilot_overlap']}")
    if checks["e3_additions_pilot_overlap"] > 0:
        raise ValueError(f"E3 additions overlap with pilot: {checks['e3_additions_pilot_overlap']}")

    # Check NEEDS_SECOND_REVIEW exclusion
    all_all_vids = e1_tr_vids.union(va_vids).union(e2_tr_vids).union(e3_tr_vids)
    checks["ja_Qxq5NAoE_excluded"] = "ja-Qxq5NAoE" not in all_all_vids
    if "ja-Qxq5NAoE" in all_all_vids:
        raise ValueError("ja-Qxq5NAoE was not excluded!")

    # Check baseline sample preservation
    checks["baseline_train_preserved_in_e2"] = e1_tr_vids.issubset(e2_tr_vids)
    checks["baseline_train_preserved_in_e3"] = e1_tr_vids.issubset(e3_tr_vids)
    checks["shared_val_count"] = len(shared_val)

    return checks


def run_pipeline() -> Dict[str, Any]:
    """Execute complete dataset preparation pipeline with shared validation set."""
    print("=" * 80)
    print("STARTING EXPERIMENTAL DATASET PREPARATION (SHARED VALIDATION DESIGN)")
    print("=" * 80)

    # Step 1: Resolve review status
    print("\n[Step 1] Resolving human content validation review status...")
    usable_other_records, review_audit = resolve_review_status()
    print(f"  Total reviewed: {review_audit['total_reviewed']}")
    print(f"  Verdicts: {review_audit['verdict_counts']}")
    print(f"  NEEDS_SECOND_REVIEW: {review_audit['needs_second_review']} (strictly excluded)")
    print(f"  Reclassified (mapped to 'other'): {list(review_audit['reclassified'].keys())}")
    print(f"  Usable targeted hard-negative 'other' samples: {len(usable_other_records)}")

    # Step 2: Build E1 Baseline manifests and Shared Validation Manifest
    print("\n[Step 2] Building E1 (Baseline) and Shared Validation manifests...")
    e1_master_df, e1_train_df, shared_val_df = build_e1_baseline_dataset()

    # Save shared validation manifest
    shared_val_path = PROCESSED_DIR / "shared_val_split_manifest.csv"
    shared_val_df.to_csv(shared_val_path, index=False)
    print(f"  Shared Validation Set: {len(shared_val_df)} samples -> {shared_val_path.name}")

    e1_master_path = PROCESSED_DIR / "e1_baseline_manifest.csv"
    e1_train_path = PROCESSED_DIR / "e1_train_split_manifest.csv"
    e1_val_path = PROCESSED_DIR / "e1_val_split_manifest.csv"
    e1_master_df.to_csv(e1_master_path, index=False)
    e1_train_df.to_csv(e1_train_path, index=False)
    shared_val_df.to_csv(e1_val_path, index=False)
    print(f"  E1 Master: {len(e1_master_df)} samples -> {e1_master_path.name}")
    print(f"  E1 Train:  {len(e1_train_df)} samples -> {e1_train_path.name}")
    print(f"  E1 Val:    {len(shared_val_df)} samples -> {e1_val_path.name}")

    # Step 3: Build E2 Targeted Hard-Negatives manifests (646 train / 105 val)
    print("\n[Step 3] Building E2 (Targeted Hard-Negatives) manifests...")
    e2_master_df, e2_train_df, e2_val_df = build_e2_targeted_dataset(
        e1_train_df, shared_val_df, usable_other_records
    )
    e2_master_path = PROCESSED_DIR / "e2_targeted_manifest.csv"
    e2_train_path = PROCESSED_DIR / "e2_train_split_manifest.csv"
    e2_val_path = PROCESSED_DIR / "e2_val_split_manifest.csv"
    e2_master_df.to_csv(e2_master_path, index=False)
    e2_train_df.to_csv(e2_train_path, index=False)
    e2_val_df.to_csv(e2_val_path, index=False)
    print(f"  E2 Master: {len(e2_master_df)} samples -> {e2_master_path.name}")
    print(f"  E2 Train:  {len(e2_train_df)} samples -> {e2_train_path.name} (422 baseline + 224 targeted)")
    print(f"  E2 Val:    {len(e2_val_df)} samples -> {e2_val_path.name} (shared validation set)")

    # Step 4: Build E3 General Background candidate manifests (646 train / 105 val)
    print("\n[Step 4] Building E3 (General Background Audio) candidate manifests...")
    forbidden_vids = set(e2_master_df["video_id"]).union(
        set(pd.read_csv(TARGET_MANIFEST_PATH)["video_id"].dropna().str.strip())
    ).union(
        set(pd.read_csv(EVAL_MANIFEST_PATH)["video_id"].dropna().str.strip())
    )
    if (PROCESSED_DIR / "targeted_expansion_candidates.csv").exists():
        exp_cands = pd.read_csv(PROCESSED_DIR / "targeted_expansion_candidates.csv")
        forbidden_vids.update(exp_cands["video_id"].dropna().str.strip())
    if PILOT_MANIFEST_PATH.exists():
        pilot_df = pd.read_csv(PILOT_MANIFEST_PATH)
        if "video_id" in pilot_df.columns:
            forbidden_vids.update(pilot_df["video_id"].dropna().str.strip())

    e3_master_df, e3_train_df, e3_val_df, e3_download_df = build_e3_general_dataset(
        e1_train_df, shared_val_df, forbidden_video_ids=forbidden_vids, seed=42
    )
    e3_master_path = PROCESSED_DIR / "e3_general_candidate_manifest.csv"
    e3_train_path = PROCESSED_DIR / "e3_general_train_candidates.csv"
    e3_val_path = PROCESSED_DIR / "e3_general_val_candidates.csv"
    e3_download_path = PROCESSED_DIR / "e3_candidates_to_download.csv"

    e3_master_df.to_csv(e3_master_path, index=False)
    e3_train_df.to_csv(e3_train_path, index=False)
    e3_val_df.to_csv(e3_val_path, index=False)
    e3_download_df.to_csv(e3_download_path, index=False)

    print(f"  E3 Master:   {len(e3_master_df)} candidates -> {e3_master_path.name} (PENDING_DOWNLOAD)")
    print(f"  E3 Train:    {len(e3_train_df)} candidates -> {e3_train_path.name} (422 baseline + 224 general)")
    print(f"  E3 Val:      {len(e3_val_df)} candidates -> {e3_val_path.name} (shared validation set)")
    print(f"  E3 Download: {len(e3_download_df)} candidates -> {e3_download_path.name} (downloader manifest)")

    # Step 5: Build E4 Subdomain Ablation manifests and Matched Controls
    print("\n[Step 5] Building E4 (Subdomain Ablations & Matched Controls) manifests and configuration...")
    ablation_summary = build_e4_ablations(e2_train_df, shared_val_df, seed=42)
    for exp_name, info in ablation_summary.items():
        type_str = "Ablation" if info["type"] == "category_ablation" else "Control"
        print(f"  [{type_str:<8}] {exp_name:<38}: {info['train_samples']} train + {info['val_samples']} val = {info['total_samples']} samples (excluded {info['excluded_samples']} {info['target_subdomain']})")

    # Step 6: Verify isolation & leakage
    print("\n[Step 6] Running comprehensive data leakage & isolation verification...")
    checks = verify_all_isolation(
        e1_train_df, shared_val_df,
        e2_train_df, e3_train_df,
    )
    for k, v in checks.items():
        print(f"  Check: {k} -> {v}")

    # Step 7: Export summary metadata
    summary = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": 42,
        "validation_design": "shared_fixed_benchmark",
        "shared_validation_samples": len(shared_val_df),
        "review_audit": review_audit,
        "e1_baseline": {
            "total_samples": len(e1_master_df),
            "train_samples": len(e1_train_df),
            "val_samples": len(shared_val_df),
            "classes_train": e1_train_df["target_class"].value_counts().to_dict(),
            "classes_val": shared_val_df["target_class"].value_counts().to_dict(),
        },
        "e2_targeted": {
            "total_samples": len(e2_master_df),
            "train_samples": len(e2_train_df),
            "val_samples": len(e2_val_df),
            "classes_train": e2_train_df["target_class"].value_counts().to_dict(),
            "classes_val": e2_val_df["target_class"].value_counts().to_dict(),
            "subdomains_other_train": e2_train_df[e2_train_df["target_class"] == "other"]["subdomain"].value_counts().to_dict(),
        },
        "e3_general_candidates": {
            "total_samples": len(e3_master_df),
            "train_samples": len(e3_train_df),
            "val_samples": len(e3_val_df),
            "classes_train": e3_train_df["target_class"].value_counts().to_dict(),
            "classes_val": e3_val_df["target_class"].value_counts().to_dict(),
            "subdomains_other_train": e3_train_df[e3_train_df["target_class"] == "other"]["subdomain"].value_counts().to_dict(),
            "download_status": "PENDING_DOWNLOAD",
        },
        "e4_ablations": ablation_summary,
        "isolation_checks": checks,
    }

    summary_path = PROCESSED_DIR / "experimental_manifests_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Summary exported to {summary_path}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    run_pipeline()
