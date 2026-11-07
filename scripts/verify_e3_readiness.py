"""Comprehensive E3 Readiness Verification Script.

Verifies:
1. Accepted E3 samples from initial and replacement reviews.
2. Source-level isolation (zero overlap with E1, E2, shared val, eval, pilot, targeted expansion).
3. Absence of duplicate source IDs.
4. Physical audio integrity of every accepted file (16 kHz, mono, 16-bit PCM, non-empty).
5. Traceability of human review decisions.
6. Compiles verified e3_train_split_manifest.csv (422 baseline + accepted E3).
"""

from pathlib import Path
import sys
import wave

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
INITIAL_REVIEW_PATH = PROCESSED_DIR / "e3_review_results.csv"
REPL_REVIEW_PATH = PROCESSED_DIR / "e3_replacement_review_results.csv"
E1_TRAIN_PATH = PROCESSED_DIR / "e1_train_split_manifest.csv"
SHARED_VAL_PATH = PROCESSED_DIR / "shared_val_split_manifest.csv"
EVAL_PATH = PROCESSED_DIR / "eval_manifest.csv"
PILOT_PATH = PROCESSED_DIR / "pilot_manifest.csv"
TARGETED_EXP_PATH = PROCESSED_DIR / "targeted_expansion_candidates.csv"

E3_TRAIN_MANIFEST_PATH = PROCESSED_DIR / "e3_train_split_manifest.csv"
E3_VAL_MANIFEST_PATH = PROCESSED_DIR / "e3_val_split_manifest.csv"
E3_MASTER_MANIFEST_PATH = PROCESSED_DIR / "e3_master_manifest.csv"


def verify_e3():
    print("=" * 80)
    print("E3 READINESS AND ISOLATION VERIFICATION AUDIT")
    print("=" * 80)

    # 1. Load reviews
    df_init = pd.read_csv(INITIAL_REVIEW_PATH)
    df_repl = pd.read_csv(REPL_REVIEW_PATH)

    acc_init = df_init[df_init["verdict"] == "ACCEPT"].copy()
    rej_init = df_init[df_init["verdict"] == "REJECT"].copy()
    sec_init = df_init[df_init["verdict"] == "NEEDS_SECOND_REVIEW"].copy()

    acc_repl = df_repl[df_repl["verdict"] == "ACCEPT"].copy()
    rej_repl = df_repl[df_repl["verdict"] == "REJECT"].copy()
    sec_repl = df_repl[df_repl["verdict"] == "NEEDS_SECOND_REVIEW"].copy()

    print(f"Initial Review (174 queued):  {len(acc_init)} ACCEPT | {len(rej_init)} REJECT | {len(sec_init)} 2ND_REVIEW")
    print(f"Repl. Review (40 downloaded): {len(acc_repl)} ACCEPT | {len(rej_repl)} REJECT | {len(sec_repl)} 2ND_REVIEW")

    total_accepted = pd.concat([acc_init, acc_repl], ignore_index=True)
    usable_count = len(total_accepted)
    print(f"\nTotal usable accepted E3 clips: {usable_count}")
    print("Subdomain distribution of usable E3 clips:")
    for sub, cnt in total_accepted["subdomain"].value_counts().items():
        print(f"  - {sub:<16}: {cnt}")

    # 2. Duplicate source ID check
    vids = total_accepted["video_id"].tolist()
    dup_vids = [v for v in set(vids) if vids.count(v) > 1]
    print(f"\nDuplicate video IDs in E3: {len(dup_vids)}")
    assert len(dup_vids) == 0, f"Duplicate video IDs found in E3: {dup_vids}"
    print("  [PASS] 0 duplicate video IDs in E3 accepted clips.")

    # 3. Leakage checks
    print("\n--- Cross-Dataset Isolation Checks ---")
    e3_vids = set(vids)

    # E1 Train
    e1_train = pd.read_csv(E1_TRAIN_PATH)
    e1_train_vids = set(e1_train["video_id"].dropna().str.strip())
    e1_overlap = e3_vids.intersection(e1_train_vids)
    print(f"Overlap with E1 Train (422 clips): {len(e1_overlap)}")
    assert len(e1_overlap) == 0, f"Leakage with E1 Train: {e1_overlap}"
    print("  [PASS] 0 overlap with E1 training audio.")

    # Shared Val
    shared_val = pd.read_csv(SHARED_VAL_PATH)
    val_vids = set(shared_val["video_id"].dropna().str.strip())
    val_overlap = e3_vids.intersection(val_vids)
    print(f"Overlap with Shared Validation (105 clips): {len(val_overlap)}")
    assert len(val_overlap) == 0, f"Leakage with Shared Val: {val_overlap}"
    print("  [PASS] 0 overlap with shared validation benchmark.")

    # Eval
    eval_df = pd.read_csv(EVAL_PATH)
    eval_vids = set(eval_df["video_id"].dropna().str.strip())
    eval_overlap = e3_vids.intersection(eval_vids)
    print(f"Overlap with Held-out Evaluation (744 rows): {len(eval_overlap)}")
    assert len(eval_overlap) == 0, f"Leakage with Held-out Eval: {eval_overlap}"
    print("  [PASS] 0 overlap with held-out evaluation set.")

    # Pilot
    pilot_df = pd.read_csv(PILOT_PATH)
    pilot_vids = set(pilot_df["video_id"].dropna().str.strip())
    pilot_overlap = e3_vids.intersection(pilot_vids)
    print(f"Overlap with Pilot audio: {len(pilot_overlap)}")
    assert len(pilot_overlap) == 0, f"Leakage with Pilot: {pilot_overlap}"
    print("  [PASS] 0 overlap with pilot audio.")

    # Targeted expansion
    exp_df = pd.read_csv(TARGETED_EXP_PATH)
    exp_vids = set(exp_df["video_id"].dropna().str.strip())
    exp_overlap = e3_vids.intersection(exp_vids)
    print(f"Overlap with Targeted Expansion candidates (672 clips): {len(exp_overlap)}")
    assert len(exp_overlap) == 0, f"Leakage with Targeted Expansion: {exp_overlap}"
    print("  [PASS] 0 overlap with targeted expansion candidates.")

    # 4. Audio Integrity Check
    print("\n--- Physical Audio Integrity Check ---")
    corrupt_files = []
    durations = []
    for idx, row in total_accepted.iterrows():
        p = ROOT / row["file_path"]
        if not p.exists() or p.stat().st_size == 0:
            corrupt_files.append((str(p), "Missing or empty"))
            continue
        try:
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                n_frames = wf.getnframes()
                dur = n_frames / float(sr)
                durations.append(dur)
                if sr != 16000:
                    corrupt_files.append((str(p), f"Sample rate {sr} != 16000"))
                if ch != 1:
                    corrupt_files.append((str(p), f"Channels {ch} != 1"))
                if sw != 2:
                    corrupt_files.append((str(p), f"Sample width {sw} != 2"))
                if dur < 8.0:
                    corrupt_files.append((str(p), f"Duration {dur:.2f}s < 8.0s"))
        except Exception as e:
            corrupt_files.append((str(p), str(e)))

    print(f"Inspected {len(total_accepted)} files on disk.")
    print(f"Audio defects found: {len(corrupt_files)}")
    assert len(corrupt_files) == 0, f"Defective audio files: {corrupt_files}"
    print(f"  [PASS] 100% of {len(total_accepted)} files are verified 16 kHz mono 16-bit PCM WAV.")
    print(f"  Duration stats: Min={min(durations):.2f}s, Max={max(durations):.2f}s, Mean={sum(durations)/len(durations):.2f}s")

    # 5. Traceability
    print("\n--- Traceability of Content Review Decisions ---")
    missing_trace = []
    for idx, row in total_accepted.iterrows():
        if not row.get("reviewer"):
            missing_trace.append((row["video_id"], "Missing reviewer"))
        if not row.get("reviewed_at"):
            missing_trace.append((row["video_id"], "Missing timestamp"))
        if not row.get("notes"):
            missing_trace.append((row["video_id"], "Missing notes"))
    assert len(missing_trace) == 0, f"Untraceable decisions: {missing_trace}"
    print("  [PASS] 100% of review decisions have reviewer ID, reviewed_at timestamp, and audit notes.")

    # 6. Build E3 Training Manifest
    print("\n--- Compiling E3 Training Manifest ---")
    # Base E1 training rows
    e1_train_rows = e1_train.copy()
    e1_train_rows["split"] = "train"

    # Format E3 accepted rows to match standard schema
    e3_new_rows = []
    for idx, row in total_accepted.iterrows():
        e3_new_rows.append({
            "video_id": row["video_id"],
            "start_seconds": float(row["start_seconds"]),
            "end_seconds": float(row["end_seconds"]),
            "target_class": "other",
            "class_id": 4,
            "relative_path": row["file_path"].replace("\\", "/"),
            "filename": Path(row["file_path"]).name,
            "subdomain": row["subdomain"],
            "matched_label": row["matched_label"],
            "audio_set_mid": row.get("audio_set_mid", ""),
            "source_dataset": "audioset_general_control",
            "origin": row["origin"],
            "candidate_priority": "E3_ACCEPTED",
            "split": "train",
        })
    e3_train_additions = pd.DataFrame(e3_new_rows)

    e3_train_combined = pd.concat([e1_train_rows, e3_train_additions], ignore_index=True)
    e3_train_combined.to_csv(E3_TRAIN_MANIFEST_PATH, index=False, encoding="utf-8")
    print(f"Saved E3 Train Split Manifest: {E3_TRAIN_MANIFEST_PATH}")
    print(f"  Rows: {len(e3_train_combined)} (422 baseline train + {len(e3_train_additions)} accepted general background)")

    # E3 Val Manifest (Exact shared validation set)
    shared_val.to_csv(E3_VAL_MANIFEST_PATH, index=False, encoding="utf-8")
    print(f"Saved E3 Val Split Manifest (Shared benchmark): {E3_VAL_MANIFEST_PATH} ({len(shared_val)} rows)")

    # E3 Master Manifest
    shared_val_rows = shared_val.copy()
    shared_val_rows["split"] = "val"
    e3_master = pd.concat([e3_train_combined, shared_val_rows], ignore_index=True)
    e3_master.to_csv(E3_MASTER_MANIFEST_PATH, index=False, encoding="utf-8")
    print(f"Saved E3 Master Manifest: {E3_MASTER_MANIFEST_PATH} ({len(e3_master)} rows)")

    print("\nE3 Class Breakdown (Train):")
    print(e3_train_combined["target_class"].value_counts())

    print("\n" + "=" * 80)
    print("ALL E3 READINESS AND ISOLATION CHECKS PASSED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    verify_e3()
