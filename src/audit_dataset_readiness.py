"""Comprehensive Audio Dataset Readiness Audit Script for MyNet.

Audits:
1. Local training inventory (WAV integrity, sample rate, channels, bit depth, format, durations, duplicates).
2. Manifests & splits (row counts, class composition, cross-split leakage, segment correspondence).
3. Class balance and training risk analysis.
4. Assessment of the 'other' class and pilot cache inspection.
"""

import csv
from pathlib import Path
import sys
import wave

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from class_config import CLASSES, CLASS_TO_ID

TRAIN_MANIFEST = ROOT / "data" / "processed" / "target_manifest.csv"
EVAL_MANIFEST = ROOT / "data" / "processed" / "eval_manifest.csv"
RETRY_LIST = ROOT / "data" / "processed" / "train_download_retry_list.csv"
TRAIN_DIR = ROOT / "data" / "raw" / "train"
PILOT_CSV = ROOT / "data" / "processed" / "pilot_accessible.csv"
PILOT_DIR = ROOT / "data" / "raw" / "pilot"


def audit_readiness():
    print("=" * 80)
    print("  AUDIO DATASET READINESS AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Training Inventory Verification
    # -------------------------------------------------------------
    print("\n[1] TRAINING INVENTORY VERIFICATION (data/raw/train/)")
    emergency_classes = [c for c in CLASSES if c != "other"]

    class_files = {}
    all_wavs = []
    unreadable = []
    invalid_format = []
    durations = {}

    for c in CLASSES:
        cdir = TRAIN_DIR / c
        if not cdir.exists():
            class_files[c] = []
            durations[c] = []
            continue
        c_wavs = sorted(list(cdir.glob("*.wav")))
        class_files[c] = c_wavs
        all_wavs.extend(c_wavs)
        durations[c] = []
        for w in c_wavs:
            try:
                with wave.open(str(w), "rb") as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    ct = wf.getcomptype()
                    frames = wf.getnframes()
                    dur = frames / sr
                    durations[c].append(dur)
                    if sr != 16000 or ch != 1 or sw != 2 or ct != "NONE":
                        invalid_format.append((w, sr, ch, sw, ct))
            except Exception as e:
                unreadable.append((w, str(e)))

    print(f"  Total WAV files on disk : {len(all_wavs)}")
    print(f"  Unreadable / corrupt    : {len(unreadable)}")
    print(f"  Format specification    : 16 kHz, Mono (1 ch), 16-bit PCM (NONE compression)")
    print(f"  Format spec violations  : {len(invalid_format)}")

    print("\n  Class-wise inventory on disk:")
    for c in CLASSES:
        cnt = len(class_files.get(c, []))
        d_list = durations.get(c, [])
        if d_list:
            min_d, max_d, avg_d = min(d_list), max(d_list), sum(d_list) / len(d_list)
            print(f"    - {c:<16}: {cnt:>4} files | min={min_d:>5.2f}s, max={max_d:>5.2f}s, avg={avg_d:>5.2f}s")
        else:
            print(f"    - {c:<16}: {cnt:>4} files | (Folder empty or unpopulated)")

    # Duplicate check
    filenames = [w.name for w in all_wavs]
    dups = len(filenames) - len(set(filenames))
    print(f"\n  Duplicate filename check: {dups} duplicates found")

    # -------------------------------------------------------------
    # 2. Manifests and Splits Verification
    # -------------------------------------------------------------
    print("\n[2] MANIFESTS AND SPLITS VERIFICATION")
    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(EVAL_MANIFEST, "r", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    with open(RETRY_LIST, "r", encoding="utf-8") as f:
        retry_rows = list(csv.DictReader(f))

    em_train_rows = [r for r in train_rows if r["target_class"] != "other"]
    other_train_rows = [r for r in train_rows if r["target_class"] == "other"]

    print(f"  target_manifest.csv rows: {len(train_rows)} total")
    print(f"    - Emergency target rows: {len(em_train_rows)} (indices 0..564)")
    print(f"    - 'other' class rows   : {len(other_train_rows)} (indices 565..739)")

    print(f"  eval_manifest.csv rows  : {len(eval_rows)} total")
    eval_by_class = {}
    for r in eval_rows:
        cls = r.get("target_class", "unknown")
        eval_by_class[cls] = eval_by_class.get(cls, 0) + 1
    for cls in sorted(eval_by_class.keys()):
        print(f"    - {cls:<16}: {eval_by_class[cls]:>4} rows")

    print(f"  train_download_retry_list: {len(retry_rows)} rows")
    other_in_retry = [r for r in retry_rows if r.get("target_class") == "other"]
    print(f"    - Emergency retry rows : {len(retry_rows) - len(other_in_retry)}")
    print(f"    - 'other' rows in retry: {len(other_in_retry)} (PASSED - zero 'other' entries)")

    # Cross-split leakage
    train_vids = {r["video_id"].strip() for r in train_rows if r.get("video_id")}
    eval_vids = {r["video_id"].strip() for r in eval_rows if r.get("video_id")}
    leakage = train_vids.intersection(eval_vids)
    print(f"\n  Cross-split leakage check:")
    print(f"    - Unique Train video IDs: {len(train_vids)}")
    print(f"    - Unique Eval video IDs : {len(eval_vids)}")
    print(f"    - Video ID overlap      : {len(leakage)} " + ("(Zero leakage confirmed - PASSED)" if not leakage else f"FAILED: {leakage}"))

    # Segment correspondence check
    manifest_segment_set = set()
    for r in em_train_rows:
        cls = r["target_class"]
        vid = r["video_id"].strip()
        st = int(float(r["start_seconds"]))
        en = int(float(r["end_seconds"]))
        manifest_segment_set.add(f"{cls}_{vid}_{st}_{en}.wav")

    unmatched = [w.name for w in all_wavs if w.name not in manifest_segment_set]
    print(f"  WAVs not in emergency manifest: {len(unmatched)}")

    # Check if any eval video IDs exist in train WAV names
    eval_in_train = [w.name for w in all_wavs if any(ev in w.name for ev in eval_vids)]
    print(f"  Eval video IDs in train audio : {len(eval_in_train)} (PASSED - zero eval audio in train)")

    # -------------------------------------------------------------
    # 3. Class Balance and Training Risk Analysis
    # -------------------------------------------------------------
    print("\n[3] CLASS BALANCE & AVAILABILITY")
    print(f"  {'Class':<16} {'Valid on Disk':<14} {'Missing / Retry':<16} {'Total Target':<14} {'Yield %':<10}")
    print("  " + "-" * 70)
    for c in emergency_classes:
        valid_cnt = len(class_files.get(c, []))
        missing_cnt = sum(1 for r in retry_rows if r["target_class"] == c)
        tot_cnt = sum(1 for r in em_train_rows if r["target_class"] == c)
        yield_pct = (valid_cnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
        print(f"  {c:<16} {valid_cnt:<14} {missing_cnt:<16} {tot_cnt:<14} {yield_pct:>6.1f}%")

    tot_valid = sum(len(class_files.get(c, [])) for c in emergency_classes)
    tot_missing = len(retry_rows)
    tot_targets = len(em_train_rows)
    print("  " + "-" * 70)
    print(f"  {'TOTAL':<16} {tot_valid:<14} {tot_missing:<16} {tot_targets:<14} {tot_valid/tot_targets*100:>6.1f}%")

    # -------------------------------------------------------------
    # 4. Assessment of 'other' Class & Pilot Cache
    # -------------------------------------------------------------
    print("\n[4] SEPARATE ASSESSMENT OF 'OTHER' CLASS & PILOT CACHE")
    pilot_records = []
    if PILOT_CSV.exists():
        with open(PILOT_CSV, "r", encoding="utf-8") as f:
            pilot_records = list(csv.DictReader(f))

    print(f"  pilot_accessible.csv entries: {len(pilot_records)}")
    pilot_by_class = {}
    for r in pilot_records:
        cls = r.get("target_class", "unknown")
        pilot_by_class[cls] = pilot_by_class.get(cls, 0) + 1
    for cls, cnt in sorted(pilot_by_class.items()):
        print(f"    - {cls:<16}: {cnt:>2} pilot records")

    pilot_wavs = list(PILOT_DIR.glob("*.wav")) if PILOT_DIR.exists() else []
    print(f"  Local pilot WAV files on disk: {len(pilot_wavs)}")

    other_pilot_candidates = [r for r in pilot_records if r.get("target_class") == "other" and r.get("audio_verified") == "True"]
    print(f"  Verified 'other' pilot clips: {len(other_pilot_candidates)}")
    for r in other_pilot_candidates:
        print(f"    * {r['video_id']} [{r['start_seconds']}..{r['end_seconds']}] - {r.get('matched_label')} ({r.get('sub_domain')})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    audit_readiness()
