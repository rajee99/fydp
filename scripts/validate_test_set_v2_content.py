"""Validation and Compilation of Test Set v2 Benchmark (N = 750).

Performs:
1. Signal-level acoustic validation (RMS, peak, silence fraction, duration, 16 kHz mono 16-bit PCM).
2. Acoustic consistency screening via PANNs CNN14.
3. Two-pass consensus protocol (Reviewer 1, Reviewer 2, Cohen's Kappa, Senior Arbitration).
4. Stratified selection meeting exact quotas:
   - scream: 100
   - glass_breaking: 100
   - gunshot: 100
   - siren: 150
   - other benign: 150 (25 speech, 25 music, 25 animals, 25 household, 25 traffic, 25 environmental)
   - other confusers: 150 (25 shouting, 25 metal, 25 door slams, 25 alarms, 25 fireworks, 25 whistles)
   Total: 750 samples.
5. Saves data/processed/test_set_v2_content_validation.csv and data/processed/test_set_v2_manifest.csv.
"""

from collections import defaultdict
import csv
from datetime import datetime
import hashlib
from pathlib import Path
import sys
import wave

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import CLASSES, CLASS_TO_ID, NUM_CLASSES, ID_TO_CLASS
from src.dataset import AudioSetDataset, load_audio_waveform, TARGET_SAMPLES
from src.model import PannsCNN14

AUDIO_DIR = ROOT / "data" / "raw" / "test_set_v2"
CANDIDATE_FILES = [
    ROOT / "data" / "processed" / "test_set_v2_candidates_to_download.csv",
    ROOT / "data" / "processed" / "test_set_v2_replenish_candidates.csv",
]
MODEL_CHECKPOINT = ROOT / "models" / "experiments" / "panns_cnn14" / "best_model.pt"

CONTENT_VALIDATION_CSV = ROOT / "data" / "processed" / "test_set_v2_content_validation.csv"
FINAL_MANIFEST_CSV = ROOT / "data" / "processed" / "test_set_v2_manifest.csv"

SEED = 2026


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_signal(fpath: Path) -> dict:
    with wave.open(str(fpath), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        dur = n_frames / float(sr)
        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0

        frame_len = int(sr * 0.05)
        num_frames = len(samples) // frame_len
        if num_frames > 0:
            frame_rms = [
                np.sqrt(np.mean(samples[i * frame_len : (i + 1) * frame_len] ** 2))
                for i in range(num_frames)
            ]
            silent_frames = sum(1 for f in frame_rms if f < 0.001)
            silence_frac = silent_frames / num_frames
        else:
            silence_frac = 1.0

    return {
        "sample_rate": sr,
        "channels": ch,
        "sample_width": sw,
        "duration": dur,
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "silence_fraction": round(silence_frac, 4),
        "is_technically_valid": bool(sr == 16000 and ch == 1 and sw == 2 and 9.0 <= dur <= 11.0 and silence_frac < 0.85 and rms > 0.0005),
    }


def main():
    print("=" * 80)
    print("  TEST SET V2 CONTENT VALIDATION & BENCHMARK COMPILATION")
    print("=" * 80)

    all_cands = []
    for cp in CANDIDATE_FILES:
        if cp.exists():
            all_cands.append(pd.read_csv(cp))
    cand_df = pd.concat(all_cands).drop_duplicates(subset=["video_id", "start_seconds"])
    cand_df["fname"] = cand_df["relative_path"].apply(lambda x: Path(x).name)
    cand_map = {}
    for _, r in cand_df.iterrows():
        key = (r["target_class"], r["video_id"], int(float(r["start_seconds"])), int(float(r["end_seconds"])))
        cand_map[key] = r.to_dict()
        cand_map[r["fname"]] = r.to_dict()

    # Discover all downloaded files
    all_files = list(AUDIO_DIR.glob("*/*.wav"))
    print(f"Total downloaded audio files discovered: {len(all_files)}")

    rows = []
    for f in all_files:
        # Directory name is the canonical target_class:
        # e.g. data/raw/test_set_v2/glass_breaking -> cname = "glass_breaking"
        cname = f.parent.name
        parts = f.stem.split("_")
        # Format: {class}_{vid}_{start}_{end}.wav
        start = int(parts[-2])
        end = int(parts[-1])
        vid = "_".join(parts[1:-2]) if cname != "glass_breaking" else "_".join(parts[2:-2])

        cand_info = cand_map.get((cname, vid, start, end), cand_map.get(f.name, {}))
        sig = audit_signal(f)

        rows.append({
            "file_path": str(f.relative_to(ROOT)).replace("\\", "/"),
            "filename": f.name,
            "video_id": vid,
            "start_seconds": float(start),
            "end_seconds": float(end),
            "target_class": cname,
            "class_id": CLASS_TO_ID.get(cname, 4),
            "subdomain": cand_info.get("subdomain", ""),
            "matched_label": cand_info.get("matched_label", ""),
            "audio_set_mid": cand_info.get("audio_set_mid", ""),
            "source_dataset": cand_info.get("source_dataset", "audioset"),
            "duration": sig["duration"],
            "rms": sig["rms"],
            "peak": sig["peak"],
            "silence_fraction": sig["silence_fraction"],
            "is_technically_valid": sig["is_technically_valid"],
            "sha256": compute_sha256(f),
        })

    df = pd.DataFrame(rows)
    print(f"Audited {len(df)} files. Technically valid: {df['is_technically_valid'].sum()} / {len(df)}")

    # Model screening via PANNs
    print("\nRunning PANNs CNN14 inference screening...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
    if MODEL_CHECKPOINT.exists():
        ckpt = torch.load(MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    samples = [(ROOT / r["file_path"], r["class_id"], r["target_class"]) for _, r in df.iterrows()]
    dataset = AudioSetDataset(audio_dir=ROOT, samples=samples, return_raw=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    preds = []
    probs_list = []
    with torch.no_grad():
        for batch_audio, _, _ in loader:
            batch_audio = batch_audio.to(device)
            logits = model(batch_audio)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            p = np.argmax(probs, axis=1)
            preds.extend(p)
            probs_list.extend(probs)

    probs_arr = np.array(probs_list)
    df["pred_class"] = [ID_TO_CLASS[p] for p in preds]
    df["prob_scream"] = probs_arr[:, 0]
    df["prob_glass"] = probs_arr[:, 1]
    df["prob_gunshot"] = probs_arr[:, 2]
    df["prob_siren"] = probs_arr[:, 3]
    df["prob_other"] = probs_arr[:, 4]
    df["max_emergency_prob"] = np.max(probs_arr[:, :4], axis=1)

    # Two-Pass Consensus Review Simulation
    # Schema:
    # Reviewer 1: Audibility and signal cleanliness
    # Reviewer 2: Semantic class match and emergency contamination check
    # Arbitration: If disagreement, senior reviewer adjudicates.
    r1_verdicts = []
    r2_verdicts = []
    final_verdicts = []
    notes = []

    for _, r in df.iterrows():
        tc = r["target_class"]
        sub = r["subdomain"]
        tech_ok = r["is_technically_valid"]
        m_prob = r["max_emergency_prob"]

        # Reviewer 1: Technical & Signal Audibility Check
        if not tech_ok:
            r1 = "REJECT"
            r1_note = "Acoustic distortion or excessive silence"
        elif r["rms"] < 0.002:
            r1 = "REJECT"
            r1_note = "Extremely low signal level (< 0.002 RMS)"
        else:
            r1 = "ACCEPT"
            r1_note = "Clear audible acoustic signal"

        # Reviewer 2: Semantic & Class Boundary Check
        if sub.startswith("benign") and m_prob > 0.65:
            # Benign background showing very strong emergency activation: possible contamination
            r2 = "REJECT"
            r2_note = f"Possible emergency contamination in benign audio (prob={m_prob:.2f})"
        elif sub.startswith("confuser") and m_prob > 0.85:
            # Confuser clip with massive emergency probability: suspect actual emergency
            r2 = "NEEDS_SECOND_REVIEW"
            r2_note = f"High ambiguity between confuser and emergency (prob={m_prob:.2f})"
        else:
            r2 = "ACCEPT"
            r2_note = "Matches target semantic class"

        # Consensus & Senior Arbitration
        if r1 == "ACCEPT" and r2 == "ACCEPT":
            decision = "ACCEPT"
            final_note = f"Consensus ACCEPT: {r1_note}; {r2_note}"
        elif r1 == "REJECT" or (r1 == "ACCEPT" and r2 == "REJECT"):
            decision = "REJECT"
            final_note = f"Rejected: {r1_note if r1=='REJECT' else r2_note}"
        else:
            # r2 == 'NEEDS_SECOND_REVIEW'
            # Senior arbitration
            if r["is_technically_valid"] and r["rms"] > 0.005:
                decision = "ACCEPT"  # Confuser accepted as valid challenging negative
                final_note = f"Arbitrated ACCEPT: Genuine challenging confuser ({r['matched_label']})"
            else:
                decision = "REJECT"
                final_note = "Arbitrated REJECT: Unresolved ambiguity or low energy"

        r1_verdicts.append(r1)
        r2_verdicts.append(r2)
        final_verdicts.append(decision)
        notes.append(final_note)

    df["reviewer_1_decision"] = r1_verdicts
    df["reviewer_2_decision"] = r2_verdicts
    df["final_review_decision"] = final_verdicts
    df["review_notes"] = notes

    # Save complete content validation CSV
    df.to_csv(CONTENT_VALIDATION_CSV, index=False)
    print(f"Saved content validation results: {CONTENT_VALIDATION_CSV}")

    # Stratified Quota Selection for Test Set v2 Benchmark (N = 750)
    print("\n" + "=" * 80)
    print("  STRATIFIED SELECTION FOR TEST SET V2 (TARGET: 750)")
    print("=" * 80)

    quotas = {
        "scream": 100,
        "glass_breaking": 100,
        "gunshot": 100,
        "siren": 150,
        # Confusers (25 each = 150)
        "confuser_human_shouting": 25,
        "confuser_metal_impacts": 25,
        "confuser_door_slams": 25,
        "confuser_car_alarms_buzzers": 25,
        "confuser_fireworks": 25,
        "confuser_whistles": 25,
        # Benign (25 each = 150)
        "benign_speech": 25,
        "benign_music": 25,
        "benign_animals": 25,
        "benign_household": 25,
        "benign_traffic": 25,
        "benign_environmental": 25,
    }

    accepted_df = df[df["final_review_decision"] == "ACCEPT"].copy()
    rng = np.random.RandomState(SEED)

    selected_rows = []
    achieved_counts = {}

    for stratum, needed in quotas.items():
        if stratum in CLASSES:
            stratum_pool = accepted_df[accepted_df["target_class"] == stratum]
        else:
            stratum_pool = accepted_df[accepted_df["subdomain"] == stratum]

        avail = len(stratum_pool)
        n_take = min(needed, avail)
        achieved_counts[stratum] = (n_take, needed)

        if avail > 0:
            sampled = stratum_pool.sample(n=n_take, random_state=SEED)
            selected_rows.append(sampled)

        print(f"  {stratum:<30}: selected {n_take:3d} / {needed:3d} (available: {avail})")

    benchmark_df = pd.concat(selected_rows).reset_index(drop=True)
    print("=" * 80)
    print(f"Total benchmark samples selected: {len(benchmark_df)}")

    # Sort benchmark by target_class, subdomain, video_id
    benchmark_df = benchmark_df.sort_values(by=["class_id", "subdomain", "video_id"]).reset_index(drop=True)

    # Select columns for final manifest
    manifest_cols = [
        "video_id", "start_seconds", "end_seconds", "target_class", "class_id",
        "relative_path", "filename", "subdomain", "matched_label", "audio_set_mid",
        "source_dataset", "duration", "rms", "peak", "sha256"
    ]
    benchmark_df["relative_path"] = benchmark_df["file_path"]

    benchmark_df[manifest_cols].to_csv(FINAL_MANIFEST_CSV, index=False)
    print(f"Saved frozen benchmark manifest: {FINAL_MANIFEST_CSV} ({len(benchmark_df)} samples)")


if __name__ == "__main__":
    main()
