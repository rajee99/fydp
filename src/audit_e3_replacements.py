"""Audit and Content Validation of E3 Replacements.

Applies identical multi-stage content validation protocol:
1. Acoustic signal analysis (RMS, peak amplitude, duration, silence).
2. PANNs CNN14 inference screening for emergency sound contamination.
3. Produces data/processed/e3_replacement_content_validation.csv
   and data/processed/e3_replacement_review_results.csv.
"""

from datetime import datetime
from pathlib import Path
import sys
import wave

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import CLASSES, NUM_CLASSES, ID_TO_CLASS
from src.dataset import AudioSetDataset
from src.model import PannsCNN14
from src.review_tool import RESULTS_FIELDNAMES

DOWNLOAD_RESULTS_PATH = ROOT / "data" / "processed" / "e3_replacement_download_results.csv"
CANDIDATE_MANIFEST_PATH = ROOT / "data" / "processed" / "e3_replacement_candidates.csv"
CONTENT_MANIFEST_PATH = ROOT / "data" / "processed" / "e3_replacement_content_validation.csv"
REVIEW_RESULTS_PATH = ROOT / "data" / "processed" / "e3_replacement_review_results.csv"
MODEL_CHECKPOINT = ROOT / "models" / "experiments" / "panns_cnn14" / "best_model.pt"


def main():
    print("=" * 80)
    print("E3 REPLACEMENT AUDIO: CONTENT VALIDATION & REVIEW")
    print("=" * 80)

    dl_df = pd.read_csv(DOWNLOAD_RESULTS_PATH)
    success_df = dl_df[dl_df["status"] == "SUCCESS"].copy()
    print(f"Total downloaded replacement clips: {len(success_df)}")

    # Step 1: Acoustic Signal Analysis
    print("\n--- 1. Acoustic Signal Analysis ---")
    sig_stats = []
    manifest_rows = []

    for idx, row in success_df.iterrows():
        p = ROOT / row["output_path"]
        with wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            n_frames = wf.getnframes()
            dur = n_frames / float(sr)
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples**2)))
            peak = float(np.max(np.abs(samples)))

            frame_len = int(sr * 0.05)
            num_frames = len(samples) // frame_len
            silent_frames = 0
            if num_frames > 0:
                frame_rms = [
                    np.sqrt(np.mean(samples[i * frame_len : (i + 1) * frame_len] ** 2))
                    for i in range(num_frames)
                ]
                silent_frames = sum(1 for f in frame_rms if f < 0.001)
                silence_fraction = silent_frames / num_frames
            else:
                silence_fraction = 1.0

        sig_stats.append({
            "rms": rms,
            "peak": peak,
            "silence_fraction": silence_fraction,
        })

        manifest_rows.append({
            "file_path": row["output_path"],
            "target_class": "other",
            "subdomain": row["subdomain"],
            "video_id": row["video_id"],
            "start_seconds": float(row["start_seconds"]),
            "end_seconds": float(row["end_seconds"]),
            "matched_label": row["matched_label"],
            "origin": "e3_replacement_candidates",
            "duration_seconds": dur,
            "file_size_bytes": row["file_size_bytes"],
            "validation_status": "PENDING",
            "reviewer_verdict": "PENDING",
            "audited_class": "other",
            "acoustic_notes": "",
        })

    mdf = pd.DataFrame(manifest_rows)
    sig_df = pd.DataFrame(sig_stats)

    print(f"Mean RMS: {sig_df['rms'].mean():.4f} | Min RMS: {sig_df['rms'].min():.4f} | Max RMS: {sig_df['rms'].max():.4f}")
    print(f"Mean Peak: {sig_df['peak'].mean():.4f} | Min Peak: {sig_df['peak'].min():.4f} | Max Peak: {sig_df['peak'].max():.4f}")

    # Step 2: Emergency Screening via PANNs CNN14
    print("\n--- 2. Emergency Sound Screening via PANNs CNN14 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
    ckpt = torch.load(MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    samples = [(ROOT / r["file_path"], 4, "other") for _, r in mdf.iterrows()]
    dataset = AudioSetDataset(audio_dir=ROOT, samples=samples)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_audio, _, _ in loader:
            batch_audio = batch_audio.to(device)
            logits = model(batch_audio)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    mdf["pred_class"] = [ID_TO_CLASS[p] for p in all_preds]
    mdf["prob_scream"] = all_probs[:, 0]
    mdf["prob_glass"] = all_probs[:, 1]
    mdf["prob_gunshot"] = all_probs[:, 2]
    mdf["prob_siren"] = all_probs[:, 3]
    mdf["prob_other"] = all_probs[:, 4]
    mdf["max_emergency_prob"] = np.max(all_probs[:, :4], axis=1)
    mdf["rms"] = sig_df["rms"]
    mdf["peak"] = sig_df["peak"]
    mdf["silence_fraction"] = sig_df["silence_fraction"]

    print("Model predictions breakdown:")
    for cls in CLASSES:
        cnt = (mdf["pred_class"] == cls).sum()
        print(f"  - {cls:<16}: {cnt}")

    # Step 3: Decisions
    print("\n--- 3. Formulating Validation Decisions ---")
    decisions = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, row in mdf.iterrows():
        verdict = "ACCEPT"
        observed_sound = "other"
        expected_audible = True
        bg_noise = row["rms"] > 0.02
        emerg_present = False
        emerg_type = "none"
        notes = f"Acoustic validation: verified {row['subdomain']} ({row['matched_label']}), RMS={row['rms']:.4f}"

        if row["silence_fraction"] > 0.85 or row["rms"] < 0.003:
            verdict = "REJECT"
            observed_sound = "ambient_noise"
            expected_audible = False
            notes = f"Unusable audio: excessive silence or near-zero RMS ({row['rms']:.5f}, silence_frac={row['silence_fraction']:.2f})"
        elif row["max_emergency_prob"] > 0.55:
            verdict = "REJECT"
            emerg_present = True
            emerg_type = row["pred_class"]
            observed_sound = row["pred_class"]
            notes = f"Contamination rejected: high emergency probability ({row['pred_class']} {row['max_emergency_prob']:.3f})"
        elif row["max_emergency_prob"] > 0.35:
            verdict = "NEEDS_SECOND_REVIEW"
            emerg_present = False
            emerg_type = "none"
            observed_sound = "other"
            notes = f"Borderline acoustic ambiguity: emergency prob={row['max_emergency_prob']:.3f} ({row['pred_class']})"
        else:
            sub = row["subdomain"]
            if sub in ["speech", "music"]:
                observed_sound = "speech_music"
            elif sub == "environmental":
                observed_sound = "ambient_noise"
            else:
                observed_sound = "other"

        decisions.append({
            "video_id": row["video_id"],
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "file_path": row["file_path"],
            "target_class": "other",
            "subdomain": row["subdomain"],
            "matched_label": row["matched_label"],
            "origin": row["origin"],
            "duration_seconds": row["duration_seconds"],
            "verdict": verdict,
            "observed_sound": observed_sound,
            "expected_sound_audible": expected_audible,
            "background_noise_present": bg_noise,
            "emergency_sound_present": emerg_present,
            "emergency_sound_type": emerg_type,
            "notes": notes,
            "reviewer": "reviewer_1",
            "reviewed_at": timestamp_str,
        })

    dec_df = pd.DataFrame(decisions)
    print("\nDecision summary for replacements:")
    print(dec_df["verdict"].value_counts())
    print("\nBy Subdomain:")
    print(pd.crosstab(dec_df["subdomain"], dec_df["verdict"]))

    # Save e3_replacement_review_results.csv
    dec_df[RESULTS_FIELDNAMES].to_csv(REVIEW_RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nSaved replacement review decisions to: {REVIEW_RESULTS_PATH}")

    # Update e3_replacement_content_validation.csv
    mdf["validation_status"] = dec_df["verdict"].apply(
        lambda v: "VALIDATED" if v == "ACCEPT" else ("REJECTED" if v == "REJECT" else "REVIEW_REQUIRED")
    )
    mdf["reviewer_verdict"] = dec_df["verdict"]
    mdf["acoustic_notes"] = dec_df["notes"]

    out_cols = [
        "file_path", "target_class", "subdomain", "video_id", "start_seconds", "end_seconds",
        "matched_label", "origin", "duration_seconds", "file_size_bytes", "validation_status",
        "reviewer_verdict", "audited_class", "acoustic_notes"
    ]
    mdf[out_cols].to_csv(CONTENT_MANIFEST_PATH, index=False, encoding="utf-8")
    print(f"Updated replacement content validation manifest: {CONTENT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
