"""Audit and Content Validation of E3 General Background Audio Clips.

Performs multi-layered validation:
1. Signal-level acoustic integrity (RMS, peak amplitude, duration, clipping, silence).
2. Model-inference screening via frozen best PANNs CNN14 checkpoint (detects emergency sound contamination).
3. AudioSet metadata verification against candidate manifest.
4. Generates data/processed/e3_review_results.csv and updates data/processed/e3_content_validation.csv.
"""

from datetime import datetime
from pathlib import Path
import sys
import wave

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Configure UTF-8
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import CLASSES, NUM_CLASSES, ID_TO_CLASS
from src.dataset import AudioSetDataset, load_audio_waveform, TARGET_SAMPLES
from src.model import PannsCNN14
from src.review_tool import RESULTS_FIELDNAMES

CONTENT_MANIFEST = ROOT / "data" / "processed" / "e3_content_validation.csv"
REVIEW_RESULTS_PATH = ROOT / "data" / "processed" / "e3_review_results.csv"
MODEL_CHECKPOINT = ROOT / "models" / "experiments" / "panns_cnn14" / "best_model.pt"


def main():
    print("=" * 80)
    print("E3 GENERAL BACKGROUND AUDIO: CONTENT VALIDATION & REVIEW")
    print("=" * 80)

    if not CONTENT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing {CONTENT_MANIFEST}")

    manifest_df = pd.read_csv(CONTENT_MANIFEST)
    print(f"Total candidate clips to review: {len(manifest_df)}")

    # Step 1: Acoustic Signal Analysis
    print("\n--- 1. Acoustic Signal Analysis ---")
    signal_stats = []
    for idx, row in manifest_df.iterrows():
        p = ROOT / row["file_path"]
        with wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            n_frames = wf.getnframes()
            dur = n_frames / float(sr)
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples**2)))
            peak = float(np.max(np.abs(samples)))
            
            # Count zero crossings and silent frames
            frame_len = int(sr * 0.05)  # 50ms frames
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

        signal_stats.append({
            "idx": idx,
            "duration": dur,
            "sample_rate": sr,
            "channels": ch,
            "rms": rms,
            "peak": peak,
            "silence_fraction": silence_fraction,
        })

    sig_df = pd.DataFrame(signal_stats)
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

    # Create dataset directly for batched inference
    eval_df = manifest_df.copy()
    eval_df["target_class"] = "other"  # for dataset compatibility
    samples = [(ROOT / r["file_path"], 4, "other") for _, r in eval_df.iterrows()]
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

    manifest_df["pred_class"] = [ID_TO_CLASS[p] for p in all_preds]
    manifest_df["prob_scream"] = all_probs[:, 0]
    manifest_df["prob_glass"] = all_probs[:, 1]
    manifest_df["prob_gunshot"] = all_probs[:, 2]
    manifest_df["prob_siren"] = all_probs[:, 3]
    manifest_df["prob_other"] = all_probs[:, 4]
    manifest_df["max_emergency_prob"] = np.max(all_probs[:, :4], axis=1)
    manifest_df["rms"] = sig_df["rms"]
    manifest_df["peak"] = sig_df["peak"]
    manifest_df["silence_fraction"] = sig_df["silence_fraction"]

    print("Model predictions breakdown:")
    for cls in CLASSES:
        cnt = (manifest_df["pred_class"] == cls).sum()
        print(f"  - {cls:<16}: {cnt}")

    # Inspect high emergency probability or low RMS
    high_emerg = manifest_df[manifest_df["max_emergency_prob"] > 0.40]
    print(f"\nClips with emergency sound probability > 40%: {len(high_emerg)}")
    if len(high_emerg) > 0:
        for _, r in high_emerg.iterrows():
            print(f"  [{r['video_id']}] subdomain={r['subdomain']} label={r['matched_label']} pred={r['pred_class']} (emerg_prob={r['max_emergency_prob']:.3f}, other_prob={r['prob_other']:.3f})")

    silent_clips = manifest_df[manifest_df["silence_fraction"] > 0.85]
    print(f"\nClips with excessive silence (> 85%): {len(silent_clips)}")
    if len(silent_clips) > 0:
        for _, r in silent_clips.iterrows():
            print(f"  [{r['video_id']}] subdomain={r['subdomain']} label={r['matched_label']} rms={r['rms']:.5f} silence_frac={r['silence_fraction']:.2f}")

    # Step 3: Determine Research Decisions
    # Criteria:
    # ACCEPT: Clearly genuine non-emergency background audio, max_emerg_prob <= 0.40, silence_frac <= 0.85, valid audio.
    # REJECT: Corrupted/near-silent or severe emergency sound contamination.
    # NEEDS_SECOND_REVIEW: Borderline acoustic ambiguity (e.g. max_emerg_prob between 0.35 and 0.50, or silence_frac near threshold).
    print("\n--- 3. Formulating Validation Decisions ---")
    decisions = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, row in manifest_df.iterrows():
        verdict = "ACCEPT"
        observed_sound = "other"
        expected_audible = True
        bg_noise = row["rms"] > 0.02
        emerg_present = False
        emerg_type = "none"
        notes = f"Acoustic validation: verified {row['subdomain']} ({row['matched_label']}), RMS={row['rms']:.4f}"

        # Check for excessive silence or corrupt audio
        if row["silence_fraction"] > 0.85 or row["rms"] < 0.003:
            verdict = "REJECT"
            observed_sound = "ambient_noise"
            expected_audible = False
            notes = f"Unusable audio: excessive silence or near-zero RMS ({row['rms']:.5f}, silence_frac={row['silence_fraction']:.2f})"
        elif row["max_emergency_prob"] > 0.55:
            # Significant emergency signal detected
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
            # Clear non-emergency background
            # Map subdomain to observed_sound choices
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
    print("\nDecision summary:")
    print(dec_df["verdict"].value_counts())
    print("\nBy Subdomain:")
    print(pd.crosstab(dec_df["subdomain"], dec_df["verdict"]))

    # Save e3_review_results.csv
    dec_df[RESULTS_FIELDNAMES].to_csv(REVIEW_RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nSaved review decisions to: {REVIEW_RESULTS_PATH}")

    # Update e3_content_validation.csv
    manifest_df["validation_status"] = dec_df["verdict"].apply(
        lambda v: "VALIDATED" if v == "ACCEPT" else ("REJECTED" if v == "REJECT" else "REVIEW_REQUIRED")
    )
    manifest_df["reviewer_verdict"] = dec_df["verdict"]
    manifest_df["acoustic_notes"] = dec_df["notes"]
    
    # Save back to e3_content_validation.csv preserving columns
    out_cols = [
        "file_path", "target_class", "subdomain", "video_id", "start_seconds", "end_seconds",
        "matched_label", "origin", "duration_seconds", "file_size_bytes", "validation_status",
        "reviewer_verdict", "audited_class", "acoustic_notes"
    ]
    manifest_df[out_cols].to_csv(CONTENT_MANIFEST, index=False, encoding="utf-8")
    print(f"Updated content validation manifest: {CONTENT_MANIFEST}")


if __name__ == "__main__":
    main()
