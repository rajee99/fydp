"""Prepare Human Content Validation Manifest for E3 General Background Audio.

Consumes the results from data/processed/e3_download_results.csv and
generates data/processed/e3_content_validation.csv ready for review via
src/review_tool.py.
"""

import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent

DOWNLOAD_RESULTS_PATH = ROOT / "data" / "processed" / "e3_download_results.csv"
CONTENT_VALIDATION_PATH = ROOT / "data" / "processed" / "e3_content_validation.csv"
AUDIO_DIR = ROOT / "data" / "raw" / "e3_general_background"


def prepare_e3_content_validation(
    results_path: Path = DOWNLOAD_RESULTS_PATH,
    output_path: Path = CONTENT_VALIDATION_PATH,
) -> pd.DataFrame:
    """Generate content validation CSV for downloaded E3 audio."""
    if not results_path.exists():
        raise FileNotFoundError(f"Download results file not found: {results_path}")

    df_results = pd.read_csv(results_path)

    # Filter for successful downloads
    successful = df_results[df_results["status"].isin(["SUCCESS", "ALREADY_EXISTS"])].copy()

    records = []
    for _, row in successful.iterrows():
        vid = row["video_id"].strip()
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        target_class = row["target_class"]
        subdomain = row.get("subdomain", "general")
        matched_label = row.get("matched_label", "")

        rel_path = row["output_path"]
        abs_path = ROOT / rel_path

        if not abs_path.exists():
            continue

        try:
            info = sf.info(str(abs_path))
            if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.duration <= 0:
                continue
            dur = round(info.duration, 2)
            sz = abs_path.stat().st_size
        except Exception:
            continue

        records.append({
            "file_path": rel_path.replace("\\", "/"),
            "target_class": target_class,
            "subdomain": subdomain,
            "video_id": vid,
            "start_seconds": start,
            "end_seconds": end,
            "matched_label": matched_label,
            "origin": "e3_general_candidates",
            "duration_seconds": dur,
            "file_size_bytes": sz,
            "validation_status": "PENDING",
            "reviewer_verdict": "PENDING",
            "audited_class": target_class,
            "acoustic_notes": "",
        })

    out_df = pd.DataFrame(records)
    out_df.sort_values(by=["subdomain", "video_id"], inplace=True, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(out_df)} content validation records -> {output_path.name}")
    return out_df


if __name__ == "__main__":
    prepare_e3_content_validation()
