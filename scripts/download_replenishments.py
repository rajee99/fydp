"""Download Test Set v2 Replenishments quickly with ThreadPoolExecutor."""

import csv
from datetime import datetime
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.batch_downloader import download_single_segment, validate_audio_file, save_result_row

MANIFEST = ROOT / "data" / "processed" / "test_set_v2_replenish_candidates.csv"
RESULTS = ROOT / "data" / "processed" / "test_set_v2_replenish_download_results.csv"
DEST_DIR = ROOT / "data" / "raw" / "test_set_v2"


def download_item(row):
    vid = row["video_id"].strip()
    start = float(row["start_seconds"])
    end = float(row["end_seconds"])
    tc = row["target_class"]
    expected_dur = end - start
    out_filename = f"{tc}_{vid}_{int(start)}_{int(end)}.wav"
    target_wav = DEST_DIR / tc / out_filename
    rel_path = target_wav.relative_to(ROOT).as_posix()

    if target_wav.exists():
        ok, _, stats = validate_audio_file(target_wav, expected_dur)
        if ok:
            return {"video_id": vid, "status": "ALREADY_EXISTS", "output_path": rel_path}

    succ, err, att = download_single_segment(vid, start, end, target_wav, max_retries=2)
    if not succ:
        return {"video_id": vid, "status": "FAILED", "error": err, "output_path": rel_path}

    ok, v_err, stats = validate_audio_file(target_wav, expected_dur)
    if not ok:
        target_wav.unlink(missing_ok=True)
        return {"video_id": vid, "status": "VALIDATION_FAILED", "error": v_err, "output_path": rel_path}

    return {"video_id": vid, "status": "SUCCESS", "output_path": rel_path}


def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    print(f"Downloading {len(records)} replenishment items with 4 threads...")

    fieldnames = ["video_id", "status", "output_path", "error"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(download_item, records))

    with open(RESULTS, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    success_cnt = sum(1 for r in results if r["status"] in ("SUCCESS", "ALREADY_EXISTS"))
    print(f"Replenishment complete: {success_cnt} / {len(records)} successfully verified.")


if __name__ == "__main__":
    main()
