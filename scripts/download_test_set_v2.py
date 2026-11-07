"""Parallel Downloader and Verifier for Test Set v2.

Orchestrates multi-worker downloads of Test Set v2 candidates into:
  data/raw/test_set_v2/{target_class}/

Records download results to:
  data/processed/test_set_v2_download_results.csv
"""

import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import wave
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = ROOT / "data" / "processed" / "test_set_v2_candidates_to_download.csv"
DEST_DIR = ROOT / "data" / "raw" / "test_set_v2"
RESULTS_PATH = ROOT / "data" / "processed" / "test_set_v2_download_results.csv"


def run_parallel_downloads(num_workers: int = 6):
    print("=" * 80)
    print("  TEST SET V2 PARALLEL AUDIO DOWNLOAD")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    total_records = len(records)
    print(f"Total candidates in manifest: {total_records}")
    print(f"Destination audio directory:  {DEST_DIR}")
    print(f"Results destination:          {RESULTS_PATH}")
    print(f"Launching {num_workers} parallel workers...\n")

    chunk_size = (total_records + num_workers - 1) // num_workers
    processes = []
    part_files = []

    for i in range(num_workers):
        start_idx = i * chunk_size
        if start_idx >= total_records:
            break
        limit = min(chunk_size, total_records - start_idx)
        part_file = ROOT / "data" / "processed" / f"test_set_v2_download_part{i}.csv"
        part_files.append(part_file)

        cmd = [
            sys.executable,
            str(ROOT / "src" / "batch_downloader.py"),
            "--manifest", str(MANIFEST_PATH),
            "--dest-dir", str(DEST_DIR),
            "--results-file", str(part_file),
            "--start-index", str(start_idx),
            "--limit", str(limit),
            "--delay", "0.3",
            "--no-reuse-pilot",
        ]

        print(f"  Worker {i}: rows [{start_idx}..{start_idx + limit - 1}] ({limit} items) -> {part_file.name}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        processes.append((i, p, part_file))

    print("\nAll workers active. Waiting for completion...\n")

    for idx, p, part_f in processes:
        stdout_data, _ = p.communicate()
        if p.returncode != 0:
            print(f"[Worker {idx} ERROR] Exit code {p.returncode}:\n{stdout_data[-500:]}")
        else:
            print(f"[Worker {idx} OK] Completed successfully.")

    # Merge part results
    print("\nMerging part results into test_set_v2_download_results.csv...")
    merged_rows = {}
    fieldnames = []

    for _, _, part_f in processes:
        if part_f.exists():
            with open(part_f, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    key = (row["video_id"], float(row["start_seconds"]), float(row["end_seconds"]))
                    merged_rows[key] = row
            part_f.unlink(missing_ok=True)

    ordered_results = []
    for r in records:
        key = (r["video_id"].strip(), float(r["start_seconds"]), float(r["end_seconds"]))
        if key in merged_rows:
            ordered_results.append(merged_rows[key])

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_results)

    print(f"Merged {len(ordered_results)} results into {RESULTS_PATH}")

    # Summary
    success_cnt = sum(1 for r in ordered_results if r["status"] in ("SUCCESS", "ALREADY_EXISTS"))
    fail_cnt = len(ordered_results) - success_cnt
    print("\n" + "=" * 80)
    print("  DOWNLOAD RUN SUMMARY")
    print("=" * 80)
    print(f"Total candidates:      {len(ordered_results)}")
    print(f"Successfully verified: {success_cnt} ({success_cnt / len(ordered_results) * 100:.1f}%)")
    print(f"Failed / Unavailable:  {fail_cnt}")

    # Audit by class
    print("\nAvailable files by class and subdomain:")
    df = pd.DataFrame(ordered_results)
    succ_df = df[df["status"].isin(["SUCCESS", "ALREADY_EXISTS"])]
    for tc in sorted(succ_df["target_class"].unique()):
        tc_df = succ_df[succ_df["target_class"] == tc]
        print(f"  {tc:<18}: {len(tc_df)} verified files")
        if tc == "other":
            for sub in sorted(tc_df["subdomain"].unique()):
                sub_count = len(tc_df[tc_df["subdomain"] == sub])
                print(f"    - {sub:<28}: {sub_count}")


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run_parallel_downloads(num_workers=workers)
