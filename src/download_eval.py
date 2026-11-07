"""Orchestrates parallel chunked downloading for AudioSet evaluation split.

Executes disjoint slices via `src/batch_downloader.py` and unifies the results
into `data/processed/eval_download_results.csv`.
"""

import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent

EVAL_MANIFEST = ROOT / "data" / "processed" / "eval_manifest.csv"
TRAIN_MANIFEST = ROOT / "data" / "processed" / "target_manifest.csv"
EVAL_RESULTS = ROOT / "data" / "processed" / "eval_download_results.csv"
EVAL_AUDIO_DIR = ROOT / "data" / "raw" / "eval"


def run_parallel_eval_download(num_workers: int = 4):
    print("=" * 80)
    print("  EVALUATION AUDIOSET DATASET ACQUISITION (PARALLEL)")
    print("=" * 80)

    with open(EVAL_MANIFEST, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    total_records = len(records)
    print(f"Total evaluation manifest records: {total_records}")

    chunk_size = (total_records + num_workers - 1) // num_workers
    processes = []
    part_files = []

    print(f"Launching {num_workers} parallel downloader workers (chunk size: ~{chunk_size})...\n")

    for i in range(num_workers):
        start_idx = i * chunk_size
        if start_idx >= total_records:
            break
        limit = min(chunk_size, total_records - start_idx)
        part_file = ROOT / "data" / "processed" / f"eval_download_results_part{i}.csv"
        part_files.append(part_file)

        cmd = [
            sys.executable,
            str(ROOT / "src" / "batch_downloader.py"),
            "--split", "eval",
            "--start-index", str(start_idx),
            "--limit", str(limit),
            "--results-file", str(part_file),
            "--delay", "0.5",
        ]

        print(f"  Worker {i}: rows [{start_idx}..{start_idx + limit - 1}] ({limit} items) -> {part_file.name}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        processes.append((i, p, part_file))

    print("\nAll workers active. Streaming execution status...\n")

    for idx, p, part_f in processes:
        stdout_data, _ = p.communicate()
        if p.returncode != 0:
            print(f"[Worker {idx} ERROR] Exit code {p.returncode}:\n{stdout_data[-500:]}")
        else:
            print(f"[Worker {idx} OK] Processed chunk to {part_f.name}")

    # Merge part results into unified eval_download_results.csv
    print("\nMerging chunk results into unified eval_download_results.csv...")
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

    # Sort merged rows in exact order of eval_manifest
    ordered_results = []
    for r in records:
        key = (r["video_id"].strip(), float(r["start_seconds"]), float(r["end_seconds"]))
        if key in merged_rows:
            ordered_results.append(merged_rows[key])

    with open(EVAL_RESULTS, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_results)

    print(f"Unified results saved to: {EVAL_RESULTS} ({len(ordered_results)} records)")

    # Audit summary
    success_cnt = sum(1 for r in ordered_results if r["status"] in ("SUCCESS", "ALREADY_EXISTS"))
    fail_cnt = sum(1 for r in ordered_results if r["status"] not in ("SUCCESS", "ALREADY_EXISTS"))
    print("\n" + "=" * 80)
    print("  EVALUATION DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Total manifest entries: {total_records}")
    print(f"Successfully available: {success_cnt} ({success_cnt / total_records * 100:.1f}%)")
    print(f"Failed / Unavailable:   {fail_cnt} ({fail_cnt / total_records * 100:.1f}%)")

    # Count WAV files by class
    from dataset import run_preflight_dataset_check
    is_ready, issues, stats = run_preflight_dataset_check(EVAL_AUDIO_DIR)
    print("\nVerified Audio Files in data/raw/eval/:")
    for cname, cnt in stats["classes_found"].items():
        print(f"  {cname:<16}: {cnt:4d} files")
    print(f"Total Valid Files:    {stats['total_valid_files']}")
    print(f"Format Errors:        {stats['format_errors']}")
    print(f"Rate Violations:      {stats['sample_rate_violations']}")
    print("=" * 80)


if __name__ == "__main__":
    run_parallel_eval_download(num_workers=4)
