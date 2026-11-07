"""Robust, Resumable Batch Audio Downloader for AudioSet Manifests.

This script processes `data/processed/target_manifest.csv` (train) and
`data/processed/eval_manifest.csv` (eval) to download and extract normalized
10-second audio segments (16 kHz, mono, 16-bit PCM WAV).

Features:
- Strict pre-flight cross-split data leakage prevention.
- Segment-level extraction via yt-dlp (no full-video downloads).
- Safe resumption: skips pre-existing files only after validating audio format.
- Audio validation: checks 16 kHz, mono, duration, non-empty, wave readability.
- Re-uses verified pilot audio for training segments where identical.
- Structured directory layout: data/raw/{split}/{class_name}/
- Comprehensive status tracking: data/processed/{split}_download_results.csv
- CLI options for dry-run, pilot batches, and resumable full runs.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

# Central Class Configuration
try:
    from class_config import CLASSES, CLASS_TO_ID
except ImportError:
    from src.class_config import CLASSES, CLASS_TO_ID

TRAIN_MANIFEST = ROOT / "data" / "processed" / "target_manifest.csv"
EVAL_MANIFEST = ROOT / "data" / "processed" / "eval_manifest.csv"
PILOT_MANIFEST = ROOT / "data" / "processed" / "pilot_accessible.csv"

# Permanent error signatures in yt-dlp stderr/stdout
PERMANENT_ERROR_PATTERNS = [
    "video unavailable",
    "video is unavailable",
    "this video is unavailable",
    "video is not available",
    "this video is not available",
    "private video",
    "video is private",
    "this video is private",
    "this video has been removed",
    "account associated with this video has been terminated",
    "not available in your country",
    "available in your country",
    "sign in to confirm your age",
    "please sign in",
    "copyright ground",
    "terms of service",
    "community guidelines",
]


def validate_audio_file(file_path: Path, expected_duration: float, tolerance: float = 0.8):
    """Validate that a WAV file exists, is non-empty, and matches 16 kHz mono 16-bit PCM specs.

    Returns:
        (is_valid: bool, error_msg: str, stats: dict)
    """
    if not file_path.exists():
        return False, "File does not exist", {}

    size = file_path.stat().st_size
    if size == 0:
        return False, "File is empty (0 bytes)", {}

    try:
        with wave.open(str(file_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            comptype = wf.getcomptype() if hasattr(wf, "getcomptype") else "NONE"
            if sample_rate == 0:
                return False, "Invalid sample rate (0)", {}
            duration = frames / float(sample_rate)

        if channels != 1:
            return False, f"Expected 1 channel (mono), got {channels}", {}
        if sample_rate != 16000:
            return False, f"Expected 16,000 Hz, got {sample_rate}", {}
        if sampwidth != 2:
            return False, f"Expected 16-bit PCM (sample width 2), got sample width {sampwidth} ({sampwidth * 8}-bit)", {}
        if comptype != "NONE":
            return False, f"Expected uncompressed PCM (compression type 'NONE'), got '{comptype}'", {}
        if frames <= 0:
            return False, "Audio file contains 0 frames", {}
        if abs(duration - expected_duration) > tolerance:
            return False, f"Duration {duration:.2f}s deviates from expected {expected_duration:.2f}s", {}

        return True, "", {
            "channels": channels,
            "sample_rate": sample_rate,
            "sample_width": sampwidth,
            "compression_type": comptype,
            "duration": round(duration, 3),
            "size_bytes": size,
        }
    except Exception as e:
        return False, f"Corrupted or invalid WAV file: {str(e)}", {}


def check_preflight_leakage(train_path: Path, eval_path: Path):
    """Ensure zero video ID overlap between training and evaluation manifests."""
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Manifest missing: train={train_path.exists()}, eval={eval_path.exists()}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_vids = set(r["video_id"].strip() for r in csv.DictReader(f) if r.get("video_id"))

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_vids = set(r["video_id"].strip() for r in csv.DictReader(f) if r.get("video_id"))

    overlap = train_vids.intersection(eval_vids)
    if overlap:
        raise RuntimeError(
            f"FATAL: Cross-split data leakage detected! {len(overlap)} video IDs appear in both "
            f"train and eval manifests: {list(overlap)[:5]}"
        )

    return train_vids, eval_vids


def load_pilot_cache():
    """Map (video_id, start_seconds, end_seconds) to verified local audio paths in pilot."""
    cache = {}
    if not PILOT_MANIFEST.exists():
        return cache

    with open(PILOT_MANIFEST, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("audio_verified") == "True" and r.get("local_audio_path"):
                vid = r.get("video_id", "").strip()
                start = r.get("start_seconds", "").strip()
                end = r.get("end_seconds", "").strip()
                if vid and start and end:
                    try:
                        key = (vid, float(start), float(end))
                        local_path = ROOT / r["local_audio_path"].strip()
                        if local_path.exists():
                            cache[key] = local_path
                    except ValueError:
                        pass
    return cache


def load_previous_results(results_path: Path):
    """Load previously recorded results to enable smart resumption."""
    results_map = {}
    if not results_path.exists():
        return results_map

    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["video_id"], float(row["start_seconds"]), float(row["end_seconds"]))
            results_map[key] = row
    return results_map


def save_result_row(results_path: Path, row_dict: dict, fieldnames: list):
    """Append a single result row safely to the results CSV."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists() or results_path.stat().st_size == 0

    for attempt in range(10):
        try:
            with open(results_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row_dict)
            break
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))


def download_single_segment(video_id: str, start: float, end: float, output_path: Path, max_retries: int = 2):
    """Download a specific time segment with yt-dlp and convert to 16 kHz mono WAV."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestaudio",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
        "-o", str(output_path.parent / f"{output_path.stem}.%(ext)s"),
        url,
    ]

    last_error = ""
    for attempt in range(1, max_retries + 1):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return True, "", attempt

        raw_err = res.stderr.strip() or res.stdout.strip()
        last_error = raw_err.splitlines()[-1] if raw_err else f"Exit code {res.returncode}"

        # If error is permanent (video deleted, private, blocked), don't retry
        err_lower = raw_err.lower()
        if any(pat in err_lower for pat in PERMANENT_ERROR_PATTERNS):
            return False, f"PERMANENT: {last_error}", attempt

        if attempt < max_retries:
            time.sleep(2.0 * attempt)

    return False, f"TRANSIENT_EXHAUSTED: {last_error}", max_retries


def select_manifest_records(
    manifest_records: list,
    class_name: str = None,
    video_id: str = None,
    start_index: int = None,
    limit: int = None,
    priority: str = None,
) -> list:
    """Select and slice manifest records based on filtering and range parameters.

    Args:
        manifest_records: Full list of manifest dict rows.
        class_name: Optional target class filter (must be in CLASSES).
        video_id: Optional video ID filter (targets a specific video). Incompatible with range option start_index.
        start_index: 0-based starting index into filtered rows for range batching (defaults to 0).
                     Cannot be combined with video_id.
        limit: Maximum number of rows to process from start_index.
        priority: Optional candidate_priority filter (e.g. 'P1_PRIMARY').

    Returns:
        Sliced list of manifest record dicts.
    """
    records = list(manifest_records)

    if priority:
        records = [r for r in records if r.get("candidate_priority") == priority]

    if class_name:
        if class_name not in CLASSES:
            raise ValueError(f"Invalid class '{class_name}'. Must be one of: {CLASSES}")
        records = [r for r in records if r["target_class"] == class_name]

    if video_id:
        if start_index is not None:
            raise ValueError(
                f"Incompatible options: cannot combine --video-id ('{video_id}') with range option "
                f"--start-index ({start_index}). Use --video-id for single clip extraction or --start-index for batch range processing."
            )
        records = [r for r in records if r["video_id"].strip() == video_id.strip()]
        if not records:
            raise ValueError(f"Video ID '{video_id}' not found in eligible manifest records.")

    idx = 0 if start_index is None else start_index
    if idx < 0:
        raise ValueError(f"Invalid --start-index {idx}: must be a non-negative integer (>= 0).")

    total_eligible = len(records)
    if total_eligible == 0 and idx > 0:
        raise ValueError(f"Start index {idx} is out of range for 0 eligible records.")
    if total_eligible > 0 and idx >= total_eligible:
        raise ValueError(f"Start index {idx} is out of range for {total_eligible} eligible records (valid: 0..{total_eligible - 1}).")

    if limit is not None:
        if limit < 0:
            raise ValueError(f"Invalid --limit {limit}: must be a non-negative integer (>= 0).")
        records = records[idx : idx + limit]
    else:
        records = records[idx:]

    return records


def run_batch_downloader(
    split: str = "train",
    limit: int = None,
    class_name: str = None,
    video_id: str = None,
    dry_run: bool = False,
    max_retries: int = 2,
    delay: float = 0.5,
    reuse_pilot: bool = True,
    start_index: int = None,
    results_file: str = None,
    manifest: str = None,
    dest_dir: str = None,
    priority: str = None,
):
    print("=" * 80)
    split_display = split.upper() if not manifest else f"CUSTOM: {Path(manifest).name}"
    print(f"  AUDIOSET BATCH AUDIO DOWNLOADER [{split_display}]")
    print("=" * 80)

    # 1. Pre-flight cross-split leakage check
    print("Executing pre-flight cross-split leakage check...")
    train_vids, eval_vids = check_preflight_leakage(TRAIN_MANIFEST, EVAL_MANIFEST)
    print(f"  [OK] Train manifest unique videos: {len(train_vids)}")
    print(f"  [OK] Eval manifest unique videos: {len(eval_vids)}")
    print(f"  [OK] Cross-split video ID overlap: 0 (No leakage confirmed)\n")

    if manifest:
        manifest_file = Path(manifest)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Custom manifest file not found: {manifest}")
        # Check custom manifest against evaluation videos
        with open(manifest_file, "r", encoding="utf-8") as f:
            custom_vids = set(r["video_id"].strip() for r in csv.DictReader(f) if r.get("video_id"))
        custom_overlap = custom_vids.intersection(eval_vids)
        if custom_overlap:
            raise RuntimeError(
                f"FATAL: Custom manifest contains {len(custom_overlap)} evaluation video IDs: {list(custom_overlap)[:5]}"
            )
        print(f"  [OK] Custom manifest cross-check: 0 overlap with evaluation videos ({len(custom_vids)} unique vids)")
    else:
        manifest_file = TRAIN_MANIFEST if split == "train" else EVAL_MANIFEST

    if dest_dir:
        dest_dir = Path(dest_dir)
        if not dest_dir.is_absolute():
            dest_dir = ROOT / dest_dir
    else:
        dest_dir = ROOT / "data" / "raw" / ("expansion" if manifest else split)

    if results_file:
        results_path = Path(results_file)
        if not results_path.is_absolute():
            results_path = ROOT / results_path
    else:
        results_path = ROOT / "data" / "processed" / (f"{split}_download_results.csv" if not manifest else "expansion_download_results.csv")

    # 2. Load manifest entries
    with open(manifest_file, "r", encoding="utf-8") as f:
        all_records = list(csv.DictReader(f))

    records = select_manifest_records(
        all_records,
        class_name=class_name,
        video_id=video_id,
        start_index=start_index,
        limit=limit,
        priority=priority,
    )

    actual_start = 0 if start_index is None else start_index
    print(f"Processing manifest: '{manifest_file.name}' ({len(records)} segments queued, start_index={actual_start})")
    if priority:
        print(f"Priority filter: {priority}")
    print(f"Target directory: {dest_dir.relative_to(ROOT) if dest_dir.is_relative_to(ROOT) else dest_dir}")
    print(f"Results log: {results_path.relative_to(ROOT) if results_path.is_relative_to(ROOT) else results_path}")
    print(f"Dry run mode: {dry_run}\n")

    # Never reuse pilot audio for custom expansion manifests
    if manifest:
        reuse_pilot = False

    pilot_cache = load_pilot_cache() if (reuse_pilot and split == "train") else {}
    if pilot_cache:
        print(f"Loaded {len(pilot_cache)} verified pilot audio segments available for reuse in train.")

    previous_results = load_previous_results(results_path)

    fieldnames = [
        "video_id", "start_seconds", "end_seconds", "target_class",
        "subdomain", "matched_label", "audio_set_mid", "output_path", "status",
        "sample_rate", "channels", "duration_seconds", "file_size_bytes",
        "error_message", "attempts", "timestamp"
    ]

    summary = {
        "total_queued": len(records),
        "already_exists": 0,
        "copied_pilot": 0,
        "downloaded_success": 0,
        "failed": 0,
        "skipped_dry_run": 0,
    }

    for i, r in enumerate(records, 1):
        vid = r["video_id"].strip()
        start = float(r["start_seconds"])
        end = float(r["end_seconds"])
        target_class = r["target_class"]
        subdomain = r.get("subdomain", "")
        matched_label = r["matched_label"]
        mid = r["audio_set_mid"]
        expected_dur = end - start

        # Organized path: data/raw/{split}/{target_class}/{target_class}_{vid}_{start}_{end}.wav
        out_filename = f"{target_class}_{vid}_{int(start)}_{int(end)}.wav"
        target_wav = dest_dir / target_class / out_filename
        rel_target_wav = target_wav.relative_to(ROOT).as_posix()

        print(f"[{i:03d}/{len(records)}] {target_class:<14} | {vid} ({start}s–{end}s) -> {out_filename}")

        # Check A: Does a valid file already exist at destination?
        is_valid, val_err, stats = validate_audio_file(target_wav, expected_dur)
        if is_valid:
            print(f"  -> [ALREADY_EXISTS] Valid audio file verified ({stats['duration']}s, {stats['size_bytes']}B)")
            summary["already_exists"] += 1
            if not dry_run and (vid, start, end) not in previous_results:
                save_result_row(results_path, {
                    "video_id": vid, "start_seconds": start, "end_seconds": end,
                    "target_class": target_class, "subdomain": subdomain,
                    "matched_label": matched_label, "audio_set_mid": mid,
                    "output_path": rel_target_wav, "status": "ALREADY_EXISTS",
                    "sample_rate": stats["sample_rate"], "channels": stats["channels"],
                    "duration_seconds": stats["duration"], "file_size_bytes": stats["size_bytes"],
                    "error_message": "", "attempts": 0, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, fieldnames)
            continue

        # Check B: Can we reuse an already verified pilot WAV for training?
        pilot_key = (vid, start, end)
        if pilot_key in pilot_cache:
            pilot_src = pilot_cache[pilot_key]
            p_valid, p_err, p_stats = validate_audio_file(pilot_src, expected_dur)
            if p_valid:
                print(f"  -> [REUSE_PILOT] Copying from verified pilot: {pilot_src.name}")
                if not dry_run:
                    target_wav.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pilot_src, target_wav)
                    save_result_row(results_path, {
                        "video_id": vid, "start_seconds": start, "end_seconds": end,
                        "target_class": target_class, "subdomain": subdomain,
                        "matched_label": matched_label, "audio_set_mid": mid,
                        "output_path": rel_target_wav, "status": "COPIED_FROM_PILOT",
                        "sample_rate": p_stats["sample_rate"], "channels": p_stats["channels"],
                        "duration_seconds": p_stats["duration"], "file_size_bytes": p_stats["size_bytes"],
                        "error_message": "", "attempts": 0, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, fieldnames)
                summary["copied_pilot"] += 1
                continue

        # Check C: Dry-run check
        if dry_run:
            print("  -> [DRY_RUN] Would execute segment download via yt-dlp")
            summary["skipped_dry_run"] += 1
            continue

        # Execution: Download segment with yt-dlp
        print(f"  -> [DOWNLOADING] Fetching {expected_dur}s segment from YouTube...")
        success, err_msg, attempts = download_single_segment(vid, start, end, target_wav, max_retries=max_retries)

        if not success:
            print(f"  -> [FAIL] {err_msg}")
            summary["failed"] += 1
            save_result_row(results_path, {
                "video_id": vid, "start_seconds": start, "end_seconds": end,
                "target_class": target_class, "subdomain": subdomain,
                "matched_label": matched_label, "audio_set_mid": mid,
                "output_path": rel_target_wav, "status": "FAILED",
                "sample_rate": None, "channels": None, "duration_seconds": None,
                "file_size_bytes": None, "error_message": err_msg, "attempts": attempts,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, fieldnames)
            time.sleep(delay)
            continue

        # Check D: Post-download strict validation
        v_ok, v_err, v_stats = validate_audio_file(target_wav, expected_dur)
        if not v_ok:
            print(f"  -> [FAIL_POST_VALIDATION] {v_err}")
            summary["failed"] += 1
            target_wav.unlink(missing_ok=True)
            save_result_row(results_path, {
                "video_id": vid, "start_seconds": start, "end_seconds": end,
                "target_class": target_class, "subdomain": subdomain,
                "matched_label": matched_label, "audio_set_mid": mid,
                "output_path": rel_target_wav, "status": "VALIDATION_FAILED",
                "sample_rate": None, "channels": None, "duration_seconds": None,
                "file_size_bytes": None, "error_message": v_err, "attempts": attempts,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, fieldnames)
        else:
            print(f"  -> [SUCCESS] 16kHz mono verified ({v_stats['duration']}s, {v_stats['size_bytes']}B)")
            summary["downloaded_success"] += 1
            save_result_row(results_path, {
                "video_id": vid, "start_seconds": start, "end_seconds": end,
                "target_class": target_class, "subdomain": subdomain,
                "matched_label": matched_label, "audio_set_mid": mid,
                "output_path": rel_target_wav, "status": "SUCCESS",
                "sample_rate": v_stats["sample_rate"], "channels": v_stats["channels"],
                "duration_seconds": v_stats["duration"], "file_size_bytes": v_stats["size_bytes"],
                "error_message": "", "attempts": attempts,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, fieldnames)

        time.sleep(delay)

    print("\n" + "=" * 80)
    print("  DOWNLOAD RUN SUMMARY")
    print("=" * 80)
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Batch Audio Downloader for AudioSet Manifests")
    parser.add_argument("--split", default="train", help="Manifest split to process (default: train)")
    parser.add_argument("--manifest", type=str, default=None, help="Custom manifest CSV file path")
    parser.add_argument("--dest-dir", type=str, default=None, help="Custom destination directory for audio files")
    parser.add_argument("--priority", type=str, default=None, help="Filter candidate rows by priority (e.g. P1_PRIMARY)")
    parser.add_argument("--start-index", type=int, default=None, help="Zero-based starting index into filtered manifest rows (default: 0)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of segments to process from start-index (e.g. 20)")
    parser.add_argument("--class-name", choices=CLASSES, default=None, help="Filter for a specific target class")
    parser.add_argument("--video-id", type=str, default=None, help="Filter for a specific video ID")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without downloading or writing files")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum download retry attempts for transient errors")
    parser.add_argument("--delay", type=float, default=0.5, help="Pause in seconds between downloads to avoid rate limits")
    parser.add_argument("--no-reuse-pilot", dest="reuse_pilot", action="store_false", help="Disable copying pre-verified pilot files for train split")
    parser.add_argument("--results-file", type=str, default=None, help="Custom results CSV file path")

    args = parser.parse_args()
    run_batch_downloader(
        split=args.split,
        limit=args.limit,
        class_name=args.class_name,
        video_id=args.video_id,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        delay=args.delay,
        reuse_pilot=args.reuse_pilot,
        start_index=args.start_index,
        results_file=args.results_file,
        manifest=args.manifest,
        dest_dir=args.dest_dir,
        priority=args.priority,
    )


if __name__ == "__main__":
    main()
