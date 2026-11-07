"""Dedicated Non-Emergency ('other') Audio Dataset Collection Workflow.

Collects non-emergency audio from AudioSet balanced train segments across
six diverse acoustic subdomains:
  1. speech
  2. music
  3. animals
  4. household
  5. traffic
  6. environmental

Enforces strict exclusion of:
- All 4 emergency target classes (scream, glass_breaking, gunshot, siren)
- Confuser MIDs (alarms, explosions, fireworks, yelling, shouting, etc.)
- All 740 video IDs from target_manifest.csv (emergency + existing 175 other)
- All 744 video IDs from eval_manifest.csv (zero eval split leakage)
- All 20 video IDs from pilot_accessible.csv (zero pilot reuse)

All downloaded files are validated to 16 kHz mono 16-bit PCM WAV before being
accepted into data/raw/train/other/.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import subprocess
import sys
import time
import wave
from typing import Dict, List, Set, Tuple

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

try:
    from class_config import (
        OTHER_SUBDOMAINS,
        get_all_exclusion_mids,
    )
    from batch_downloader import validate_audio_file, PERMANENT_ERROR_PATTERNS
except ImportError:
    from src.class_config import (
        OTHER_SUBDOMAINS,
        get_all_exclusion_mids,
    )
    from src.batch_downloader import validate_audio_file, PERMANENT_ERROR_PATTERNS

BALANCED_SEGMENTS = ROOT / "data" / "raw" / "balanced_train_segments.csv"
TRAIN_MANIFEST = ROOT / "data" / "processed" / "target_manifest.csv"
EVAL_MANIFEST = ROOT / "data" / "processed" / "eval_manifest.csv"
PILOT_MANIFEST = ROOT / "data" / "processed" / "pilot_accessible.csv"

OTHER_DIR = ROOT / "data" / "raw" / "train" / "other"
RESULTS_CSV = ROOT / "data" / "processed" / "other_download_results.csv"
MANIFEST_CSV = ROOT / "data" / "processed" / "other_manifest.csv"


def load_excluded_video_ids() -> Set[str]:
    """Gather all video IDs from train, eval, and pilot manifests to prevent leakage."""
    excluded = set()
    manifests = [TRAIN_MANIFEST, EVAL_MANIFEST, PILOT_MANIFEST]
    for mf in manifests:
        if mf.exists():
            with open(mf, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("video_id", "").strip()
                    if vid:
                        excluded.add(vid)
    return excluded


def load_candidates(
    segments_path: Path,
    excluded_vids: Set[str],
    exclusion_mids: Set[str],
    subdomains_config: Dict[str, Dict[str, str]],
) -> Dict[str, List[dict]]:
    """Parse balanced_train_segments.csv and extract clean candidates grouped by subdomain."""
    # Map MID -> (subdomain, display_name)
    mid_to_sub = {}
    for sub, mids in subdomains_config.items():
        for mid, name in mids.items():
            mid_to_sub[mid] = (sub, name)

    candidates = {sub: [] for sub in subdomains_config}
    seen_vids = set(excluded_vids)

    with open(segments_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            ytid = parts[0]
            try:
                start = float(parts[1])
                end = float(parts[2])
                if (end - start) < 9.0:
                    continue
            except ValueError:
                continue

            labels_raw = ",".join(parts[3:]).strip('"').strip()
            mids = [m.strip().strip('"') for m in labels_raw.split(",") if m.strip()]

            # 1. Leakage & duplicate check
            if ytid in seen_vids:
                continue

            # 2. Exclusion MIDs check (emergency + confusers)
            if any(m in exclusion_mids for m in mids):
                continue

            # 3. Match against subdomains
            matched_subs = set()
            matched_details = []
            for m in mids:
                if m in mid_to_sub:
                    sub, name = mid_to_sub[m]
                    matched_subs.add(sub)
                    matched_details.append((sub, m, name))

            # Accept only unambiguous single-subdomain segments
            if len(matched_subs) == 1:
                sub = list(matched_subs)[0]
                candidates[sub].append({
                    "video_id": ytid,
                    "start_seconds": start,
                    "end_seconds": end,
                    "target_class": "other",
                    "subdomain": sub,
                    "matched_mid": matched_details[0][1],
                    "matched_label": matched_details[0][2],
                    "all_mids": ";".join(mids),
                })
                seen_vids.add(ytid)

    return candidates


def download_segment(
    video_id: str,
    start: float,
    end: float,
    output_path: Path,
    max_retries: int = 2,
) -> Tuple[bool, str, int]:
    """Download a segment via yt-dlp and transcode to 16 kHz mono WAV."""
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
        if res.returncode == 0 and output_path.exists():
            return True, "", attempt

        raw_err = res.stderr.strip() or res.stdout.strip()
        last_error = raw_err.splitlines()[-1] if raw_err else f"Exit code {res.returncode}"

        # Check for permanent errors to skip pointless retries
        err_lower = raw_err.lower()
        if any(pat in err_lower for pat in PERMANENT_ERROR_PATTERNS):
            return False, f"PERMANENT: {last_error}", attempt

        if attempt < max_retries:
            time.sleep(2.0 * attempt)

    return False, f"FAILED: {last_error}", max_retries


def log_result(results_path: Path, row: dict, fieldnames: List[str]):
    """Append a log entry to results CSV."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with open(results_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_manifest_row(manifest_path: Path, row: dict, fieldnames: List[str]):
    """Append a validated entry to the other manifest CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with open(manifest_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def collect_other_dataset(
    target_per_subdomain: int = 20,
    delay: float = 1.0,
    seed: int = 42,
    subdomain_filter: str = None,
    dry_run: bool = False,
    max_retries: int = 2,
):
    print("=" * 80)
    print("  AUDIOSET 'OTHER' DATASET COLLECTOR (NON-EMERGENCY CLASS 4)")
    print("=" * 80)

    # 1. Leakage and exclusion preparation
    print("Loading exclusion lists and split leakage guards...")
    excluded_vids = load_excluded_video_ids()
    exclusion_mids = get_all_exclusion_mids()
    print(f"  [OK] Excluded Video IDs: {len(excluded_vids)} (train, eval, pilot)")
    print(f"  [OK] Excluded Confuser MIDs: {len(exclusion_mids)}")

    # 2. Extract and sample candidates
    print(f"\nScanning candidate segments from {BALANCED_SEGMENTS.name}...")
    all_candidates = load_candidates(
        BALANCED_SEGMENTS,
        excluded_vids=excluded_vids,
        exclusion_mids=exclusion_mids,
        subdomains_config=OTHER_SUBDOMAINS,
    )

    subdomains_to_process = (
        [subdomain_filter] if subdomain_filter else list(OTHER_SUBDOMAINS.keys())
    )

    for sub in subdomains_to_process:
        c_list = all_candidates.get(sub, [])
        random.Random(seed).shuffle(c_list)
        all_candidates[sub] = c_list
        print(f"  [{sub:<14}] {len(c_list)} clean candidates queued (seed={seed})")

    OTHER_DIR.mkdir(parents=True, exist_ok=True)

    result_fields = [
        "video_id", "start_seconds", "end_seconds", "target_class",
        "subdomain", "matched_label", "audio_set_mid", "output_path",
        "status", "sample_rate", "channels", "duration_seconds",
        "file_size_bytes", "error_message", "attempts", "timestamp",
    ]

    manifest_fields = [
        "video_id", "start_seconds", "end_seconds", "target_class",
        "subdomain", "matched_label", "audio_set_mid", "output_path",
        "duration_seconds", "sample_rate", "channels",
    ]

    # Check already verified files in data/raw/train/other/
    existing_valid = {}
    if OTHER_DIR.exists():
        for wav_f in OTHER_DIR.glob("*.wav"):
            is_val, _, stats = validate_audio_file(wav_f, expected_duration=10.0, tolerance=1.5)
            if is_val:
                existing_valid[wav_f.name] = stats

    print(f"\nExisting valid 'other' WAVs on disk: {len(existing_valid)}")

    overall_collected = {}

    for sub in subdomains_to_process:
        print("\n" + "-" * 80)
        print(f"  SUBDOMAIN: {sub.upper()} (Target: {target_per_subdomain} valid files)")
        print("-" * 80)

        # Count how many we already have for this subdomain
        sub_valid_count = 0
        for fname in existing_valid:
            if fname.startswith(f"other_{sub}_"):
                sub_valid_count += 1

        print(f"Current valid for {sub}: {sub_valid_count} / {target_per_subdomain}")
        if sub_valid_count >= target_per_subdomain:
            print(f"Target already satisfied for {sub} ({sub_valid_count} valid files).")
            overall_collected[sub] = sub_valid_count
            continue

        candidates = all_candidates.get(sub, [])
        if not candidates:
            print(f"[WARNING] No candidates available for subdomain '{sub}'!")
            overall_collected[sub] = sub_valid_count
            continue

        cand_idx = 0
        attempts_in_sub = 0

        while sub_valid_count < target_per_subdomain and cand_idx < len(candidates):
            c = candidates[cand_idx]
            cand_idx += 1
            attempts_in_sub += 1

            vid = c["video_id"]
            start = c["start_seconds"]
            end = c["end_seconds"]
            expected_dur = end - start
            label = c["matched_label"]
            mid = c["matched_mid"]

            filename = f"other_{sub}_{vid}_{int(start)}_{int(end)}.wav"
            out_path = OTHER_DIR / filename
            rel_out = out_path.relative_to(ROOT).as_posix()

            print(f"[{sub}:{sub_valid_count+1:02d}/{target_per_subdomain}] {vid} ({start}s–{end}s, {label}) -> {filename}")

            # Check if already exists and valid
            if out_path.name in existing_valid:
                print(f"  -> [ALREADY_VALID] Re-using verified file on disk.")
                continue

            if dry_run:
                print("  -> [DRY_RUN] Would download and validate audio.")
                sub_valid_count += 1
                continue

            # Execute segment download
            ok, err_msg, attempts = download_segment(
                vid, start, end, out_path, max_retries=max_retries
            )

            if not ok:
                print(f"  -> [FAIL_DOWNLOAD] {err_msg}")
                # Clean up any leftover partial file
                out_path.unlink(missing_ok=True)
                log_result(RESULTS_CSV, {
                    "video_id": vid, "start_seconds": start, "end_seconds": end,
                    "target_class": "other", "subdomain": sub, "matched_label": label,
                    "audio_set_mid": mid, "output_path": rel_out, "status": "FAILED",
                    "sample_rate": None, "channels": None, "duration_seconds": None,
                    "file_size_bytes": None, "error_message": err_msg,
                    "attempts": attempts, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, result_fields)
                time.sleep(delay)
                continue

            # Strict validation
            v_ok, v_err, stats = validate_audio_file(out_path, expected_duration=expected_dur, tolerance=0.8)
            if not v_ok:
                print(f"  -> [FAIL_VALIDATION] {v_err}")
                out_path.unlink(missing_ok=True)
                log_result(RESULTS_CSV, {
                    "video_id": vid, "start_seconds": start, "end_seconds": end,
                    "target_class": "other", "subdomain": sub, "matched_label": label,
                    "audio_set_mid": mid, "output_path": rel_out, "status": "FAILED_VALIDATION",
                    "sample_rate": stats.get("sample_rate"), "channels": stats.get("channels"),
                    "duration_seconds": stats.get("duration"), "file_size_bytes": stats.get("size_bytes"),
                    "error_message": v_err, "attempts": attempts,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, result_fields)
                time.sleep(delay)
                continue

            # Valid clip accepted!
            sub_valid_count += 1
            existing_valid[out_path.name] = stats
            print(f"  -> [VALID_AUDIO] {stats['duration']}s, 16kHz mono, {stats['size_bytes']} bytes. ({sub_valid_count}/{target_per_subdomain})")

            log_result(RESULTS_CSV, {
                "video_id": vid, "start_seconds": start, "end_seconds": end,
                "target_class": "other", "subdomain": sub, "matched_label": label,
                "audio_set_mid": mid, "output_path": rel_out, "status": "SUCCESS",
                "sample_rate": stats["sample_rate"], "channels": stats["channels"],
                "duration_seconds": stats["duration"], "file_size_bytes": stats["size_bytes"],
                "error_message": "", "attempts": attempts,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, result_fields)

            write_manifest_row(MANIFEST_CSV, {
                "video_id": vid, "start_seconds": start, "end_seconds": end,
                "target_class": "other", "subdomain": sub, "matched_label": label,
                "audio_set_mid": mid, "output_path": rel_out,
                "duration_seconds": stats["duration"],
                "sample_rate": stats["sample_rate"], "channels": stats["channels"],
            }, manifest_fields)

            time.sleep(delay)

        overall_collected[sub] = sub_valid_count
        print(f"\nSubdomain '{sub}' completed with {sub_valid_count} valid audio files.")

    print("\n" + "=" * 80)
    print("  COLLECTION SUMMARY")
    print("=" * 80)
    total_valid = 0
    for sub, count in overall_collected.items():
        print(f"  {sub:<15}: {count:3d} valid files")
        total_valid += count
    print("-" * 80)
    print(f"  TOTAL 'OTHER' VALID FILES: {total_valid}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Collect clean AudioSet non-emergency audio for Class 4 ('other').")
    parser.add_argument("--target-per-subdomain", type=int, default=20, help="Target valid files per subdomain (default: 20).")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between downloads (default: 1.0).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed for sampling (default: 42).")
    parser.add_argument("--subdomain", type=str, default=None, choices=list(OTHER_SUBDOMAINS.keys()), help="Collect only a specific subdomain.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate collection without downloading.")
    parser.add_argument("--max-retries", type=int, default=2, help="Max yt-dlp download attempts per clip.")

    args = parser.parse_args()
    collect_other_dataset(
        target_per_subdomain=args.target_per_subdomain,
        delay=args.delay,
        seed=args.seed,
        subdomain_filter=args.subdomain,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
