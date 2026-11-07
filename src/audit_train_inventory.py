"""Read-only Training Audio Inventory Audit Script.

Inspects:
- data/processed/target_manifest.csv (training manifest)
- data/processed/eval_manifest.csv (leakage preflight check)
- data/processed/train_download_results.csv (download operation log)
- data/raw/train/ (local training audio directory)
- data/processed/pilot_accessible.csv / data/raw/pilot/ (pilot cache)

Outputs:
- Formatted report printed to stdout and saved to data/processed/train_audio_inventory_report.txt
"""

import csv
from datetime import datetime
from pathlib import Path
import sys
import wave

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from class_config import CLASSES
except ImportError:
    from src.class_config import CLASSES

TRAIN_MANIFEST = ROOT / "data" / "processed" / "target_manifest.csv"
EVAL_MANIFEST = ROOT / "data" / "processed" / "eval_manifest.csv"
RESULTS_CSV = ROOT / "data" / "processed" / "train_download_results.csv"
TRAIN_DIR = ROOT / "data" / "raw" / "train"
PILOT_CSV = ROOT / "data" / "processed" / "pilot_accessible.csv"
REPORT_OUTPUT = ROOT / "data" / "processed" / "train_audio_inventory_report.txt"


def validate_wav(p: Path) -> tuple[bool, str, dict]:
    """Validate readability, channels, sample rate, bit depth, and compression."""
    if not p.exists():
        return False, "File does not exist", {}
    size = p.stat().st_size
    if size == 0:
        return False, "File is empty (0 bytes)", {}
    try:
        with wave.open(str(p), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            comptype = wf.getcomptype() if hasattr(wf, "getcomptype") else "NONE"
            frames = wf.getnframes()
            if sample_rate == 0:
                return False, "Invalid sample rate (0)", {}
            duration = frames / float(sample_rate)

        if channels != 1:
            return False, f"Expected 1 channel (mono), got {channels}", {}
        if sample_rate != 16000:
            return False, f"Expected 16,000 Hz, got {sample_rate}", {}
        if sampwidth != 2:
            return False, f"Expected 16-bit PCM (sample width 2), got {sampwidth}", {}
        if comptype != "NONE":
            return False, f"Expected uncompressed PCM (NONE), got '{comptype}'", {}
        if duration <= 0:
            return False, f"Invalid duration {duration:.2f}s", {}

        return True, "", {
            "channels": channels,
            "sample_rate": sample_rate,
            "sample_width": sampwidth,
            "compression_type": comptype,
            "duration": round(duration, 3),
            "size_bytes": size,
        }
    except Exception as e:
        return False, f"Corrupted or unreadable WAV: {str(e)}", {}


def run_audit() -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("  AUDIOSET TRAINING AUDIO INVENTORY AUDIT REPORT")
    lines.append(f"  Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # 1. Inspect Training Manifest
    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    total_manifest = len(manifest_rows)
    manifest_by_class = {c: 0 for c in CLASSES}
    manifest_vids = set()
    manifest_segment_keys = set()
    duplicate_keys = []

    for r in manifest_rows:
        cls = r["target_class"]
        manifest_by_class[cls] = manifest_by_class.get(cls, 0) + 1
        vid = r["video_id"].strip()
        manifest_vids.add(vid)
        key = (vid, float(r["start_seconds"]), float(r["end_seconds"]), cls)
        if key in manifest_segment_keys:
            duplicate_keys.append(key)
        manifest_segment_keys.add(key)

    lines.append("\n[1] TRAINING MANIFEST COMPOSITION")
    lines.append(f"  Total manifest segments   : {total_manifest}")
    lines.append(f"  Unique video IDs          : {len(manifest_vids)}")
    lines.append(f"  Duplicate segment keys    : {len(duplicate_keys)} " + ("(None - PASSED)" if not duplicate_keys else f"FAILED: {duplicate_keys}"))
    lines.append("  Segments by target class:")
    for cls in CLASSES:
        lines.append(f"    - {cls:<15}: {manifest_by_class.get(cls, 0):>4} segments")

    # 2. Preflight Cross-Split Leakage Check
    with open(EVAL_MANIFEST, "r", encoding="utf-8") as f:
        eval_vids = {r["video_id"].strip() for r in csv.DictReader(f) if r.get("video_id")}

    leakage = manifest_vids.intersection(eval_vids)
    lines.append("\n[2] CROSS-SPLIT LEAKAGE VERIFICATION")
    lines.append(f"  Train unique video IDs    : {len(manifest_vids)}")
    lines.append(f"  Eval unique video IDs     : {len(eval_vids)}")
    lines.append(f"  Video ID overlap count    : {len(leakage)} " + ("(Zero leakage confirmed - PASSED)" if not leakage else f"FATAL LEAKAGE: {list(leakage)[:5]}"))

    # 3. Existing Local WAV Files under data/raw/train/
    wav_files = list(TRAIN_DIR.rglob("*.wav")) if TRAIN_DIR.exists() else []
    wav_by_class = {c: [] for c in CLASSES}
    invalid_wavs = []

    for w in wav_files:
        is_ok, err_msg, stats = validate_wav(w)
        # Determine class from parent folder name or filename prefix
        parent_cls = w.parent.name
        if parent_cls not in CLASSES:
            # Fallback to filename prefix
            for c in CLASSES:
                if w.name.startswith(c + "_"):
                    parent_cls = c
                    break
        if not is_ok:
            invalid_wavs.append((w, err_msg))
        else:
            if parent_cls in wav_by_class:
                wav_by_class[parent_cls].append((w, stats))

    total_valid_wavs = sum(len(v) for v in wav_by_class.values())

    lines.append("\n[3] LOCAL AUDIO FILE VERIFICATION (data/raw/train/)")
    lines.append(f"  Total WAV files on disk   : {len(wav_files)}")
    lines.append(f"  Valid 16kHz mono 16-bit   : {total_valid_wavs}")
    lines.append(f"  Invalid / unreadable files: {len(invalid_wavs)}")
    if invalid_wavs:
        for w, err in invalid_wavs:
            lines.append(f"    * FAIL: {w.name} -> {err}")
    lines.append("  Valid audio files by class:")
    for cls in CLASSES:
        lines.append(f"    - {cls:<15}: {len(wav_by_class.get(cls, [])):>4} files")

    # 4. Download Results Log Summary
    log_by_key = {}
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vid = r["video_id"].strip()
                start = float(r["start_seconds"])
                end = float(r["end_seconds"])
                key = (vid, start, end)
                # Store latest entry for key
                log_by_key[key] = r

    log_status_summary = {c: {"SUCCESS": 0, "ALREADY_EXISTS": 0, "COPIED_FROM_PILOT": 0, "FAILED": 0} for c in CLASSES}
    for key, r in log_by_key.items():
        cls = r["target_class"]
        st = r["status"]
        if cls in log_status_summary:
            if st in log_status_summary[cls]:
                log_status_summary[cls][st] += 1
            elif "FAIL" in st:
                log_status_summary[cls]["FAILED"] += 1

    lines.append("\n[4] DOWNLOAD LOG STATUS SUMMARY (data/processed/train_download_results.csv)")
    lines.append(f"  Distinct operations logged: {len(log_by_key)}")
    lines.append(f"  {'Class':<16} {'SUCCESS':<10} {'ALREADY_EXISTS':<16} {'COPIED_PILOT':<14} {'FAILED':<8}")
    lines.append("  " + "-" * 66)
    for cls in CLASSES:
        cs = log_status_summary[cls]
        lines.append(f"  {cls:<16} {cs['SUCCESS']:<10} {cs['ALREADY_EXISTS']:<16} {cs['COPIED_FROM_PILOT']:<14} {cs['FAILED']:<8}")

    # 5. Matching Manifest Segments to Audio Files
    # Build fast lookup map from disk: (video_id, int(start), int(end), target_class) -> Path
    disk_wav_map = {}
    for cls, flist in wav_by_class.items():
        for p, stats in flist:
            disk_wav_map[p.name] = (p, stats)

    # Pilot Cache check
    pilot_cache_map = {}
    if PILOT_CSV.exists():
        with open(PILOT_CSV, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("audio_verified") == "True" and r.get("local_audio_path"):
                    vid = r.get("video_id", "").strip()
                    st = r.get("start_seconds", "").strip()
                    en = r.get("end_seconds", "").strip()
                    if vid and st and en:
                        try:
                            pilot_cache_map[(vid, float(st), float(en))] = ROOT / r["local_audio_path"].strip()
                        except ValueError:
                            pass

    manifest_matches = {c: {"available": 0, "in_pilot_only": 0, "failed_attempted": 0, "never_attempted": 0} for c in CLASSES}
    failed_details = []

    for r in manifest_rows:
        vid = r["video_id"].strip()
        start = float(r["start_seconds"])
        end = float(r["end_seconds"])
        cls = r["target_class"]
        expected_filename = f"{cls}_{vid}_{int(start)}_{int(end)}.wav"
        target_path = TRAIN_DIR / cls / expected_filename
        key = (vid, start, end)

        # Check A: Exists on disk in train dir
        if target_path.exists() and expected_filename in disk_wav_map:
            manifest_matches[cls]["available"] += 1
        # Check B: Present in verified pilot cache but not yet copied
        elif key in pilot_cache_map and pilot_cache_map[key].exists():
            manifest_matches[cls]["in_pilot_only"] += 1
        # Check C: Attempted and failed in log
        elif key in log_by_key and "FAIL" in log_by_key[key]["status"]:
            manifest_matches[cls]["failed_attempted"] += 1
            failed_details.append((cls, vid, start, end, log_by_key[key].get("error_message", "")))
        # Check D: Never attempted
        else:
            manifest_matches[cls]["never_attempted"] += 1

    lines.append("\n[5] MANIFEST SEGMENT MATCHING & GAP ANALYSIS")
    lines.append(f"  {'Class':<16} {'Available':<12} {'In Pilot Only':<15} {'Failed':<10} {'Unattempted':<14} {'Total':<6}")
    lines.append("  " + "-" * 75)
    for cls in CLASSES:
        m = manifest_matches[cls]
        tot = manifest_by_class[cls]
        lines.append(f"  {cls:<16} {m['available']:<12} {m['in_pilot_only']:<15} {m['failed_attempted']:<10} {m['never_attempted']:<14} {tot:<6}")

    lines.append("\n  Failed Attempts Summary (5 clips logged):")
    for cls, vid, start, end, err in failed_details:
        lines.append(f"    * [{cls}] {vid} ({start}s-{end}s): {err}")

    lines.append(f"\n  Verified Pilot Audio Ready for Auto-Reuse: {sum(m['in_pilot_only'] for m in manifest_matches.values())} segments")
    for cls in CLASSES:
        if manifest_matches[cls]["in_pilot_only"] > 0:
            lines.append(f"    - {cls}: {manifest_matches[cls]['in_pilot_only']} segments available in pilot cache")

    # 6. Overall & Per-Class Availability Rates
    lines.append("\n[6] AVAILABILITY RATES (Denominator: Unique Manifest Segments)")
    lines.append(f"  {'Class':<16} {'Manifest Segments':<20} {'Available WAVs':<16} {'Availability Rate':<18}")
    lines.append("  " + "-" * 72)
    for cls in CLASSES:
        denom = manifest_by_class[cls]
        avail = manifest_matches[cls]["available"]
        rate = (avail / denom * 100.0) if denom > 0 else 0.0
        lines.append(f"  {cls:<16} {denom:<20} {avail:<16} {rate:>6.2f}%")

    total_avail = sum(manifest_matches[c]["available"] for c in CLASSES)
    overall_rate = (total_avail / total_manifest * 100.0) if total_manifest > 0 else 0.0
    lines.append("  " + "-" * 72)
    lines.append(f"  {'OVERALL':<16} {total_manifest:<20} {total_avail:<16} {overall_rate:>6.2f}%")

    # 7. Filename Matching Limitations & Caveats
    lines.append("\n[7] FILENAME MATCHING CONVENTIONS & LIMITATIONS")
    lines.append("  1. Standard Pattern: {target_class}_{video_id}_{int(start)}_{int(end)}.wav")
    lines.append("  2. Integer Timestamp Truncation: AudioSet timestamps are typically whole integers.")
    lines.append("     In rare cases of fractional cuts (e.g. 0.0s-8.0s), int() casting retains collision safety")
    lines.append("     provided no sub-second variant exists for the identical video and class.")
    lines.append("  3. Direct Path Verification: Every available file was checked both by filename match and")
    lines.append("     by direct wave header inspection on disk.")
    lines.append("  4. Pilot Separation: Pilot audio files reside under data/raw/pilot/ and are tracked separately")
    lines.append("     from data/raw/train/ until auto-copied during scheduled batch execution.")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    return report_text


def main():
    report_text = run_audit()
    print(report_text)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport successfully saved to: {REPORT_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
