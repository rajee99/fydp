import csv
from pathlib import Path
import subprocess
import sys
import wave

# Ensure stdout handles UTF-8 on Windows
sys.stdout.reconfigure(encoding="utf-8")

def extract_pilot_other():
    root = Path(__file__).resolve().parent.parent
    manifest_path = root / "data" / "processed" / "pilot_other_candidates.csv"
    results_path = root / "data" / "processed" / "pilot_other_download_results.csv"

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        candidates = list(csv.DictReader(f))

    print(f"Loaded {len(candidates)} candidates from {manifest_path.name}")

    results = []

    for i, c in enumerate(candidates, 1):
        vid = c["video_id"].strip()
        start = float(c["start_seconds"])
        end = float(c["end_seconds"])
        sub = c["sub_domain"]
        label = c["audio_set_label"]
        rel_out = c["intended_output_path"]
        output_wav = root / rel_out
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(candidates)}] Processing '{sub}' candidate {vid} ({start}-{end}s)")
        print(f"  Target file: {rel_out}")

        record = {
            "video_id": vid,
            "sub_domain": sub,
            "audio_set_label": label,
            "start_seconds": start,
            "end_seconds": end,
            "output_path": rel_out,
            "status": "UNKNOWN",
            "sample_rate": None,
            "channels": None,
            "duration_seconds": None,
            "file_size_bytes": None,
            "error_message": "",
        }

        # Overwrite protection
        if output_wav.exists() and output_wav.stat().st_size > 0:
            print(f"  [INFO] File already exists ({output_wav.name}). Skipping download to prevent overwrite.")
            record["status"] = "SKIPPED_EXISTING"
        else:
            url = f"https://www.youtube.com/watch?v={vid}"
            cmd = [
                "yt-dlp",
                "--no-playlist",
                "-f", "bestaudio",
                "--download-sections", f"*{start}-{end}",
                "--force-keyframes-at-cuts",
                "-x",
                "--audio-format", "wav",
                "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
                "-o", str(output_wav.parent / f"{output_wav.stem}.%(ext)s"),
                url,
            ]

            print("  Downloading and converting section with yt-dlp / FFmpeg...")
            res = subprocess.run(cmd, capture_output=True, text=True)

            if res.returncode != 0:
                err_msg = res.stderr.strip() or res.stdout.strip() or "Extraction process failed."
                first_err_line = err_msg.splitlines()[-1] if err_msg else "Unknown error"
                print(f"  [FAIL] Extraction failed for {vid}: {first_err_line}")
                record["status"] = "FAILED"
                record["error_message"] = first_err_line
                results.append(record)
                continue

        # Verification of the WAV file
        if not output_wav.exists():
            print(f"  [FAIL] Expected output WAV not found on disk: {output_wav}")
            record["status"] = "FAILED"
            record["error_message"] = "Output WAV file missing after download"
            results.append(record)
            continue

        file_size = output_wav.stat().st_size
        if file_size == 0:
            print("  [FAIL] Extracted audio file is empty (0 bytes).")
            record["status"] = "FAILED"
            record["error_message"] = "Extracted WAV file is 0 bytes"
            results.append(record)
            continue

        try:
            with wave.open(str(output_wav), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                frames = wf.getnframes()
                duration = frames / float(sample_rate)

            # Verification assertions
            is_mono = (channels == 1)
            is_16k = (sample_rate == 16000)
            valid_duration = (abs(duration - (end - start)) <= 0.8)

            record["sample_rate"] = sample_rate
            record["channels"] = channels
            record["duration_seconds"] = round(duration, 3)
            record["file_size_bytes"] = file_size

            if is_mono and is_16k and valid_duration:
                record["status"] = "SUCCESS"
                print(f"  [SUCCESS] {output_wav.name}: {channels}ch, {sample_rate}Hz, {duration:.2f}s, {file_size} bytes")
            else:
                record["status"] = "FAILED"
                record["error_message"] = f"Validation failed: mono={is_mono}, 16kHz={is_16k}, dur={duration:.2f}s"
                print(f"  [FAIL] Validation criteria not met: {record['error_message']}")

        except Exception as e:
            record["status"] = "FAILED"
            record["error_message"] = f"Error reading WAV: {str(e)}"
            print(f"  [FAIL] Error reading WAV: {e}")

        results.append(record)

    # Save results to data/processed/pilot_other_download_results.csv
    fieldnames = [
        "video_id", "sub_domain", "audio_set_label",
        "start_seconds", "end_seconds", "output_path",
        "status", "sample_rate", "channels",
        "duration_seconds", "file_size_bytes", "error_message"
    ]

    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll operations finished. Results written to: {results_path}")
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"Summary: {success_count}/{len(candidates)} successful.")


if __name__ == "__main__":
    extract_pilot_other()
