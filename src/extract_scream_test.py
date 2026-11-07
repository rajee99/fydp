import csv
from pathlib import Path
import subprocess
import sys


def extract_scream_samples(manifest_file=None):
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data" / "raw" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest_file:
        manifest_path = Path(manifest_file)
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
    else:
        manifest_path = root / "data" / "processed" / "scream_pilot.csv"
        if not manifest_path.exists():
            manifest_path = root / "data" / "processed" / "scream_replacements_checked.csv"

    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading candidates from: {manifest_path.relative_to(root)}")
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        candidates = list(reader)[:2]

    if not candidates:
        print("[ERROR] No candidates found in manifest.", file=sys.stderr)
        sys.exit(1)

    for i, row in enumerate(candidates, 1):
        video_id = str(row["video_id"]).strip()
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        title = row.get("video_title", "Unknown title")

        filename = f"scream_{video_id}_{int(start)}_{int(end)}.wav"
        target_wav = out_dir / filename

        print(f"\n[{i}/{len(candidates)}] Candidate: {video_id} ('{title}')")
        print(f"Segment: {start}s - {end}s (10s) -> Destination: data/raw/pilot/{filename}")

        # Do not overwrite existing audio files
        if target_wav.exists():
            print(f"[SUCCESS] {filename} already exists (skipping overwrite).")
            continue

        url = f"https://www.youtube.com/watch?v={video_id}"

        # Download only the selected 10s segment and convert to mono 16kHz WAV via FFmpeg
        output_template = str(out_dir / f"{target_wav.stem}.%(ext)s")
        command = [
            "yt-dlp",
            "--no-playlist",
            "-f", "bestaudio",
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-x",
            "--audio-format", "wav",
            "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
            "-o", output_template,
            url,
        ]

        print("Extracting 10-second segment...")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0 and target_wav.exists():
            print(f"[SUCCESS] Successfully extracted: {filename}")
        else:
            err = result.stderr.strip() or result.stdout.strip() or "Extraction process failed."
            last_err = err.splitlines()[-1] if err.splitlines() else err
            print(f"[ERROR] Failed extracting {video_id}: {last_err}")


if __name__ == "__main__":
    manifest_arg = sys.argv[1] if len(sys.argv) > 1 else None
    extract_scream_samples(manifest_arg)

