import csv
from pathlib import Path
import subprocess
import sys
import wave


def extract_first_gunshot_candidate():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data" / "raw" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "data" / "processed" / "gunshot_candidates.csv"
    output_wav = out_dir / "gunshot_test.wav"

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        candidates = list(reader)

    if not candidates:
        print("[ERROR] No candidates found in manifest.", file=sys.stderr)
        sys.exit(1)

    candidate = candidates[0]
    video_id = candidate["video_id"].strip()
    start = float(candidate["start_seconds"])
    end = float(candidate["end_seconds"])
    title = candidate.get("video_title", "Unknown")

    print(f"Candidate: {video_id} ('{title}')")
    print(f"Segment: {start}s - {end}s ({end - start}s) -> Target: {output_wav.relative_to(root)}")

    # Overwrite protection
    if output_wav.exists():
        print(f"[INFO] {output_wav.name} already exists. Skipping download to prevent overwrite.")
    else:
        url = f"https://www.youtube.com/watch?v={video_id}"
        command = [
            "yt-dlp",
            "--no-playlist",
            "-f", "bestaudio",
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-x",
            "--audio-format", "wav",
            "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
            "-o", str(out_dir / f"{output_wav.stem}.%(ext)s"),
            url,
        ]

        print("Requesting selected audio segment with yt-dlp / FFmpeg...")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip() or "Extraction process failed."
            print(f"[ERROR] Extraction failed for {video_id}: {err}", file=sys.stderr)
            sys.exit(1)

    # Verification of the resulting WAV file
    if not output_wav.exists():
        print(f"[ERROR] Expected output file not found: {output_wav}", file=sys.stderr)
        sys.exit(1)

    file_size = output_wav.stat().st_size
    if file_size == 0:
        print("[ERROR] Extracted audio file is empty (0 bytes).", file=sys.stderr)
        sys.exit(1)

    with wave.open(str(output_wav), "rb") as wf:
        channels = wf.getnchannels()
        framerate = wf.getframerate()
        frames = wf.getnframes()
        duration = frames / float(framerate)

    print("\n=== WAV File Verification ===")
    print(f"File: {output_wav.relative_to(root)}")
    print(f"File Size: {file_size} bytes (Non-empty: True)")
    print(f"Channels: {channels} ({'Mono' if channels == 1 else 'Stereo/Multi'})")
    print(f"Sample Rate: {framerate} Hz")
    print(f"Frames: {frames}")
    print(f"Duration: {duration:.2f} seconds")

    assert channels == 1, f"Expected 1 channel (mono), got {channels}"
    assert framerate == 16000, f"Expected 16000 Hz, got {framerate}"
    assert frames > 0, "No audio frames in WAV"
    assert abs(duration - (end - start)) <= 0.5, (
        f"Duration {duration}s deviates significantly from requested {end - start}s"
    )

    print("\n[SUCCESS] Extraction and verification completed successfully.")


if __name__ == "__main__":
    extract_first_gunshot_candidate()
