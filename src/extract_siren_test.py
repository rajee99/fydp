import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out_dir = root / "data" / "raw" / "pilot"
out_dir.mkdir(parents=True, exist_ok=True)

video_id = "-4RWqM0UCCY"
start = 0
end = 10
output = out_dir / "siren_test.wav"

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
    "-o", str(out_dir / "siren_source.%(ext)s"),
    url,
]

print("Requesting the selected audio segment...")
result = subprocess.run(command)

if result.returncode != 0:
    raise SystemExit("Extraction failed. No training data was changed.")

print("Check the pilot folder for the extracted audio.")
