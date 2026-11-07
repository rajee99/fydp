import json
import subprocess
import pandas as pd

df = pd.read_csv("data/processed/pilot_manifest.csv")
accessible = []
failed = []

for video_id in df["video_id"]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    result = subprocess.run(
        [
            "yt-dlp",
            "--skip-download",
            "--dump-single-json",
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        try:
            info = json.loads(result.stdout)
            accessible.append((video_id, info.get("title", "Unknown title")))
        except json.JSONDecodeError:
            failed.append((video_id, "Invalid metadata response"))
    else:
        error = result.stderr.strip().splitlines()
        failed.append((video_id, error[-1] if error else "Unknown error"))

print(f"Accessible: {len(accessible)}")
for video_id, title in accessible:
    print(f"OK: {video_id} | {title}")

print(f"\nUnavailable or failed: {len(failed)}")
for video_id, error in failed:
    print(f"FAIL: {video_id} | {error}")
