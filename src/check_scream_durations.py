import json
import subprocess

video_ids = ["3sYPuktlyDY", "9ZBR-dTCPrg"]

for video_id in video_ids:
    result = subprocess.run(
        [
            "yt-dlp",
            "--skip-download",
            "--dump-single-json",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        data = json.loads(result.stdout)
        print(f"\nVideo ID: {video_id}")
        print(f"Title: {data.get('title')}")
        print(f"Duration: {data.get('duration')} seconds")
    else:
        print(f"\nVideo ID: {video_id}")
        print("Metadata check failed:")
        print(result.stderr.strip().splitlines()[-1])
