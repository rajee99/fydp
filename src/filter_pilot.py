import json
import subprocess
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / "data" / "processed" / "pilot_manifest.csv"
output = root / "data" / "processed" / "pilot_accessible.csv"

df = pd.read_csv(source)
accessible = []

for _, row in df.iterrows():
    video_id = str(row["video_id"])
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
        try:
            info = json.loads(result.stdout)
            record = row.to_dict()
            record["video_title"] = info.get("title", "")
            accessible.append(record)
            print(f"OK: {video_id}")
        except json.JSONDecodeError:
            print(f"SKIP: {video_id} — invalid metadata")
    else:
        print(f"SKIP: {video_id} — unavailable or restricted")

pd.DataFrame(accessible).to_csv(output, index=False)
print(f"\nSaved {len(accessible)} accessible records to: {output}")
print("\nAccessible records by class:")
if accessible:
    print(pd.DataFrame(accessible).groupby("target_class").size().to_string())
