import json
import subprocess
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / "data" / "processed" / "scream_replacements.csv"
output = root / "data" / "processed" / "scream_replacements_checked.csv"

df = pd.read_csv(source)
results = []

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
            results.append(record)
            print(f"OK: {video_id} | {info.get('title', '')}")
        except json.JSONDecodeError:
            print(f"SKIP: {video_id} | Invalid metadata")
    else:
        print(f"SKIP: {video_id} | Unavailable or restricted")

pd.DataFrame(results).to_csv(output, index=False)
print(f"\nAccessible candidates: {len(results)} / {len(df)}")
print(f"Saved to: {output}")
