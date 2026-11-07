import json
import subprocess
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
target_manifest_path = root / "data" / "processed" / "target_manifest.csv"
pilot_accessible_path = root / "data" / "processed" / "pilot_accessible.csv"
output_path = root / "data" / "processed" / "gunshot_candidates.csv"

target_df = pd.read_csv(target_manifest_path)
pilot_df = pd.read_csv(pilot_accessible_path)

# Exclude any source already represented in the pilot manifest
pilot_video_ids = set(pilot_df["video_id"].dropna().astype(str))
print(f"Sources in pilot manifest to exclude: {len(pilot_video_ids)}")

gunshot_candidates = target_df[
    (target_df["target_class"] == "gunshot")
    & (~target_df["video_id"].astype(str).isin(pilot_video_ids))
]
print(f"Gunshot candidates to evaluate: {len(gunshot_candidates)}")

suitable_candidates = []

for _, row in gunshot_candidates.iterrows():
    vid = str(row["video_id"]).strip()
    start = float(row["start_seconds"])
    end = float(row["end_seconds"])

    # Metadata check using yt-dlp only (no audio download)
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--dump-single-json",
        f"https://www.youtube.com/watch?v={vid}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode == 0:
        try:
            info = json.loads(res.stdout)
            duration = info.get("duration")
            title = info.get("title", "")

            # Confirm segment falls within video duration
            if duration is not None and duration >= end:
                record = {
                    "video_id": vid,
                    "start_seconds": start,
                    "end_seconds": end,
                    "target_class": row["target_class"],
                    "matched_label": row["matched_label"],
                    "audio_set_mid": row["audio_set_mid"],
                    "video_title": title,
                }
                suitable_candidates.append(record)
                print(f"[FOUND] {vid} | {title} | Segment: {start}s-{end}s | Duration: {duration}s")
                # Save first suitable candidate
                break
            else:
                print(f"[SKIP] Segment out of bounds: {vid} (Duration: {duration}s < End: {end}s)")
        except json.JSONDecodeError:
            print(f"[SKIP] JSON decode error: {vid}")
    else:
        print(f"[SKIP] Inaccessible/restricted: {vid}")

if not suitable_candidates:
    raise SystemExit("No suitable gunshot candidate found.")

out_df = pd.DataFrame(suitable_candidates)
out_df.to_csv(output_path, index=False)
print(f"\nSaved first suitable candidate to: {output_path}")
print(out_df.to_string(index=False))
