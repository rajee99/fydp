import json
import subprocess
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
target_manifest_path = root / "data" / "processed" / "target_manifest.csv"
pilot_accessible_path = root / "data" / "processed" / "pilot_accessible.csv"
output_path = root / "data" / "processed" / "glass_candidates.csv"

target_df = pd.read_csv(target_manifest_path)
pilot_df = pd.read_csv(pilot_accessible_path)

# Identify local audio files already present in the pilot manifest
registered_audio = set(pilot_df["local_audio_path"].dropna().astype(str))
registered_vids_with_audio = set(
    pilot_df[pilot_df["local_audio_path"].notna()]["video_id"].dropna().astype(str)
)

print(f"Registered audio files in pilot manifest: {len(registered_audio)}")

# Filter glass-breaking candidates from target_manifest
glass_candidates = target_df[target_df["target_class"] == "glass_breaking"]

# Filter out any candidate that already has a local audio file
pool = []
for _, row in glass_candidates.iterrows():
    vid = str(row["video_id"]).strip()
    if vid in registered_vids_with_audio:
        continue
    pool.append(row.to_dict())

print(f"Total glass-breaking records without local audio: {len(pool)}")

valid_candidates = []
for item in pool:
    vid = item["video_id"]
    start = float(item["start_seconds"])
    end = float(item["end_seconds"])

    # Metadata check with yt-dlp
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

            # Ensure valid segment within video duration
            if duration is not None and duration >= end:
                record = {
                    "video_id": vid,
                    "start_seconds": start,
                    "end_seconds": end,
                    "target_class": item["target_class"],
                    "matched_label": item["matched_label"],
                    "audio_set_mid": item["audio_set_mid"],
                    "video_title": title,
                }
                valid_candidates.append(record)
                print(f"OK: {vid} | {title} | Segment: {start}s-{end}s (Duration: {duration}s)")
            else:
                print(f"SKIP (Invalid segment): {vid} (Duration: {duration}s, End: {end}s)")
        except json.JSONDecodeError:
            print(f"SKIP (JSON parse error): {vid}")
    else:
        print(f"SKIP (Unavailable/restricted): {vid}")

    if len(valid_candidates) >= 5:
        break

output_df = pd.DataFrame(valid_candidates)
output_df.to_csv(output_path, index=False)

print(f"\nSaved {len(valid_candidates)} candidates to: {output_path}")
print(output_df[["video_id", "start_seconds", "end_seconds", "matched_label", "video_title"]].to_string(index=False))
