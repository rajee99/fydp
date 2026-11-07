import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
manifest = pd.read_csv(root / "data" / "processed" / "target_manifest.csv")
previous = pd.read_csv(root / "data" / "processed" / "pilot_manifest.csv")

attempted_ids = set(previous["video_id"].astype(str))
candidates = manifest[
    (manifest["target_class"] == "scream")
    & (~manifest["video_id"].astype(str).isin(attempted_ids))
].head(10).copy()

output = root / "data" / "processed" / "scream_replacements.csv"
candidates.to_csv(output, index=False)

print(f"Saved {len(candidates)} replacement candidates to: {output}")
print(candidates[["video_id", "start_seconds", "end_seconds", "matched_label"]].to_string(index=False))
