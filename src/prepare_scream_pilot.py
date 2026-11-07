import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / "data" / "processed" / "scream_replacements_checked.csv"
output = root / "data" / "processed" / "scream_pilot.csv"

df = pd.read_csv(source)
chosen = df[df["video_id"].astype(str).isin(["3sYPuktlyDY", "9ZBR-dTCPrg"])].copy()

if len(chosen) != 2:
    raise SystemExit(f"Expected 2 candidates, found {len(chosen)}")

chosen.to_csv(output, index=False)
print(f"Saved {len(chosen)} candidates to: {output}")
print(chosen[["video_id", "start_seconds", "end_seconds", "target_class", "video_title"]].to_string(index=False))
