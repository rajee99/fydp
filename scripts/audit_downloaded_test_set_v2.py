"""Audit all downloaded WAV files in data/raw/test_set_v2."""

from pathlib import Path
import pandas as pd
import wave

ROOT = Path(__file__).resolve().parent.parent

CAND_FILES = [
    ROOT / "data" / "processed" / "test_set_v2_candidates_to_download.csv",
    ROOT / "data" / "processed" / "test_set_v2_replenish_candidates.csv",
]

all_cands = []
for p in CAND_FILES:
    if p.exists():
        all_cands.append(pd.read_csv(p))
cand_df = pd.concat(all_cands).drop_duplicates(subset=["video_id", "start_seconds"])
cand_df["fname"] = cand_df["relative_path"].apply(lambda x: Path(x).name)
cand_map = {}
for _, r in cand_df.iterrows():
    cand_map[r["fname"]] = r.to_dict()

AUDIO_DIR = ROOT / "data" / "raw" / "test_set_v2"
wavs = list(AUDIO_DIR.glob("*/*.wav"))
print(f"Total WAV files in test_set_v2: {len(wavs)}")

class_counts = {}
subdomain_counts = {}
valid_count = 0

for w in wavs:
    cname = w.parent.name
    class_counts[cname] = class_counts.get(cname, 0) + 1
    info = cand_map.get(w.name, {})
    sub = info.get("subdomain", "unknown")
    subdomain_counts[sub] = subdomain_counts.get(sub, 0) + 1

    try:
        with wave.open(str(w), "rb") as wf:
            if wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2:
                valid_count += 1
    except Exception:
        pass

print("\n--- Class Counts ---")
for c, cnt in sorted(class_counts.items()):
    print(f"  {c:<18}: {cnt:3d}")

print(f"\nTotal technically valid: {valid_count} / {len(wavs)}")

print("\n--- Subdomain Counts in 'other' ---")
for s, cnt in sorted(subdomain_counts.items()):
    if s.startswith("benign") or s.startswith("confuser"):
        print(f"  {s:<30}: {cnt:3d}")
