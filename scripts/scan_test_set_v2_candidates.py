"""Scan available AudioSet pools for Test Set v2 candidate counts.

Audits candidate counts across:
1. scream
2. glass_breaking
3. gunshot
4. siren
5. other (Benign background & Targeted confusers)

Strictly excludes all 2,577 historical forbidden video IDs.
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import (
    EMERGENCY_MIDS, AMBIGUOUS_MIDS, OTHER_SUBDOMAINS, get_all_exclusion_mids
)

EVAL_SEGMENTS_PATH = ROOT / "data" / "raw" / "eval_segments.csv"
UNBALANCED_SEGMENTS_PATH = ROOT / "data" / "raw" / "unbalanced_train_segments.csv"
BALANCED_SEGMENTS_PATH = ROOT / "data" / "raw" / "balanced_train_segments.csv"


def collect_all_forbidden() -> set:
    forbidden = set()
    for p in (ROOT / "data" / "processed").glob("*.csv"):
        try:
            df = pd.read_csv(p)
            if "video_id" in df.columns:
                forbidden.update(df["video_id"].dropna().astype(str).str.strip())
        except Exception:
            pass
    for p in (ROOT / "data" / "raw").glob("*manifest*.csv"):
        try:
            df = pd.read_csv(p)
            if "video_id" in df.columns:
                forbidden.update(df["video_id"].dropna().astype(str).str.strip())
        except Exception:
            pass
    return forbidden


def parse_segments(path: Path, forbidden: set):
    print(f"\nParsing {path.name}...")
    excl_mids = get_all_exclusion_mids()

    scream_mids = {"/m/03qc9zr"}
    glass_mids = {"/m/07rn7sz", "/m/039jq"}
    gunshot_mids = {"/m/032s66", "/m/04zjc", "/m/02z32qm", "/m/0_1c", "/m/073cg4", "/m/014zdl"}
    siren_mids = {"/m/03kmc9", "/m/012n7d", "/m/04qvtq", "/m/012ndj", "/m/0dgbq"}
    
    # Confuser MIDs
    confuser_mids = {
        "human_shouting": {"/m/07p6fty", "/m/07sr1lc", "/t/dd00135"},
        "metal_impacts": {"/m/07pws3f", "/m/07pjjrj"},
        "door_slams": {"/m/02dgv", "/m/07r4wb8"},
        "car_alarms_buzzers": {"/m/02mfyn", "/m/01y3hg", "/m/0c3f7m", "/m/07pp_mv"},
        "fireworks": {"/m/0g6b5", "/g/122z_qxw"},
        "whistles": {"/m/01557"},
    }
    all_confuser_mids = set.union(*confuser_mids.values())

    counts = {
        "scream": 0,
        "glass_breaking": 0,
        "gunshot": 0,
        "siren": 0,
        "other_benign": 0,
        "other_confusers": 0,
    }

    vids_by_class = {c: set() for c in counts}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 4:
                continue
            vid = parts[0]
            if vid in forbidden:
                continue
            labels = set([l.strip().strip('"') for l in parts[3:]])

            # Classify
            if labels.intersection(scream_mids):
                vids_by_class["scream"].add(vid)
            elif labels.intersection(glass_mids):
                vids_by_class["glass_breaking"].add(vid)
            elif labels.intersection(gunshot_mids):
                vids_by_class["gunshot"].add(vid)
            elif labels.intersection(siren_mids):
                vids_by_class["siren"].add(vid)
            elif labels.intersection(all_confuser_mids) and not labels.intersection(set(EMERGENCY_MIDS.keys())):
                vids_by_class["other_confusers"].add(vid)
            elif not labels.intersection(excl_mids):
                # Check benign subdomains
                is_benign = False
                for sub, sub_m in OTHER_SUBDOMAINS.items():
                    if labels.intersection(set(sub_m.keys())):
                        is_benign = True
                        break
                if is_benign:
                    vids_by_class["other_benign"].add(vid)

    for c, vids in vids_by_class.items():
        print(f"  {c:<18}: {len(vids)} unique available video IDs")
    return vids_by_class


def main():
    forbidden = collect_all_forbidden()
    print(f"Total forbidden video IDs: {len(forbidden)}")

    if EVAL_SEGMENTS_PATH.exists():
        parse_segments(EVAL_SEGMENTS_PATH, forbidden)

    if BALANCED_SEGMENTS_PATH.exists():
        parse_segments(BALANCED_SEGMENTS_PATH, forbidden)

    if UNBALANCED_SEGMENTS_PATH.exists():
        parse_segments(UNBALANCED_SEGMENTS_PATH, forbidden)


if __name__ == "__main__":
    main()
