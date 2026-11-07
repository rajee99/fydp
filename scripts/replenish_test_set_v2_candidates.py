"""Sample Replenishments for Test Set v2 to ensure every stratum hits exact target quota.

Quotas needed:
- scream: 15 candidates (need 8)
- gunshot: 8 candidates (need 4)
- siren: 8 candidates (need 4)
- benign_environmental: 6 candidates (need 3)
- benign_speech: 4 candidates (need 1)
- confuser_fireworks: 8 candidates (need 4)
- confuser_metal_impacts: 6 candidates (need 2)
Total: 55 candidates.
"""

from collections import defaultdict
import csv
from pathlib import Path
import random
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_test_set_v2_candidates import (
    collect_all_forbidden, define_categories, parse_and_collect_candidates,
    EVAL_SEGMENTS_PATH, UNBALANCED_SEGMENTS_PATH, SEED
)

OUT_REPLENISH_MANIFEST = ROOT / "data" / "processed" / "test_set_v2_replenish_candidates.csv"
OUT_REPLENISH_RESULTS = ROOT / "data" / "processed" / "test_set_v2_replenish_download_results.csv"
DEST_DIR = ROOT / "data" / "raw" / "test_set_v2"


def main():
    forbidden = collect_all_forbidden()
    # Add already queued candidate video IDs
    cand_df = pd.read_csv(ROOT / "data" / "processed" / "test_set_v2_candidates_to_download.csv")
    forbidden.update(cand_df["video_id"].dropna().str.strip())
    print(f"Total forbidden video IDs including initial candidates: {len(forbidden)}")

    cats = define_categories()

    replenish_targets = {
        "scream": 15,
        "gunshot": 8,
        "siren": 8,
        "benign_environmental": 6,
        "benign_speech": 4,
        "confuser_fireworks": 8,
        "confuser_metal_impacts": 6,
    }

    eval_cands = parse_and_collect_candidates(EVAL_SEGMENTS_PATH, "audioset_eval", cats, forbidden)
    unbal_cands = parse_and_collect_candidates(UNBALANCED_SEGMENTS_PATH, "audioset_unbalanced", cats, forbidden)

    rng = random.Random(SEED + 100)
    selected = []
    selected_vids = set()

    for stratum, needed in replenish_targets.items():
        pool = []
        ev = eval_cands.get(stratum, [])
        rng.shuffle(ev)
        pool.extend(ev)

        unb = unbal_cands.get(stratum, [])
        rng.shuffle(unb)
        pool.extend(unb)

        chosen = []
        for cand in pool:
            vid = cand["video_id"]
            if vid not in forbidden and vid not in selected_vids:
                chosen.append(cand)
                selected_vids.add(vid)
                if len(chosen) >= needed:
                    break

        print(f"  {stratum:<30}: queued {len(chosen):2d} / {needed:2d}")
        selected.extend(chosen)

    fieldnames = [
        "video_id", "start_seconds", "end_seconds", "target_class", "class_id",
        "relative_path", "subdomain", "matched_label", "audio_set_mid",
        "source_dataset"
    ]
    with open(OUT_REPLENISH_MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Saved {len(selected)} replenishment candidates to: {OUT_REPLENISH_MANIFEST}")


if __name__ == "__main__":
    main()
