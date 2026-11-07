"""Prepare Test Set v2 Candidate Manifest.

Samples candidates across:
- scream (target: 100, queue: 150)
- glass_breaking (target: 100, queue: 150)
- gunshot (target: 100, queue: 140)
- siren (target: 150, queue: 200)
- other benign (target: 150 [25x6], queue: 210 [35x6])
- other confusers (target: 150 [25x6], queue: 210 [35x6])

Strictly enforces:
1. Zero overlap with all 2,577 historical forbidden video IDs.
2. Stratified sampling with fixed seed (2026).
3. AudioSet eval segments prioritized where available, followed by unbalanced train.
"""

from collections import defaultdict
import csv
from pathlib import Path
import random
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import (
    CLASSES, CLASS_TO_ID, EMERGENCY_MIDS, AMBIGUOUS_MIDS, OTHER_SUBDOMAINS
)

EVAL_SEGMENTS_PATH = ROOT / "data" / "raw" / "eval_segments.csv"
UNBALANCED_SEGMENTS_PATH = ROOT / "data" / "raw" / "unbalanced_train_segments.csv"
BALANCED_SEGMENTS_PATH = ROOT / "data" / "raw" / "balanced_train_segments.csv"
CLASS_MAP_PATH = ROOT / "data" / "raw" / "yamnet_class_map.csv"

OUT_CANDIDATES_CSV = ROOT / "data" / "processed" / "test_set_v2_candidates_to_download.csv"

SEED = 2026


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


def load_yamnet_map():
    df = pd.read_csv(CLASS_MAP_PATH)
    return dict(zip(df["mid"], df["display_name"]))


def define_categories():
    mid2name = load_yamnet_map()

    scream_mids = {"/m/03qc9zr"}
    glass_mids = {"/m/07rn7sz", "/m/039jq"}
    gunshot_mids = {"/m/032s66", "/m/04zjc", "/m/02z32qm", "/m/0_1c", "/m/073cg4", "/m/014zdl"}
    siren_mids = {"/m/03kmc9", "/m/012n7d", "/m/04qvtq", "/m/012ndj", "/m/0dgbq"}

    # Strict 4-class emergency target MIDs (must be absent from any 'other' candidate)
    pure_emergency_mids = scream_mids | glass_mids | gunshot_mids | siren_mids

    confuser_categories = {
        "human_shouting": {"/m/07p6fty", "/m/07sr1lc", "/t/dd00135"},
        "metal_impacts": {"/m/07pws3f", "/m/07pjjrj"},
        "door_slams": {"/m/02dgv", "/m/07r4wb8", "/m/07rjzl8"},
        "car_alarms_buzzers": {"/m/02mfyn", "/m/01y3hg", "/m/0c3f7m", "/m/07pp_mv", "/m/030rvx"},
        "fireworks": {"/m/0g6b5", "/g/122z_qxw"},
        "whistles": {"/m/0l156k", "/m/06hck5", "/m/04zmvq"},
    }

    all_confuser_mids = set.union(*confuser_categories.values())

    # Benign subdomains (strictly exclude doors/knocks from household since they are confusers)
    benign_subdomains = {
        "speech": {"/m/09x0r", "/m/01h8n0", "/m/02qldy", "/m/02rtxlg"},
        "music": {"/m/04rlf", "/m/042v_gx", "/m/05r5c", "/m/015lz1"},
        "animals": {"/m/0bt9lr", "/m/05tny_", "/m/01yrx", "/m/07qrkrw", "/m/015p6", "/m/020bb7"},
        "household": {"/m/04brg2", "/m/023pjk", "/m/01x3z", "/m/02jz0l", "/m/0642b4"},
        "traffic": {"/m/0btp2", "/m/07yv9", "/m/0k4j", "/m/01bjv", "/m/04_sv"},
        "environmental": {"/m/03m9d0z", "/m/0838f", "/m/06mb1", "/m/0j6m2", "/m/05kq4", "/m/034srq"},
    }

    benign_exclusion = pure_emergency_mids | set(EMERGENCY_MIDS.keys()) | set(AMBIGUOUS_MIDS.keys()) | all_confuser_mids

    return {
        "scream_mids": scream_mids,
        "glass_mids": glass_mids,
        "gunshot_mids": gunshot_mids,
        "siren_mids": siren_mids,
        "pure_emergency_mids": pure_emergency_mids,
        "confuser_categories": confuser_categories,
        "all_confuser_mids": all_confuser_mids,
        "benign_subdomains": benign_subdomains,
        "benign_exclusion": benign_exclusion,
        "mid2name": mid2name,
    }


def parse_and_collect_candidates(segments_path: Path, source_name: str, cats: dict, forbidden: set):
    print(f"Collecting candidates from {source_name} ({segments_path.name})...")
    candidates = defaultdict(list)

    with open(segments_path, "r", encoding="utf-8") as f:
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
            try:
                start = float(parts[1])
                end = float(parts[2])
            except ValueError:
                continue

            labels = set([l.strip().strip('"') for l in parts[3:]])

            # 1. Scream
            if labels.intersection(cats["scream_mids"]):
                mid = list(labels.intersection(cats["scream_mids"]))[0]
                candidates["scream"].append({
                    "video_id": vid, "start_seconds": start, "end_seconds": end,
                    "target_class": "scream", "class_id": 0,
                    "subdomain": "human_vocal", "matched_label": cats["mid2name"].get(mid, "Screaming"),
                    "audio_set_mid": mid, "source_dataset": source_name,
                    "relative_path": f"data/raw/test_set_v2/scream/scream_{vid}_{int(start)}_{int(end)}.wav"
                })
            # 2. Glass breaking
            elif labels.intersection(cats["glass_mids"]):
                mid = list(labels.intersection(cats["glass_mids"]))[0]
                candidates["glass_breaking"].append({
                    "video_id": vid, "start_seconds": start, "end_seconds": end,
                    "target_class": "glass_breaking", "class_id": 1,
                    "subdomain": "impact", "matched_label": cats["mid2name"].get(mid, "Glass"),
                    "audio_set_mid": mid, "source_dataset": source_name,
                    "relative_path": f"data/raw/test_set_v2/glass_breaking/glass_breaking_{vid}_{int(start)}_{int(end)}.wav"
                })
            # 3. Gunshot
            elif labels.intersection(cats["gunshot_mids"]):
                mid = list(labels.intersection(cats["gunshot_mids"]))[0]
                candidates["gunshot"].append({
                    "video_id": vid, "start_seconds": start, "end_seconds": end,
                    "target_class": "gunshot", "class_id": 2,
                    "subdomain": "weapon", "matched_label": cats["mid2name"].get(mid, "Gunshot"),
                    "audio_set_mid": mid, "source_dataset": source_name,
                    "relative_path": f"data/raw/test_set_v2/gunshot/gunshot_{vid}_{int(start)}_{int(end)}.wav"
                })
            # 4. Siren
            elif labels.intersection(cats["siren_mids"]):
                mid = list(labels.intersection(cats["siren_mids"]))[0]
                candidates["siren"].append({
                    "video_id": vid, "start_seconds": start, "end_seconds": end,
                    "target_class": "siren", "class_id": 3,
                    "subdomain": "emergency_vehicle", "matched_label": cats["mid2name"].get(mid, "Siren"),
                    "audio_set_mid": mid, "source_dataset": source_name,
                    "relative_path": f"data/raw/test_set_v2/siren/siren_{vid}_{int(start)}_{int(end)}.wav"
                })
            # 5. Other Confusers (Strictly no pure emergency MIDs)
            elif labels.intersection(cats["all_confuser_mids"]) and not labels.intersection(cats["pure_emergency_mids"]):
                for conf_sub, conf_mids in cats["confuser_categories"].items():
                    hit = labels.intersection(conf_mids)
                    if hit:
                        mid = list(hit)[0]
                        candidates[f"confuser_{conf_sub}"].append({
                            "video_id": vid, "start_seconds": start, "end_seconds": end,
                            "target_class": "other", "class_id": 4,
                            "subdomain": f"confuser_{conf_sub}", "matched_label": cats["mid2name"].get(mid, conf_sub),
                            "audio_set_mid": mid, "source_dataset": source_name,
                            "relative_path": f"data/raw/test_set_v2/other/other_{vid}_{int(start)}_{int(end)}.wav"
                        })
                        break
            # 6. Other Benign (Strictly no emergency MIDs and no ambiguous/confusers)
            elif not labels.intersection(cats["benign_exclusion"]):
                for ben_sub, ben_mids in cats["benign_subdomains"].items():
                    hit = labels.intersection(ben_mids)
                    if hit:
                        mid = list(hit)[0]
                        candidates[f"benign_{ben_sub}"].append({
                            "video_id": vid, "start_seconds": start, "end_seconds": end,
                            "target_class": "other", "class_id": 4,
                            "subdomain": f"benign_{ben_sub}", "matched_label": cats["mid2name"].get(mid, ben_sub),
                            "audio_set_mid": mid, "source_dataset": source_name,
                            "relative_path": f"data/raw/test_set_v2/other/other_{vid}_{int(start)}_{int(end)}.wav"
                        })
                        break

    return candidates


def main():
    print("=" * 80)
    print("  PREPARING TEST SET V2 CANDIDATES")
    print("=" * 80)

    forbidden = collect_all_forbidden()
    print(f"Total forbidden video IDs: {len(forbidden)}")

    cats = define_categories()

    # Quotas for queuing candidates (target * 1.4-1.5 buffer for download yield)
    queue_targets = {
        "scream": 150,               # target: 100
        "glass_breaking": 150,       # target: 100
        "gunshot": 140,              # target: 100
        "siren": 200,                # target: 150
        # Confusers (target 25 each = 150)
        "confuser_human_shouting": 35,
        "confuser_metal_impacts": 35,
        "confuser_door_slams": 35,
        "confuser_car_alarms_buzzers": 35,
        "confuser_fireworks": 35,
        "confuser_whistles": 35,
        # Benign (target 25 each = 150)
        "benign_speech": 35,
        "benign_music": 35,
        "benign_animals": 35,
        "benign_household": 35,
        "benign_traffic": 35,
        "benign_environmental": 35,
    }

    # 1. Parse eval_segments first (highest priority)
    eval_candidates = parse_and_collect_candidates(EVAL_SEGMENTS_PATH, "audioset_eval", cats, forbidden)
    print("Eval candidates available:")
    for k, v in eval_candidates.items():
        print(f"  {k:<30}: {len(v)}")

    # 2. Parse unbalanced_train_segments
    unbal_candidates = parse_and_collect_candidates(UNBALANCED_SEGMENTS_PATH, "audioset_unbalanced", cats, forbidden)
    print("\nUnbalanced candidates available:")
    for k, v in unbal_candidates.items():
        print(f"  {k:<30}: {len(v)}")

    # Merge candidates: use eval first, then append unbalanced
    rng = random.Random(SEED)
    selected_candidates = []

    print("\n" + "=" * 80)
    print("  SAMPLING TEST SET V2 QUEUE (SEED = 2026)")
    print("=" * 80)

    selected_video_ids = set()

    for stratum, needed in queue_targets.items():
        pool = []
        # Eval candidates
        ev = eval_candidates.get(stratum, [])
        rng.shuffle(ev)
        pool.extend(ev)

        # Unbalanced candidates
        unb = unbal_candidates.get(stratum, [])
        rng.shuffle(unb)
        pool.extend(unb)

        # Deduplicate by video_id within pool and against previously selected
        chosen = []
        for cand in pool:
            vid = cand["video_id"]
            if vid not in forbidden and vid not in selected_video_ids:
                chosen.append(cand)
                selected_video_ids.add(vid)
                if len(chosen) >= needed:
                    break

        print(f"  {stratum:<30}: queued {len(chosen):3d} / {needed} (eval: {sum(1 for c in chosen if c['source_dataset']=='audioset_eval')}, unbal: {sum(1 for c in chosen if c['source_dataset']=='audioset_unbalanced')})")
        selected_candidates.extend(chosen)

    print("=" * 80)
    print(f"Total candidates queued: {len(selected_candidates)}")

    # Final check on forbidden leakage
    leakage = set(c["video_id"] for c in selected_candidates).intersection(forbidden)
    if leakage:
        raise RuntimeError(f"FATAL: {len(leakage)} forbidden video IDs leaked into candidate queue: {leakage}")
    print("[LEAKAGE CHECK] PASSED: Exactly 0 overlapping video IDs with forbidden set!")

    # Write out candidates CSV
    fieldnames = [
        "video_id", "start_seconds", "end_seconds", "target_class", "class_id",
        "relative_path", "subdomain", "matched_label", "audio_set_mid",
        "source_dataset"
    ]
    with open(OUT_CANDIDATES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_candidates)

    print(f"Saved candidate manifest to: {OUT_CANDIDATES_CSV}")


if __name__ == "__main__":
    main()
