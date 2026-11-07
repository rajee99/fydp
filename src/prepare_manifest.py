import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "target_manifest.csv"

# Use central class configuration
try:
    from class_config import (
        EMERGENCY_TARGET_LABELS as TARGETS,
        OTHER_SUBDOMAINS,
        get_all_exclusion_mids,
    )
except ImportError:
    from src.class_config import (
        EMERGENCY_TARGET_LABELS as TARGETS,
        OTHER_SUBDOMAINS,
        get_all_exclusion_mids,
    )

with (RAW / "yamnet_class_map.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    class_map = {
        row["mid"]: row["display_name"]
        for row in csv.DictReader(f)
    }

label_to_target = {
    label: target
    for target, labels in TARGETS.items()
    for label in labels
}

rows_out = []
counts = Counter()
seen_emergency = set()
emergency_vids = set()

# 1. Extract emergency target rows
with (RAW / "balanced_train_segments.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    lines = (line for line in f if line.strip() and not line.startswith("#"))
    reader = csv.reader(lines, skipinitialspace=True)

    for row in reader:
        if len(row) < 4:
            continue

        video_id = row[0].strip()
        start = float(row[1])
        end = float(row[2])

        for mid in row[3].strip().split(","):
            mid = mid.strip()
            label = class_map.get(mid)
            target = label_to_target.get(label)

            if target is None:
                continue

            key = (video_id, start, end, target)
            if key in seen_emergency:
                continue
            seen_emergency.add(key)
            emergency_vids.add(video_id)

            rows_out.append({
                "video_id": video_id,
                "start_seconds": start,
                "end_seconds": end,
                "target_class": target,
                "matched_label": label,
                "audio_set_mid": mid,
            })
            counts[target] += 1

# 2. Extract and sample 'other' candidates
exclusion_mids = get_all_exclusion_mids()
mid_to_subdomain = {
    mid: domain
    for domain, mids in OTHER_SUBDOMAINS.items()
    for mid in mids
}

candidates_by_domain = defaultdict(list)
excluded_count = 0

with (RAW / "balanced_train_segments.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    lines = (line for line in f if line.strip() and not line.startswith("#"))
    reader = csv.reader(lines, skipinitialspace=True)

    for row in reader:
        if len(row) < 4:
            continue

        video_id = row[0].strip()
        start = float(row[1])
        end = float(row[2])
        row_mids = [m.strip() for m in row[3].strip().split(",") if m.strip()]

        # Exclude segments carrying emergency or ambiguous MIDs
        if any(m in exclusion_mids for m in row_mids):
            excluded_count += 1
            continue

        # Prevent assigning 'other' to a video already used in emergency targets
        if video_id in emergency_vids:
            continue

        # Identify matching non-emergency sub-domains
        matched_domains = set()
        matched_mid_for_domain = {}
        for m in row_mids:
            if m in mid_to_subdomain:
                d = mid_to_subdomain[m]
                matched_domains.add(d)
                if d not in matched_mid_for_domain:
                    matched_mid_for_domain[d] = m

        # Require exactly one matching sub-domain to guarantee acoustic purity
        if len(matched_domains) == 1:
            domain = list(matched_domains)[0]
            chosen_mid = matched_mid_for_domain[domain]
            candidates_by_domain[domain].append({
                "video_id": video_id,
                "start_seconds": start,
                "end_seconds": end,
                "target_class": "other",
                "matched_label": class_map.get(chosen_mid, OTHER_SUBDOMAINS[domain].get(chosen_mid, "other")),
                "audio_set_mid": chosen_mid,
                "subdomain": domain,
            })

# Deterministic sampling: 35 per sub-domain (175 total)
TARGET_PER_DOMAIN = 35
rng = random.Random(42)
sampled_other = []
used_vids = set(emergency_vids)
subdomain_counts = Counter()

for d in sorted(OTHER_SUBDOMAINS.keys()):
    pool = candidates_by_domain[d]
    pool.sort(key=lambda x: (x["video_id"], x["start_seconds"]))
    rng.shuffle(pool)

    sampled_count = 0
    for item in pool:
        if item["video_id"] not in used_vids:
            used_vids.add(item["video_id"])
            sampled_other.append({
                "video_id": item["video_id"],
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
                "target_class": item["target_class"],
                "matched_label": item["matched_label"],
                "audio_set_mid": item["audio_set_mid"],
            })
            subdomain_counts[d] += 1
            sampled_count += 1
            if sampled_count >= TARGET_PER_DOMAIN:
                break

    if sampled_count < TARGET_PER_DOMAIN:
        raise RuntimeError(f"Insufficient pure candidates for sub-domain '{d}': requested {TARGET_PER_DOMAIN}, found {sampled_count}")

counts["other"] = len(sampled_other)
rows_out.extend(sampled_other)

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "video_id", "start_seconds", "end_seconds",
        "target_class", "matched_label", "audio_set_mid"
    ])
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Manifest saved: {OUT}")
print(f"Total labelled segments: {len(rows_out)}")
all_classes = list(TARGETS.keys()) + ["other"]
for target in all_classes:
    print(f"  {target}: {counts[target]}")
print(f"Sub-domain distribution for 'other' ({len(sampled_other)} total):")
for d, cnt in sorted(subdomain_counts.items()):
    print(f"  {d}: {cnt}")
print(f"Excluded segments (emergency/ambiguous): {excluded_count}")
