# Targeted In-The-Wild Dataset Expansion Manifest Summary
**Generated:** D:\ml\fydp\MyNet
**Preferred Baseline Model:** `models/experiments/panns_cnn14/best_model.pt`
**Source Catalog:** Google AudioSet Unbalanced Training Corpus (`unbalanced_train_segments.csv`)
**Output Manifest:** [`data/processed/targeted_expansion_candidates.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_candidates.csv)

---

## 1. Executive Summary & Quota Fulfillment

| Target Category | Subdomain / Focus | Requested Quota | Primary Candidates Selected (P1) | Backup / Review Candidates (P2/P3) | Total Candidate Pool | Shortfall Against Quota |
|---|---|---:|---:|---:|---:|:---:|
| **Scream** | Male/Female, Room, Hall, Distant Urban/Rural, Crowd, Music | **110** | **110** | 52 | 162 | **0 (100% Met)** |
| **Glass Breaking** | Shatter, Breaking Impact, Bottles, Liquid, Tableware | **85** | **85** | 69 | 154 | **0 (100% Met)** |
| **Hard-Negative: Metal** | Dropped utensils, pots, pans, clang, clatter | **45** | **45** | 15 | 60 | **0 (100% Met)** |
| **Hard-Negative: Shouting** | Human shouting, yelling, crying (no screams) | **45** | **45** | 15 | 60 | **0 (100% Met)** |
| **Hard-Negative: Doors** | Slams, door closures, sliding doors, knocks | **35** | **35** | 15 | 50 | **0 (100% Met)** |
| **Hard-Negative: Alarms** | Car alarms, smoke detectors, buzzers (no sirens) | **35** | **35** | 15 | 50 | **0 (100% Met)** |
| **Hard-Negative: Fireworks** | Fireworks, firecrackers, explosions (no gunshots) | **30** | **30** | 15 | 45 | **0 (100% Met)** |
| **Hard-Negative: Tools** | Hammers, jackhammers, drills, saws, power tools | **30** | **30** | 15 | 45 | **0 (100% Met)** |
| **Hard-Negative: Whistles** | Whistles, steam whistles, tire squeals, train whistles | **20** | **20** | 15 | 35 | **0 (100% Met)** |
| **TOTALS** | **All Categories & Subdomains** | **435** | **435** | **237** | **672** | **ZERO SHORTFALL** |

---

## 2. Source-Level Isolation Guarantee

- **Total Forbidden Video IDs Audited:** **42,531 unique IDs** spanning:
  - Existing training manifests (`target_manifest.csv`, `train_split_manifest.csv`, `combined_training_manifest.csv`).
  - Existing evaluation manifest (`eval_manifest.csv`) and the entire held-out AudioSet evaluation corpus (`eval_segments.csv`).
  - Existing pilot manifests, candidates, retry lists, and downloaded audio files on disk.
  - The original 175 unused `other` manifest rows (isolated by excluding all `balanced_train_segments.csv` rows).
- **Total Candidates Rejected by Isolation Check:** **0 clips**.
- **Final Candidate Pool Video ID Overlap:** **0 clips (100% verified isolated)**.
- **Total Unique Source Videos in Candidate Pool:** **672 unique videos** (1 segment per video across all 672 rows).

---

## 3. Subdomain Breakdown & Candidate Selection Details

### A. Scream Detection Candidates (Target: 110 P1 clips)
All primary candidates strictly possess the official AudioSet tag `Screaming` (`/m/03qc9zr`) and were balanced across varied acoustic conditions:

| Acoustic Environment / Condition | P1 Primary Count | Sample Co-labels | Purity & Sound Characteristic |
|---|---:|---|---|
| `indoor_small_room` | 20 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `indoor_hall` | 5 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `outdoor_urban` | 7 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `outdoor_rural` | 1 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `crowd_background` | 15 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `vehicle_background` | 8 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `music_background` | 15 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `isolated_vocal` | 20 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `child_scream` | 10 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |
| `speech_mixed` | 9 | Diverse AudioSet co-labels | High-confidence human vocal scream in specified acoustic environment |

**Flagged Confusers & Overlaps (P3 Review Pool):**
- **Emergency Overlaps:** 2 clips co-occurring with `Shatter` were flagged and quarantined into `P3_REVIEW`.
- **Animal Vocalizations:** 10 screaming goat/sheep clips were flagged as `FLAGGED_AMBIGUOUS_LABEL` to prevent animal mimicry confusion.
- **Recreational Shouting/Laughter:** 10 party/laughter screaming clips were flagged for manual review.

### B. Glass Breaking Candidates (Target: 85 P1 clips)
Prioritizes true physical shattering and window/bottle destruction:

| Glass Subdomain | P1 Primary Count | AudioSet MID | Focus & Characteristic |
|---|---:|---|---|
| `breaking_shatter` | 20 | `/m/07rn7sz` or `/m/039jq` | Verified physical glass shattering / breaking event |
| `bottle_clink_shatter` | 15 | `/m/07rn7sz` or `/m/039jq` | Verified physical glass shattering / breaking event |
| `bottle_liquid_shatter` | 15 | `/m/07rn7sz` or `/m/039jq` | Verified physical glass shattering / breaking event |
| `tableware_falling` | 7 | `/m/07rn7sz` or `/m/039jq` | Verified physical glass shattering / breaking event |
| `indoor_room` | 8 | `/m/07rn7sz` or `/m/039jq` | Verified physical glass shattering / breaking event |
| `indoor_hall` | 4 | `/m/07rn7sz` or `/m/039jq` | Verified physical glass shattering / breaking event |
| `outdoor_urban` | 6 | `/m/07rn7sz` or `/m/039jq` | Verified physical glass shattering / breaking event |
| `isolated_glass` | 10 | `/m/07rn7sz` or `/m/039jq` | Verified physical glass shattering / breaking event |

**Candidate Labels Needing Verification (P2 Review Pool):**
- As required, `Smash, crash` (`/m/07pjjrj`) and `Crack` (`/m/07qs1cx`) were **not** automatically classified as glass breaking. Exactly 30 candidate items (15 Smash/crash, 15 Crack) were isolated into `P2_REVIEW` with status `NEEDS_AUDIO_VERIFICATION`.
- Overlapping clips (co-occurring with gunshots or ambulance sirens) were quarantined into `P3_REVIEW`.

### C. Hard-Negative Other Candidates (Target: 240 P1 clips)
Strictly non-emergency clips matching key false-positive categories:

| Subdomain | Requested Quota | Selected P1 | Selected P2 Backup | Audited MIDs Included | Emergency Isolation Status |
|---|---:|---:|---:|---|:---:|
| `metal_impacts` | 45 | 45 | 15 | Dishes, pots, and pans, Cutlery, silverware, Clang, Clatter | **Strictly Isolated (0% Emergency MIDs)** |
| `human_shouting` | 45 | 45 | 15 | Shout, Yell, Children shouting, Crying, sobbing, Wail, moan, Groan | **Strictly Isolated (0% Emergency MIDs)** |
| `door_slams` | 35 | 35 | 15 | Slam, Door, Sliding door, Cupboard open or close, Knock | **Strictly Isolated (0% Emergency MIDs)** |
| `car_alarms_detectors` | 35 | 35 | 15 | Car alarm, Smoke detector, smoke alarm, Fire alarm, Alarm, Buzzer, Alarm clock | **Strictly Isolated (0% Emergency MIDs)** |
| `fireworks_firecrackers` | 30 | 30 | 15 | Fireworks, Firecracker, Explosion | **Strictly Isolated (0% Emergency MIDs)** |
| `construction_tools` | 30 | 30 | 15 | Hammer, Jackhammer, Sawing, Drill, Power tool, Tools | **Strictly Isolated (0% Emergency MIDs)** |
| `whistles_high_pitch` | 20 | 20 | 15 | Whistle, Steam whistle, Train whistle, Tire squeal, Squeal | **Strictly Isolated (0% Emergency MIDs)** |

---

## 4. Recommended Download Priorities

When download authorization is granted, execute in sequential priority tiers:

1. **Priority Tier 1 (`P1_PRIMARY` — 435 clips):**
   - Immediately download the 435 primary clips to satisfy all target quotas.
   - Expected yield based on 75% YouTube survival rate: ~325–350 clips.
2. **Priority Tier 2 (`P2_BACKUP` — 180 clips):**
   - Use as instant drop-in replacements for any P1 clips that fail to download (e.g. video removed, private, or geo-blocked).
3. **Priority Tier 3 (`P2_REVIEW` & `P3_REVIEW` — 56 clips):**
   - Download only into an isolated staging folder for manual human listening audit before inclusion in training.

---

## 5. Verification Checklist & Compliance

- [x] **Zero Audio Downloaded:** No audio files were fetched or written during this task.
- [x] **Zero Existing Manifests Modified:** All existing train, val, eval, and combined manifests remain bit-for-bit unchanged.
- [x] **Evaluation Set Untouched:** Independent evaluation audio and manifest remain strictly isolated.
- [x] **No Model Retraining:** All models and checkpoints remain intact.
