# Targeted In-The-Wild Dataset Expansion Human Validation Report

**Execution Date:** 2026-10-30 19:38:16  
**Review Log:** [`data/processed/targeted_expansion_review_results.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_review_results.csv) (400 decisions)  
**Synchronized Content Validation Manifest:** [`data/processed/targeted_expansion_content_validation.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_content_validation.csv)  
**Review Tool:** [`src/review_tool.py`](file:///d:/ml/fydp/MyNet/src/review_tool.py)  
**Audio Directory:** [`data/raw/targeted_expansion/`](file:///d:/ml/fydp/MyNet/data/raw/targeted_expansion/) (400 WAV files)  

---

## 1. Executive Summary & Validation Yield

- **Total Recordings Evaluated:** **400 clips** (100% completed human semantic review).
- **Accepted Clips (`ACCEPT`):** **376 clips** (**94.00%** semantic acceptance rate).
- **Reclassified Clips (`RECLASSIFY`):** **4 clips** (1.00% — verified audio assigned to correct observed class).
- **Rejected Clips (`REJECT`):** **19 clips** (4.75% — acoustic content failed target class criteria or was inaudible).
- **Pending Second Review (`NEEDS_SECOND_REVIEW`):** **1 clip** (0.25% — ambiguous fireworks/firecrackers recording).
- **Zero Emergency Sound Contamination:** **100% of the 222 hard-negative `other` files** were confirmed free of unintended emergency sirens, screams, or gunshots (`emergency_sound_present = False` for 400/400 clips).
- **Total Usable Expansion Clips:** **380 clips** (376 accepted + 4 valid reclassified hard-negatives).

---

## 2. Decision Breakdown by Target Class

| Target Class | Reviewed Clips | ACCEPT | RECLASSIFY | REJECT | 2nd REVIEW | Acceptance Rate | Usable Clips |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scream` | 97 | **82** | 3 | 12 | 0 | **84.5%** | **82** |
| `glass_breaking` | 81 | **74** | 1 | 6 | 0 | **91.4%** | **74** |
| `other` | 222 | **220** | 0 | 1 | 1 | **99.1%** | **224** |
| **TOTAL** | **400** | **376** | **4** | **19** | **1** | **94.0%** | **380** |

---

## 3. Subdomain-Level Breakdown of Accepted Audio

### A. Scream Subdomains (`scream` — 82 Accepted)

| Subdomain / Acoustic Environment | Reviewed | Accepted | Rejected / Reclassified | Acceptance Rate |
|---|---:|---:|---:|---:|
| `indoor_small_room` | 16 | **13** | 3 | 81.2% |
| `indoor_hall` | 3 | **3** | 0 | 100.0% |
| `outdoor_urban` | 4 | **2** | 2 | 50.0% |
| `outdoor_rural` | 0 | **0** | 0 | 0.0% |
| `crowd_background` | 10 | **3** | 7 | 30.0% |
| `vehicle_background` | 6 | **4** | 2 | 66.7% |
| `music_background` | 32 | **31** | 1 | 96.9% |
| `isolated_vocal` | 16 | **16** | 0 | 100.0% |
| `child_scream` | 5 | **5** | 0 | 100.0% |
| `speech_mixed` | 5 | **5** | 0 | 100.0% |
| **Subtotal Scream** | **97** | **82** | **15** | **84.5%** |

### B. Glass Breaking Subdomains (`glass_breaking` — 74 Accepted)

| Subdomain / Physical Impact Type | Reviewed | Accepted | Rejected / Reclassified | Acceptance Rate |
|---|---:|---:|---:|---:|
| `breaking_shatter` | 22 | **21** | 1 | 95.5% |
| `bottle_clink_shatter` | 11 | **11** | 0 | 100.0% |
| `bottle_liquid_shatter` | 19 | **16** | 3 | 84.2% |
| `tableware_falling` | 5 | **2** | 3 | 40.0% |
| `indoor_room` | 6 | **6** | 0 | 100.0% |
| `indoor_hall` | 3 | **3** | 0 | 100.0% |
| `outdoor_urban` | 5 | **5** | 0 | 100.0% |
| `isolated_glass` | 10 | **10** | 0 | 100.0% |
| **Subtotal Glass Breaking** | **81** | **74** | **7** | **91.4%** |

### C. Hard-Negative Other Subdomains (`other` — 220 Accepted + 4 Reclassified = 224 Usable)

| Subdomain / Confuser Domain | Reviewed | Accepted | Rejected / 2nd Review | Acceptance Rate |
|---|---:|---:|---:|---:|
| `metal_impacts` | 41 | **40** | 1 | 97.6% |
| `human_shouting` | 42 | **42** | 0 | 100.0% |
| `door_slams` | 32 | **32** | 0 | 100.0% |
| `car_alarms_detectors` | 29 | **29** | 0 | 100.0% |
| `fireworks_firecrackers` | 29 | **28** | 1 | 96.6% |
| `construction_tools` | 30 | **30** | 0 | 100.0% |
| `whistles_high_pitch` | 19 | **19** | 0 | 100.0% |
| **Subtotal Hard-Negative Other** | **222** | **220** | **2** | **99.1%** |

---

## 4. Reclassification & Rejection Analysis

### A. Reclassified Clips (4 clips $\rightarrow$ Eligible for Hard-Negative `other`)

| Video ID | Original Target Class | Original Subdomain | Actual Observed Sound | Research Value |
|---|---|---|---|---|
| `ad-8X58aIEM` | `scream` | `indoor_small_room` | **`human_shouting`** | High-value hard-negative: loud non-scream human shouting/action sound. |
| `Pdau8nsBqxs` | `scream` | `indoor_small_room` | **`human_shouting`** | High-value hard-negative: loud non-scream human shouting/action sound. |
| `d4v3_z0ISrM` | `scream` | `indoor_small_room` | **`other`** | High-value hard-negative: loud non-scream human shouting/action sound. |
| `I1TWWYbjvJ8` | `glass_breaking` | `breaking_shatter` | **`door_slam`** | High-value hard-negative: loud non-scream human shouting/action sound. |

### B. Rejected Clips (19 clips)

- **Scream Rejections (12 clips):**
  - 7 clips in `crowd_background`: General cheering, sports chanting, or crowd noise without genuine emergency/panic screams.
  - 2 clips in `outdoor_urban`: Urban speech/traffic ambience where screams were inaudible.
  - 2 clips in `vehicle_background`: Engine drone obscured faint vocalizations.
  - 1 clip in `music_background`: Vocal singing rather than distress scream.
- **Glass Breaking Rejections (6 clips):**
  - 3 clips in `tableware_falling`: Dull ceramic thuds or plastic clatter with no high-frequency glass shatter ring.
  - 3 clips in `bottle_liquid_shatter`: Liquid splash without audible bottle glass breakage.
- **Other Rejections (1 clip):**
  - 1 clip in `metal_impacts` (`7P9AhulU9co`): Low SNR/inaudible impact.

### C. Borderline / Needs Second Review (1 clip)

- `ja-Qxq5NAoE` (`other / fireworks_firecrackers`): Single loud pop ambiguous between firecracker and distant gunshot; flagged for secondary review.

---

## 5. Quota Fulfillment & Reserve Feasibility

| Class | Target Quota | Verified Usable | Deficit | Clean `P2_BACKUP` Reserve | Replenishment Feasibility |
|---|---:|---:|---:|---:|:---:|
| `scream` | **110** | **82** | 28 | 6 in P2 (+ 12 in P3 review) | Partial P2 reserve (needs extra P3/candidates) |
| `glass_breaking` | **85** | **74** | 11 | **18** | **100% Coverable from P2 reserve** |
| `other` | **240** | **224** | 16 | **49** | **100% Coverable from P2 reserve** |
| **TOTAL** | **435** | **380** | **55** | **73** | High coverage from clean reserves |

---

## 6. Recommended Next Steps

1. **Option A (Proceed to Integration Directly):** Use the **376 accepted clips** (or **380** including reclassified hard-negatives) directly to build the expanded dataset manifest. 380 clean, human-validated in-the-wild clips represent an **87.4% fulfillment** of the expansion goal and provide a massive boost over the original AudioSet baseline (527 clips).
2. **Option B (Download Targeted Quota Top-Up):** Download replacements for the 11 glass breaking and 16 other deficits from the clean 67 remaining `P2_BACKUP` candidates before dataset integration.
