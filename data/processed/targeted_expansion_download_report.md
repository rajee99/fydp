# Targeted In-The-Wild Dataset Expansion Download Report
**Execution Date:** 2026-10-30 15:58:39  
**Target Expansion Manifest:** [`data/processed/targeted_expansion_candidates.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_candidates.csv)  
**Replenishment Manifest:** [`data/processed/targeted_expansion_replacements.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_replacements.csv)  
**Replenishment Mapping Log:** [`data/processed/targeted_expansion_replacements_map.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_replacements_map.csv)  
**Content Review Template:** [`data/processed/targeted_expansion_content_validation.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_content_validation.csv)  
**Downloaded Audio Location:** [`data/raw/targeted_expansion/`](file:///d:/ml/fydp/MyNet/data/raw/targeted_expansion/)  
**Download Results Log:** [`data/processed/targeted_expansion_download_results.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_download_results.csv)  
**Downloader Used:** [`src/batch_downloader.py`](file:///d:/ml/fydp/MyNet/src/batch_downloader.py)  
**Preferred Baseline Model:** [`models/experiments/panns_cnn14/best_model.pt`](file:///d:/ml/fydp/MyNet/models/experiments/panns_cnn14/best_model.pt)  

---

## 1. Executive Summary & Combined Download Yield

- **Initial Phase 1 (`P1_PRIMARY`):** 435 queued -> **328 downloaded & verified** (75.40% yield, 107 deficit).
- **Phase 2 Replenishment (`P2_BACKUP`):** 107 queued -> **72 downloaded & verified** (67.29% yield, 35 permanent/validation fails).
- **Total Audio Files on Disk:** **400 verified WAV files** (92.0% of total 435 expansion quota achieved).
- **Audio Format Compliance:** **100% of all 400 files** strictly verified: **16,000 Hz**, **Mono**, **16-bit PCM**, non-empty, matching manifest durations.
- **Total Unusable Segments Purged:** 21 duration-mismatched files (14 in P1, 7 in P2) were rejected and deleted immediately with zero silent padding.
- **Strict Physical Dataset Isolation:** All files are stored under `data/raw/targeted_expansion/`. **Zero existing training (527), evaluation (519), or pilot (15) audio files were touched or altered**.
- **No Manifest Integration:** The new recordings remain quarantined and have not been added to training manifests or used for model training.

---

## 2. Per-Class Quota & Download Results

| Target Class | Target Quota | P1 Downloaded | P2 Replenished | Total Valid Audio | Remaining Deficit | % Quota Achieved |
|---|---:|---:|---:|---:|---:|:---:|
| `scream` | **110** | 76 | 21 | **97** | 13 | **88.2%** |
| `glass_breaking` | **85** | 68 | 13 | **81** | 4 | **95.3%** |
| `other` | **240** | 184 | 38 | **222** | 18 | **92.5%** |
| **TOTAL** | **435** | **328** | **72** | **400** | **35** | **92.0%** |

---

## 3. Subdomain Breakdown

### A. Scream Subdomains (`scream`)

| Subdomain / Acoustic Environment | P1 Queued | P1 Suc | P2 Queued | P2 Suc | Total Valid Audio |
|---|---:|---:|---:|---:|---:|
| `indoor_small_room` | 20 | 16 | 0 | 0 | **16** |
| `indoor_hall` | 5 | 3 | 0 | 0 | **3** |
| `outdoor_urban` | 7 | 4 | 0 | 0 | **4** |
| `outdoor_rural` | 1 | 0 | 0 | 0 | **0** |
| `crowd_background` | 15 | 10 | 0 | 0 | **10** |
| `vehicle_background` | 8 | 6 | 0 | 0 | **6** |
| `music_background` | 15 | 11 | 34 | 21 | **32** |
| `isolated_vocal` | 20 | 16 | 0 | 0 | **16** |
| `child_scream` | 10 | 5 | 0 | 0 | **5** |
| `speech_mixed` | 9 | 5 | 0 | 0 | **5** |
| **Subtotal Scream** | **110** | **76** | **34** | **21** | **97 / 110 (88.2%)** |

### B. Glass Breaking Subdomains (`glass_breaking`)

| Subdomain / Physical Impact Type | P1 Queued | P1 Suc | P2 Queued | P2 Suc | Total Valid Audio |
|---|---:|---:|---:|---:|---:|
| `breaking_shatter` | 20 | 16 | 9 | 6 | **22** |
| `bottle_clink_shatter` | 15 | 11 | 0 | 0 | **11** |
| `bottle_liquid_shatter` | 15 | 12 | 8 | 7 | **19** |
| `tableware_falling` | 6 | 5 | 0 | 0 | **5** |
| `indoor_room` | 8 | 6 | 0 | 0 | **6** |
| `indoor_hall` | 4 | 3 | 0 | 0 | **3** |
| `outdoor_urban` | 6 | 5 | 0 | 0 | **5** |
| `isolated_glass` | 11 | 10 | 0 | 0 | **10** |
| **Subtotal Glass Breaking** | **85** | **68** | **17** | **13** | **81 / 85 (95.3%)** |

### C. Hard-Negative Other Subdomains (`other`)

| Subdomain / Confuser Domain | Target Quota | P1 Suc | P2 Suc | Total Valid Audio | % Quota Achieved |
|---|---:|---:|---:|---:|:---:|
| `metal_impacts` | 45 | 38 | 3 | **41** | **91.1%** |
| `human_shouting` | 45 | 36 | 6 | **42** | **93.3%** |
| `door_slams` | 35 | 28 | 4 | **32** | **91.4%** |
| `car_alarms_detectors` | 35 | 25 | 4 | **29** | **82.9%** |
| `fireworks_firecrackers` | 30 | 21 | 8 | **29** | **96.7%** |
| `construction_tools` | 30 | 22 | 8 | **30** | **100.0%** |
| `whistles_high_pitch` | 20 | 14 | 5 | **19** | **95.0%** |
| **Subtotal Hard-Negative Other** | **240** | **184** | **38** | **222 / 240** | **92.5%** |

---

## 4. P2 Replenishment Mapping & Execution Audit

All 107 replenishment candidates were drawn strictly from `P2_BACKUP` and mapped 1-to-1 to each failed P1 clip:

- **Hard-Negative Other (56 clips):** 100% matched to the **exact same subdomain** (7 metal impacts, 9 shouting, 7 door slams, 10 alarms, 9 fireworks, 8 construction tools, 6 whistles).
- **Glass Breaking (17 clips):** Direct subdomain matches for breaking shatter (4) and bottle liquid shatter (3); remaining 10 mapped to equivalent high-yield shatter and bottle shatter reserves.
- **Scream (34 clips):** Direct matches for music background (4); remaining 30 mapped to verified unflagged reserve screams in music background.
- **Full Mapping Records:** Preserved in [`data/processed/targeted_expansion_replacements_map.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_replacements_map.csv).

---

## 5. Comprehensive Failure Analysis

| Failure Category | P1 Failures | P2 Failures | Combined Total | Root Cause & Failure Mechanism |
|---|---:|---:|---:|---|
| `VIDEO_UNAVAILABLE` | 55 | 20 | **75** | Source video removed, deleted, or set to private by YouTube uploader. |
| `PRIVATE_VIDEO` | 18 | 7 | **25** | Source video converted to private by channel owner. |
| `DURATION_MISMATCH` | 14 | 7 | **21** | Video stream shorter than slice duration (>0.8s deviation); strictly purged per validation rules. |
| `AUTH_REQUIRED` | 11 | 1 | **12** | Age restriction or bot detection triggered YouTube authentication wall. |
| `POLICY_REMOVED` | 6 | 0 | **6** | Video removed by YouTube for Terms of Service or Community Guidelines violations. |
| `TRANSIENT_EXHAUSTED` | 2 | 0 | **2** | Remote connection reset or timeout across retries. |
| `OTHER_FAILURE` | 1 | 0 | **1** | Other remote streaming failure. |

---

## 6. Audio Validation Compliance Audit

All 400 audio files in `data/raw/targeted_expansion/` underwent binary header and payload inspection:

- **Total Files Evaluated:** **400 WAV files** (97 scream, 81 glass breaking, 222 other).
- **Sampling Rate:** 100% verified at **16,000 Hz** (400 / 400).
- **Channel Count:** 100% verified **Mono (1 channel)** (400 / 400).
- **Bit Depth / Sample Width:** 100% verified **16-bit PCM** (`sampwidth = 2`, `compression = NONE`) (400 / 400).
- **Non-empty Audio:** 100% contain valid non-zero waveform samples.
- **Duration Accuracy:** All files match expected manifest durations within 0.8s tolerance; 21 invalid/short segments were strictly purged from disk.

---

## 7. Dataset Isolation & Safeguards Verification

- [x] **P2 / P3 Review Queues Untouched:** All 30 `P2_REVIEW` and 27 `P3_REVIEW` candidates remain completely untouched.
- [x] **Remaining P2 Reserve Preserved:** 73 unqueued `P2_BACKUP` candidates remain available in reserve.
- [x] **Original Training Data Untouched:** `data/raw/train/` contains exactly 527 WAV files, unchanged.
- [x] **Evaluation Set Untouched:** `data/raw/eval/` contains exactly 519 WAV files, unchanged.
- [x] **Pilot Data Untouched:** `data/raw/pilot/` contains exactly 15 WAV files, unchanged.
- [x] **Original Manifests Preserved:** `target_manifest.csv`, `eval_manifest.csv`, and `targeted_expansion_candidates.csv` remain unaltered.
- [x] **No Dataset Merge:** Newly downloaded audio files remain quarantined under `data/raw/targeted_expansion/` and have **not** been added to training manifests.
- [x] **No Model Training:** No models were retrained or altered.

