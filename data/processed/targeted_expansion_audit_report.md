# Targeted In-The-Wild Dataset Expansion Audit Report

**Audit Timestamp:** 2026-10-30 16:00:11  
**Audit Scope:** Targeted Dataset Expansion Post-Download Verification & Subdomain Reconciliation  
**Primary Candidates Manifest:** [`data/processed/targeted_expansion_candidates.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_candidates.csv)  
**Replenishment Manifest:** [`data/processed/targeted_expansion_replacements.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_replacements.csv)  
**Replenishment Mapping Log:** [`data/processed/targeted_expansion_replacements_map.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_replacements_map.csv)  
**Content Review Template:** [`data/processed/targeted_expansion_content_validation.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_content_validation.csv)  
**Downloaded Audio Directory:** [`data/raw/targeted_expansion/`](file:///d:/ml/fydp/MyNet/data/raw/targeted_expansion/)  
**Download Results Log:** [`data/processed/targeted_expansion_download_results.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_download_results.csv)  

---

## 1. Root-Cause Explanation of Subdomain Count Discrepancies

### A. The Discrepancy Observed
In the initial post-download reporting, a discrepancy was observed between class totals and the sum of subdomain rows:
- **Scream:** Total accepted files was **97**, but the listed subdomains in the report table summed to **105** (+8 overcount).
- **Glass Breaking:** Total accepted files was **81**, but the listed subdomains in the report table summed to **88** (+7 overcount).
- **Other:** Total accepted files was **222**, and its subdomains summed exactly to **222** (0 discrepancy).

### B. Root-Cause Mechanism: Shared Subdomain Key Collisions
Investigation of the reporting script (`scratch/generate_replenishment_report.py`) revealed that subdomain aggregators were stored in a flat dictionary keyed solely by `subdomain` string without partitioning by `target_class`:

```python
# Defective reporting aggregation (flat key)
sub_p1 = defaultdict(lambda: Counter())
for r in p1_results:
    sub_p1[r['subdomain']]['total'] += 1
    if is_success:
        sub_p1[r['subdomain']]['success'] += 1
```

However, across the AudioSet expansion ontology, two environmental subdomain names were shared across both classes:
1. **`indoor_hall`**: Present in both `scream` and `glass_breaking`.
2. **`outdoor_urban`**: Present in both `scream` and `glass_breaking`.

Because `sub_p1['indoor_hall']` and `sub_p1['outdoor_urban']` accumulated counts across **both** classes simultaneously:
- When printing the Scream table, `sub_p1['indoor_hall']` reported **6 successes** (3 scream + 3 glass) instead of **3 scream** (+3 error).
- When printing the Scream table, `sub_p1['outdoor_urban']` reported **9 successes** (4 scream + 5 glass) instead of **4 scream** (+5 error).
- **Total Scream Table Overcount:** `+3 + 5 = +8` $\rightarrow$ `97 + 8 = 105`.
- When printing the Glass table, `sub_p1['indoor_hall']` reported **6 successes** (3 scream + 3 glass) instead of **3 glass** (+3 error).
- When printing the Glass table, `sub_p1['outdoor_urban']` reported **9 successes** (4 scream + 5 glass) instead of **5 glass** (+4 error).
- **Total Glass Table Overcount:** `+3 + 4 = +7` $\rightarrow$ `81 + 7 = 88`.
- In contrast, none of the 7 subdomains in `other` collided with scream or glass, so `other` reconciled with 100% precision (222 / 222).

### C. Subdomain Exclusivity & Source of Truth
- **Subdomain Exclusivity:** Every single audio recording belongs to **exactly one target class** and **exactly one acoustic subdomain**. There is zero multi-label assignment in this expansion set.
- **Source of Truth:** Keying aggregations strictly by the tuple `(target_class, subdomain)` eliminates key collisions and demonstrates that the actual audio files on disk and results CSV reconcile perfectly to 97 scream, 81 glass breaking, and 222 other.

---

## 2. Reconciled Class and Subdomain Inventory

### A. Reconciled Class-Level Summary

| Target Class | Initial Target Quota | P1 Accepted | P2 Accepted | Total Verified Audio | Remaining Deficit | % Quota Achieved |
|---|---:|---:|---:|---:|---:|:---:|
| `scream` | **110** | 76 | 21 | **97** | 13 | **88.2%** |
| `glass_breaking` | **85** | 68 | 13 | **81** | 4 | **95.3%** |
| `other` | **240** | 184 | 38 | **222** | 18 | **92.5%** |
| **TOTAL** | **435** | **328** | **72** | **400** | **35** | **92.0%** |

### B. Reconciled Scream Subdomains (`scream` — Total: 97)

| Subdomain / Acoustic Environment | P1 Queued | P1 Accepted | P2 Queued | P2 Accepted | Total Verified Audio |
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
| **Reconciled Scream Total** | **110** | **76** | **34** | **21** | **97 (100% Reconciled)** |

### C. Reconciled Glass Breaking Subdomains (`glass_breaking` — Total: 81)

| Subdomain / Physical Impact Type | P1 Queued | P1 Accepted | P2 Queued | P2 Accepted | Total Verified Audio |
|---|---:|---:|---:|---:|---:|
| `breaking_shatter` | 20 | 16 | 9 | 6 | **22** |
| `bottle_clink_shatter` | 15 | 11 | 0 | 0 | **11** |
| `bottle_liquid_shatter` | 15 | 12 | 8 | 7 | **19** |
| `tableware_falling` | 6 | 5 | 0 | 0 | **5** |
| `indoor_room` | 8 | 6 | 0 | 0 | **6** |
| `indoor_hall` | 4 | 3 | 0 | 0 | **3** |
| `outdoor_urban` | 6 | 5 | 0 | 0 | **5** |
| `isolated_glass` | 11 | 10 | 0 | 0 | **10** |
| **Reconciled Glass Total** | **85** | **68** | **17** | **13** | **81 (100% Reconciled)** |

### D. Reconciled Hard-Negative Other Subdomains (`other` — Total: 222)

| Subdomain / Confuser Domain | Target Quota | P1 Accepted | P2 Accepted | Total Verified Audio | % Quota Met |
|---|---:|---:|---:|---:|:---:|
| `metal_impacts` | 45 | 38 | 3 | **41** | **91.1%** |
| `human_shouting` | 45 | 36 | 6 | **42** | **93.3%** |
| `door_slams` | 35 | 28 | 4 | **32** | **91.4%** |
| `car_alarms_detectors` | 35 | 25 | 4 | **29** | **82.9%** |
| `fireworks_firecrackers` | 30 | 21 | 8 | **29** | **96.7%** |
| `construction_tools` | 30 | 22 | 8 | **30** | **100.0%** |
| `whistles_high_pitch` | 20 | 14 | 5 | **19** | **95.0%** |
| **Reconciled Other Total** | **240** | **184** | **38** | **222** | **92.5% (100% Reconciled)** |

---

## 3. Duplicate and Isolation Verification Audit

A five-point integrity audit was conducted across all 400 audio files on disk and all download result rows:

| Verification Criterion | Expected | Verified Result | Audit Verdict |
|---|:---:|:---:|:---:|
| **Files on Disk vs. Results Log** | 400 | Exactly 400 files present; 0 missing; 0 untracked | ✅ PASS |
| **P1 + P2 Reconciliation** | 328 P1 + 72 P2 | 328 P1 accepted + 72 P2 accepted = 400 total | ✅ PASS |
| **Unique Source Video IDs** | 400 unique | Exactly 400 unique video IDs; 0 duplicate IDs | ✅ PASS |
| **Unique Audio File Paths** | 400 unique | Exactly 400 unique disk paths; 0 collisions | ✅ PASS |
| **Single Class Membership** | 1 class / file | 100% of files match directory and filename class | ✅ PASS |
| **Overlap with Training (`train`)** | 0 overlap | 0 video ID overlap with `target_manifest.csv` (527 clips) | ✅ PASS |
| **Overlap with Evaluation (`eval`)** | 0 overlap | 0 video ID overlap with `eval_manifest.csv` (519 clips) | ✅ PASS |
| **Overlap with Pilot (`pilot`)** | 0 overlap | 0 video ID overlap with `pilot_accessible.csv` (15 clips) | ✅ PASS |
| **Untouched Reserves** | 130 clips | 73 `P2_BACKUP`, 30 `P2_REVIEW`, 27 `P3_REVIEW` untouched | ✅ PASS |

---

## 4. Technical Audio Validation Summary

Every accepted audio recording was subjected to binary inspection:
- **Sampling Rate:** 100% verified at **16,000 Hz** (400 / 400).
- **Channel Count:** 100% verified **Mono (1 channel)** (400 / 400).
- **Bit Depth / Encoding:** 100% verified **16-bit linear PCM** (`sampwidth = 2`, `compression = NONE`) (400 / 400).
- **Non-Empty Payload:** All files have non-zero frames (mean duration: 9.87s, mean file size: ~316 KB).
- **Strict Purge of Defective Streams:** 21 audio streams whose video source was shorter than manifest timestamps (>0.8s deviation) were rejected and deleted immediately, ensuring **zero silent padding artifacts** enter the dataset.

---

## 5. Semantic / Content Validation Readiness

### A. Why Format Checks Are Insufficient
While 100% of the 400 files pass technical format validation (16 kHz, mono, 16-bit PCM), **technical compliance does not guarantee acoustic correctness**:
- A file labeled `Screaming` in AudioSet may contain music vocals, cheering crowds, or singing rather than panic/distress screams.
- A file labeled `Shatter` may contain falling plastic, ceramic plates, or metallic clatter rather than glass breaking.
- A hard-negative `Other` file (e.g. human shouting or construction) might contain faint background sirens, screeches, or gunshots that could contaminate model training.

### B. Dedicated Human Review Template Generated
To support systematic, reviewable validation, a structured CSV was generated:
- **Review File:** [`data/processed/targeted_expansion_content_validation.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_content_validation.csv)
- **Total Items:** Exactly 400 rows (328 P1, 72 P2).
- **Schema:** `file_path`, `target_class`, `subdomain`, `video_id`, `start_seconds`, `end_seconds`, `matched_label`, `origin`, `duration_seconds`, `file_size_bytes`, `validation_status`, `reviewer_verdict`, `audited_class`, `acoustic_notes`.
- **Initial Status:** All 400 rows are initialized to `PENDING`.

### C. Class-Wise Human Review Checklist

#### 1. Scream Checklist (97 files)
- [ ] **Audibility:** Is a human vocal scream clearly audible above any background music/speech/crowd?
- [ ] **Acoustic Signature:** High fundamental pitch, harsh phonation, chaotic formant structure.
- [ ] **Exclusions (Reject or reclassify as Other):** Normal singing, prolonged shouting without scream timbre, laughter, sports cheering, baby cooing/babble.

#### 2. Glass Breaking Checklist (81 files)
- [ ] **Audibility:** Is the impact and resonant shattering of glass distinctly identifiable?
- [ ] **Acoustic Signature:** High-frequency impact burst (>3 kHz) followed by rapid secondary shards tinkling / scattering.
- [ ] **Exclusions (Reject or reclassify as Other):** Metal-on-metal clang, plastic snapping, ceramic mug dull thud with no glass ring, gunshots, explosions.

#### 3. Hard-Negative Other Checklist (222 files)
- [ ] **Zero Emergency Sound Contamination:** Verify that **no** emergency signals (scream, gunshot, glass breaking, emergency vehicle siren) are audible in the background.
- [ ] **Target Confuser Integrity:** Verify that the audio contains the expected hard-negative event (shouting, door slam, metal strike, vehicle alarm horn, firecracker, construction tool, whistle).
- [ ] **Exclusions:** If an emergency event is audible (even in the background), flag as `CONTAINS_EMERGENCY_EVENT` to prevent training corruption.

---

## 6. Preservation of Existing Work & Safeguards

- [x] `target_manifest.csv` (527 clips) and `eval_manifest.csv` (519 clips) are **100% unaltered**.
- [x] Audio in `data/raw/train/`, `data/raw/eval/`, and `data/raw/pilot/` are **100% unaltered**.
- [x] Baseline and combined PANNs CNN14 model checkpoints are **100% unaltered**.
- [x] No expansion files have been merged into training manifests.
- [x] No new train-validation splits have been generated.
- [x] No model training has been started.

---

## 7. Recommended Next Steps

1. **Execute Content Review:** Use [`data/processed/targeted_expansion_content_validation.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_content_validation.csv) with the review tool to verify the acoustic validity of the 400 audio files.
2. **Replenish Deficits if Needed:** If any files are rejected during human content review, draw replacement candidates from the remaining 73 clean `P2_BACKUP` reserves.
3. **Merge Approved Files:** After content validation approval, merge the clean in-the-wild clips with the baseline training set into a fresh expansion manifest.
