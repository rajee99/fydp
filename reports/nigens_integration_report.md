# NIGENS Dataset Integration Audit Report
**Date:** 2026-10-30 12:02:12  
**Target Project:** `D:\ml\fydp\MyNet`  
**Source Dataset:** NIGENS General Sounds Database (`D:\ml\fydp\MyNet\NIGENS`)  

---

## 1. Executive Summary

- **Total Audio Files Inspected:** 898
- **Audio File Integrity:** 898 valid / 0 invalid
- **Directly Approved / Included Samples:** **366**
  - Target `scream`: **49** clips
  - Target `other`: **317** clips
- **Samples Requiring Manual Review:** **308**
- **Excluded Samples:** **224**
  - Original evaluation folds (Folds 7 & 8): 217 clips
  - Weapon/gunshot confusers in general class: 5 clips
  - Exact duplicate within NIGENS: 1 clip
  - Unlisted auxiliary file: 1 clip
- **Cross-Dataset Leakage / Overlap with Project Audio:** **0 clips** (Strict zero-overlap verified)

---

## 2. Audio Integrity & Format Statistics

All inspected files were parsed using `soundfile` and validated against audio header constraints:

| Audio Property | Distribution Across Inspected Dataset |
|---|---|
| **Sample Rates** | 44,100 Hz (814 files), 25,000 Hz (80 GRID speech files), 48,000 Hz (4 files) |
| **Channel Layout** | Stereo / 2-channel (708 files), Mono / 1-channel (190 files) |
| **Bit Depth / Format** | PCM_16 (890 files), FLOAT 32-bit (4 files), PCM_24 (2 files), PCM_32 (1 file), IMA_ADPCM (1 file) |
| **Duration Range** | Min: 0.16s (`BottleSoap+6102_80_2.wav`), Max: 196.02s (`MusicPiano_DIGIVG1-76.wav`), Mean: 18.66s |
| **Total Audio Material** | 4.66 hours (16,759.3 seconds) |
| **Corrupt / Unreadable Files** | 0 |

---

## 3. Original Class Inventory & Mapping Breakdown

| Original NIGENS Folder | Total on Disk | Fold 1–6 (Train) | Fold 7–8 (Eval) | Status in Pipeline | FYDP Target Class | Eligible (Included) |
|---|:---:|:---:|:---:|---|:---:|:---:|
| `alarm` | 50 | - | - | Manual Review Required (37) | `other` | **0** |
| `baby` | 40 | - | - | Include (Approved) | `other` | **30** |
| `crash` | 50 | - | - | Manual Review Required (38) | `other` | **0** |
| `dog` | 45 | - | - | Include (Approved) | `other` | **35** |
| `engine` | 39 | - | - | Include (Approved) | `other` | **30** |
| `femaleScream` | 45 | - | - | Include (29) / Review (6) | `scream` | **29** |
| `femaleSpeech` | 40 | - | - | Include (Approved) | `other` | **31** |
| `fire` | 51 | - | - | Include (Approved) | `other` | **38** |
| `footsteps` | 42 | - | - | Include (Approved) | `other` | **32** |
| `general` | 303 | - | - | Inspect & Verify (223) | `other` | **0** |
| `knock` | 40 | - | - | Include (Approved) | `other` | **30** |
| `maleScream` | 31 | - | - | Include (20) / Review (4) | `scream` | **20** |
| `maleSpeech` | 40 | - | - | Include (Approved) | `other` | **29** |
| `phone` | 40 | - | - | Include (Approved) | `other` | **30** |
| `piano` | 42 | - | - | Include (Approved) | `other` | **32** |

---

## 4. Original 8-Fold Split Preservation

NIGENS is officially structured into an 8-fold cross-validation scheme (`NIGENS_8-foldSplit_fold<1..8>_wo_timit.flist`).
In accordance with the author's technical specification (Trowitzsch et al., 2019):
- **Folds 1 to 6 (Training & Cross-Validation):** 680 files total.
- **Folds 7 and 8 (Evaluation / Testing):** 217 files total. **Strictly excluded from integration to prevent evaluation contamination.**

| Fold Index | Flist File | Total Files | Role in Integration | Status |
|:---:|---|:---:|---|---|
| Fold 1 | `NIGENS_8-foldSplit_fold1_wo_timit.flist` | 117 | Training / Development | Eligible for Integration |
| Fold 2 | `NIGENS_8-foldSplit_fold2_wo_timit.flist` | 117 | Training / Development | Eligible for Integration |
| Fold 3 | `NIGENS_8-foldSplit_fold3_wo_timit.flist` | 114 | Training / Development | Eligible for Integration |
| Fold 4 | `NIGENS_8-foldSplit_fold4_wo_timit.flist` | 113 | Training / Development | Eligible for Integration |
| Fold 5 | `NIGENS_8-foldSplit_fold5_wo_timit.flist` | 110 | Training / Development | Eligible for Integration |
| Fold 6 | `NIGENS_8-foldSplit_fold6_wo_timit.flist` | 109 | Training / Development | Eligible for Integration |
| Fold 7 | `NIGENS_8-foldSplit_fold7_wo_timit.flist` | 109 | Held-out Evaluation | **EXCLUDED (Evaluation Isolation)** |
| Fold 8 | `NIGENS_8-foldSplit_fold8_wo_timit.flist` | 108 | Held-out Evaluation | **EXCLUDED (Evaluation Isolation)** |
| Unmapped | `alarm/Alarm+8001_04.wm.wav` | 1 | Unlisted auxiliary file | **EXCLUDED (Unlisted)** |

---

## 5. Duplicate and Cross-Dataset Isolation Audit

### 5.1 Cross-Dataset Isolation (Zero Leakage)
Every file in NIGENS was checked against SHA256 hashes of all existing project WAV files:
- `data/raw/train/`: 527 files checked -> **0 overlaps**
- `data/raw/eval/`: 519 files checked -> **0 overlaps**
- `data/raw/pilot/`: 15 files checked -> **0 overlaps**
Cross-dataset leakage is **0.00%**. Source separation is guaranteed.

### 5.2 Internal Exact Duplicates
NIGENS contains exactly **1 duplicate pair** across its 898 files:
- `fire/Fire_CampfireLoop2_SFX-KIT.wav` (Fold 1) - **RETAINED** (`INCLUDED`)
- `fire/Fire_Loop12_SFX-KIT.wav` (Fold 5) - **EXCLUDED** (`exact_duplicate_hash`)

### 5.3 Unlisted Auxiliary Files
- `alarm/Alarm+8001_04.wm.wav` (not listed in `all.flist` or `all_wo_timit.flist`) - **EXCLUDED** (`unlisted_file_not_in_official_flists`)

---

## 6. Detailed Breakdown of Manually Reviewable Samples (308 files)

Samples flagged as `MANUAL_REVIEW_REQUIRED` require explicit user decisions before injection into training:

1. **`crash` Class (38 training fold clips):**
   - **10 clips** contain glass, window, ceramic, or monitor destruction (e.g. `smash-window+shatter_CAP01-330.wav`, `GLASS-CRASH_GEN-HDF-12879.wav`, `CrashGlass+6004_94_1.wav`).
   - *Risk:* Mapping these to `other` would inject true glass-breaking sounds into the negative class. Conversely, mapping them to `glass_breaking` requires verifying they meet our project's strict glass shattering definition.
   - **28 clips** contain structural, metal, wood, or vehicle crashes (e.g. `AutoCrash`, `CrashWood`, `BuildingCrash`).

2. **`alarm` Class (37 training fold clips):**
   - **1 clip** is explicitly named `Siren+6086_53.wav` (Fold 1). *Risk:* Cannot be mapped to `other` as it is an emergency siren.
   - **36 clips** contain burglar alarms, car alarms, smoke detectors, and electronic beeps. *Risk:* Highly valuable hard-negatives for `other`, but must be vetted to prevent siren confusion.

3. **`general` Class (223 training fold clips):**
   - Contains heterogeneous non-target sounds. **5 clips** with gunshot/weapon keywords (`GUN-BULLET`, `GUN-MACHINE-GUN`, `GUN-RICOCHET`, `GUN-TASER`, `GunScreamer`) were **automatically excluded**.
   - Remaining 223 clips include vehicle horns, whistles, clinking glass, applause, dog barks, etc.

4. **Ambiguous Screams (10 clips in training folds):**
   - `femaleScream` (6 clips): `shout`, `19-yeard-old-girl-shouting`, `oneshout`, `womanshout`, `another-surprised-oooh`, `vocal-joy-and-excitement2`.
   - `maleScream` (4 clips): `sad-cry-7`, `manshout`, `shout`, `wahha-yanga-ripi-a-raja-bip-gonog`.
   - *Risk:* Distinguishing emotional speech/shouting from distress screaming.

---

## 7. Final Eligible Sample Summary

Under the conservative, high-precision filtering rules:
- **Eligible Scream Samples:** **+49 verified clips** (29 female, 20 male).
  - *Impact:* Increases our current training inventory of 37 screams by **+132.4%** to **86 clips**.
- **Eligible Hard-Negative 'Other' Samples:** **+317 verified clips**.
  - Includes door knocks (30), crying babies (30), barking dogs (35), running engines (30), speech sentences (60), fire crackling (38), footsteps (32), ringing phones (30), and piano music (32).
  - *Impact:* Increases our current training inventory of 120 'other' clips to **437 clips**.

---

## 8. Licensing and Dataset Use Restrictions

1. **License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (**CC BY-NC-ND 4.0**).
   - **Non-Commercial:** Research, academic, and non-commercial educational use only.
   - **Attribution:** Must cite: `Trowitzsch, I., Taghia, J., Kashef, Y., & Obermayer, K. (2019). The NIGENS general sound events database. TU Berlin, arXiv:1902.08314`.
   - **No Derivatives Redistribution:** While transformation/preprocessing for model training within the local project pipeline is permitted under fair use / research, modified raw audio files cannot be redistributed publicly.
2. **Watermarking:** Sound files acquired from StockMusic.com contain digital watermarking to identify unauthorized redistribution.
3. **TIMIT Speech Sounds:** 120 speech files from the TIMIT corpus were excluded by the original authors due to LDC licensing restrictions. Only the 80 unencumbered GRID speech files are included.
4. **Freesound Attribution:** Scream and selected general audio files were obtained from Freesound.org under Attribution licenses; original creator URLs are preserved in `NIGENS/README.md`.