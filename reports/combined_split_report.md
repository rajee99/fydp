# Combined Dataset Source-Grouped Train/Validation Split Report
**Date:** 2026-10-30 12:29:00  
**Target Project:** `D:\ml\fydp\MyNet`  
**Random Seed:** `42` (Fixed & Fully Reproducible)  
**Source Manifest:** [`data/processed/combined_training_manifest.csv`](file:///d:/ml/fydp/MyNet/data/processed/combined_training_manifest.csv)  

---

## 1. Executive Summary & Split Statistics

- **Total Combined Samples Assigned:** **946**
- **Training Partition:** **756 samples** (79.92%) across **614 unique source groups**
- **Validation Partition:** **190 samples** (20.08%) across **165 unique source groups**
- **Target Ratio:** 80% Train / 20% Validation (Achieved: **79.92% / 20.08%**)
- **Source-Level Isolation:** **100% Guaranteed** (0 group overlap, 0 video ID overlap, 0 file hash overlap)
- **Evaluation Isolation:** NIGENS folds 7 & 8 are strictly absent; zero collision with independent evaluation audio.

---

## 2. Per-Class Distribution across Splits

Grouped stratification successfully maintained close alignment with the 80/20 target across all five classes:

| Class Name | Combined Total | Training Set | Validation Set | Train Ratio (%) | Val Ratio (%) | Class Share (%) |
|---|---:|---:|---:|---:|---:|---:|
| `scream` | 88 | 70 | 18 | 79.55% | 20.45% | 9.3% |
| `glass_breaking` | 103 | 82 | 21 | 79.61% | 20.39% | 10.9% |
| `gunshot` | 115 | 92 | 23 | 80.00% | 20.00% | 12.2% |
| `siren` | 159 | 127 | 32 | 79.87% | 20.13% | 16.8% |
| `other` | 481 | 385 | 96 | 80.04% | 19.96% | 50.8% |
| **Total** | **946** | **756** | **190** | **79.92%** | **20.08%** | **100.0%** |

> [!NOTE]
> Maximum class-level deviation from the target 20.00% validation ratio is **0.45%** (`scream`: 20.45%), which occurred because source isolation strictly prevented splitting multi-sample scream recording sessions.

---

## 3. Dataset Origin Breakdown across Splits

| Source Dataset | Total Samples | Training Set | Validation Set | Train % | Val % |
|---|---:|---:|---:|---:|---:|
| **AudioSet** | 527 | 418 | 109 | 79.32% | 20.68% |
| **NIGENS** | 419 | 338 | 81 | 80.67% | 19.33% |
| **Combined** | **946** | **756** | **190** | **79.92%** | **20.08%** |

---

## 4. Source-Group Isolation & Leakage Audit

| Verification Dimension | Result | Observed Metrics | Audit Guarantee |
|---|:---:|---|---|
| **Total Sample Coverage** | **PASS** | 946 / 946 samples assigned | Every sample appears in exactly one split |
| **Sample ID Disjointness** | **PASS** | 0 overlapping IDs | Zero duplicate samples between train & val |
| **File Hash Disjointness** | **PASS** | 0 overlapping SHA256 hashes | Zero acoustic byte collisions between splits |
| **Source Group Disjointness** | **PASS** | 0 overlapping source groups | All takes/crops from same source stay together |
| **AudioSet Video ID Isolation** | **PASS** | 0 overlapping video IDs | Zero YouTube video cross-split leakage |
| **Evaluation Folds Isolation** | **PASS** | 0 samples from folds 7/8 | Held-out evaluation set remains strictly isolated |
| **5-Class Representation** | **PASS** | 5 classes in train, 5 classes in val | All classes adequately represented in both splits |
| **Split Column Integrity** | **PASS** | 100% assigned (`train` / `validation`) | Preserves all 23 schema columns |

---

## 5. Output Manifest Reference

1. **Training Manifest:** [`data/processed/combined_train_split_manifest.csv`](file:///d:/ml/fydp/MyNet/data/processed/combined_train_split_manifest.csv)  
   - **756 rows**, `split='train'`
2. **Validation Manifest:** [`data/processed/combined_val_split_manifest.csv`](file:///d:/ml/fydp/MyNet/data/processed/combined_val_split_manifest.csv)  
   - **190 rows**, `split='validation'`

---

## 6. Confirmation of Reproducibility

The split execution is completely deterministic and reproducible. Rerunning:
```powershell
python src/split_combined_dataset.py --seed 42
```
produces identical SHA256 file digests for both `combined_train_split_manifest.csv` and `combined_val_split_manifest.csv`.