# Training Experiment Report: PANNs CNN14 (Transfer Learning)

**Execution Timestamp:** 2026-11-02 19:18:58  
**Model Architecture:** `PannsCNN14`  
**Total / Trainable Parameters:** 79,963,653 / 10,245  
**Optimizer:** ADAM  
**Head Learning Rate:** `0.001`  
**Backbone Learning Rate:** `None`  
**Checkpoint Directory:** [`models\experiments\phase2_e4\e4_matched_control_construction_tools_seed42`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_matched_control_construction_tools_seed42)  
**Best Checkpoint File:** [`best_model.pt`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_matched_control_construction_tools_seed42\best_model.pt)  
**Training Manifest:** [`e4_matched_control_construction_tools_train.csv`](file:///D:\ml\fydp\MyNet\data\processed\e4_ablations\e4_matched_control_construction_tools_train.csv)  
**Validation Manifest:** [`shared_val_split_manifest.csv`](file:///D:\ml\fydp\MyNet\data\processed\shared_val_split_manifest.csv)  

---

## 1. Executive Summary

- **Best Epoch:** 1 (out of 8 epochs completed)
- **Best Validation Accuracy:** **89.52%**
- **Best Validation Macro F1:** **0.8965**
- **Best Validation Loss:** 0.8433 (Selection Metric: macro_f1)
- **Total Training Duration:** 38.4s (0.64 min)
- **Peak GPU Memory:** 1363.30 MB

---

## 2. Dataset Distribution

- **Training Partition:** 616 samples
- **Validation Partition:** 105 samples

| Class Name | Train Samples | Val Samples | Total Samples | Train Class Weight |
|---|---:|---:|---:|---:|
| `scream` | 30 | 7 | 37 | 2.4327 |
| `glass_breaking` | 77 | 19 | 96 | 0.9478 |
| `gunshot` | 92 | 23 | 115 | 0.7933 |
| `siren` | 127 | 32 | 159 | 0.5746 |
| `other` | 290 | 24 | 314 | 0.2517 |
| **Total** | **616** | **105** | **721** | - |

---

## 3. Comparison Against Frozen AudioSet Baseline

| Metric / Class | Baseline (AudioSet Val N=105) | Combined Model (AudioSet+NIGENS Val N=105) | Delta |
|---|---:|---:|---:|
| **Overall Accuracy** | 89.52% | 89.52% | +0.00% |
| **Macro F1 Score** | 0.8995 | 0.8965 | -0.0030 |
| `scream` F1 | 1.0000 | 1.0000 | +0.0000 |
| `glass_breaking` F1 | 0.7619 | 0.7273 | -0.0346 |
| `gunshot` F1 | 0.9787 | 0.9787 | +0.0000 |
| `siren` F1 | 0.9841 | 0.9688 | -0.0153 |
| `other` F1 | 0.7727 | 0.8077 | +0.0350 |

---

## 4. Best Epoch Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Validation Support |
|---|---:|---:|---:|---:|
| `scream` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `glass_breaking` | 0.8571 | 0.6316 | 0.7273 | 19 |
| `gunshot` | 0.9583 | 1.0000 | 0.9787 | 23 |
| `siren` | 0.9688 | 0.9688 | 0.9688 | 32 |
| `other` | 0.7500 | 0.8750 | 0.8077 | 24 |

---

## 5. Validation Confusion Matrix

Rows indicate Ground Truth, columns indicate Predictions:

| True \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---:|---:|---:|---:|
| `scream` | 7 | 0 | 0 | 0 | 0 |
| `glass_breaking` | 0 | 12 | 1 | 0 | 6 |
| `gunshot` | 0 | 0 | 23 | 0 | 0 |
| `siren` | 0 | 0 | 0 | 31 | 1 |
| `other` | 0 | 2 | 0 | 1 | 21 |

