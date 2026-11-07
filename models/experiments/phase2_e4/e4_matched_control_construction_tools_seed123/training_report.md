# Training Experiment Report: PANNs CNN14 (Transfer Learning)

**Execution Timestamp:** 2026-11-02 19:20:12  
**Model Architecture:** `PannsCNN14`  
**Total / Trainable Parameters:** 79,963,653 / 10,245  
**Optimizer:** ADAM  
**Head Learning Rate:** `0.001`  
**Backbone Learning Rate:** `None`  
**Checkpoint Directory:** [`models\experiments\phase2_e4\e4_matched_control_construction_tools_seed123`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_matched_control_construction_tools_seed123)  
**Best Checkpoint File:** [`best_model.pt`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_matched_control_construction_tools_seed123\best_model.pt)  
**Training Manifest:** [`e4_matched_control_construction_tools_train.csv`](file:///D:\ml\fydp\MyNet\data\processed\e4_ablations\e4_matched_control_construction_tools_train.csv)  
**Validation Manifest:** [`shared_val_split_manifest.csv`](file:///D:\ml\fydp\MyNet\data\processed\shared_val_split_manifest.csv)  

---

## 1. Executive Summary

- **Best Epoch:** 4 (out of 11 epochs completed)
- **Best Validation Accuracy:** **89.52%**
- **Best Validation Macro F1:** **0.8907**
- **Best Validation Loss:** 0.4479 (Selection Metric: macro_f1)
- **Total Training Duration:** 71.1s (1.18 min)
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
| **Macro F1 Score** | 0.8995 | 0.8907 | -0.0088 |
| `scream` F1 | 1.0000 | 0.9333 | -0.0667 |
| `glass_breaking` F1 | 0.7619 | 0.7805 | +0.0186 |
| `gunshot` F1 | 0.9787 | 1.0000 | +0.0213 |
| `siren` F1 | 0.9841 | 0.9841 | +0.0000 |
| `other` F1 | 0.7727 | 0.7556 | -0.0171 |

---

## 4. Best Epoch Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Validation Support |
|---|---:|---:|---:|---:|
| `scream` | 0.8750 | 1.0000 | 0.9333 | 7 |
| `glass_breaking` | 0.7273 | 0.8421 | 0.7805 | 19 |
| `gunshot` | 1.0000 | 1.0000 | 1.0000 | 23 |
| `siren` | 1.0000 | 0.9688 | 0.9841 | 32 |
| `other` | 0.8095 | 0.7083 | 0.7556 | 24 |

---

## 5. Validation Confusion Matrix

Rows indicate Ground Truth, columns indicate Predictions:

| True \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---:|---:|---:|---:|
| `scream` | 7 | 0 | 0 | 0 | 0 |
| `glass_breaking` | 0 | 16 | 0 | 0 | 3 |
| `gunshot` | 0 | 0 | 23 | 0 | 0 |
| `siren` | 0 | 0 | 0 | 31 | 1 |
| `other` | 1 | 6 | 0 | 0 | 17 |

