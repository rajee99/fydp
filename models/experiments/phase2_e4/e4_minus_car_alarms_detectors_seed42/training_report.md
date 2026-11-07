# Training Experiment Report: PANNs CNN14 (Transfer Learning)

**Execution Timestamp:** 2026-11-02 18:31:49  
**Model Architecture:** `PannsCNN14`  
**Total / Trainable Parameters:** 79,963,653 / 10,245  
**Optimizer:** ADAM  
**Head Learning Rate:** `0.001`  
**Backbone Learning Rate:** `None`  
**Checkpoint Directory:** [`models\experiments\phase2_e4\e4_minus_car_alarms_detectors_seed42`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_minus_car_alarms_detectors_seed42)  
**Best Checkpoint File:** [`best_model.pt`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_minus_car_alarms_detectors_seed42\best_model.pt)  
**Training Manifest:** [`e4_minus_car_alarms_detectors_train.csv`](file:///D:\ml\fydp\MyNet\data\processed\e4_ablations\e4_minus_car_alarms_detectors_train.csv)  
**Validation Manifest:** [`shared_val_split_manifest.csv`](file:///D:\ml\fydp\MyNet\data\processed\shared_val_split_manifest.csv)  

---

## 1. Executive Summary

- **Best Epoch:** 26 (out of 30 epochs completed)
- **Best Validation Accuracy:** **92.38%**
- **Best Validation Macro F1:** **0.9273**
- **Best Validation Loss:** 0.2554 (Selection Metric: macro_f1)
- **Total Training Duration:** 201.9s (3.37 min)
- **Peak GPU Memory:** 1363.30 MB

---

## 2. Dataset Distribution

- **Training Partition:** 617 samples
- **Validation Partition:** 105 samples

| Class Name | Train Samples | Val Samples | Total Samples | Train Class Weight |
|---|---:|---:|---:|---:|
| `scream` | 30 | 7 | 37 | 2.4331 |
| `glass_breaking` | 77 | 19 | 96 | 0.9480 |
| `gunshot` | 92 | 23 | 115 | 0.7934 |
| `siren` | 127 | 32 | 159 | 0.5747 |
| `other` | 291 | 24 | 315 | 0.2508 |
| **Total** | **617** | **105** | **722** | - |

---

## 3. Comparison Against Frozen AudioSet Baseline

| Metric / Class | Baseline (AudioSet Val N=105) | Combined Model (AudioSet+NIGENS Val N=105) | Delta |
|---|---:|---:|---:|
| **Overall Accuracy** | 89.52% | 92.38% | +2.86% |
| **Macro F1 Score** | 0.8995 | 0.9273 | +0.0278 |
| `scream` F1 | 1.0000 | 1.0000 | +0.0000 |
| `glass_breaking` F1 | 0.7619 | 0.8293 | +0.0674 |
| `gunshot` F1 | 0.9787 | 0.9787 | +0.0000 |
| `siren` F1 | 0.9841 | 0.9841 | +0.0000 |
| `other` F1 | 0.7727 | 0.8444 | +0.0717 |

---

## 4. Best Epoch Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Validation Support |
|---|---:|---:|---:|---:|
| `scream` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `glass_breaking` | 0.7727 | 0.8947 | 0.8293 | 19 |
| `gunshot` | 0.9583 | 1.0000 | 0.9787 | 23 |
| `siren` | 1.0000 | 0.9688 | 0.9841 | 32 |
| `other` | 0.9048 | 0.7917 | 0.8444 | 24 |

---

## 5. Validation Confusion Matrix

Rows indicate Ground Truth, columns indicate Predictions:

| True \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---:|---:|---:|---:|
| `scream` | 7 | 0 | 0 | 0 | 0 |
| `glass_breaking` | 0 | 17 | 1 | 0 | 1 |
| `gunshot` | 0 | 0 | 23 | 0 | 0 |
| `siren` | 0 | 0 | 0 | 31 | 1 |
| `other` | 0 | 5 | 0 | 0 | 19 |

