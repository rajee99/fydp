# Training Experiment Report: PANNs CNN14 (Transfer Learning)

**Execution Timestamp:** 2026-11-02 18:34:37  
**Model Architecture:** `PannsCNN14`  
**Total / Trainable Parameters:** 79,963,653 / 10,245  
**Optimizer:** ADAM  
**Head Learning Rate:** `0.001`  
**Backbone Learning Rate:** `None`  
**Checkpoint Directory:** [`models\experiments\phase2_e4\e4_minus_car_alarms_detectors_seed123`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_minus_car_alarms_detectors_seed123)  
**Best Checkpoint File:** [`best_model.pt`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_minus_car_alarms_detectors_seed123\best_model.pt)  
**Training Manifest:** [`e4_minus_car_alarms_detectors_train.csv`](file:///D:\ml\fydp\MyNet\data\processed\e4_ablations\e4_minus_car_alarms_detectors_train.csv)  
**Validation Manifest:** [`shared_val_split_manifest.csv`](file:///D:\ml\fydp\MyNet\data\processed\shared_val_split_manifest.csv)  

---

## 1. Executive Summary

- **Best Epoch:** 16 (out of 23 epochs completed)
- **Best Validation Accuracy:** **91.43%**
- **Best Validation Macro F1:** **0.9187**
- **Best Validation Loss:** 0.2935 (Selection Metric: macro_f1)
- **Total Training Duration:** 165.5s (2.76 min)
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
| **Overall Accuracy** | 89.52% | 91.43% | +1.91% |
| **Macro F1 Score** | 0.8995 | 0.9187 | +0.0192 |
| `scream` F1 | 1.0000 | 1.0000 | +0.0000 |
| `glass_breaking` F1 | 0.7619 | 0.8095 | +0.0476 |
| `gunshot` F1 | 0.9787 | 1.0000 | +0.0213 |
| `siren` F1 | 0.9841 | 0.9841 | +0.0000 |
| `other` F1 | 0.7727 | 0.8000 | +0.0273 |

---

## 4. Best Epoch Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Validation Support |
|---|---:|---:|---:|---:|
| `scream` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `glass_breaking` | 0.7391 | 0.8947 | 0.8095 | 19 |
| `gunshot` | 1.0000 | 1.0000 | 1.0000 | 23 |
| `siren` | 1.0000 | 0.9688 | 0.9841 | 32 |
| `other` | 0.8571 | 0.7500 | 0.8000 | 24 |

---

## 5. Validation Confusion Matrix

Rows indicate Ground Truth, columns indicate Predictions:

| True \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---:|---:|---:|---:|
| `scream` | 7 | 0 | 0 | 0 | 0 |
| `glass_breaking` | 0 | 17 | 0 | 0 | 2 |
| `gunshot` | 0 | 0 | 23 | 0 | 0 |
| `siren` | 0 | 0 | 0 | 31 | 1 |
| `other` | 0 | 6 | 0 | 0 | 18 |

