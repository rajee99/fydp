# Training Experiment Report: PANNs CNN14 (Transfer Learning)

**Execution Timestamp:** 2026-11-02 18:09:32  
**Model Architecture:** `PannsCNN14`  
**Total / Trainable Parameters:** 79,963,653 / 10,245  
**Optimizer:** ADAM  
**Head Learning Rate:** `0.001`  
**Backbone Learning Rate:** `None`  
**Checkpoint Directory:** [`models\experiments\phase2_e4\e4_minus_metal_impacts_seed123`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_minus_metal_impacts_seed123)  
**Best Checkpoint File:** [`best_model.pt`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_minus_metal_impacts_seed123\best_model.pt)  
**Training Manifest:** [`e4_minus_metal_impacts_train.csv`](file:///D:\ml\fydp\MyNet\data\processed\e4_ablations\e4_minus_metal_impacts_train.csv)  
**Validation Manifest:** [`shared_val_split_manifest.csv`](file:///D:\ml\fydp\MyNet\data\processed\shared_val_split_manifest.csv)  

---

## 1. Executive Summary

- **Best Epoch:** 12 (out of 19 epochs completed)
- **Best Validation Accuracy:** **90.48%**
- **Best Validation Macro F1:** **0.9092**
- **Best Validation Loss:** 0.3832 (Selection Metric: macro_f1)
- **Total Training Duration:** 149.4s (2.49 min)
- **Peak GPU Memory:** 1363.30 MB

---

## 2. Dataset Distribution

- **Training Partition:** 606 samples
- **Validation Partition:** 105 samples

| Class Name | Train Samples | Val Samples | Total Samples | Train Class Weight |
|---|---:|---:|---:|---:|
| `scream` | 30 | 7 | 37 | 2.4283 |
| `glass_breaking` | 77 | 19 | 96 | 0.9461 |
| `gunshot` | 92 | 23 | 115 | 0.7918 |
| `siren` | 127 | 32 | 159 | 0.5736 |
| `other` | 280 | 24 | 304 | 0.2602 |
| **Total** | **606** | **105** | **711** | - |

---

## 3. Comparison Against Frozen AudioSet Baseline

| Metric / Class | Baseline (AudioSet Val N=105) | Combined Model (AudioSet+NIGENS Val N=105) | Delta |
|---|---:|---:|---:|
| **Overall Accuracy** | 89.52% | 90.48% | +0.96% |
| **Macro F1 Score** | 0.8995 | 0.9092 | +0.0097 |
| `scream` F1 | 1.0000 | 1.0000 | +0.0000 |
| `glass_breaking` F1 | 0.7619 | 0.8000 | +0.0381 |
| `gunshot` F1 | 0.9787 | 1.0000 | +0.0213 |
| `siren` F1 | 0.9841 | 0.9841 | +0.0000 |
| `other` F1 | 0.7727 | 0.7619 | -0.0108 |

---

## 4. Best Epoch Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Validation Support |
|---|---:|---:|---:|---:|
| `scream` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `glass_breaking` | 0.6923 | 0.9474 | 0.8000 | 19 |
| `gunshot` | 1.0000 | 1.0000 | 1.0000 | 23 |
| `siren` | 1.0000 | 0.9688 | 0.9841 | 32 |
| `other` | 0.8889 | 0.6667 | 0.7619 | 24 |

---

## 5. Validation Confusion Matrix

Rows indicate Ground Truth, columns indicate Predictions:

| True \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---:|---:|---:|---:|
| `scream` | 7 | 0 | 0 | 0 | 0 |
| `glass_breaking` | 0 | 18 | 0 | 0 | 1 |
| `gunshot` | 0 | 0 | 23 | 0 | 0 |
| `siren` | 0 | 0 | 0 | 31 | 1 |
| `other` | 0 | 8 | 0 | 0 | 16 |

