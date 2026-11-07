# Training Experiment Report: PANNs CNN14 (Transfer Learning)

**Execution Timestamp:** 2026-11-02 18:54:08  
**Model Architecture:** `PannsCNN14`  
**Total / Trainable Parameters:** 79,963,653 / 10,245  
**Optimizer:** ADAM  
**Head Learning Rate:** `0.001`  
**Backbone Learning Rate:** `None`  
**Checkpoint Directory:** [`models\experiments\phase2_e4\e4_minus_whistles_high_pitch_seed123`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_minus_whistles_high_pitch_seed123)  
**Best Checkpoint File:** [`best_model.pt`](file:///D:\ml\fydp\MyNet\models\experiments\phase2_e4\e4_minus_whistles_high_pitch_seed123\best_model.pt)  
**Training Manifest:** [`e4_minus_whistles_high_pitch_train.csv`](file:///D:\ml\fydp\MyNet\data\processed\e4_ablations\e4_minus_whistles_high_pitch_train.csv)  
**Validation Manifest:** [`shared_val_split_manifest.csv`](file:///D:\ml\fydp\MyNet\data\processed\shared_val_split_manifest.csv)  

---

## 1. Executive Summary

- **Best Epoch:** 1 (out of 8 epochs completed)
- **Best Validation Accuracy:** **89.52%**
- **Best Validation Macro F1:** **0.9009**
- **Best Validation Loss:** 0.8939 (Selection Metric: macro_f1)
- **Total Training Duration:** 38.9s (0.65 min)
- **Peak GPU Memory:** 1360.99 MB

---

## 2. Dataset Distribution

- **Training Partition:** 627 samples
- **Validation Partition:** 105 samples

| Class Name | Train Samples | Val Samples | Total Samples | Train Class Weight |
|---|---:|---:|---:|---:|
| `scream` | 30 | 7 | 37 | 2.4371 |
| `glass_breaking` | 77 | 19 | 96 | 0.9495 |
| `gunshot` | 92 | 23 | 115 | 0.7947 |
| `siren` | 127 | 32 | 159 | 0.5757 |
| `other` | 301 | 24 | 325 | 0.2429 |
| **Total** | **627** | **105** | **732** | - |

---

## 3. Comparison Against Frozen AudioSet Baseline

| Metric / Class | Baseline (AudioSet Val N=105) | Combined Model (AudioSet+NIGENS Val N=105) | Delta |
|---|---:|---:|---:|
| **Overall Accuracy** | 89.52% | 89.52% | +0.00% |
| **Macro F1 Score** | 0.8995 | 0.9009 | +0.0014 |
| `scream` F1 | 1.0000 | 1.0000 | +0.0000 |
| `glass_breaking` F1 | 0.7619 | 0.7568 | -0.0051 |
| `gunshot` F1 | 0.9787 | 0.9565 | -0.0222 |
| `siren` F1 | 0.9841 | 0.9677 | -0.0164 |
| `other` F1 | 0.7727 | 0.8235 | +0.0508 |

---

## 4. Best Epoch Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Validation Support |
|---|---:|---:|---:|---:|
| `scream` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `glass_breaking` | 0.7778 | 0.7368 | 0.7568 | 19 |
| `gunshot` | 0.9565 | 0.9565 | 0.9565 | 23 |
| `siren` | 1.0000 | 0.9375 | 0.9677 | 32 |
| `other` | 0.7778 | 0.8750 | 0.8235 | 24 |

---

## 5. Validation Confusion Matrix

Rows indicate Ground Truth, columns indicate Predictions:

| True \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---:|---:|---:|---:|
| `scream` | 7 | 0 | 0 | 0 | 0 |
| `glass_breaking` | 0 | 14 | 1 | 0 | 4 |
| `gunshot` | 0 | 1 | 22 | 0 | 0 |
| `siren` | 0 | 0 | 0 | 30 | 2 |
| `other` | 0 | 3 | 0 | 0 | 21 |

