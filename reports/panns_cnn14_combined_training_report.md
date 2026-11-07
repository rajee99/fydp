# Training Experiment Report: PANNs CNN14 (Combined AudioSet + NIGENS Transfer Learning)

**Execution Timestamp:** 2026-10-30 12:56:02  
**Model Architecture:** `PannsCNN14`  
**Total / Trainable Parameters:** 79,963,653 / 10,245  
**Optimizer:** ADAM  
**Head Learning Rate:** `0.0001`  
**Backbone Learning Rate:** `None`  
**Checkpoint Directory:** [`models\experiments\panns_cnn14_combined`](file:///models\experiments\panns_cnn14_combined)  
**Best Checkpoint File:** [`best_model.pt`](file:///models\experiments\panns_cnn14_combined\best_model.pt)  
**Training Manifest:** [`combined_train_split_manifest.csv`](file:///data\processed\combined_train_split_manifest.csv)  
**Validation Manifest:** [`combined_val_split_manifest.csv`](file:///data\processed\combined_val_split_manifest.csv)  

---

## 1. Executive Summary

- **Best Epoch:** 21 (out of 28 epochs completed)
- **Best Validation Accuracy:** **87.89%**
- **Best Validation Macro F1:** **0.8881**
- **Best Validation Loss:** 0.6697 (Selection Metric: macro_f1)
- **Total Training Duration:** 588.9s (9.81 min)
- **Peak GPU Memory:** 1356.18 MB

---

## 2. Dataset Distribution

- **Training Partition:** 756 samples
- **Validation Partition:** 190 samples

| Class Name | Train Samples | Val Samples | Total Samples | Train Class Weight |
|---|---:|---:|---:|---:|
| `scream` | 70 | 18 | 88 | 1.4936 |
| `glass_breaking` | 82 | 21 | 103 | 1.2751 |
| `gunshot` | 92 | 23 | 115 | 1.1365 |
| `siren` | 127 | 32 | 159 | 0.8233 |
| `other` | 385 | 96 | 481 | 0.2716 |
| **Total** | **756** | **190** | **946** | - |

---

## 3. Comparison Against Frozen AudioSet Baseline

| Metric / Class | Baseline (AudioSet Val N=105) | Combined Model (AudioSet+NIGENS Val N=190) | Delta |
|---|---:|---:|---:|
| **Overall Accuracy** | 89.52% | 87.89% | -1.63% |
| **Macro F1 Score** | 0.8995 | 0.8881 | -0.0114 |
| `scream` F1 | 1.0000 | 0.9474 | -0.0526 |
| `glass_breaking` F1 | 0.7619 | 0.7692 | +0.0073 |
| `gunshot` F1 | 0.9787 | 0.9787 | +0.0000 |
| `siren` F1 | 0.9841 | 0.8732 | -0.1109 |
| `other` F1 | 0.7727 | 0.8721 | +0.0994 |

---

## 4. Best Epoch Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Validation Support |
|---|---:|---:|---:|---:|
| `scream` | 0.9000 | 1.0000 | 0.9474 | 18 |
| `glass_breaking` | 0.6452 | 0.9524 | 0.7692 | 21 |
| `gunshot` | 0.9583 | 1.0000 | 0.9787 | 23 |
| `siren` | 0.7949 | 0.9688 | 0.8732 | 32 |
| `other` | 0.9868 | 0.7812 | 0.8721 | 96 |

---

## 5. Validation Confusion Matrix

Rows indicate Ground Truth, columns indicate Predictions:

| True \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---:|---:|---:|---:|
| `scream` | 18 | 0 | 0 | 0 | 0 |
| `glass_breaking` | 0 | 20 | 1 | 0 | 0 |
| `gunshot` | 0 | 0 | 23 | 0 | 0 |
| `siren` | 0 | 0 | 0 | 31 | 1 |
| `other` | 2 | 11 | 0 | 8 | 75 |

