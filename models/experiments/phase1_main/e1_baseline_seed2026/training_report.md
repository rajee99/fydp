# Training Experiment Report: PANNs CNN14 (Transfer Learning)

**Execution Timestamp:** 2026-10-30 22:18:49  
**Model Architecture:** `PannsCNN14`  
**Total / Trainable Parameters:** 79,963,653 / 10,245  
**Optimizer:** ADAM  
**Head Learning Rate:** `0.001`  
**Backbone Learning Rate:** `None`  
**Checkpoint Directory:** [`models\experiments\phase1_main\e1_baseline_seed2026`](file:///D:\ml\fydp\MyNet\models\experiments\phase1_main\e1_baseline_seed2026)  
**Best Checkpoint File:** [`best_model.pt`](file:///D:\ml\fydp\MyNet\models\experiments\phase1_main\e1_baseline_seed2026\best_model.pt)  
**Training Manifest:** [`e1_train_split_manifest.csv`](file:///D:\ml\fydp\MyNet\data\processed\e1_train_split_manifest.csv)  
**Validation Manifest:** [`shared_val_split_manifest.csv`](file:///D:\ml\fydp\MyNet\data\processed\shared_val_split_manifest.csv)  

---

## 1. Executive Summary

- **Best Epoch:** 11 (out of 18 epochs completed)
- **Best Validation Accuracy:** **89.52%**
- **Best Validation Macro F1:** **0.8995**
- **Best Validation Loss:** 0.3683 (Selection Metric: macro_f1)
- **Total Training Duration:** 527.9s (8.80 min)
- **Peak GPU Memory:** 1358.41 MB

---

## 2. Dataset Distribution

- **Training Partition:** 422 samples
- **Validation Partition:** 105 samples

| Class Name | Train Samples | Val Samples | Total Samples | Train Class Weight |
|---|---:|---:|---:|---:|
| `scream` | 30 | 7 | 37 | 2.2081 |
| `glass_breaking` | 77 | 19 | 96 | 0.8603 |
| `gunshot` | 92 | 23 | 115 | 0.7200 |
| `siren` | 127 | 32 | 159 | 0.5216 |
| `other` | 96 | 24 | 120 | 0.6900 |
| **Total** | **422** | **105** | **527** | - |

---

## 3. Comparison Against Frozen AudioSet Baseline

| Metric / Class | Baseline (AudioSet Val N=105) | Combined Model (AudioSet+NIGENS Val N=105) | Delta |
|---|---:|---:|---:|
| **Overall Accuracy** | 89.52% | 89.52% | +0.00% |
| **Macro F1 Score** | 0.8995 | 0.8995 | +0.0000 |
| `scream` F1 | 1.0000 | 1.0000 | +0.0000 |
| `glass_breaking` F1 | 0.7619 | 0.7619 | +0.0000 |
| `gunshot` F1 | 0.9787 | 0.9787 | +0.0000 |
| `siren` F1 | 0.9841 | 0.9841 | +0.0000 |
| `other` F1 | 0.7727 | 0.7727 | +0.0000 |

---

## 4. Best Epoch Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Validation Support |
|---|---:|---:|---:|---:|
| `scream` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `glass_breaking` | 0.6957 | 0.8421 | 0.7619 | 19 |
| `gunshot` | 0.9583 | 1.0000 | 0.9787 | 23 |
| `siren` | 1.0000 | 0.9688 | 0.9841 | 32 |
| `other` | 0.8500 | 0.7083 | 0.7727 | 24 |

---

## 5. Validation Confusion Matrix

Rows indicate Ground Truth, columns indicate Predictions:

| True \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---:|---:|---:|---:|
| `scream` | 7 | 0 | 0 | 0 | 0 |
| `glass_breaking` | 0 | 16 | 1 | 0 | 2 |
| `gunshot` | 0 | 0 | 23 | 0 | 0 |
| `siren` | 0 | 0 | 0 | 31 | 1 |
| `other` | 0 | 7 | 0 | 0 | 17 |

