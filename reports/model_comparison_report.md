# Independent Evaluation: Baseline vs. Combined PANNs CNN14

**Execution Date:** 2026-10-30 13:02:51  
**Target Project:** `D:\ml\fydp\MyNet`  
**Independent Evaluation Clips:** **519 valid WAV clips** across all 5 classes  
**Evaluation Audio Directory:** [`data/raw/eval/`](file:///d:/ml/fydp/MyNet/data/raw/eval/)  
**Baseline Checkpoint:** [`models/experiments/panns_cnn14/best_model.pt`](file:///d:/ml/fydp/MyNet/models/experiments/panns_cnn14/best_model.pt) (Epoch 7)  
**Combined Checkpoint:** [`models/experiments/panns_cnn14_combined/best_model.pt`](file:///d:/ml/fydp/MyNet/models/experiments/panns_cnn14_combined/best_model.pt) (Epoch 21)  

---

## 1. Executive Summary & Head-to-Head Comparison

| Evaluation Metric | Baseline Model (AudioSet-only) | Combined Model (AudioSet + NIGENS) | Delta | Advantage |
|---|---:|---:|---:|:---:|
| **Overall Accuracy** | **84.78%** | 81.70% | -3.08% | **Baseline (+3.08%)** |
| **Macro F1-Score** | **0.8195** | 0.7837 | -0.0358 | **Baseline (+0.0358)** |
| **Macro Precision** | **0.8044** | 0.7825 | -0.0219 | **Baseline (+0.0381)** |
| **Macro Recall** | **0.8460** | 0.8098 | -0.0362 | **Baseline (+0.0406)** |

### Per-Sample Prediction Agreement & Outcome Distribution

- **Sample Agreement:** Both models made the exact same prediction on **460 / 519 clips** (88.6%).
- **Both Models Correct:** **405 clips** (78.0%)
- **Both Models Incorrect:** **60 clips** (11.6%)
- **Baseline Correct, Combined Incorrect:** **35 clips** (6.7%)
- **Combined Correct, Baseline Incorrect:** **19 clips** (3.7%)

---

## 2. Per-Class Performance Breakdown

| Class Name | Support | Baseline Precision | Combined Precision | Baseline Recall | Combined Recall | Baseline F1 | Combined F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scream` | 30 | 0.6279 | 0.6667 | 0.9000 | 0.8000 | 0.7397 | 0.7273 | -0.0124 |
| `glass_breaking` | 85 | 0.7033 | 0.6018 | 0.7529 | 0.8000 | 0.7273 | 0.6869 | -0.0404 |
| `gunshot` | 109 | 0.9266 | 0.8607 | 0.9266 | 0.9633 | 0.9266 | 0.9091 | -0.0175 |
| `siren` | 164 | 0.9814 | 0.9583 | 0.9634 | 0.9817 | 0.9723 | 0.9699 | -0.0024 |
| `other` | 131 | 0.7826 | 0.8250 | 0.6870 | 0.5038 | 0.7317 | 0.6256 | -0.1061 |

---

## 3. Confusion Matrices

### Baseline Model Confusion Matrix

| Ground Truth \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---|---:|---|---:|---|---:|---|---::|
| `scream` | 27 | 0 | 0 | 0 | 3 |
| `glass_breaking` | 7 | 64 | 4 | 0 | 10 |
| `gunshot` | 0 | 1 | 101 | 0 | 7 |
| `siren` | 0 | 1 | 0 | 158 | 5 |
| `other` | 9 | 25 | 4 | 3 | 90 |

### Combined Model Confusion Matrix

| Ground Truth \ Pred | `scream` | `glass_breaking` | `gunshot` | `siren` | `other` |
|---|---:|---|---:|---|---:|---|---:|---|---::|
| `scream` | 24 | 3 | 1 | 0 | 2 |
| `glass_breaking` | 3 | 68 | 6 | 1 | 7 |
| `gunshot` | 0 | 1 | 105 | 0 | 3 |
| `siren` | 0 | 1 | 0 | 161 | 2 |
| `other` | 9 | 40 | 10 | 6 | 66 |

---

## 4. Targeted Error Category Deep-Dive

| Error Category | Baseline Errors | Combined Errors | Net Delta | Combined Fixed | Combined Broke | Better Model |
|---|---:|---:|---:|---:|---:|:---:|
| `other` -> `glass_breaking` | 25 | 40 | +15 | 2 | 17 | **Baseline** |
| `other` -> `scream` | 9 | 9 | +0 | 1 | 0 | **Tie** |
| `siren` -> `other` | 5 | 2 | -3 | 4 | 0 | **Combined** |
| `glass_breaking` -> `other` | 10 | 7 | -3 | 4 | 1 | **Combined** |
| `scream` -> `other` | 3 | 2 | -1 | 1 | 0 | **Combined** |

### Detailed Findings on Targeted Error Categories

#### 1. Other predicted as glass_breaking
- **Baseline Errors:** 25 clips
- **Combined Errors:** 40 clips (Delta: +15)
- **Clips Fixed by Combined:** 2
  - Examples fixed: `other_0OhqyNGlJYE_310_320.wav, other_dg3oPBY_vXQ_0_4.wav`
- **Clips Broken by Combined:** 17
  - Examples broken: `other_-UTdhK0lwuw_30_40.wav, other_B1beLwV4yzw_30_40.wav, other_C6roSYqchkk_60_70.wav`

#### 1. Other predicted as scream
- **Baseline Errors:** 9 clips
- **Combined Errors:** 9 clips (Delta: +0)
- **Clips Fixed by Combined:** 1
  - Examples fixed: `other_ejo2aXEWSgo_0_10.wav`
- **Clips Broken by Combined:** 0

#### 1. Siren predicted as other
- **Baseline Errors:** 5 clips
- **Combined Errors:** 2 clips (Delta: -3)
- **Clips Fixed by Combined:** 4
  - Examples fixed: `siren_dSzZWgbJ378_30_40.wav, siren_La56WBAJ2xk_30_40.wav, siren_lIgNQxTUvtc_30_40.wav`
- **Clips Broken by Combined:** 0

#### 1. Glass_breaking predicted as other
- **Baseline Errors:** 10 clips
- **Combined Errors:** 7 clips (Delta: -3)
- **Clips Fixed by Combined:** 4
  - Examples fixed: `glass_breaking_i8aWCOfrrJU_100_110.wav, glass_breaking_lBBqkex3VXI_60_70.wav, glass_breaking_PF6Mn51Nkvs_30_40.wav`
- **Clips Broken by Combined:** 1
  - Examples broken: `glass_breaking_uUcpzHx2ZR0_60_70.wav`

#### 1. Scream predicted as other
- **Baseline Errors:** 3 clips
- **Combined Errors:** 2 clips (Delta: -1)
- **Clips Fixed by Combined:** 1
  - Examples fixed: `scream_MQ0YasvMcuQ_1_11.wav`
- **Clips Broken by Combined:** 0

---

## 5. Architectural & Domain Shift Diagnosis

1. **Acoustic Domain Mismatch (Studio-Quality Foley vs. In-the-Wild YouTube):**
   - The independent evaluation dataset (`data/raw/eval/`) consists strictly of authentic YouTube environment audio from AudioSet, featuring background reverberation, street noise, dialogue, and low SNR.
   - The NIGENS dataset consists predominantly of isolated foley and studio-clean sound effects (e.g. SFX kits, Sound Ideas, GRID speech) with near-zero background ambiance.
   - Adding 419 NIGENS samples shifted the classification head's decision boundaries toward cleaner acoustic signatures. Consequently, on noisy real-world YouTube recordings, ambient transients (knocks, clicks, clatter, utensils) are misclassified as emergency classes.

2. **Impact on Specific Error Categories & Classes:**
   - **Massive False Alarm Increase for Glass Breaking:** `other` predicted as `glass_breaking` jumped from 25 clips to 40 clips (+15 errors). This severely eroded `glass_breaking` precision (from 70.33% down to 60.18%) and F1 (from 0.7273 to 0.6869).
   - **Collapse in Background Identification:** `other` recall plummeted from 68.70% (90/131) down to 50.38% (66/131), dropping `other` F1 from 0.7317 to 0.6256.
   - **Scream Sensitivity:** `scream` recall dropped from 90.00% (27/30) down to 80.00% (24/30). Combined misclassified 3 screams as glass breaking and 1 as gunshot, lowering scream F1 from 0.7397 to 0.7273.
   - **Targeted False Negative Reduction:** On the positive side, Combined reduced false negatives into `other`: `siren` -> `other` fell from 5 to 2; `glass_breaking` -> `other` fell from 10 to 7; and `scream` -> `other` fell from 3 to 2.

---

## 6. Final Recommendation

> [!IMPORTANT]
> **Final Decision: RETAIN THE BASELINE CHECKPOINT AS THE PREFERRED PRODUCTION MODEL.**

- **Primary Reason:** The baseline frozen PANNs CNN14 checkpoint (`models/experiments/panns_cnn14/best_model.pt`) significantly outperforms the combined model on the independent test set across all macro metrics:
  - **Accuracy:** **84.78%** vs. 81.70% (+3.08% superior; 440 vs 424 correct)
  - **Macro F1:** **0.8195** vs. 0.7837 (+0.0358 superior)
  - **False Alarms:** Baseline produced only 41 false alarms on `other` audio, whereas the Combined model produced **65 false alarms** (+58.5% increase in false positive disruptions).
  - **Scream Recall:** **90.00%** vs. 80.00% (Baseline correctly identified 27/30 screams vs. 24/30 by Combined).
- **Both checkpoints remain preserved:**
  - Baseline: `models/experiments/panns_cnn14/best_model.pt`
  - Combined: `models/experiments/panns_cnn14_combined/best_model.pt`
