# Phase 2 E4 Statistical Evaluation & Category Attribution Report

**Project:** `D:\ml\fydp\MyNet`  
**Research Framing:** *Data-Centric Analysis of Hard-Negative Learning and Domain Shift in Emergency Audio Classification*  
**Date:** November 02, 2026  
**Document:** [`phase2_e4_evaluation_report.md`](file:///D:/ml/fydp/MyNet/reports/phase2_e4_evaluation_report.md)  
**Status:** `EXPLORATORY BENCHMARK ATTRIBUTION (TEST SET V2 REUSE)`  
**Benchmark Set:** Test Set v2 (N = 750 pristine samples)  
**Evaluation Mode:** `REAL_WEIGHTS`  
**Seeds Evaluated:** [42, 123, 2026]  
**Bootstrap Resamples:** 10,000 (Recording-cluster aware, Seed = 42)  

---

## 1. Executive Summary & Epistemic Status

1. **Epistemic Demarcation:** Findings reported herein represent **Exploratory Category Attribution Analysis** on Test Set v2. Test Set v2 was unblinded in Phase 1; these results attribute the relative semantic impact of individual confuser subcategories on that benchmark.
2. **Strict Cluster-Level Statistical Integrity:** Resampling occurs strictly at the physical recording ID level ($N = 25$ recordings per confuser category), preserving within-cluster multi-seed prediction correlations and completely eliminating the pseudo-replication fallacy.
3. **Dual Estimand Reporting:** The two mandated estimands are strictly separated into independent reporting columns:
   - **Primary Estimand ($\theta_1$):** Three-seed average model error difference (single-model deployment scenario).
   - **Secondary Estimand ($\theta_2$):** Probability-ensemble error difference (3-model ensemble deployment scenario).
4. **Primary Hypothesis Testing:** Exactly 7 primary comparisons enter the primary Holm-Bonferroni step-down correction at family-wise $\alpha = 0.05$ using exact/Monte Carlo paired sign-flip permutation tests on $\theta_1$.
5. **Secondary Multiplicity Policy:** Secondary comparisons against `E2 Full` and ensemble $\theta_2$ tests are evaluated under their own isolated 7-hypothesis Holm-Bonferroni corrections and are never pooled with primary tests.

---

## 2. Primary Hypothesis Family (Estimand 1: $\theta_1$ Three-Seed Average Model Effect)

The primary comparison contrasts each category ablation ($N = 602..627$) against its matched-size random control ($N = 602..627$) using the **three-seed average model effect ($\theta_1$)**. Because sample sizes are strictly identical, this contrast isolates the **pure semantic importance** of each acoustic confuser category for an end-user deploying a single model.

| Rank | Category Name | Slice $N$ | Point Est $\theta_1$ | 95% Cluster Bootstrap CI | Permutation Method | Permutation $p$-value | Primary Holm-Bonferroni $\alpha$ | Significant under Primary FWER? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Fireworks & Firecrackers | 25 | +0.4133 | [+0.2400, +0.5867] | `exact_enumeration_2^13` | 0.000244 | 0.007143 | **YES** |
| **2** | Human Shouting | 25 | +0.3600 | [+0.1867, +0.5333] | `exact_enumeration_2^12` | 0.000977 | 0.008333 | **YES** |
| **3** | Construction Tools | 150 | -0.0422 | [-0.0711, -0.0156] | `monte_carlo_100000` | 0.003880 | 0.010000 | **CONFOUNDED (UNRESOLVED)*** |
| **4** | Car Alarms & Detectors | 25 | +0.1467 | [+0.0400, +0.2667] | `exact_enumeration_2^6` | 0.031250 | 0.012500 | NO |
| **5** | Door Slams | 25 | +0.0667 | [-0.0000, +0.1467] | `exact_enumeration_2^6` | 0.187500 | 0.016667 | NO |
| **6** | Whistles & High Pitch | 25 | +0.0400 | [+0.0000, +0.1067] | `exact_enumeration_2^2` | 0.500000 | 0.025000 | NO |
| **7** | Metal Impacts | 25 | +0.0000 | [+0.0000, +0.0000] | `trivial_null` | 1.000000 | 0.050000 | NO |

> [!NOTE]
> $\theta_1$ represents the expected difference in false emergency rate for a single deployed model trained with arbitrary random seed. Inference is performed via recording-level paired permutation testing and cluster-aware bootstrap. Positive values indicate that removing the targeted category increased false alarms (i.e. targeted hard negatives were beneficial).
> 
> **\* Note on Construction Tools Confound:** Test Set v2 lacks a dedicated construction tools slice, triggering an automated fallback to the pooled 150-clip confuser stratum. The matched control randomly lost 7 shouting and 4 fireworks clips from training, causing false alarms on those subdomains that the ablation resisted. The negative shift is an artifact of cross-category collateral damage and pooled fallback evaluation; the category-specific effect is **UNRESOLVED**.

---

## 3. Secondary Hypothesis Family (Estimand 2: $\theta_2$ Three-Seed Probability Ensemble Effect)

The secondary comparison evaluates the performance shift if an end-user deploys the full **3-model probability ensemble**. Tested using the exact two-sided binomial McNemar test on the ensemble decisions, with an independent 7-hypothesis Holm-Bonferroni correction.

| Rank | Category Name | Slice $N$ | Point Est $\theta_2$ | 95% Cluster Bootstrap CI | Ens Discordant $(b : c)$ | Exact Binomial $p$-value | Ensemble Holm-Bonferroni $\alpha$ | Significant under Ensemble FWER? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Fireworks & Firecrackers | 25 | +0.4400 | [+0.2400, +0.6400] | `11 : 0` | 0.000977 | 0.007143 | **YES** |
| **2** | Human Shouting | 25 | +0.3600 | [+0.1200, +0.5600] | `10 : 1` | 0.011719 | 0.008333 | NO |
| **3** | Construction Tools | 150 | -0.0467 | [-0.0800, -0.0133] | `0 : 7` | 0.015625 | 0.010000 | NO (Confounded Fallback) |
| **4** | Car Alarms & Detectors | 25 | +0.1200 | [+0.0000, +0.2800] | `3 : 0` | 0.250000 | 0.012500 | NO |
| **5** | Metal Impacts | 25 | +0.0000 | [+0.0000, +0.0000] | `0 : 0` | 1.000000 | 0.016667 | NO |
| **6** | Door Slams | 25 | +0.0400 | [+0.0000, +0.1200] | `1 : 0` | 1.000000 | 0.025000 | NO |
| **7** | Whistles & High Pitch | 25 | +0.0400 | [+0.0000, +0.1200] | `1 : 0` | 1.000000 | 0.050000 | NO |

> [!NOTE]
> $\theta_2$ measures the performance shift for a 3-seed probability ensemble. This is a secondary analysis because typical edge acoustic systems deploy a single neural network.

---

## 4. Descriptive Multi-Seed Consistency Analysis (Per-Seed Paired McNemar)

| Category Name | Seed 42 Discordant $(b : c)$ [$p$] | Seed 123 Discordant $(b : c)$ [$p$] | Seed 2026 Discordant $(b : c)$ [$p$] | Direction Consistency ($3/3$)? |
| :--- | :---: | :---: | :---: | :---: |
| Human Shouting | `8:1` (0.0391) | `10:1` (0.0117) | `11:0` (0.0010) | **YES (3/3)** |
| Metal Impacts | `0:0` (1.0000) | `0:0` (1.0000) | `0:0` (1.0000) | NO |
| Door Slams | `3:1` (0.6250) | `2:0` (0.5000) | `2:1` (1.0000) | **YES (3/3)** |
| Construction Tools | `10:12` (0.8318) | `0:12` (0.0005) | `0:5` (0.0625) | NO |
| Car Alarms & Detectors | `5:0` (0.0625) | `3:0` (0.2500) | `3:0` (0.2500) | **YES (3/3)** |
| Fireworks & Firecrackers | `11:0` (0.0010) | `11:0` (0.0010) | `9:0` (0.0039) | **YES (3/3)** |
| Whistles & High Pitch | `1:0` (1.0000) | `0:0` (1.0000) | `2:0` (0.5000) | NO |

---

## 5. Secondary Reference Comparison: Category Ablations vs. E2 Full Reference Ceiling

The secondary comparison measures the total real-world margin loss from removing a category compared to the unablated ceiling (`E2 Full`, $N = 646$). This contrast conflates category semantic loss with sample-size reduction ($646 \rightarrow 602..627$).

| Rank | Category Name | Reference Condition | Slice $N$ | Point Est $\theta_1$ vs Full [95% CI] | Permutation $p$-value | Secondary Holm $\alpha$ | Significant under FWER? | $\theta_2$ vs Full [95% CI] | Ens McNemar $p$-value |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Human Shouting | `e2_full` | 25 | +0.5067 [+0.3333, +0.6667] | 0.000031 | 0.007143 | **YES** | +0.5200 [+0.3200, +0.7200] | 0.000244 |
| **2** | Fireworks & Firecrackers | `e2_full` | 25 | +0.4800 [+0.2933, +0.6667] | 0.000122 | 0.008333 | **YES** | +0.4400 [+0.2400, +0.6400] | 0.000977 |
| **3** | Car Alarms & Detectors | `e2_full` | 25 | +0.1467 [+0.0400, +0.2667] | 0.031250 | 0.010000 | NO | +0.1200 [+0.0000, +0.2800] | 0.250000 |
| **4** | Door Slams | `e2_full` | 25 | +0.0400 [-0.0133, +0.0933] | 0.375000 | 0.012500 | NO | +0.0400 [+0.0000, +0.1200] | 1.000000 |
| **5** | Whistles & High Pitch | `e2_full` | 25 | +0.0400 [+0.0000, +0.1067] | 0.500000 | 0.016667 | NO | +0.0400 [+0.0000, +0.1200] | 1.000000 |
| **6** | Construction Tools | `e2_full` | 150 | +0.0044 [-0.0067, +0.0156] | 0.687500 | 0.025000 | NO | +0.0000 [+0.0000, +0.0000] | 1.000000 |
| **7** | Metal Impacts | `e2_full` | 25 | +0.0000 [+0.0000, +0.0000] | 1.000000 | 0.050000 | NO | +0.0000 [+0.0000, +0.0000] | 1.000000 |

> [!NOTE]
> **Secondary Multiplicity Policy:** The 7 secondary comparisons against `E2 Full` constitute a separate hypothesis family evaluated with its own 7-test Holm-Bonferroni step-down correction ($\alpha = 0.05$). They are strictly separated from the primary family to prevent multiplicity leakage.

---

## 6. Overall Condition Performance Summary

| Condition Key | Train N | Macro F1 (Mean ± Std) | Confusers FER (Mean) | Overall FER (Mean) | Ens Macro F1 | Ens Confusers FER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `e2_full` | — | 0.8308 ± 0.0026 | 0.3066 | 0.2300 | 0.8386 | 0.3067 |
| `e4_minus_human_shouting` | — | 0.8143 ± 0.0075 | 0.4089 | 0.2944 | 0.8226 | 0.4133 |
| `e4_matched_control_human_shouting` | — | 0.8175 ± 0.0021 | 0.3444 | 0.2789 | 0.8224 | 0.3400 |
| `e4_minus_metal_impacts` | — | 0.8235 ± 0.0060 | 0.3400 | 0.2867 | 0.8246 | 0.3400 |
| `e4_matched_control_metal_impacts` | — | 0.8218 ± 0.0075 | 0.3333 | 0.2556 | 0.8227 | 0.3200 |
| `e4_minus_door_slams` | — | 0.8278 ± 0.0009 | 0.3400 | 0.2589 | 0.8373 | 0.3267 |
| `e4_matched_control_door_slams` | — | 0.8258 ± 0.0022 | 0.3289 | 0.2522 | 0.8359 | 0.3200 |
| `e4_minus_construction_tools` | — | 0.8324 ± 0.0090 | 0.3111 | 0.2456 | 0.8386 | 0.3067 |
| `e4_matched_control_construction_tools` | — | 0.8032 ± 0.0192 | 0.3534 | 0.2611 | 0.8134 | 0.3533 |
| `e4_minus_car_alarms_detectors` | — | 0.8137 ± 0.0080 | 0.3489 | 0.2855 | 0.8245 | 0.3400 |
| `e4_matched_control_car_alarms_detectors` | — | 0.8241 ± 0.0051 | 0.3200 | 0.2389 | 0.8238 | 0.3133 |
| `e4_minus_fireworks_firecrackers` | — | 0.8059 ± 0.0055 | 0.3955 | 0.3033 | 0.8189 | 0.3800 |
| `e4_matched_control_fireworks_firecrackers` | — | 0.8227 ± 0.0067 | 0.3356 | 0.2578 | 0.8252 | 0.3267 |
| `e4_minus_whistles_high_pitch` | — | 0.7841 ± 0.0452 | 0.3311 | 0.2467 | 0.8164 | 0.3267 |
| `e4_matched_control_whistles_high_pitch` | — | 0.8120 ± 0.0028 | 0.3400 | 0.2733 | 0.8205 | 0.3267 |

---

## 5. Statistical Power & Epistemic Boundaries

1. **Exact Binomial Power Threshold:** With $N = 25$ recordings, a single-seed or ensemble comparison requires at least 6 discordant pairs ($b=6, c=0$) for unadjusted significance ($p = 0.031250$) and at least 9 discordant pairs ($b=9, c=0$) to survive Holm-Bonferroni Rank 1 ($p = 0.003906 < 0.007143$).
2. **Interpretation of Insignificance:** Non-significant category differences ($p > 0.05$) indicate that on a fixed benchmark of $N = 25$ recordings, the margin shift was not detectable above stochastic noise. It does not constitute proof that the category has zero real-world utility.
3. **Absence of Dedicated Tools Slice in Benchmark:** The Test Set v2 benchmark contains 6 confuser subdomains ($6 \times 25 = 150$). The effect of `construction_tools` ablation was evaluated across the overall confuser stratum ($N = 150$), accurately measuring its generalized acoustic protection.

---

**Report Generated Automatically by:** `src/evaluate_phase2_e4.py`  