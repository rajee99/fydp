# Phase 2 E4 Final Experimental Results & Scientific Report

**Project:** `D:\ml\fydp\MyNet`  
**Research Framing:** *Data-Centric Analysis of Hard-Negative Learning and Domain Shift in Emergency Audio Classification*  
**Date:** November 2, 2026  
**Document:** [`reports/phase2_e4_final_results.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_final_results.md)  
**Correction & Audit Documents:**
- [`reports/phase2_e4_final_interpretation_correction.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_final_interpretation_correction.md) *(Definitive Scientific Interpretation Correction)*
- [`reports/phase2_e4_independent_audit.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_independent_audit.md) *(Independent Forensic Audit)*
- [`reports/phase2_e4_evaluation_report.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_evaluation_report.md)
- [`reports/phase2_e4_live_execution_status.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_live_execution_status.md)
- [`models/experiments/phase2_e4/phase2_e4_evaluation_summary.json`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/phase2_e4_evaluation_summary.json)  
**Preceding Audit & Protocol Documents:**
- [`reports/phase2_e4_primary_estimand_decision.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_primary_estimand_decision.md)
- [`reports/phase2_e4_permutation_verification.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_permutation_verification.md)
- [`reports/phase2_e4_statistical_design_correction.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_statistical_design_correction.md)
- [`reports/phase2_e4_final_execution_contract.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_final_execution_contract.md)  
**Benchmark:** Test Set v2 ($N = 750$ independently sourced audio recordings)  
**Epistemic Status:** **EXPLORATORY BENCHMARK ATTRIBUTION (TEST SET V2 REUSE)**  
**Training Sweep Status:** **ALL 45 / 45 RUNS COMPLETED SUCCESSFULLY (100%)**

---

## 1. Executive Summary & Core Research Findings

```mermaid
flowchart TD
    subgraph Core_Research_Question ["Core Phase 2 E4 Research Question"]
        Q["Which specific acoustic hard-negative categories drive the reliability gains of E2?<br/>Are hard negatives acoustically specific, or are they fungible with arbitrary background audio?"]
    end

    subgraph Experimental_Evidence ["Phase 2 Empirical Evidence (45 Trained Models, 7 Ablation Categories)"]
        FW["Fireworks & Firecrackers:<br/>+41.3% False Alarm Surge (p = 0.000244, Holm adj p = 0.0017)<br/>95% CI: [+0.240, +0.587]<br/>Status: Strong exploratory evidence of contribution"]
        SH["Human Shouting:<br/>+36.0% False Alarm Surge (p = 0.000977, Holm adj p = 0.0059)<br/>95% CI: [+0.187, +0.533]<br/>Status: Strong exploratory evidence of contribution"]
        CA["Car Alarms & Detectors:<br/>+14.7% False Alarm Surge (p = 0.03125, Holm adj p = 0.125)<br/>95% CI: [+0.040, +0.267]<br/>Status: Positive but inconclusive after multiple-comparison correction"]
        DS["Door Slams & Whistles:<br/>Delta in [+4.0%, +6.7%], p >= 0.1875<br/>Status: No statistically significant effect detected"]
        MI["Metal Impacts:<br/>Delta = 0.0%, p = 1.0000<br/>Status: No observed difference in this benchmark comparison"]
        CT["Construction Tools:<br/>Delta = -4.2%, p = 0.00388 (Pooled N=150 Fallback)<br/>Status: Category-specific effect unresolved due to pooled evaluation and collateral control removals"]
    end

    subgraph Scientific_Conclusion ["Scientific Conclusion"]
        Verdict["Acoustic Specificity Supported:<br/>Hard negatives are NOT fungible general audio.<br/>Critical acoustic overlap categories (Fireworks -> Gunshot, Shouting -> Scream)<br/>drive the targeted hard-negative advantage on this benchmark."]
    end

    Q --> FW & SH & CA & DS & MI & CT
    FW & SH --> Verdict
```

### 1.1. Summary of Scientific Outcomes

1. **Fireworks & Firecrackers and Human Shouting Provide Strong Evidence of Contribution:**
   - Ablating **Fireworks & Firecrackers** against matched-size random controls causes a **$+41.33\%$ surge in false alarm rate** on fireworks sounds under the primary estimand $\theta_1$ ($p = 0.000244$, primary Holm-Bonferroni adjusted $p = 0.001708$). This survives family-wise error rate control at $\alpha = 0.05$.
   - Ablating **Human Shouting** against matched-size random controls causes a **$+36.00\%$ surge in false alarm rate** on shouting confusers under the primary estimand $\theta_1$ ($p = 0.000977$, primary Holm-Bonferroni adjusted $p = 0.005862$). This also survives family-wise error rate control at $\alpha = 0.05$.
   - **Conclusion:** Both categories exhibit **strong exploratory evidence of contribution**, carving crucial decision boundaries against Gunshot and Scream, respectively.

2. **Car Alarms Exhibit Directional Sensitivity but Fail FWER Multiplicity Penalty:**
   - Ablating **Car Alarms & Detectors** causes a $+14.67\%$ increase in false alarm rate on siren-confuser sounds ($p = 0.031250$, unadjusted significant at nominal $\alpha = 0.05$).
   - However, under the 7-hypothesis primary Holm-Bonferroni step-down correction, its rank 4 threshold is $\alpha / 4 = 0.012500$. Because $0.031250 > 0.012500$, it is **not statistically significant after multiplicity adjustment** (adjusted $p = 0.125000$).
   - Multi-seed consistency is directionally unanimous ($3/3$ seeds: $b:c = 5:0, 3:0, 3:0$), indicating positive sensitivity that remains underpowered at $N = 25$ recordings. Status: **Positive but inconclusive after multiple-comparison correction**.

3. **Door Slams, Whistles, and Metal Impacts Show Non-Significant Differences:**
   - **Door Slams:** $\Delta = +6.67\%$ ($p = 0.187500$, adjusted $p = 0.562500$). Status: **No statistically significant effect detected**.
   - **Whistles & High Pitch:** $\Delta = +4.00\%$ ($p = 0.500000$, adjusted $p = 1.000000$). Status: **No statistically significant effect detected**.
   - **Metal Impacts:** $\Delta = 0.00\%$ ($p = 1.000000$, zero discordant pairs across all 3 seeds). Status: **No observed difference in this benchmark comparison**.
   - **Important Inferential Note:** Non-significance is **not** proof of zero effect. The limited test sample size ($N = 25$ recordings per slice) provides restricted statistical power to detect small or moderate effect sizes. For Metal Impacts, no performance difference was detected on the 25 benchmark clips under this transfer setup, indicating no measurable marginal benefit on this specific benchmark.

4. **Construction Tools: Confounded Fallback Comparison:**
   - Test Set v2 lacks a dedicated construction tools confuser slice, causing an automated evaluation fallback to the pooled confuser stratum ($N = 150$).
   - To match training size ($N = 616$), the matched random control condition randomly lost **7 human shouting clips** and **4 fireworks clips** from its training inventory, while the ablation condition retained 100% of shouting (44) and fireworks (28).
   - On the pooled 150 confusers (which contain 25 shouting and 25 fireworks clips), the control model suffered false alarms due to this collateral training loss (15 gunshot and 8 scream false alarms), producing an apparent negative shift ($\hat{\theta}_1 = -0.0422, p = 0.003880$).
   - **Resolution:** Any claim that removing construction tools improves performance is scientifically invalid. The comparison is severely confounded by cross-category collateral damage and pooled fallback evaluation. Status: **Category-specific effect unresolved due to pooled evaluation and collateral control removals**.

### 1.2. Standardized Category Interpretation Taxonomy

| Category | Slice $N$ | Point Estimate $\hat{\theta}_1$ [95% CI] | Permutation $p$-value | Holm-Bonferroni FWER | Standardized Scientific Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Fireworks & Firecrackers** | 25 | **+0.4133** [+0.2400, +0.5867] | **0.000244** | **Survives Rank 1** | **Strong exploratory evidence of contribution.** |
| **Human Shouting** | 25 | **+0.3600** [+0.1867, +0.5333] | **0.000977** | **Survives Rank 2** | **Strong exploratory evidence of contribution.** |
| **Car Alarms & Detectors** | 25 | +0.1467 [+0.0400, +0.2667] | 0.031250 | Fails Rank 4 | **Positive but inconclusive after multiple-comparison correction.** |
| **Door Slams** | 25 | +0.0667 [-0.0000, +0.1467] | 0.187500 | Fails Rank 5 | **No statistically significant effect detected.** |
| **Whistles & High Pitch** | 25 | +0.0400 [+0.0000, +0.1067] | 0.500000 | Fails Rank 6 | **No statistically significant effect detected.** |
| **Metal Impacts** | 25 | +0.0000 [+0.0000, +0.0000] | 1.000000 | Fails Rank 7 | **No observed difference in this benchmark comparison.** |
| **Construction Tools** | 150 | -0.0422 [-0.0711, -0.0156] | 0.003880 | Confounded Slice | **Category-specific effect unresolved due to pooled evaluation and collateral control removals.** |

> [!NOTE]
> **Epistemic Demarcation:** In accordance with the project research protocol, Test Set v2 was unblinded in Phase 1. Consequently, all Phase 2 category-attribution findings reported herein are formally designated **Exploratory Benchmark Attribution on Test Set v2**. Strict cluster-aware bootstrap and non-parametric permutation tests were enforced to prevent inferential overclaiming. Results demonstrate empirical performance sensitivity under this specific model architecture; they do not constitute universal physical causality. Non-significance is not proof of no effect.

---

## 2. Experimental Design & Implementation Verification

### 2.1. Complete 45-Run Execution Matrix

All 15 experimental conditions were trained across 3 random seeds (42, 123, 2026), totaling 45 independent models:

- **Backbone Architecture:** PANNs CNN14 (frozen pretrained backbone, 5-class linear transfer head, Adam optimizer, lr=0.001, batch size 16, weight decay $10^{-4}$).
- **Shared Validation Invariant:** Exactly 105 samples ([`shared_val_split_manifest.csv`](file:///d:/ml/fydp/MyNet/data/processed/shared_val_split_manifest.csv), SHA-256 `38569ae4...`) evaluated identically across all 45 runs.
- **Early Stopping:** Monitored validation macro-F1 with patience = 7 epochs (max 30 epochs).
- **Execution Host:** NVIDIA GeForce RTX 5060 (8.52 GB VRAM) on CUDA.
- **Run Artifacts:** Every single run directory preserves `best_model.pt` (320.03 MB), `val_predictions.csv`, `metrics.json`, `training_history.json`, and `run_meta.json`.

```
====================================================================================================
Execution Summary: 45 Planned Runs | 45 Succeeded | 0 Failed | 0 Incomplete | 0 Regressions
Storage Footprint: 45 x best_model.pt = 14.40 GB | Free Drive Space: 43.20 GB
====================================================================================================
```

---

## 3. Primary Hypothesis Family: Estimand $\theta_1$ (Single-Model Deployment)

### 3.1. Estimand Definition & Mathematical Foundations

$$\theta_1 = \mathbb{E}_{\text{models}, \text{recordings}} \left[ \mathbb{I}(\hat{y}_{i, s}^{\text{Abl}} \in \text{Emerg}) - \mathbb{I}(\hat{y}_{i, s}^{\text{Ctl}} \in \text{Emerg}) \right]$$

For each physical recording $i \in \{1 \dots N\}$, the three seed decisions are averaged:
$$d_i = \frac{1}{3} \sum_{s \in \{42, 123, 2026\}} \left( \mathbb{I}(\hat{y}_{i, s}^{\text{Abl}} \in \text{Emerg}) - \mathbb{I}(\hat{y}_{i, s}^{\text{Ctl}} \in \text{Emerg}) \right)$$
$d_i \in \left\{ -1.0, -\frac{2}{3}, -\frac{1}{3}, 0.0, +\frac{1}{3}, +\frac{2}{3}, +1.0 \right\}$.

Statistical inference:
1. **Hypothesis Testing:** Exact exhaustive ($2^K$ for $K \le 16$) or Monte Carlo ($R = 100,000$) paired sign-flip permutation tests testing $\mathcal{H}_0: \mathbb{E}[d_i] = 0$.
2. **Confidence Intervals:** 10,000 recording-cluster bootstrap iterations resampling physical recording IDs with replacement.
3. **Multiplicity Correction:** Primary Holm-Bonferroni step-down correction across exactly the 7 category pairs at $\alpha = 0.05$.

### 3.2. Primary Statistical Results Table

| Rank | Confuser Category | Slice $N$ | Point Estimate $\hat{\theta}_1$ | 95% Cluster Bootstrap CI | Discordant Recordings ($K$) | Permutation Method | Permutation $p$-value | Holm $\alpha$ Threshold | Statistically Significant (FWER $\alpha=0.05$)? |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Fireworks & Firecrackers** | 25 | **+0.4133** | **[+0.2400, +0.5867]** | 13 | `exact_enumeration_2^13` | **0.000244** | 0.007143 | **YES (SURVIVES HOLM)** |
| **2** | **Human Shouting** | 25 | **+0.3600** | **[+0.1867, +0.5333]** | 12 | `exact_enumeration_2^12` | **0.000977** | 0.008333 | **YES (SURVIVES HOLM)** |
| **3** | Construction Tools | 150 | -0.0422 | [-0.0711, -0.0156] | 23 | `monte_carlo_100000` | 0.003880 | 0.010000 | **CONFOUNDED (UNRESOLVED)*** |
| **4** | Car Alarms & Detectors | 25 | +0.1467 | [+0.0400, +0.2667] | 6 | `exact_enumeration_2^6` | 0.031250 | 0.012500 | **NO (Fails Holm Rank 4)** |
| **5** | Door Slams | 25 | +0.0667 | [-0.0000, +0.1467] | 6 | `exact_enumeration_2^6` | 0.187500 | 0.016667 | **NO** |
| **6** | Whistles & High Pitch | 25 | +0.0400 | [+0.0000, +0.1067] | 2 | `exact_enumeration_2^2` | 0.500000 | 0.025000 | **NO** |
| **7** | Metal Impacts | 25 | +0.0000 | [+0.0000, +0.0000] | 0 | `trivial_null` | 1.000000 | 0.050000 | **NO** |

> [!NOTE]
> **\* Note on Construction Tools Confound:** As documented in the independent forensic audit, Test Set v2 lacks a dedicated construction tools slice, triggering an automated fallback to the pooled 150-clip confuser stratum. The matched random control randomly lost 7 shouting and 4 fireworks clips from its training set, causing 15 gunshot and 8 scream false alarms on those test subdomains that the ablation model resisted. The apparent negative shift is an artifact of cross-category collateral damage in the control and pooled fallback evaluation; the category-specific effect is **UNRESOLVED**.

---

## 4. Secondary Hypothesis Family: Estimand $\theta_2$ (Ensemble Deployment)

### 4.1. Estimand Definition & McNemar Exact Test

$$\theta_2 = \mathbb{E}_{\text{recordings}} \left[ \mathbb{I}(\hat{y}_{i, \text{ens}}^{\text{Abl}} \in \text{Emerg}) - \mathbb{I}(\hat{y}_{i, \text{ens}}^{\text{Ctl}} \in \text{Emerg}) \right]$$
where $\hat{y}_{i, \text{ens}} = \arg\max_c \frac{1}{3}\sum_{s} \text{Softmax}(f(x_i; \mathbf{w}_s))_c$.

Because ensemble decisions are discrete binary outputs per recording, discordances form integer contingency cell counts $(b, c) \in \mathbb{N}^2$, tested via exact two-sided binomial McNemar (`scipy.stats.binomtest(b, b + c, p=0.5)`).

### 4.2. Secondary Ensemble Results Table

| Rank | Confuser Category | Slice $N$ | Point Estimate $\hat{\theta}_2$ | 95% Cluster Bootstrap CI | Discordant Pairs $(b : c)$ | Exact Binomial $p$-value | Ensemble Holm $\alpha$ | Statistically Significant (Ensemble FWER)? |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Fireworks & Firecrackers** | 25 | **+0.4400** | **[+0.2400, +0.6400]** | `11 : 0` | **0.000977** | 0.007143 | **YES (SURVIVES HOLM)** |
| **2** | Human Shouting | 25 | +0.3600 | [+0.1200, +0.5600] | `10 : 1` | 0.011719 | 0.008333 | **NO (Fails Holm Rank 2: 0.0117 > 0.0083)** |
| **3** | Construction Tools | 150 | -0.0467 | [-0.0800, -0.0133] | `0 : 7` | 0.015625 | 0.010000 | **NO (Confounded Fallback)** |
| **4** | Car Alarms & Detectors | 25 | +0.1200 | [+0.0000, +0.2800] | `3 : 0` | 0.250000 | 0.012500 | **NO** |
| **5** | Metal Impacts | 25 | +0.0000 | [+0.0000, +0.0000] | `0 : 0` | 1.000000 | 0.016667 | **NO** |
| **6** | Door Slams | 25 | +0.0400 | [+0.0000, +0.1200] | `1 : 0` | 1.000000 | 0.025000 | **NO** |
| **7** | Whistles & High Pitch | 25 | +0.0400 | [+0.0000, +0.1200] | `1 : 0` | 1.000000 | 0.050000 | **NO** |

> [!IMPORTANT]
> **Key Methodological Insight:** Under single-model deployment ($\theta_1$), **Human Shouting is statistically significant** ($p = 0.000977 < 0.008333$). Under ensemble deployment ($\theta_2$), probability averaging rescues one borderline case ($b=10, c=1$), shifting the $p$-value to $0.011719$, which narrowly misses the rank 2 threshold ($0.008333$). This demonstrates the vital importance of formalizing the primary estimand to reflect the actual single-model edge deployment scenario.

---

## 5. Descriptive Multi-Seed Consistency Analysis

To evaluate whether the observed effects are stable across random initializations and SGD trajectories, each seed was evaluated independently using exact paired binomial McNemar tests.

In the paired contingency counts $(b : c)$, $b$ denotes the number of recordings where the Ablation model falsely alarmed on a confuser and the Control model did not; $c$ denotes the number of recordings where the Control model falsely alarmed and the Ablation did not. Two-sided exact binomial $p$-values test $\mathcal{H}_0: b = c$:

| Confuser Category | Seed 42 Discordant $(b:c)$ [$p$] | Seed 123 Discordant $(b:c)$ [$p$] | Seed 2026 Discordant $(b:c)$ [$p$] | 3/3 Directional Concordance? | Seed Variance Interpretation |
|:---|:---:|:---:|:---:|:---:|:---|
| **Fireworks & Firecrackers** | `11:0` ($p = 0.00098$) | `11:0` ($p = 0.00098$) | `9:0` ($p = 0.00391$) | **YES (3/3)** | Perfectly stable; zero false alarms produced under control that were not produced under ablation. |
| **Human Shouting** | `8:1` ($p = 0.03906$) | `10:1` ($p = 0.01172$) | `11:0` ($p = 0.00098$) | **YES (3/3)** | Highly consistent; significant in all 3 individual seeds. |
| **Car Alarms & Detectors** | `5:0` ($p = 0.06250$) | `3:0` ($p = 0.25000$) | `3:0` ($p = 0.25000$) | **YES (3/3)** | Consistent direction ($b > 0, c = 0$), but underpowered individually. |
| **Door Slams** | `3:1` ($p = 0.62500$) | `2:0` ($p = 0.50000$) | `2:1` ($p = 1.00000$) | **YES (3/3)** | Low effect magnitude; within stochastic noise range. |
| **Whistles & High Pitch** | `1:0` ($p = 1.00000$) | `0:0` ($p = 1.00000$) | `2:0` ($p = 0.50000$) | **NO** | Negligible shift. |
| **Metal Impacts** | `0:0` ($p = 1.00000$) | `0:0` ($p = 1.00000$) | `0:0` ($p = 1.00000$) | **NO** | Identical predictions across all 3 seeds. |
| **Construction Tools** | `10:12` ($p = 0.83181$) | `0:12` ($p = 0.00049$) | `0:5` ($p = 0.06250$) | **NO** | Seed 42 exhibits discordant predictions in both directions ($b=10, c=12$, two-sided exact McNemar $p=0.83181$), showing no statistically significant directional shift. Seeds 123 ($b=0, c=12$) and 2026 ($b=0, c=5$) exhibit unidirectional control false alarms driven by cross-category collateral loss. |

---

## 6. Secondary Reference Comparison: Ablation vs. E2 Full Reference Ceiling

This comparison contrasts each category ablation against the unablated reference model [`E2 Full`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e2_full_seed42) ($N = 646$). This measures the total margin degradation relative to deploying the complete targeted training set:

| Rank | Category Name | Ref Condition | Slice $N$ | Point Estimate $\theta_1$ [95% CI] | Permutation $p$-value | Holm $\alpha$ Threshold | Significant under FWER? | $\theta_2$ Ensemble Shift [95% CI] |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Human Shouting** | `e2_full` | 25 | **+0.5067 [+0.3333, +0.6667]** | **0.000031** | 0.007143 | **YES** | +0.5200 [+0.3200, +0.7200] |
| **2** | **Fireworks & Firecrackers** | `e2_full` | 25 | **+0.4800 [+0.2933, +0.6667]** | **0.000122** | 0.008333 | **YES** | +0.4400 [+0.2400, +0.6400] |
| **3** | Car Alarms & Detectors | `e2_full` | 25 | +0.1467 [+0.0400, +0.2667] | 0.031250 | 0.010000 | **NO** | +0.1200 [+0.0000, +0.2800] |
| **4** | Door Slams | `e2_full` | 25 | +0.0400 [-0.0133, +0.0933] | 0.375000 | 0.012500 | **NO** | +0.0400 [+0.0000, +0.1200] |
| **5** | Whistles & High Pitch | `e2_full` | 25 | +0.0400 [+0.0000, +0.1067] | 0.500000 | 0.016667 | **NO** | +0.0400 [+0.0000, +0.1200] |
| **6** | Construction Tools | `e2_full` | 150 | +0.0044 [-0.0067, +0.0156] | 0.687500 | 0.025000 | **NO** | +0.0000 [+0.0000, +0.0000] |
| **7** | Metal Impacts | `e2_full` | 25 | +0.0000 [+0.0000, +0.0000] | 1.000000 | 0.050000 | **NO** | +0.0000 [+0.0000, +0.0000] |

---

## 7. Global Condition Performance Across Full Test Set v2 (N = 750)

| Experimental Condition Key | Train Samples | Test Macro-F1 (Mean ± Std) | Confusers FER (Mean) | Overall Test FER (Mean) | Ensemble Macro-F1 | Ensemble Confusers FER |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `e2_full` (Reference Ceiling) | 646 | **0.8308 ± 0.0026** | **0.3066** | **0.2300** | **0.8386** | **0.3067** |
| `e4_minus_human_shouting` | 602 | 0.8143 ± 0.0075 | 0.4089 | 0.2944 | 0.8226 | 0.4133 |
| `e4_matched_control_human_shouting` | 602 | 0.8175 ± 0.0021 | 0.3444 | 0.2789 | 0.8224 | 0.3400 |
| `e4_minus_fireworks_firecrackers` | 618 | 0.8059 ± 0.0055 | 0.3955 | 0.3033 | 0.8189 | 0.3800 |
| `e4_matched_control_fireworks_firecrackers` | 618 | 0.8227 ± 0.0067 | 0.3356 | 0.2578 | 0.8252 | 0.3267 |
| `e4_minus_metal_impacts` | 606 | 0.8235 ± 0.0060 | 0.3400 | 0.2867 | 0.8246 | 0.3400 |
| `e4_matched_control_metal_impacts` | 606 | 0.8218 ± 0.0075 | 0.3333 | 0.2556 | 0.8227 | 0.3200 |
| `e4_minus_door_slams` | 613 | 0.8278 ± 0.0009 | 0.3400 | 0.2589 | 0.8373 | 0.3267 |
| `e4_matched_control_door_slams` | 613 | 0.8258 ± 0.0022 | 0.3289 | 0.2522 | 0.8359 | 0.3200 |
| `e4_minus_car_alarms_detectors` | 617 | 0.8137 ± 0.0080 | 0.3489 | 0.2855 | 0.8245 | 0.3400 |
| `e4_matched_control_car_alarms_detectors` | 617 | 0.8241 ± 0.0051 | 0.3200 | 0.2389 | 0.8238 | 0.3133 |
| `e4_minus_whistles_high_pitch` | 627 | 0.7841 ± 0.0452 | 0.3311 | 0.2467 | 0.8164 | 0.3267 |
| `e4_matched_control_whistles_high_pitch` | 627 | 0.8120 ± 0.0028 | 0.3400 | 0.2733 | 0.8205 | 0.3267 |
| `e4_minus_construction_tools` | 616 | 0.8324 ± 0.0090 | 0.3111 | 0.2456 | 0.8386 | 0.3067 |
| `e4_matched_control_construction_tools` | 616 | 0.8032 ± 0.0192 | 0.3534 | 0.2611 | 0.8134 | 0.3533 |

---

## 8. Scientific Insights & Discussion

### 8.1. Resolving the Central Research Question
The core FYDP research question asks:
> *"Does model-error-guided hard-negative data selection improve the reliability of emergency audio classification more effectively than simply increasing the amount of general background audio?"*

Phase 1 established that **E2 Matched Targeted Hard Negatives outperformed E3 General Background Audio** by +4.8% Macro-F1 and achieved an 8.6x greater reduction in confuser false emergency errors, despite identical dataset size ($N = 646$).

Phase 2 E4 now resolves the internal mechanism of this advantage on the benchmark:
1. **Hard Negatives are Semantically and Acoustically Specific:**
   The substantial false-alarm surges observed upon ablating **Fireworks** ($+41.33\%$) and **Shouting** ($+36.00\%$) demonstrate that under this transfer setup, reliability gains are driven by targeted acoustic boundary carving, rather than fungible non-emergency audio volume.
2. **Spectral-Temporal Confusion Mechanism:**
   - Gunshots and fireworks share rapid impulsive acoustic rise times and broad-spectrum explosive energy. Without fireworks in training, CNN14 transfer models collapse into classifying fireworks as gunshots.
   - Screams and shouting share human vocal tract formant resonances and intense pitch harmonics. Without shouting in training, CNN14 transfer models misclassify emotional vocalizations as screams.
   - Generic background audio (speech, traffic, rain, ambient room noise) contains none of these specific boundary characteristics and fails to suppress these high-risk false alarms.
3. **Construction Tools Confound:**
   The apparent negative shift for Construction Tools ($\hat{\theta}_1 = -0.0422, p = 0.003880$) was empirically traced to cross-category collateral sample loss in the matched control (which lost 7 shouting and 4 fireworks clips) evaluated across a pooled 150-clip confuser set. It does not represent a true benefit of removing construction tools; the category-specific effect remains **unresolved**.

### 8.2. Acoustic Redundancy and Annotation Prioritization
Conversely, the finding that **Metal Impacts** produced zero observable difference on this benchmark ($\Delta = 0.0\%$, $p = 1.0$) demonstrates that:
- Under this transfer setup, additional metal impact negatives provided no measurable marginal benefit on the 25 evaluated test recordings.
- While non-significance is not proof of zero effect, pretrained representations (from AudioSet) appear sufficiently resilient to metal impacts on these test clips that further data curation yielded negligible return.
- Data-centric curation pipelines should concentrate annotation resources on **empirically verified high-impact acoustic confusion modes** (fireworks, human shouting, and resonant alarm tones) rather than categories with negligible empirical sensitivity.

---

## 9. Conclusion & Research State

1. **Phase 2 E4 is Formally Complete:**
   All 45 experimental models have been trained, verified, and evaluated on Test Set v2. All primary and secondary hypothesis tests have been independently reproduced with 100% precision.
2. **Standardized Category Taxonomy Codified:**
   - **Fireworks & Firecrackers:** *Strong exploratory evidence of contribution.*
   - **Human Shouting:** *Strong exploratory evidence of contribution.*
   - **Car Alarms & Detectors:** *Positive but inconclusive after multiple-comparison correction.*
   - **Door Slams:** *No statistically significant effect detected.*
   - **Whistles & High Pitch:** *No statistically significant effect detected.*
   - **Metal Impacts:** *No observed difference in this benchmark comparison.*
   - **Construction Tools:** *Category-specific effect unresolved due to pooled evaluation and collateral control removals.*
3. **Epistemic Demarcation & Next Steps:**
   In accordance with the project protocol, all Phase 2 findings on Test Set v2 represent **exploratory benchmark attribution**. True confirmatory validation requires future unblinded evaluation on newly acquired independent physical recordings. No further training or data collection will be launched without faculty and team review.
