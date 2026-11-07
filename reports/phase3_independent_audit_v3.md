# Phase 3 Independent Protocol & Evidence Audit Report (v3.0)

**Project:** `D:\ml\fydp\MyNet`  
**Research Framing:** *Data-Centric Analysis of Hard-Negative Learning and Domain Shift in Emergency Audio Classification*  
**Date:** November 3, 2026  
**Auditor Role:** Independent Senior Machine Learning & Biostatistical Methodology Auditor  
**Primary Audit Target:** Proposed Phase 3 Confirmatory Protocol & Underlying Research Evidence  
**Audited Artifacts:**
- [`reports/FYDP_Advisor_Brief.docx`](file:///d:/ml/fydp/MyNet/reports/FYDP_Advisor_Brief.docx) *(Advisor Discussion Memo, Nov 2026)*
- [`reports/phase3_independent_confirmation_protocol_v2_1.md`](file:///d:/ml/fydp/MyNet/reports/phase3_independent_confirmation_protocol_v2_1.md) *(Pre-Registration Draft v2.1)*
- [`master_experiment_registry_reconciled.csv`](file:///d:/ml/fydp/MyNet/master_experiment_registry_reconciled.csv) *(37 Experiment Records)*
- [`reports/master_experiment_registry_final_qc_audit.md`](file:///d:/ml/fydp/MyNet/reports/master_experiment_registry_final_qc_audit.md) *(Latest QC Audit Report)*
- [`FYDP_Complete_Research_Work_Plan.md`](file:///C:/Users/Rajee/Downloads/FYDP_Complete_Research_Work_Plan.md) *(Master Work Plan v2, Nov 3, 2026)*
- [`scripts/simulate_phase3_power_comprehensive.py`](file:///d:/ml/fydp/MyNet/scripts/simulate_phase3_power_comprehensive.py) & [`reports/phase3_power_simulation_comprehensive.json`](file:///d:/ml/fydp/MyNet/reports/phase3_power_simulation_comprehensive.json)
- Historical manifests, prediction tables, and checkpoints across `models/experiments/` and `data/processed/`

**Machine-Readable Deliverable:** [`reports/phase3_document_conflict_matrix_v3.csv`](file:///d:/ml/fydp/MyNet/reports/phase3_document_conflict_matrix_v3.csv)  
**Companion Assessment:** [`reports/phase3_publication_readiness_v3.md`](file:///d:/ml/fydp/MyNet/reports/phase3_publication_readiness_v3.md)  
**Safety & Integrity Constraints:** Strictly read-only audit. Zero models trained, zero datasets downloaded or modified, zero manifests altered, zero checkpoints touched, zero planned experiments marked complete.

---

## 1. Executive Summary & Epistemic Verdict

This independent forensic audit evaluated whether the proposed Phase 3 study and current empirical evidence base can support a defensible publication claim regarding **model-error-guided hard-negative data selection** in emergency audio classification.

### 1.1. Core Audit Finding
The central research inquiry—whether mining hard negatives from baseline model errors restructures decision boundaries more effectively than adding general background audio under matched data budgets—is scientifically compelling, coherent, and highly relevant to data-centric ML. 

However, **in its current proposed form, the Phase 3 evaluation cannot definitively prove that model-error guidance is the active causal ingredient.** 
Specifically:
1. **Causal Confounding (E2 vs. E3):** Comparing E2 (targeted hard negatives) against E3 (general background) confounds *confuser category coverage* with *model-error-guided selection*. Reviewers will correctly object that E2 outperformed E3 simply because it was exposed to shouting and fireworks, not because those clips were mined via model errors.
2. **Document Inconsistencies:** Critical operational discrepancies exist between the Advisor Brief, Protocol v2.1, and the Work Plan regarding Gate 2 safety endpoints, non-inferiority margins, and reporting mandates.
3. **Statistical Power Conditionality:** The claimed >99% statistical power for Gate 1 is strictly conditional on an assumed true effect size (Delta >= 0.08) and homogeneous category performance. If true unexposed performance shrinks to Delta = 0.05, power drops to 89.9%; at Delta = 0.03, power is only 52.4%. Power for Gates 2, 3, and 4 has not been mathematically simulated.
4. **Missing Equivalence Boundary:** The current protocol lacks a pre-registered equivalence margin. If Gate 1 fails to demonstrate superiority, the study cannot distinguish between a truly null effect and an underpowered inconclusive test.

---

## 2. Document Reconciliation & Conflict Matrix Analysis

A systematic line-by-line cross-examination was conducted across [`FYDP_Advisor_Brief.docx`](file:///d:/ml/fydp/MyNet/reports/FYDP_Advisor_Brief.docx), [`reports/phase3_independent_confirmation_protocol_v2_1.md`](file:///d:/ml/fydp/MyNet/reports/phase3_independent_confirmation_protocol_v2_1.md), and [`FYDP_Complete_Research_Work_Plan.md`](file:///C:/Users/Rajee/Downloads/FYDP_Complete_Research_Work_Plan.md). All 15 conflict dimensions are codified in [`reports/phase3_document_conflict_matrix_v3.csv`](file:///d:/ml/fydp/MyNet/reports/phase3_document_conflict_matrix_v3.csv).

### Summary Table of Critical Discrepancies

| Conflict ID | Dimension | Source A (Advisor Brief / Work Plan) | Source B (Protocol v2.1) | Methodological Impact | Recommended Resolution | Advisor Sign-Off? |
|:---|:---|:---|:---|:---|:---|:---:|
| **CONF-001** | **Gate Definitions & Order** | Brief §7 flags that an interim summary replaced benign B-FER with E2 vs E1 emergency recall. | Protocol §5.1 retains Gate 1 (A-FER sup), Gate 2 (B-FER non-inf), Gate 3 (Rec non-inf), Gate 4 (Macro-F1 sup). | Dropping benign B-FER removes the primary safety guard against false alarm inflation on clean audio. | Reaffirm Protocol v2.1 4-gate order; both safety gates must compare E2 against matched E3. | **YES** |
| **CONF-002** | **Primary vs. Secondary Estimands** | Brief §10 asks if pooled A-FER is the primary claim and whether subdomains must remain exploratory. | Protocol §1.2 & §6 strictly confines confirmatory claims to pooled sequence (N=350); 7 subdomains (N=50) are exploratory. | Slices of N=50 have inadequate power (25.8%–40.6%) under multiplicity control. | Formally freeze pooled theta_1 as the sole confirmatory claim; subdomains are descriptive only. | **YES** |
| **CONF-003** | **Gate 1 Decision Rule** | Brief §7 notes summary conflates p-values with upper bounds ('Upper 1-sided p < 0.050'). | Protocol §5.1 states 'One-sided permutation p < 0.05; H_0,1: theta_1 <= 0'. | Conflating p-values and confidence bounds damages academic credibility. | Codify rule: Reject H_0,1 if one-sided permutation p < 0.05 and theta_hat_1 > 0; report 2-sided 95% CI. | **NO** |
| **CONF-004** | **Gate 2 Comparator** | Brief §7 notes summary changed Gate 2 to E2 vs E1 emergency recall. | Protocol §5.1 tests E2 vs E3 B-FER on N=350 benign clips (theta_2 = E[B-FER_E2 - B-FER_E3]). | E1 (N=422) confounds data addition with data strategy; E3 (N=634) isolates selection strategy. | Retain E3 as comparator for Gate 2 B-FER non-inferiority (Upper 95% bound < delta_B). | **YES** |
| **CONF-005** | **Gate 3 Test & Estimand** | Brief §7 flags that summary specified McNemar for Gate 3; McNemar is micro-recall only. | Protocol §5.1 specifies 4-class Macro-Recall evaluated via class-stratified bootstrap inversion. | Micro-recall allows easy classes (sirens) to mask recall collapse on critical rare classes (screams). | Retain unweighted 4-class macro-recall evaluated via class-stratified bootstrap; reject McNemar. | **YES** |
| **CONF-006** | **Gate 4 Metric & Degeneracy** | Brief §6 requires full confusion matrix recomputation per bootstrap draw. | Protocol §5.2 defines Gate 4 Macro-F1 with explicit penalty (F1_c = 0) if TP_c + FP_c = 0. | Prevents division-by-zero when a bootstrap replicate drops a class; penalizes class abandonment. | Reaffirm Protocol v2.1 formulation with explicit zero-F1 penalty for unpredicted classes. | **NO** |
| **CONF-007** | **Non-Inferiority Margins** | Brief §7 flags delta_B previously denoted benign margin but was applied to recall; asks if margins are defensible. | Protocol §5.1 assigns delta_B = 0.020 (2.0%) to benign B-FER and delta_R = 0.025 (2.5%) to recall. | Margin conflation invalidates formal hypothesis testing. Margins require operational defense. | Decouple symbols strictly: delta_B = 0.020 (benign FER), delta_R = 0.025 (macro-recall). | **YES** |
| **CONF-008** | **Equivalence Margin** | Work Plan §4 & §5 mandate adding an equivalence margin to interpret a null result. | Protocol v2.1 specifies one-sided superiority only; no equivalence margin defined. | If Gate 1 fails, superiority testing yields an uninformative null ('absence of evidence'). | Pre-register SESOI Delta_equiv = 0.030 (3.0 pp A-FER); evaluate TOST if Gate 1 fails superiority. | **YES** |
| **CONF-009** | **Statistical Test Mismatch** | Brief §7 notes interim summary used McNemar for Gates 2 & 3. | Protocol §5.1 uses cluster bootstrap for Gates 2, 3, 4; permutation for Gate 1. | Ratio metrics violate paired binary contingency assumptions of McNemar's test. | Formally purge McNemar references for Gates 2, 3, 4; reserve McNemar solely for discrete ensemble theta_2. | **NO** |
| **CONF-010** | **Reporting Rule After Failed Gate** | Work Plan §4 mandates pre-specifying that all 4 gate estimates and CIs are always reported. | Protocol §5 states 'testing halts immediately and no subsequent confirmatory claims are permitted'. | Halting without reporting downstream metrics suppresses critical safety and recall data. | Pre-specify that point estimates and 95% CIs for ALL 4 gates will be published regardless of pass/fail. | **YES** |
| **CONF-011** | **Conditional Power Claims** | Brief §6 & §7 state power is conditional on simulation assumptions, not guaranteed. | Protocol §1.2 states Gatekeeper sequence is 'fully powered (>99%)'. | Misrepresents optimistic simulation scenarios as guaranteed benchmark performance. | State power transparently: 99.8% conditional on Delta >= 0.08; 89.9% at Delta = 0.05; 52.4% at Delta = 0.03. | **NO** |
| **CONF-012** | **Sampling Unit & Clustering** | Brief §6 & Work Plan §10 note physical recordings are units; seeds are repeated measures. | Protocol §4.3 & §5.1 specify physical recording clusters for all permutation and bootstrap draws. | Treating seeds or windowed clips as independent samples deflates standard errors and inflates Type I error. | Unify on recording-level cluster units across all code; enforce clustering by original_recording_id. | **NO** |
| **CONF-013** | **Seed Variation & Ensemble Role** | Work Plan §2 flags unusually low Phase 1 E3 seed variance; Brief §9 notes seeds don't increase N. | Protocol §2.2 uses 3 seeds (42, 123, 2026); primary estimand averages seeds; ensemble theta_2 is secondary. | Confusing individual model expectation with ensemble deployment muddles the primary claim. | Primary claim tests individual model expectation theta_1; ensemble theta_2 is secondary exploratory. | **NO** |
| **CONF-014** | **Benchmark Prevalence vs Stream** | Brief §5 & §9 emphasize balanced N=1,400 does not reflect real-world prevalence. | Protocol §3 uses balanced N=1,400; Section 6.3 proposes 72-hour continuous stream simulation. | Balanced test measures acoustic sensitivity/specificity; streaming measures operational false alarm burden. | Confirm balanced N=1,400 for Gates 1-4, paired with mandatory 72-hour streaming evaluation. | **YES** |
| **CONF-015** | **Audio Licensing & Open Release** | Brief §8 asks whether raw audio, metadata-only, or tiered release is acceptable. | Protocol §9 specifies 5-tier open-access release on Zenodo with CC attribution manifest; BBC excluded. | Commercial redistribution of CC-BY-NC audio creates legal liability. Metadata-only adds friction. | Present Option 1 (5-tier Zenodo release with attribution CSV) vs Option 2 (Metadata-only). | **YES** |

---

## 3. In-Depth Statistical Methodology Audit

### 3.1. McNemar's Test Applicability & Limitations
McNemar's test evaluates paired binary discordant pairs ($b$ and $c$) in a $2 \times 2$ contingency table. 
- **Valid Use Case:** Secondary estimand $\theta_2$ (discrete majority-vote 3-model ensemble on physical recordings, where each recording is classified as either a False Emergency Alert ($y = 1$) or Correct Rejection ($y = 0$)).
- **Fatal Misapplication:** Any application of McNemar's test to **macro-recall** (Gate 3) or **macro-F1** (Gate 4). Macro-recall is an unweighted average of class-specific recall ratios:
  $$\text{Rec}_{\text{macro}} = \frac{1}{4} \sum_{c=1}^4 \frac{\sum_{i \in \mathcal{C}_c} I(\hat{y}_i = c)}{N_c}$$
  Because macro-recall is a ratio of random variables across heterogeneous classes, it cannot be cast as a single $2 \times 2$ paired binary table. Applying McNemar collapses class boundaries, converting macro-recall into micro-recall, which allows high-prevalence or acoustically salient classes (e.g. sirens) to conceal catastrophic recall collapse on rare classes (e.g. screams). Protocol v2.1 correctly abandoned McNemar for Gates 2, 3, and 4 in favor of cluster bootstrapping.

### 3.2. Clustered Multi-Seed Permutation Testing (Gate 1)
For Gate 1, each physical recording $i \in \{1, \dots, N\}$ ($N = 350$) is evaluated across $S = 3$ random training seeds for both E3 and E2:
$$d_i = \frac{1}{3} \sum_{s=1}^3 \left[ I(\hat{y}_{i, s}^{\text{E3}} \in \mathcal{E}) - I(\hat{y}_{i, s}^{\text{E2}} \in \mathcal{E}) \right] \in \left\{ -1, -\frac{2}{3}, -\frac{1}{3}, 0, \frac{1}{3}, \frac{2}{3}, 1 \right\}$$
The test statistic is $T = \sum_{i=1}^N d_i$. Under $\mathcal{H}_{0,1}: \mathbb{E}[d_i] = 0$, the signs of $d_i$ are exchangeable under random reflection:
$$T^{(r)} = \sum_{i=1}^N \sigma_i^{(r)} d_i, \quad \sigma_i^{(r)} \in \{-1, +1\} \text{ with } p = 0.5$$
- **Audit Assessment:** This implementation is exact, non-parametric, and correctly accounts for multi-seed correlation by collapsing seed outcomes into the recording-level mean $d_i$ *before* sign-flipping. It guarantees exact nominal Type I error control ($\alpha = 0.05$) regardless of intra-cluster correlation.

### 3.3. Class-Stratified Cluster Bootstrapping (Gates 2, 3, and 4)
Ratio metrics (B-FER, macro-recall, macro-F1) violate the additive sign-flip symmetry required by permutation testing under non-zero margins.
- **Protocol Formulation:** Resamples physical recording clusters with replacement within each stratum:
  - Gate 2: Resamples $N = 350$ benign recording clusters.
  - Gate 3: Resamples $N_c = 175$ clusters independently within each of the 4 emergency classes ($N = 700$ total).
  - Gate 4: Resamples clusters independently within all 5 classes ($175 \times 4$ emergency, $700$ negative controls).
- **Audit Assessment:** Class-stratified cluster bootstrapping guarantees that true class instances ($TP_c + FN_c = 175$) remain strictly positive in every bootstrap replicate $b \in \{1, \dots, 10000\}$. The non-inferiority bound inversion is mathematically sound.

### 3.4. Rigorous Examination of Statistical Power Claims
Protocol v2.1 asserts that the primary Gatekeeper sequence is "fully powered (>99%)". Forensic audit of [`scripts/simulate_phase3_power_comprehensive.py`](file:///d:/ml/fydp/MyNet/scripts/simulate_phase3_power_comprehensive.py) reveals critical nuances:
1. **Gate 1 Power is Conditional:**
   - At $\Delta = 0.08$ (assumed discordant pairs $p_{10} = 0.18, p_{01} = 0.10$), simulated power is **99.86%**.
   - At $\Delta = 0.05$ (conservative effect, $p_{10} = 0.15, p_{01} = 0.10$), simulated power is **89.98%**.
   - At $\Delta = 0.03$ (small effect, $p_{10} = 0.13, p_{01} = 0.10$), simulated power drops to **52.4%**.
   - If true performance on pristine, unexposed field audio experiences domain shrinkage (common when transitioning from development benchmarks to independent field benchmarks), a true effect of $\Delta \le 0.04$ carries substantial risk (~30%–45%) of Type II failure.
2. **Gates 2, 3, and 4 Were Never Simulated:**
   - The simulation script evaluated ONLY Gate 1 permutation testing and binary McNemar. **Zero Monte Carlo simulations were conducted for Gate 2 non-inferiority, Gate 3 macro-recall non-inferiority, or Gate 4 macro-F1 superiority.**
   - Claiming that the entire 4-gate sequence has >99% power is an unsupported assertion. Sequence power is bounded above by the joint probability of passing all 4 gates:
     $$\text{Power}_{\text{sequence}} = \mathbb{P}(\text{Pass } G_1 \cap \text{Pass } G_2 \cap \text{Pass } G_3 \cap \text{Pass } G_4) \le \min_k \mathbb{P}(\text{Pass } G_k)$$
   - If Gate 3 macro-recall non-inferiority has 90% power and Gate 4 has 85% power, overall sequence power cannot exceed ~76%–85%.

### 3.5. Equivalence vs. Non-Inferiority
The current protocol conflates the absence of a statistically significant difference with equivalence. 
- If Gate 1 fails ($p \ge 0.05$), the only valid statistical statement is: *"The trial failed to demonstrate superiority of targeted selection over general background expansion."*
- To publish an informative data-centric finding when $\mathcal{H}_{0,1}$ is not rejected, the protocol **must pre-register an Equivalence Margin** (e.g. $\Delta_{\text{equiv}} = 0.030$). This allows the Two One-Sided Tests (TOST) procedure to reject effects $|\theta_1| \ge 0.030$, proving that targeted selection is *operationally equivalent* to general expansion within a 3.0 pp tolerance.

---

## 4. Causal Identification Audit

### 4.1. The Fundamental Confound in E2 vs. E3
The central research hypothesis asks:
> *"Does model-error-guided hard-negative data selection improve the reliability of emergency audio classification more effectively than simply increasing the amount of general background audio?"*

In Phase 1 and the proposed Phase 3 protocol, the comparison is framed as **E2 (Targeted Hard Negatives) vs. E3 (Matched General Background)**:
- **E1 (Baseline):** 422 training clips (326 emergency, 96 generic background).
- **E2 (Targeted):** 422 baseline + 212 targeted confusers mined from baseline model errors across 7 acoustic categories.
- **E3 (General):** 422 baseline + 212 randomly selected general background clips (ambient sound, speech, rain, traffic).

```
                      ┌──────────────────────────────────────────────┐
                      │              E2 vs. E3 COMPARISON            │
                      └──────────────────────┬───────────────────────┘
                                             │
                   ┌─────────────────────────┴────────────────────────┐
                   ▼                                                  ▼
      FACTOR 1: Category Coverage                        FACTOR 2: Selection Policy
  (Shouting, Fireworks, Alarms vs.                 (Model-Error Guided Mining vs.
        Ambient Background)                             Random Audio Sampling)
```

**Fatal Confound:** E2 and E3 differ simultaneously in **acoustic category composition** AND in **data selection mechanism**.
If E2 significantly outperforms E3 on Test Set v3 confusers:
- Did it win because baseline error mining found informative decision-boundary boundaries?
- Or did it win simply because E2 saw shouting and fireworks during training, while E3 saw rain and traffic?

Under the current E2 vs. E3 design, **it is impossible to answer this question.** Any peer reviewer or faculty examiner will argue that the result is an expected domain-coverage effect, not a validation of model-error-guided selection.

### 4.2. Assessment of the Proposed E5 Control Arm
Work Plan Section 5 correctly identifies this flaw and proposes an **E5 Control Arm (Random Confuser-Category Negatives)**:
- **E5 Specification:** 422 baseline + 212 random confuser clips drawn from the **exact same 7 confuser categories** as E2, but sampled completely at random without baseline model error guidance.

#### Rigorous Matching Audit: E2 vs. E5
To isolate the pure causal effect of model-error guidance, E5 must match E2 along 8 strict dimensions:

| Matching Dimension | Audit Requirement for E5 Arm | Confound if Violated |
|:---|:---|:---|
| **1. Category Proportions** | Exactly matched clip quotas across the 7 subdomains (e.g., exactly ~30 shouting, ~30 fireworks, etc.). | Skewing categories attributes gains to category imbalance. |
| **2. Total Clip Budget** | Exactly 212 added clips (N = 634 total training clips). | Conbounds data volume with selection method. |
| **3. Source Repository Pool** | Sampled from the identical source pool (e.g. Freesound / FSD50K candidate pool). | Sourcing from different platforms introduces recording environment bias. |
| **4. Technical Quality Bounds** | Identical duration (10.0s), sample rate (16 kHz mono), RMS bounds, and SNR profile. | Clean audio vs noisy audio confounds quality with error-guidance. |
| **5. Model Architecture** | Identical PANNs CNN14 frozen backbone with 10,245-parameter linear transfer head. | Architectural capacity differences mask data effects. |
| **6. Training Hyperparameters** | Identical batch size (16), optimizer (Adam lr=0.001), loss weighting (mu=1.0), and epochs (20). | Optimization differences alter convergence rates. |
| **7. Seeding & Initialization** | Identical 3 random seeds (42, 123, 2026) with matched weight initializations. | Seed variance masks true treatment effect. |
| **8. Checkpoint Selection** | Early stopping based strictly on the invariant shared validation split (N = 105). | Tuning checkpoint selection on confusers creates validation leakage. |

### 4.3. The Minimum Defensible Experimental Design
To support a publication claim in a top venue (e.g. DCASE, TMLR, IEEE TASLP), the design must decompose observed gains into three distinct orthogonal effects:

$$\begin{aligned}
\text{Total Observed Gain} &= \text{Score}(E2) - \text{Score}(E1) \\
&= \underbrace{[\text{Score}(E3) - \text{Score}(E1)]}_{\text{General Data Volume Effect}} + \underbrace{[\text{Score}(E5) - \text{Score}(E3)]}_{\text{Category Coverage Effect}} + \underbrace{[\text{Score}(E2) - \text{Score}(E5)]}_{\mathbf{\text{Pure Model-Error Guidance Effect}}}
\end{aligned}$$

- **If $\text{Score}(E2) \gg \text{Score}(E5)$:** Model-error-guided selection is causally validated. The thesis defense succeeds unequivocally.
- **If $\text{Score}(E2) \approx \text{Score}(E5) \gg \text{Score}(E3)$:** Category coverage is the active ingredient; active error mining provided zero marginal utility over naive category collection. This is a publishable negative result in data-centric AI.
- **If $\text{Score}(E5) \approx \text{Score}(E3)$:** Adding confusers provides no benefit over random ambient background.

---

## 5. Master Registry Forensic Audit & Evidence Verification

A complete forensic inspection of all 37 records in [`master_experiment_registry_reconciled.csv`](file:///d:/ml/fydp/MyNet/master_experiment_registry_reconciled.csv) was conducted against underlying repository artifacts.

```
Total Master Registry Records: 37
├── COMPLETED_VERIFIED:           33 records (89.2%)
├── COMPLETED_PARTIALLY_VERIFIED:  1 record   (2.7% - EXP-001)
└── PLANNED (Unexecuted):          3 records   (8.1% - EXP-029, EXP-030, EXP-031)
```

### 5.1. Audit Findings on Specific Experiments
1. **EXP-001 (Baseline 4-Layer CNN from Scratch):**
   - **Audit Status:** `COMPLETED_PARTIALLY_VERIFIED`.
   - **Finding:** Checkpoint `models/checkpoints/best_model.pt` exists and validates. Training trajectory is fully preserved in `training_history.json` and `best_model_meta.json` (Val Acc = 0.5143, Macro-F1 = 0.4430). However, **no per-sample prediction CSV was ever exported**. Downgrading from verified to partially verified was methodologically necessary.
2. **EXP-004 & EXP-006 (AudioSet Baseline vs. Combined AudioSet + NIGENS):**
   - **Audit Status:** `COMPLETED_VERIFIED`.
   - **Finding:** Evaluated on Eval Set v1 (N = 519). Both experiments share a single paired comparison artifact: [`models/experiments/model_comparison/eval_predictions_comparison.csv`](file:///d:/ml/fydp/MyNet/models/experiments/model_comparison/eval_predictions_comparison.csv). Audited columns confirm side-by-side predictions (`pred_baseline` vs `pred_combined`). The shared nature of this artifact is now documented.
3. **EXP-008 (Targeted Hard-Negative Data Harvesting):**
   - **Audit Status:** `COMPLETED_VERIFIED`.
   - **Finding:** Sourced 212 candidate clips across Freesound and YouTube. Manifest [`targeted_expansion_review_results.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_review_results.csv) confirms 100% human audit agreement. Cross-split checks verified zero YouTube ID overlap and zero SHA-256 collisions against train/val. Pretraining overlap against AudioSet's 2M clips was not evaluated and is not claimed.
4. **EXP-009 through EXP-013 (Phase 1 Main Hypothesis Arms):**
   - **Audit Status:** `COMPLETED_VERIFIED`.
   - **Finding:** All 9 individual run directories preserve frozen weights (`best_model.pt`), validation predictions (`val_predictions.csv`, N = 105), and benchmark predictions on Test Set v2 (`test_set_v2_predictions.csv`, N = 750).
   - **Anomaly Flagged:** On the in-distribution validation set (N = 105), E3 produced identical performance across all 3 seeds (Accuracy = 0.9143, Macro-F1 = 0.9183). Investigation confirmed that with 10,245 parameters and 20 epochs, E3 reached the identical classification plateau across seeds on the small 105-clip validation slice (which had only 24 generic 'other' clips). On Test Set v2 (N = 750), seed variance was normal (Acc: 80.93%, 81.07%, 80.93%).
5. **EXP-014 through EXP-028 (Phase 2 E4 Ablations & Matched Controls):**
   - **Audit Status:** `COMPLETED_VERIFIED` (Exploratory).
   - **Finding:** All 45 models (15 conditions x 3 seeds) exist with verified checkpoints and summary JSONs.
   - **Critical Methodological Flaw Identified:** In Phase 2 E4, Construction Tools ablation was reported as N = 150. Forensic audit proved that Test Set v2 completely lacked a dedicated construction tools slice. The evaluation script fell back to the pooled confuser stratum (N = 150). Furthermore, the matched random control lost 7 shouting and 4 fireworks clips from training, confounding the comparison. The construction tools finding is formally designated **UNRESOLVED**.

---

## 6. Publication-Readiness & Claims Assessment

### 6.1. Strongest Defensible Publication Claims (Supported by Evidence)
Based on verified artifacts, the strongest claims currently supported are:
1. **Transfer Learning Dominance:** Frozen PANNs CNN14 transfer learning dramatically outperforms training from scratch on emergency audio, boosting macro-F1 from 0.4430 (EXP-001) to 0.8962 (EXP-002) on limited sample budgets (N = 527).
2. **Exploratory Proof of Vulnerability:** Baseline emergency audio classifiers exhibit extreme vulnerability to acoustically similar non-emergency confusers (Test Set v2 confuser false emergency rate = 53.78% in E3).
3. **Exploratory Confuser Hierarchy:** Fireworks (+41.33 pp A-FER reduction) and human shouting (+36.00 pp A-FER reduction) are the primary drivers of false emergency alerts; adding them to training significantly reduces confuser false alarms on exploratory Test Set v2.

### 6.2. Claims the Current Evidence CANNOT Support
The following claims must NOT be made in the thesis or paper without new experimental evidence:
1. **No Confirmatory Claim on Test Set v2:** Test Set v2 was used repeatedly during model iteration, error diagnosis, and post-hoc ablation analysis. All Test Set v2 results are strictly **exploratory**.
2. **No General Causal Claim for Model-Error Guidance:** Because E2 (confuser negatives) is only compared against E3 (general background), the study has not ruled out the rival hypothesis that adding *any* shouting and fireworks audio produces the exact same benefit.
3. **No Claim of Generalizability Across Architectures:** All transfer experiments were conducted on a single frozen backbone (PANNs CNN14) with a single linear head. The protocol does not prove that error-guided selection benefits fine-tuned backbones, AST (Audio Spectrogram Transformer), or smaller edge-native architectures.
4. **No Claim of 'Zero Pretraining Leakage':** CNN14 was pretrained on 2 million AudioSet clips. Sourcing test audio from Freesound reduces known YouTube ID overlap, but cannot rule out latent acoustic exposure during pretraining.

---

## 7. Faculty Advisor Decision Checklist

Before Test Set v3 candidate harvesting (N = 2,400) or Phase 3 model training begins, formal written approval is required on the following 6 decisions:

```
====================================================================================================
FACULTY SUPERVISOR DECISION MEMORANDUM (PHASE 3 PROTOCOL FREEZE)
====================================================================================================

[ ] DECISION 1: Primary Confirmatory Scope
    Approve restricting primary confirmatory claims strictly to the pooled Gatekeeper sequence 
    (Gates 1 to 4 on N = 350 confusers, N = 350 benign, N = 700 emergency targets), with all 7 
    individual confuser subdomains (N = 50 each) formally classified as exploratory secondary endpoints.
    APPROVED [ ]    REVISE [ ]: ____________________________________________________________________

[ ] DECISION 2: Causal Control Arm (E5 Addition)
    Approve adding the E5 Matched Random Confuser Control Arm (N = 634: 422 baseline + 212 random 
    confusers matched by category) to isolate model-error selection from confuser category coverage.
    APPROVED [ ]    REVISE [ ]: ____________________________________________________________________

[ ] DECISION 3: Non-Inferiority Margins & Equivalence Margin
    Approve pre-registered margins:
    - Benign Background False Emergency Rate (Gate 2): delta_B = 0.020 (2.0%)
    - Emergency Macro-Recall (Gate 3): delta_R = 0.025 (2.5%)
    - Gate 1 Equivalence Margin (SESOI for TOST if Gate 1 fails): Delta_equiv = 0.030 (3.0%)
    APPROVED [ ]    REVISE [ ]: ____________________________________________________________________

[ ] DECISION 4: Reporting Mandate Across All Gates
    Approve the pre-registered reporting rule: Point estimates, standard errors, and 95% CIs for 
    ALL four gates will be fully calculated and published, even if testing halts at an early gate.
    APPROVED [ ]    REVISE [ ]: ____________________________________________________________________

[ ] DECISION 5: Multi-Tiered Open Benchmark Release Policy
    Approve releasing Test Set v3 on Zenodo under the 5-tier framework (CC0 annotations/code, CC-BY/NC 
    audio with machine-readable attribution manifest), with BBC audio permanently excluded.
    APPROVED [ ]    REVISE [ ]: ____________________________________________________________________

[ ] DECISION 6: Candidate Harvest Quota (2,400 Raw Clips)
    Authorize harvesting 2,400 raw candidates across Freesound and unreleased field audio to absorb 
    an estimated 30% human validation rejection rate and achieve N = 1,400 approved benchmark clips.
    APPROVED [ ]    REVISE [ ]: ____________________________________________________________________
====================================================================================================
```
