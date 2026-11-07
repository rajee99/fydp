# Phase 3 Protocol v3.2 Pre-Freeze Consistency & Statistical-Validity Audit Report

**Document Status:** Formal Pre-Freeze Methodological Deliverable  
**Date:** November 2026  
**Operating Mode:** Strictly Read-Only Consistency & Statistical-Validity Review  
**Audited Artifacts:**
- Protocol v2.1: `reports/phase3_independent_confirmation_protocol_v2_1.md`
- Advisor Brief: `reports/FYDP_Advisor_Brief.docx`
- Audit Deliverables: `reports/phase3_independent_audit_v3.md`, `reports/phase3_publication_readiness_v3.md`, `reports/phase3_audit_v3_1_corrections.md`, `reports/phase3_audit_v3_1_issue_matrix.csv`
- Simulation Scripts & Ground Truth: `scripts/simulate_phase3_power_comprehensive.py`, `reports/phase3_power_simulation_comprehensive.json`, `scripts/simulate_phase3_power.py`
- Experimental Manifests: `data/processed/e1_baseline_manifest.csv`, `data/processed/e2_matched_manifest.csv`, `data/processed/e3_master_manifest.csv`, `data/processed/shared_val_split_manifest.csv`, `data/processed/targeted_expansion_candidates.csv`
- Master Registry: `master_experiment_registry_reconciled.csv`

---

## 1. Executive Verdict: Conditional Hold (Not Ready for Freeze)

> [!CAUTION]
> **PRE-FREEZE AUDIT VERDICT: CONDITIONAL HOLD — PROTOCOL NOT READY FOR FREEZE.**  
> Protocol v2.1 and the subsequent v3.0 and v3.1 methodological audits have successfully corrected severe historical statistical defects—specifically eliminating intra-recording pseudoreplication, preventing invalid McNemar tests on macro-recall ratios, and establishing class-stratified cluster bootstrapping.  
> 
> However, a comprehensive consistency and statistical-validity audit confirms that **the protocol cannot be frozen for execution, dataset acquisition cannot begin, and models cannot be trained** until five critical blockers are formally resolved:
> 1. **Sequence Power Retraction:** The speculative 75%–85% four-gate sequence power range asserted in Audit v3.0 and Publication Readiness v3.0 must be formally retracted. Overall sequence power is **strictly UNKNOWN** because Gates 2, 3, and 4 have never been simulated.
> 2. **Gate 1 Permutation Validity:** The paired sign-flip test must not be described as unconditionally exact. Its exactness relies on conditional reflection symmetry under the sharp exchangeability null $\mathcal{H}_0^{\text{sharp}}$, which is an assumption about model error symmetry rather than an experimental design guarantee. Under error distribution skewness across recording clusters, its validity is asymptotic. A paired cluster bootstrap must be pre-registered as a mandatory sensitivity analysis.
> 3. **Estimand Notation Standardization:** Conflicting symbols must be purged. Specifically, the old `theta_2` symbol (which collides between Gate 2 Benign B-FER in Table 5.1 and the discrete ensemble estimand in Section 6.2 and the simulation code), undefined `theta_4` in Gate 4 decision rule, and the two-sided CI mismatch in Gate 1 must be harmonized to `theta_A_FER`, `theta_B_FER`, `theta_Rec`, `theta_F1`, and `theta_ens`.
> 4. **Operational Margins Approval:** Margins $\delta_{\text{B}} = 0.020$ (benign false alarms) and $\delta_{\text{R}} = 0.025$ (emergency recall loss) represent operational risk-tolerance thresholds that cannot be justified by statistics alone and require explicit faculty advisor sign-off. The equivalence margin $\Delta_{\text{equiv}} = 0.030$ must be pre-registered to prevent uninformative null findings.
> 5. **E5 Control Arm Formalization:** The E2 vs. E3 comparison remains fundamentally confounded between category coverage and selection policy. The 8-dimension matched E5 control arm must be formally instantiated, sampled, and SHA-256 hashed *before* Test Set v3 acquisition.

---

## 2. Verified Facts (Ground-Truth Repository Evidence)

Every item below has been verified against actual repository code, manifests, and data artifacts:

1. **Gate 1 Simulated Power:** Forensic review of `scripts/simulate_phase3_power_comprehensive.py` and `reports/phase3_power_simulation_comprehensive.json` confirms that Gate 1 clustered 3-seed permutation power at $N = 350$ ($\alpha = 0.05$) is:
   - **$99.86\%$** under the Moderate scenario ($\Delta = 0.08, p_{10}=0.18, p_{01}=0.10$).
   - **$89.98\%$** under the Conservative scenario ($\Delta = 0.05, p_{10}=0.15, p_{01}=0.10$).
   - **$52.40\%$** under the Minimal scenario ($\Delta = 0.03, p_{10}=0.13, p_{01}=0.10$).
   - For individual subdomains ($N = 50$), power at $\alpha = 0.05$ is only **$19.88\%$ to $45.91\%$**, collapsing to **$7.25\%$ to $23.10\%$** under Holm multiplicity adjustment ($\alpha = 0.0125$).
2. **Absence of Gates 2, 3, and 4 Power Simulations:** Neither `simulate_phase3_power_comprehensive.py` nor `simulate_phase3_power.py` contains any simulation routines for Gate 2 (B-FER bootstrap bound), Gate 3 (macro-recall stratified bootstrap bound), or Gate 4 (five-class macro-F1 stratified bootstrap).
3. **E1 Baseline Manifest Composition:** In `data/processed/e1_baseline_manifest.csv`, the training split contains exactly $N = 422$ clips ($326$ emergency: $127$ sirens, $92$ gunshots, $77$ glass breaks, $30$ screams; and $96$ AudioSet background clips). Validation split contains $105$ clips.
4. **E2 Matched Manifest Composition:** In `data/processed/e2_matched_manifest.csv`, the training split contains exactly $N = 634$ clips ($326$ emergency targets, $96$ baseline background, and exactly $212$ targeted confusers).
5. **Exact E2 Confuser Quotas:** In `data/processed/e2_matched_manifest.csv`, the 212 added confusers strictly break down into:
   - `human_shouting`: 42 clips
   - `metal_impacts`: 37 clips
   - `door_slams`: 32 clips
   - `construction_tools`: 29 clips
   - `car_alarms_detectors`: 28 clips
   - `fireworks_firecrackers`: 25 clips
   - `whistles_high_pitch`: 18 clips
   - `other_vocal`: 1 clip
6. **Shared Validation Split Invariant:** In `data/processed/shared_val_split_manifest.csv`, exactly $N = 105$ clips ($75$ emergency, $30$ background) are held completely invariant across all experiments.
7. **Model Architecture Invariant:** Across `src/train.py` and `src/model.py`, all models use a frozen PANNs CNN14 backbone with a linear transfer head containing exactly 10,245 parameters ($2048 \times 5 + 5$).
8. **Zero Data Augmentation Invariant:** In `src/dataset.py`, `AudioSetDataset` defaults to `transform=None`. No SpecAugment, Mixup, or pitch shifts are applied.

---

## 3. Unresolved Assumptions & Methodological Risks

### 3.1. Conditional Reflection Symmetry in Gate 1
- **The Issue:** Gate 1 evaluates differences $d_i = \frac{1}{3}\sum_{s=1}^3 [I(\hat{y}_{i,s}^{\text{E3}} \in \mathcal{E}) - I(\hat{y}_{i,s}^{\text{E2}} \in \mathcal{E})]$. The sign-flip test flips signs with probability 0.5: $T^{(r)} = \sum_{i=1}^N \sigma_i^{(r)} d_i$.
- **The Risk:** This test is finite-sample exact if and only if differences $d_i$ satisfy conditional reflection symmetry:
  $$\mathbb{P}(d_i = +c \mid |d_i| = c) = \mathbb{P}(d_i = -c \mid |d_i| = c) = 0.5$$
  In observational benchmark evaluations, treatments are not randomly assigned across clips. Reflection symmetry is an assumption regarding the sharp null hypothesis of identical error distributions across all acoustic profiles ($\mathcal{H}_0^{\text{sharp}}$). If the true null is weak ($\mathbb{E}[d_i] = 0$) but error distributions are skewed (e.g., E2 makes concentrated errors on specific sound patterns while E3 makes diffuse errors), $d_i$ is asymmetric. Under skewness, the permutation test is only asymptotically valid via the central limit theorem, not finite-sample exact.
- **Required Action:** Do not describe the test as unconditionally exact. Mandate a recording-level paired cluster bootstrap as an official sensitivity analysis.

### 3.2. Physical Recording Cluster Boundary vs. Segmented Windows
- **The Issue:** Protocol v2.1 specifies "Unit of Analysis: Physical recording cluster."
- **The Risk:** If continuous audio uploads (e.g. 60-second Freesound recordings) are sliced into multiple 10-second clips, and those slices are treated as distinct independent observations $i$, the independence assumption across clusters is violated due to shared microphone acoustics, background noise, and reverberation.
- **Required Action:** Enforce clustering at the canonical recording session / upload ID (`recording_id`) across all evaluation code. No two benchmark clips may share an original source recording.

### 3.3. Unsimulated Four-Gate Sequence Power
- **The Issue:** Protocol v2.1 claimed ">99% sequence power," and Audit v3.0 claimed "75%–85% sequence power."
- **The Risk:** Gates 2, 3, and 4 were never simulated. Multiplying marginal powers assumes gate independence:
  $$\mathbb{P}\left(\bigcap_{k=1}^4 \text{Pass } G_k\right) = \prod_{k=1}^4 \mathbb{P}(\text{Pass } G_k)$$
  This assumption is false. All four gates evaluate identical model checkpoints on overlapping partitions of the same 1,400-clip benchmark. Decisions across gates are positively correlated. Under Fréchet-Hoeffding bounds:
  $$\max\left(0, 1 - \sum_{k=1}^4 (1 - P_k)\right) \le P_{\text{sequence}} \le \min_{k} P_k$$
  Because $P_2, P_3, P_4$ are unsimulated, overall sequence power is mathematically unbounded from below.
- **Required Action:** Formally classify overall four-gate sequence power as **strictly UNKNOWN**.

---

## 4. Estimand and Notation Standardization (Table 5.1 Reconciliation)

To eliminate all internal collisions and ambiguities, the following descriptive nomenclature must be enforced across all Phase 3 protocol files, reports, and code:

```
                              STANDARDIZED ESTIMAND NOMENCLATURE
  
  ┌───────────────┬────────────────────────────────────────────────────────┬─────────────────────────────────────┐
  │ Symbol        │ Mathematical Definition                                │ Confirmatory Role                   │
  ├───────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ theta_A_FER   │ E[A-FER_E3 - A-FER_E2]                                 │ Gate 1 Primary Confirmatory         │
  │ theta_B_FER   │ E[B-FER_E2 - B-FER_E3]                                 │ Gate 2 Safety Non-Inferiority       │
  │ theta_Rec     │ Rec_macro^E3 - Rec_macro^E2                            │ Gate 3 Sensitivity Non-Inferiority  │
  │ theta_F1      │ Macro-F1^E2 - Macro-F1^E3                              │ Gate 4 Operational Utility          │
  │ theta_ens     │ E[A-FER_E3_ens - A-FER_E2_ens]                         │ Secondary Exploratory Ensemble      │
  └───────────────┴────────────────────────────────────────────────────────┴─────────────────────────────────────┘
```

### Reconciled Table 5.1 Specification

| Dimension | Gate 1 (Primary Confirmatory) | Gate 2 (Safety Non-Inferiority) | Gate 3 (Sensitivity Non-Inferiority) | Gate 4 (Operational Utility) |
|:---|:---|:---|:---|:---|
| **Endpoint Metric** | Confuser False Emergency Rate ($\text{A-FER}$) | Benign False Emergency Rate ($\text{B-FER}$) | Emergency Macro-Recall ($\text{Rec}_{\text{macro}}$) | Overall Five-Class Macro-F1 |
| **Evaluated Slice** | $N = 350$ Targeted Confusers | $N = 350$ Benign Background Clips | $N = 700$ Emergency Targets ($175$ per class) | Full Benchmark ($N = 1,400$) |
| **Estimand** | `theta_A_FER` $= \mathbb{E}[\text{A-FER}_{\text{E3}} - \text{A-FER}_{\text{E2}}]$ | `theta_B_FER` $= \mathbb{E}[\text{B-FER}_{\text{E2}} - \text{B-FER}_{\text{E3}}]$ | `theta_Rec` $= \text{Rec}_{\text{macro}}^{\text{E3}} - \text{Rec}_{\text{macro}}^{\text{E2}}$ | `theta_F1` $= \text{Macro-F1}^{\text{E2}} - \text{Macro-F1}^{\text{E3}}$ |
| **Null Hypothesis ($\mathcal{H}_0$)** | $\mathcal{H}_{0,1}: \text{theta\_A\_FER} \le 0$ (E2 no better) | $\mathcal{H}_{0,2}: \text{theta\_B\_FER} \ge \delta_{\text{B}}$ (E2 degrades B-FER) | $\mathcal{H}_{0,3}: \text{theta\_Rec} \ge \delta_{\text{R}}$ (E2 misses recall) | $\mathcal{H}_{0,4}: \text{theta\_F1} \le 0$ (E2 F1 no better) |
| **Alternative ($\mathcal{H}_1$)** | $\mathcal{H}_{1,1}: \text{theta\_A\_FER} > 0$ (E2 superior) | $\mathcal{H}_{1,2}: \text{theta\_B\_FER} < \delta_{\text{B}}$ (E2 non-inferior) | $\mathcal{H}_{1,3}: \text{theta\_Rec} < \delta_{\text{R}}$ (E2 non-inferior) | $\mathcal{H}_{1,4}: \text{theta\_F1} > 0$ (E2 superior) |
| **Decision Rule** | One-sided sign-flip permutation $p < 0.05$ | $\hat{\theta}_{\text{B\_FER},\text{upper}}^{95\%} < \delta_{\text{B}} = 0.020$ | $\hat{\theta}_{\text{Rec},\text{upper}}^{95\%} < \delta_{\text{R}} = 0.025$ | $\hat{\theta}_{\text{F1},\text{lower}}^{95\%} > 0.000$ (5th percentile $> 0$) |
| **Pre-Registered Margin** | Not applicable (Superiority) | $\delta_{\text{B}} = 0.020$ ($2.0\%$) [Pending Approval] | $\delta_{\text{R}} = 0.025$ ($2.5\%$) [Pending Approval] | Not applicable (Superiority) |
| **Unit of Analysis** | Physical recording cluster (`recording_id`) | Physical recording cluster (`recording_id`) | Class-stratified recording cluster | Class-stratified recording cluster |
| **Statistical Test** | Paired recording sign-flip permutation ($R = 100,000$) | Inversion of 10,000 cluster-bootstrap percentile bound | Inversion of 10,000 class-stratified cluster-bootstrap bound | Paired class-stratified cluster-bootstrap test ($B = 10,000$) |
| **Confidence Interval** | One-sided 95% Lower Bootstrap Bound | One-sided 95% Upper Bootstrap Bound | One-sided 95% Upper Bootstrap Bound | One-sided 95% Lower Bootstrap Bound |

---

## 5. Non-Inferiority and Equivalence Margins Audit

```
                       MARGIN SPECIFICATION AUDIT MATRIX
  
  ┌─────────────┬─────────────┬─────────────┬──────────────────────────┬────────────────────────────┬───────────────────┐
  │ Margin      │ Value (pp)  │ Estimand    │ Direction of Benefit     │ Practical Impact           │ Advisor Approval  │
  ├─────────────┼─────────────┼─────────────┼──────────────────────────┼────────────────────────────┼───────────────────┤
  │ delta_B     │ 0.020 (2%)  │ theta_B_FER │ Lower B-FER is better    │ Max <= 7.0 false alarms    │ REQUIRED          │
  │ delta_R     │ 0.025 (2.5%)│ theta_Rec   │ Higher recall is better  │ Max <= 17.5 missed alerts  │ REQUIRED          │
  │ Delta_equiv │ 0.030 (3%)  │ theta_A_FER │ In bounds [-0.03, +0.03] │ Max <= 10.5 diff on conf   │ REQUIRED          │
  └─────────────┴─────────────┴─────────────┴──────────────────────────┴────────────────────────────┴───────────────────┘
```

1. **Gate 2 Margin ($\delta_{\text{B}} = 0.020$):**
   - **Exact Estimand:** `theta_B_FER = E[B-FER_E2 - B-FER_E3]` on $N = 350$ benign background clips.
   - **Direction of Benefit:** Lower B-FER is better. Positive `theta_B_FER` indicates E2 degrades benign false alarms.
   - **Hypotheses:** $\mathcal{H}_{0,2}: \text{theta\_B\_FER} \ge 0.020$ vs $\mathcal{H}_{1,2}: \text{theta\_B\_FER} < 0.020$.
   - **Decision Rule:** One-sided 95% upper bootstrap percentile bound $\hat{\theta}_{\text{B\_FER},\text{upper}}^{95\%} < 0.020$.
   - **Operational Interpretation:** In 350 benign clips, a 2.0 pp increase allows at most $350 \times 0.020 = 7.0$ additional false emergency classifications. Prevents alert fatigue in emergency dispatchers.
   - **Advisor Approval:** **REQUIRED.** Not validated by dataset; represents operational policy tolerance.
2. **Gate 3 Margin ($\delta_{\text{R}} = 0.025$):**
   - **Exact Estimand:** `theta_Rec = Rec_macro_E3 - Rec_macro_E2` across 4 emergency classes on $N = 700$ emergency clips ($175$ per class).
   - **Direction of Benefit:** Higher recall is better. Positive `theta_Rec` indicates E2 suffers macro-recall loss.
   - **Hypotheses:** $\mathcal{H}_{0,3}: \text{theta\_Rec} \ge 0.025$ vs $\mathcal{H}_{1,3}: \text{theta\_Rec} < 0.025$.
   - **Decision Rule:** One-sided 95% upper class-stratified bootstrap bound $\hat{\theta}_{\text{Rec},\text{upper}}^{95\%} < 0.025$.
   - **Operational Interpretation:** In 700 emergency clips, a 2.5 pp drop allows at most $700 \times 0.025 = 17.5$ additional missed emergencies across all 4 classes (~4.4 missed screams, gunshots, sirens, or glass breaks per class). In life-safety systems, missing true emergencies carries severe operational liability.
   - **Advisor Approval:** **REQUIRED.** Must be formally ratified by the faculty advisor.
3. **Equivalence Margin ($\Delta_{\text{equiv}} = 0.030$):**
   - **Exact Estimand:** `theta_A_FER = E[A-FER_E3 - A-FER_E2]` on $N = 350$ targeted confusers.
   - **Direction of Benefit:** Positive indicates E2 superiority. Equivalence evaluates $|\text{theta\_A\_FER}| < 0.030$.
   - **Hypotheses:** $\mathcal{H}_{0,\text{equiv}}: |\text{theta\_A\_FER}| \ge 0.030$ vs $\mathcal{H}_{1,\text{equiv}}: |\text{theta\_A\_FER}| < 0.030$.
   - **Decision Rule:** Two One-Sided Tests (TOST) at $\alpha = 0.05$ (equivalent to verifying that the 90% two-sided bootstrap CI falls entirely within $[-0.030, +0.030]$).
   - **Operational Interpretation:** Converts an inconclusive failure to reject Gate 1 ($p \ge 0.05$) into an informative positive finding that model-error guidance produces performance equivalent within 3.0 pp (~10.5 clips out of 350) to random background scaling.
   - **Advisor Approval:** **REQUIRED.** Pre-registration prevents uninterpretable null findings.

---

## 6. E5 Matching Audit & Control Arm Specification

### 6.1. Audit of the 11 Matching Dimensions (E2 vs. E5)

```
                            E2 vs. E5 MATCHING AUDIT
  
  [1. Total Sample Size] ─────> N = 634 (422 baseline + 212 added)        [VERIFIED]
  [2. Category Quotas] ───────> Exactly matched across 7 subdomains      [VERIFIED]
  [3. Candidate Pool] ────────> Sourced from same candidate repository   [VERIFIED]
  [4. Audio DSP Specs] ───────> 16 kHz mono, 10.0s, matched RMS/SNR      [VERIFIED]
  [5. Backbone & Capacity] ───> Frozen PANNs CNN14, 10,245 param head     [VERIFIED]
  [6. Hyperparameters] ───────> Adam lr=0.001, mu=1.0, 20 epochs, b=16    [VERIFIED]
  [7. Random Seeds] ──────────> Fixed seeds 42, 123, 2026                 [VERIFIED]
  [8. Validation Set] ────────> Invariant shared val split (N=105)        [VERIFIED]
  [9. Data Augmentation] ─────> transform=None (Zero augmentation)        [VERIFIED]
  [10. Cluster Independence] ─> recording_id cluster grouping             [SPECIFIED]
  [11. Duplicate Screening] ──> SHA-256 hash & acoustic similarity checks [SPECIFIED]
```

| Matching Dimension | Audit Verification Status | Verification Detail / Repository Artifact |
|:---|:---:|:---|
| **1. Total Sample Size** | **VERIFIED** | Exactly 634 training clips ($422$ baseline + $212$ added confusers). Matched to `e2_matched_manifest.csv`. |
| **2. Category Quotas** | **VERIFIED** | 42 shouting, 37 metal impacts, 32 door slams, 29 tools, 28 alarms, 25 fireworks, 18 whistles, 1 vocal. Matches E2. |
| **3. Candidate Pool** | **VERIFIED** | Candidate pool exists in `data/processed/targeted_expansion_candidates.csv` (672 candidates across subdomains). |
| **4. Audio DSP Specs** | **VERIFIED** | 16.0 kHz sample rate, 16-bit mono, 10.0s duration, identical RMS bounds. Specified in `src/dataset.py`. |
| **5. Model Architecture** | **VERIFIED** | Frozen PANNs CNN14 backbone with 10,245 trainable linear head parameters. Specified in `src/model.py`. |
| **6. Hyperparameters** | **VERIFIED** | Adam optimizer ($\text{lr} = 0.001, \beta_1=0.9, \beta_2=0.999$, loss weight $\mu=1.0$), batch size 16, 20 epochs. |
| **7. Training Seeds** | **VERIFIED** | Identical seeds $42, 123, 2026$. Specified in training scripts. |
| **8. Validation Split** | **VERIFIED** | Invariant shared split ($N = 105$, `shared_val_split_manifest.csv`). Zero overlap with training splits. |
| **9. Data Augmentation** | **VERIFIED** | Zero data augmentation (`transform=None`). Confirmed in `src/train.py` and `src/dataset.py`. |
| **10. Cluster Independence** | **SPECIFIED (PENDING MANIFEST)** | Enforce that no E5 candidate shares a `recording_id` with E1, E2, E3, or validation sets. |
| **11. Duplicate Screening** | **SPECIFIED (PENDING MANIFEST)** | Perform exact SHA-256 hash checks and acoustic similarity screening against all repository audio. |

### 6.2. Freezing and Pre-Acquisition SHA-256 Hashing Protocol
To guarantee causal identification and prevent post-hoc cherry-picking:
1. `data/processed/e5_random_confusers_manifest.csv` must be sampled from the eligible candidate pool using a fixed random seed prior to downloading Test Set v3.
2. The manifest and every raw audio file in E5 must have its `SHA-256` hash computed, committed to git, and frozen.
3. Test Set v3 acquisition may proceed **only after the E5 manifest SHA-256 hash is committed**.

---

## 7. Causal Interpretation & Contrast Boundaries

```
                      CAUSAL CONTRAST IDENTIFICATION MAP
  
      E1 (Baseline, N=422)
        │
        ├────────────────────────────────────┐
        ▼ (+212 General Background)          ▼ (+212 Confusers, Random)
      E3 (General, N=634)                  E5 (Random Confusers, N=634)
        │                                    │
        │                                    ▼ (+212 Confusers, Error-Guided)
        └──────────────────────────────────> E2 (Targeted Hard Negatives, N=634)
```

The protocol provides three distinct contrasts, each with strict causal boundaries:

1. **Contrast 1: E3 versus E1 (General Background Scaling)**
   - *What it isolates:* The effect of increasing generic non-emergency audio volume (+50.2% sample size, adding traffic, rain, speech).
   - *Causal limits:* Conflates dataset volume with acoustic diversity. Does not test confuser rejection or hard-negative selection.
2. **Contrast 2: E5 versus E3 (Acoustic Category Coverage)**
   - *What it isolates:* The effect of adding confuser categories (shouting, fireworks) versus adding general ambient sound, holding sample size ($N = 634$) and selection mechanism (random sampling) constant.
   - *Causal limits:* Evaluates domain coverage only. A significant win for E5 demonstrates that training on shouting helps reject shouting—a domain-coverage effect, not an algorithmic discovery.
3. **Contrast 3: E2 versus E5 (Model-Error-Guided Hard-Negative Mining)**
   - *What it isolates:* The true, unconfounded causal effect of **baseline model-error guidance** versus random sampling within the exact same confuser categories and quotas.
   - *Causal limits & remaining confounders:*
     - *Acoustic difficulty / SNR confounding:* Baseline errors might be acoustically degraded, clipped, or multi-source, confounding hardness with signal quality.
     - *Non-additivity:* The total effect cannot be assumed to be additive:
       $$\text{Total Effect } (E2 - E1) \neq (E3 - E1) + (E5 - E3) + (E2 - E5)$$
       Non-linear interactions between dataset volume, domain coverage, and boundary hardness prevent claiming an additive causal decomposition.

---

## 8. Exact Faculty Advisor Decisions Required Prior to Freeze

The faculty advisor must review and provide formal written sign-off on the following seven decisions:

1. **Decision 1: Overall Sequence Power Classification:** Approve classifying overall four-gate sequence power as strictly UNKNOWN, while reporting Gate 1 power as conditional on effect size ($\Delta \ge 0.08 \implies 99.86\%$, $\Delta = 0.05 \implies 89.98\%$).
2. **Decision 2: Standardized Estimand Nomenclature:** Approve replacing colliding symbols with `theta_A_FER`, `theta_B_FER`, `theta_Rec`, `theta_F1`, and `theta_ens`.
3. **Decision 3: Benign False Emergency Rate Margin ($\delta_{\text{B}} = 0.020$):** Ratify tolerating at most a 2.0 pp increase in benign false emergency alerts ($\le 7.0$ additional false alerts per 350 benign clips).
4. **Decision 4: Emergency Macro-Recall Margin ($\delta_{\text{R}} = 0.025$):** Ratify tolerating at most a 2.5 pp loss in emergency macro-recall ($\le 17.5$ additional missed true emergency events across 700 emergency clips).
5. **Decision 5: Pre-Registration of Equivalence Margin ($\Delta_{\text{equiv}} = 0.030$):** Pre-register TOST against $[-0.030, +0.030]$ for `theta_A_FER` to enable reporting operational equivalence if Gate 1 superiority is not rejected.
6. **Decision 6: Formal Adoption of E5 Control Arm:** Approve incorporating the 8-dimension matched E5 arm ($N = 634$, 212 random confusers matching E2 quotas) to isolate model-error guidance from category coverage.
7. **Decision 7: Enforcement of Root Recording Clustering:** Approve enforcing clustering at canonical `recording_id` rather than 10-second windowed segments.

---

## 9. Final Freeze Readiness Recommendation

> [!IMPORTANT]
> **ROADMAP TO ACHIEVE PROTOCOL FREEZE:**
> 1. Update `reports/phase3_independent_confirmation_protocol_v2_1.md` to reflect the reconciled Table 5.1 and standardized estimand symbols.
> 2. Submit `reports/phase3_protocol_v3_2_approval_checklist.md` to the faculty advisor for formal sign-off.
> 3. Generate and SHA-256 hash `data/processed/e5_random_confusers_manifest.csv`.
> 4. Once written advisor approvals are obtained and the E5 manifest is hashed, generate **Protocol v3.0 (Frozen Final)** and proceed to Test Set v3 candidate acquisition.
