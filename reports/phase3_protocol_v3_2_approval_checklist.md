# Phase 3 Protocol v3.2 Pre-Freeze Approval Checklist

**Document Status:** Formal Protocol Sign-Off & Governance Deliverable  
**Date:** November 2026  
**Audited Artifacts:** Protocol v2.1, Advisor Brief, Audit v3.0, Audit v3.1, Simulation Code & Registry Manifests  
**Operating Mode:** Strictly Read-Only Consistency & Statistical Validity Review  

---

## Structure of the Checklist

This checklist categorizes every methodological and experimental dimension into five distinct governance statuses:
1. **Technically Verified:** Empirically verified against actual codebase artifacts, simulation runs, manifests, or checkpoints.
2. **Documentation Correction Required:** Discrepancy or symbol collision identified across text files requiring immediate harmonization before document freeze.
3. **Statistical Justification Required:** Theoretical or distributional assumption requiring explicit formal qualification and sensitivity analysis specification.
4. **Advisor Approval Required:** Policy, risk-tolerance, or scientific scope decision requiring formal written approval from the faculty advisor.
5. **Blocker Before Protocol Freeze:** Critical unresolved issue that strictly halts protocol freeze, dataset acquisition, and preregistration.

---

## 1. Technically Verified Items

- [x] **VERIF-01: Gate 1 Simulation Results:** Verified in `reports/phase3_power_simulation_comprehensive.json`. Clustered 3-seed permutation power at $N = 350$ ($\alpha = 0.05$) is:
  - $99.86\%$ at Moderate effect ($\Delta = 0.08, p_{10}=0.18, p_{01}=0.10$).
  - $89.98\%$ at Conservative effect ($\Delta = 0.05, p_{10}=0.15, p_{01}=0.10$).
  - $52.40\%$ at Minimal effect ($\Delta = 0.03, p_{10}=0.13, p_{01}=0.10$).
  - Individual subdomains ($N = 50$) have only $19.88\%$ to $45.91\%$ power at $\alpha = 0.05$ (collapsing to $7.25\%$ to $23.10\%$ under Holm multiplicity adjustment).
- [x] **VERIF-02: Absence of Simulation for Gates 2, 3, and 4:** Forensic inspection of `scripts/simulate_phase3_power_comprehensive.py` and `scripts/simulate_phase3_power.py` confirms that zero simulation code exists for bootstrap non-inferiority bounds (Gates 2 and 3) or macro-F1 class-stratified bootstrapping (Gate 4).
- [x] **VERIF-03: E1 Baseline Manifest Composition:** Verified in `data/processed/e1_baseline_manifest.csv`. Training split contains exactly $N = 422$ clips ($326$ emergency targets: $127$ sirens, $92$ gunshots, $77$ glass breaks, $30$ screams; and $96$ generic AudioSet background clips). Validation split contains $105$ clips.
- [x] **VERIF-04: E2 Matched Manifest Composition:** Verified in `data/processed/e2_matched_manifest.csv`. Training split contains exactly $N = 634$ clips ($326$ emergency targets identical to E1, $96$ baseline background, and exactly $212$ added targeted confusers across 7 acoustic subdomains).
- [x] **VERIF-05: Confuser Subdomain Quotas in E2:** Verified in `data/processed/e2_matched_manifest.csv`. The 212 added confusers strictly break down as:
  - `human_shouting`: 42 clips
  - `metal_impacts`: 37 clips
  - `door_slams`: 32 clips
  - `construction_tools`: 29 clips
  - `car_alarms_detectors`: 28 clips
  - `fireworks_firecrackers`: 25 clips
  - `whistles_high_pitch`: 18 clips
  - `other_vocal`: 1 clip
- [x] **VERIF-06: Shared Invariant Validation Set:** Verified in `data/processed/shared_val_split_manifest.csv`. Exactly $N = 105$ clips ($75$ emergency targets, $30$ background), held completely invariant across E1, E2, E3, and proposed E5.
- [x] **VERIF-07: Invariant Backbone Architecture:** Verified in `src/train.py` and `src/model.py`. All experimental arms utilize a frozen PANNs CNN14 backbone with a linear transfer head containing exactly 10,245 trainable parameters ($2048 \times 5 + 5$).
- [x] **VERIF-08: Zero Data Augmentation Invariant:** Verified in `src/train.py` and `src/dataset.py`. `AudioSetDataset` defaults to `transform=None`, confirming that neither SpecAugment nor Mixup was applied during training. E5 must maintain this zero-augmentation invariant.
- [x] **VERIF-09: Multi-Seed Training Evaluation:** Verified in training configurations. All models are trained across 3 fixed random seeds ($42, 123, 2026$) and evaluated by averaging predictions at the recording cluster level.

---

## 2. Documentation Corrections Required

- [ ] **DOC-01: Disambiguate `theta_2` in Protocol v2.1:** In Protocol v2.1 Table 5.1, replace `theta_2` with `theta_B_FER` for Gate 2 Benign B-FER non-inferiority.
- [ ] **DOC-02: Reserve `theta_ens` for Secondary Ensemble Estimand:** In Protocol v2.1 Section 6.2 item 2 and Advisor Brief Section 7, replace `theta_2` with `theta_ens` for the 3-model majority-vote ensemble evaluated via McNemar's test.
- [ ] **DOC-03: Harmonize Gate 4 Notation:** In Protocol v2.1 Table 5.1 and Section 5.2, replace undefined `theta_4` with `theta_F1` to match Estimand row ($\Delta \text{Macro-F1}$).
- [ ] **DOC-04: Harmonize Gate 1 Confidence Bound:** In Protocol v2.1 Table 5.1 (Confidence Interval row), replace "Two-sided 95% Cluster-Bootstrap CI" with "One-sided 95% Lower Bootstrap Bound" (or two-sided 90% CI) to match the one-sided $\alpha = 0.05$ decision rule.
- [ ] **DOC-05: Retract Heuristic 75%–85% Sequence Power Estimate:** Formally strike the speculative 75%–85% overall sequence power claim from Audit v3.0 Section 3.4 and Publication Readiness v3.0 Section 4.1.
- [ ] **DOC-06: Harmonize Advisor Brief Text:** Update `FYDP_Advisor_Brief.docx` (Section 7, paras 131, 137, 138) to reflect Protocol v2.1: Gate 2 = Benign B-FER ($\delta_{\text{B}}$), Gate 3 = Emergency Macro-Recall ($\delta_{\text{R}}$), with McNemar restricted strictly to $\theta_{\text{ens}}$.

---

## 3. Statistical Justification Required

- [ ] **STAT-01: Explicit Qualification of Gate 1 Sign-Flip Exactness:** Document in Protocol Section 5.1 that the paired sign-flip permutation test is exact only under the sharp exchangeability null $\mathcal{H}_0^{\text{sharp}}: (y_i^{\text{E2}}, y_i^{\text{E3}}) \stackrel{d}{=} (y_i^{\text{E3}}, y_i^{\text{E2}})$. Clarify that in non-randomized benchmark evaluation, error distribution skewness across recording clusters makes the test asymptotically valid rather than unconditionally exact.
- [ ] **STAT-02: Pre-Register Recording-Level Paired Cluster Bootstrap Sensitivity:** Mandate a paired cluster bootstrap on $\theta_{\text{A\_FER}}$ as an official sensitivity analysis alongside the sign-flip test to verify robust Type I error control under potential error skewness.
- [ ] **STAT-03: Classification of Sequence Power as Strictly UNKNOWN:** Document in Protocol Section 1.2 and Section 7.1 that overall four-gate sequence power is formally UNKNOWN until a joint multi-gate Monte Carlo simulation modeling correlated model decisions across all 1,400 benchmark audio clips is executed.

---

## 4. Advisor Approval Required

- [ ] **ADV-01: Operational Non-Inferiority Margin for Benign Audio ($\delta_{\text{B}} = 0.020$):** Advisor sign-off on tolerating up to a 2.0 percentage point increase in false emergency alarms on benign audio ($\le 7.0$ additional false dispatches per 350 benign clips).
- [ ] **ADV-02: Operational Non-Inferiority Margin for Emergency Recall ($\delta_{\text{R}} = 0.025$):** Advisor sign-off on tolerating up to a 2.5 percentage point drop in emergency macro-recall ($\le 17.5$ additional missed true emergency events across 700 emergency clips).
- [ ] **ADV-03: Pre-Registration of Equivalence Margin ($\Delta_{\text{equiv}} = 0.030$):** Advisor sign-off on pre-registering Two One-Sided Tests (TOST) against $[-0.030, +0.030]$ for $\theta_{\text{A\_FER}}$ to enable publishing an informative finding of operational equivalence if Gate 1 superiority is not rejected.
- [ ] **ADV-04: Inclusion of E5 Random-Confuser Control Arm:** Advisor approval to incorporate the 8-dimension matched E5 arm into the formal study design to eliminate the category-coverage confound.
- [ ] **ADV-05: Enforcement of Canonical Physical Recording Clustering:** Advisor approval to enforce clustering by canonical recording session / Freesound upload ID (`recording_id`) rather than 10-second windowed segments.

---

## 5. Critical Blockers Before Protocol Freeze

The following five items represent **strict hard blockers**. Protocol v2.1 cannot be frozen, Test Set v3 cannot be acquired, and experimental training cannot commence until all five are formally resolved:

| Blocker ID | Blocker Description | Required Action to Clear | Target Document |
|:---|:---|:---|:---|
| **BLOCKER-01** | **Causal Identification Confound in E2 vs. E3** | Incorporate the 8-dimension matched E5 control arm ($N = 634$, 212 random confusers matching E2 category quotas) into Protocol Section 2. | `protocol_v2_1.md` Section 2 |
| **BLOCKER-02** | **Unjustified Operational Margins** | Obtain written faculty advisor sign-off for $\delta_{\text{B}} = 0.020$ and $\delta_{\text{R}} = 0.025$. | Advisor Brief Section 10 |
| **BLOCKER-03** | **Unsupported Sequence Power Claims** | Retract ">99% sequence power" and "75%–85% sequence power" claims; classify sequence power as UNKNOWN in all public protocol text. | `protocol_v2_1.md` Section 1.2 |
| **BLOCKER-04** | **Estimand Notation Collision** | Purge `theta_2` collision and undefined `theta_4`; enforce `theta_A_FER`, `theta_B_FER`, `theta_Rec`, `theta_F1`, and `theta_ens`. | `protocol_v2_1.md` Table 5.1 |
| **BLOCKER-05** | **E5 Manifest Instantiation and Pre-Acquisition Hashing** | Generate `data/processed/e5_matched_manifest.csv` from eligible candidate pools, verify zero overlap with E1/E2/E3, compute its `SHA-256` hash, and freeze it *strictly before* Test Set v3 audio acquisition. | Repository Data Manifest |

---

## 6. Freeze Readiness Recommendation

> [!CAUTION]
> **VERDICT: CONDITIONAL HOLD — NOT READY FOR FREEZE.**  
> The Phase 3 confirmatory protocol is in an advanced, highly refined state. However, the protocol **must NOT be frozen for execution** until the faculty advisor signs off on the five critical blockers listed above.
