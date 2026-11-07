# Phase 3 Publication-Readiness Assessment & Scientific Scope Report (v3.0)

**Project:** `D:\ml\fydp\MyNet`  
**Research Framing:** *Data-Centric Analysis of Hard-Negative Learning and Domain Shift in Emergency Audio Classification*  
**Date:** November 3, 2026  
**Auditor Role:** Independent Senior Machine Learning & Biostatistical Methodology Auditor  
**Primary Deliverable:** Definitive Publication-Readiness Assessment & Epistemic Scope Specification  
**Companion Documents:**
- [`reports/phase3_independent_audit_v3.md`](file:///d:/ml/fydp/MyNet/reports/phase3_independent_audit_v3.md) *(Comprehensive Independent Audit Report)*
- [`reports/phase3_document_conflict_matrix_v3.csv`](file:///d:/ml/fydp/MyNet/reports/phase3_document_conflict_matrix_v3.csv) *(15-Dimension Conflict Matrix)*
- [`reports/FYDP_Advisor_Brief.docx`](file:///d:/ml/fydp/MyNet/reports/FYDP_Advisor_Brief.docx) *(Advisor Discussion Memo)*
- [`master_experiment_registry_reconciled.csv`](file:///d:/ml/fydp/MyNet/master_experiment_registry_reconciled.csv) *(Master Experiment Registry)*

---

## 1. Executive Framing & Target Venue Fit

The goal of this research is to evaluate whether model-error-guided hard-negative data selection improves the reliability of edge emergency audio classifiers more effectively than conventional background data expansion.

### 1.1. Recommended Target Venues
Based on the data-centric methodology, pre-registered gatekeeper design, and empirical focus, the optimal publication targets are:
1. **DCASE Workshop / Challenge (Workshop on Detection and Classification of Acoustic Scenes and Events):** Premier venue for environmental audio; strongly values data-centric insights, acoustic confuser benchmarks, and nuisance alert suppression.
2. **Transactions on Machine Learning Research (TMLR):** Ideal for thorough, rigorous empirical studies with pre-registered hypotheses; explicitly welcomes negative results and studies distinguishing active learning ingredients from domain-coverage effects.
3. **Data-Centric AI (DCAI) / NeurIPS Workshop on Data-Centric Machine Learning:** Focuses specifically on sample selection strategies, benchmark integrity, and dataset curation quality under fixed parameter budgets.

### 1.2. The Core Scientific Contribution
The defensible publication contribution is **not** an algorithmic or architectural novelty (CNN14 and linear transfer heads are established tools). Rather, the contribution is a **controlled, data-centric empirical study of negative-sample selection policies in safety-critical edge audio monitoring**, demonstrating:
1. The empirical vulnerability of pretrained audio backbones to specific semantic confusers.
2. The exact degree to which active error mining outperforms naive category coverage and general background audio under strictly matched sample budgets ($N = 634$).
3. The operational trade-offs between false alarm suppression, emergency sensitivity, and benign environmental stability.

---

## 2. Epistemic Evidence Partition

To ensure scientific integrity and prevent reviewer rejection for overclaiming, all project findings must be partitioned into four strict epistemic categories:

```
                               ┌────────────────────────────────────────────────────────┐
                               │           EPISTEMIC EVIDENCE CLASSIFICATION            │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
        ┌───────────────────────────┬──────────────────────┴─────┬───────────────────────────┐
        ▼                           ▼                            ▼                           ▼
1. CURRENT VERIFIED         2. EXPLORATORY EVIDENCE       3. PLANNED EVIDENCE         4. UNRESOLVED DECISIONS
(Artifacts & Checkpoints    (Test Set v2 & Post-Hoc       (Test Set v3 & Causal       (Advisor Sign-Off
    Fully Validated)             E4 Ablations)                 Control Arms)               Mandatory)
```

### 2.1. Category 1: Current Verified Evidence
*Artifacts, checkpoints, and metrics independently audited and confirmed from underlying disk files:*
- **EXP-001 (Prototype Baseline):** 4-layer CNN trained from scratch achieved Accuracy = 0.5143, Macro-F1 = 0.4430 on initial dataset ($N = 527$). Checkpoint [`models/checkpoints/best_model.pt`](file:///d:/ml/fydp/MyNet/models/checkpoints/best_model.pt) verified; prediction CSV missing (documented as `COMPLETED_PARTIALLY_VERIFIED`).
- **EXP-002 (Transfer Head Baseline):** PANNs CNN14 frozen feature extractor with linear head achieved Accuracy = 0.8920, Macro-F1 = 0.8962 on validation split ($N = 105$), establishing the massive utility of pretrained acoustic representations.
- **EXP-004 & EXP-006 (Shared Evaluation Comparison):** Side-by-side evaluation on Eval Set v1 ($N = 519$) verified in [`models/experiments/model_comparison/eval_predictions_comparison.csv`](file:///d:/ml/fydp/MyNet/models/experiments/model_comparison/eval_predictions_comparison.csv). Demonstrated that combining AudioSet and NIGENS studio audio produced acoustic domain mismatch without improving emergency discrimination.
- **EXP-008 (Targeted Confuser Harvesting):** Harvested 212 candidate clips with 100% human audit consensus and verified cross-split hash isolation ([`data/processed/targeted_expansion_review_results.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_review_results.csv)).
- **EXP-009 through EXP-011 (Phase 1 In-Distribution Performance):** 9 models (E1, E2, E3 across 3 seeds) trained and evaluated on validation split ($N = 105$). Verified that E3 had lowest validation loss and highest accuracy (.9143), while E2 had highest emergency precision (.9302).

### 2.2. Category 2: Exploratory Evidence (Development Artifacts)
*Empirical findings obtained on Test Set v2 ($N = 750$) and Phase 2 E4 ablations. These cannot serve as confirmatory proof because Test Set v2 was exposed during research iteration:*
- **Phase 1 Test Set v2 Benchmark:** E2 achieved Accuracy = 82.98% / Macro-F1 = 0.8292 vs. E3 at 80.98% / 0.8136. Targeted confuser false emergency rate was 32.00% (E2) vs. 53.78% (E3).
- **Phase 2 E4 Category Ablations:**
  - Fireworks ablation: Removing 28 fireworks clips caused a +41.33 pp increase in A-FER ($p = 0.000002$), indicating fireworks are critical hard negatives.
  - Human shouting ablation: Removing 44 shouting clips caused a +36.00 pp increase in A-FER ($p = 0.000004$), indicating shouting is a critical hard negative.
  - Car alarms ablation: +14.67 pp increase ($p = 0.031250$); did not survive Holm-Bonferroni correction (Rank 4).
  - Door slams, whistles, and metal impacts: Did not reach statistical significance at $N = 25$ recordings per slice.
  - Construction tools: Evaluation script fell back to pooled $N = 150$ confusers due to missing slice; matched control suffered cross-category collateral loss. Finding is **UNRESOLVED**.

### 2.3. Category 3: Planned Evidence (Phase 3 Confirmatory Design)
*Unexecuted studies specified in protocols, awaiting unexposed Test Set v3 acquisition:*
- **Independent Test Set v3 ($N = 1,400$):** Unexposed benchmark (700 emergency targets, 350 benign background controls, 350 targeted confusers) under Two-Key cryptographic blinding.
- **Hierarchical Gatekeeper Sequence:** Formal confirmatory evaluation across Gates 1 to 4 with FWER controlled at $\alpha = 0.05$.
- **Causal E5 Control Arm ($N = 634$):** Matched random confuser additions to isolate model-error guidance from category coverage.
- **Continuous 72-Hour Streaming Simulation:** Empirical measurement of event-level recall ($R_{\text{event}}$), false alarms per 24 hours, and mean time between false alerts (MTBFA) with hysteresis persistence logic.
- **EXP-029, EXP-030, EXP-031:** Domain shift quantification, sequential adaptation/forgetting, and CKA/UMAP embedding representation geometry.

### 2.4. Category 4: Unresolved Methodological Decisions
*Items requiring formal faculty advisor sign-off before candidate downloading or model training begins:*
1. Formal pre-registration of the primary confirmatory scope (pooled Gatekeeper vs. exploratory subdomains).
2. Addition and pre-registration of the E5 causal control arm.
3. Pre-registration of the Gate 1 Equivalence Margin ($\Delta_{\text{equiv}} = 0.030$).
4. Codification of the mandatory reporting rule for all 4 gates regardless of early gate halting.
5. Decision on Zenodo public release framework (5-tier multi-license vs. metadata-only).
6. Authorization of the 2,400 raw candidate harvest quota.

---

## 3. What the Evidence CANNOT Support (Explicit Guardrails)

To protect the thesis and paper against fatal methodological critiques, the manuscript must adhere to the following negative boundaries:

1. **CANNOT claim model-error guidance is causally proven over category coverage (without E5):**  
   If E2 beats E3, the paper cannot claim active error mining is superior to category collection. Without the E5 random-confuser control, the gain could be 100% driven by simply having shouting and fireworks in the training set.
2. **CANNOT cite Test Set v2 as independent confirmatory evidence:**  
   Test Set v2 was repeatedly evaluated during development, threshold setting, and error analysis. Citing v2 as confirmatory evidence violates standard ML benchmark integrity.
3. **CANNOT claim generalizability across diverse deep learning architectures:**  
   All transfer experiments used a frozen PANNs CNN14 backbone with a single linear head (10,245 trainable parameters). The evidence does not prove that error-guided selection benefits end-to-end fine-tuned models, Audio Spectrogram Transformers (AST), or lightweight edge architectures (e.g. MobileNetV4).
4. **CANNOT claim 'Zero Pretraining Leakage':**  
   CNN14 was pretrained on 2 million AudioSet clips. Sourcing candidate audio from Freesound and excluding known YouTube video IDs reduces known overlap, but cannot guarantee that latent acoustic duplicates do not exist in AudioSet's vast uncaptioned pretraining pool. The paper must state: *"Screened for known YouTube IDs and acoustic duplicates, subject to unknown latent pretraining exposure."*
5. **CANNOT claim category-specific confirmation on $N = 50$ slices:**  
   Individual confuser subdomains ($N = 50$) have only 25.8% to 40.6% statistical power under multiplicity correction. Category findings must be explicitly presented as descriptive secondary analyses.

---

## 4. Contingency Interpretation Framework (Pass, Partial, Fail)

A major strength of a pre-registered data-centric study is that **every Phase 3 empirical outcome represents a defensible, publishable scientific finding**, provided the E5 control and equivalence bounds are pre-registered:

```
                               ┌────────────────────────────────────────────────────────┐
                               │             PHASE 3 CONFIRMATORY OUTCOMES              │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
        ┌───────────────────────────┬──────────────────────┴─────┬───────────────────────────┐
        ▼                           ▼                            ▼                           ▼
   SCENARIO A                  SCENARIO B                   SCENARIO C                  SCENARIO D
Full Superiority          Category Coverage           Null Treatment Effect        Equivalent Protection
 (E2 > E5 > E3)             (E2 ≈ E5 > E3)               (E2 ≈ E5 ≈ E3)             (Gate 1 Equivalence)
```

### Scenario A: Full Causal Confirmation ($E2 \gg E5 > E3$)
- **Empirical Finding:** E2 significantly outperforms both general background (E3) and randomly selected confusers (E5), while passing Gates 2, 3, and 4.
- **Scientific Claim:** Model-error-guided hard-negative selection is causally validated. Active error mining restructures decision boundaries far more effectively than naive category collection or general background expansion.
- **Publication Venue:** Top-tier competitive track (DCASE, TMLR, IEEE TASLP).

### Scenario B: Category Coverage Dominance ($E2 \approx E5 \gg E3$)
- **Empirical Finding:** E2 and E5 perform identically on Test Set v3, and both significantly outperform general background (E3).
- **Scientific Claim:** Category coverage is the active ingredient; active error mining provides zero marginal utility over simple random sampling within confuser categories. Practitioners do not need complex error-mining pipelines—they simply need domain-diverse confuser audio.
- **Publication Venue:** Excellent data-centric ML finding (NeurIPS DCAI, TMLR); highly cited by practitioners for debunking active learning overhead.

### Scenario C: General Background Sufficiency ($E2 \approx E5 \approx E3$)
- **Empirical Finding:** E2, E5, and E3 perform identically across confusers, benign audio, and emergency targets.
- **Scientific Claim:** Adding general background audio provides equal decision-boundary regularization to targeted confusers under matched sample budgets. Hard-negative engineering is unnecessary when operating on high-capacity pretrained acoustic representations.
- **Publication Venue:** High-impact negative result paper in data-centric AI.

### Scenario D: Non-Inferiority with Equivalence ($\text{TOST } p < 0.05$ on $[-\Delta_{\text{equiv}}, +\Delta_{\text{equiv}}]$)
- **Empirical Finding:** Gate 1 fails superiority ($p \ge 0.05$), but Two One-Sided Tests establish that the difference between E2 and E3 is bounded within $\pm 3.0$ percentage points, while Gates 2 and 3 pass non-inferiority.
- **Scientific Claim:** Targeted confusers and general background audio provide statistically equivalent protection within an operationally defined 3.0 pp boundary.
- **Publication Venue:** Rigorous empirical methodology publication.

---

## 5. Summary of Decisions for the Faculty Advisor

To advance Phase 3 from protocol draft to frozen pre-registration, the following six decisions require advisor approval:

1. **Decision 1 (Confirmatory Scope):** Restrict confirmatory claims to the pooled 4-gate sequence ($N = 350, 350, 700, 1400$); designate all 7 subdomains ($N = 50$) exploratory secondary endpoints.
2. **Decision 2 (Causal Arm E5):** Approve adding the E5 Matched Random Confuser Control ($N = 634$) to isolate model-error selection from category coverage.
3. **Decision 3 (Margins):** Approve pre-registered margins: $\delta_B = 0.020$ (benign FER), $\delta_R = 0.025$ (macro-recall), and $\Delta_{\text{equiv}} = 0.030$ (equivalence bound).
4. **Decision 4 (Full Reporting Mandate):** Pre-specify that point estimates and 95% CIs for all 4 gates will be published regardless of whether an early gate halts confirmatory testing.
5. **Decision 5 (Release Policy):** Approve 5-tier open benchmark release on Zenodo with CC attribution manifest (BBC audio excluded).
6. **Decision 6 (Harvest Quota):** Authorize downloading 2,400 candidate clips across Freesound and unreleased field recordings to achieve $N = 1,400$ approved benchmark WAVs.
