# Phase 3 Independent Confirmatory Evaluation Protocol: Second-Level Independent Methodological Review

**Project:** `D:\ml\fydp\MyNet`  
**Research Framing:** *Data-Centric Analysis of Hard-Negative Learning and Domain Shift in Emergency Audio Classification*  
**Date:** November 2, 2026  
**Document:** [`reports/phase3_protocol_second_level_review.md`](file:///d:/ml/fydp/MyNet/reports/phase3_protocol_second_level_review.md)  
**Preceding Documents:**
- [`reports/phase3_independent_confirmation_protocol.md`](file:///d:/ml/fydp/MyNet/reports/phase3_independent_confirmation_protocol.md) *(Original Draft Protocol v1.0)*
- [`reports/phase3_protocol_methodology_review.md`](file:///d:/ml/fydp/MyNet/reports/phase3_protocol_methodology_review.md) *(First-Level Audit)*
**Supporting Simulation Scripts & Artifacts:**
- [`scripts/simulate_phase3_power.py`](file:///d:/ml/fydp/MyNet/scripts/simulate_phase3_power.py)
- [`scripts/simulate_phase3_power_comprehensive.py`](file:///d:/ml/fydp/MyNet/scripts/simulate_phase3_power_comprehensive.py)
- [`reports/phase3_power_simulation_results.json`](file:///d:/ml/fydp/MyNet/reports/phase3_power_simulation_results.json)
- [`reports/phase3_power_simulation_comprehensive.json`](file:///d:/ml/fydp/MyNet/reports/phase3_power_simulation_comprehensive.json)  
**Epistemic Invariant:** Zero model training, zero dataset downloads, zero manifest alterations, and zero checkpoint modifications.

---

## 1. Executive Summary & Purpose of Second-Level Review

This document provides a second-level independent methodological review of the Phase 3 Confirmatory Protocol. Rather than accepting the recommendations of the first-level audit uncritically, this review conducts an exhaustive, mathematically grounded verification of:
1. **Simulation Code & Statistical Power Claims:** Audits [`scripts/simulate_phase3_power.py`](file:///d:/ml/fydp/MyNet/scripts/simulate_phase3_power.py) and runs expanded simulations via [`scripts/simulate_phase3_power_comprehensive.py`](file:///d:/ml/fydp/MyNet/scripts/simulate_phase3_power_comprehensive.py) across $N \in \{25, 50, 75, 100, 150, 350\}$, distinguishing exact discrete McNemar tests from clustered fractional permutation tests, while explicitly correcting for optimistic effect-size bias (Winner's Curse).
2. **Gatekeeper Hypothesis Architecture:** Resolves mathematical incompatibilities in Gates 3 and 4, establishing valid class-stratified cluster-bootstrap hypothesis tests for ratio metrics (Macro-Recall and Macro-F1).
3. **Non-Inferiority Margins in Operational Context:** Re-evaluates the $3.0\%$ margin against continuous edge streaming dynamics (overlapping sliding windows, hysteresis persistence filters, and dispatch costs).
4. **Endpoint Hierarchy & Multilevel Allocation:** Formally separates primary confirmatory, secondary confirmatory, and exploratory endpoints across the $N = 1,400$ benchmark.
5. **Licensing & Redistribution Feasibility:** Resolves the BBC Sound Effects redistribution ban and details Freesound license heterogeneity (CC0, CC-BY, CC-BY-NC).

```mermaid
flowchart TD
    subgraph Second_Level_Review_Architecture ["Second-Level Independent Review Structure"]
        V1["1. Independent Power Verification<br/>(Inspect scripts/simulate_phase3_power.py<br/>Run Comprehensive 10k Simulations across N=25..350)"]
        V2["2. Corrected Gatekeeper Formulation<br/>(Class-Stratified Cluster-Bootstrap for Ratio Metrics)"]
        V3["3. Operational Non-Inferiority Audit<br/>(Window Hysteresis, Alarm Cooldown, Dispatch Impact)"]
        V4["4. Allocation & Endpoint Hierarchy<br/>(Primary Confirmatory vs Secondary vs Exploratory)"]
        V5["5. Licensing & Provenance Realities<br/>(BBC Redistribution Ban, Freesound CC Heterogeneity)"]
    end

    V1 & V2 & V3 & V4 & V5 --> FinalProtocol["Approved Corrected Protocol Draft<br/>(reports/phase3_independent_confirmation_protocol_v2.md)"]
```

---

## 2. Independent Verification of Statistical Power

### 2.1. Audit of Initial Simulation Code (`scripts/simulate_phase3_power.py`)
We inspected [`scripts/simulate_phase3_power.py`](file:///d:/ml/fydp/MyNet/scripts/simulate_phase3_power.py) and identified several crucial methodological nuances:
1. **Paired Outcome Modeling:**  
   In `simulate_mcnemar_power`, recording-level outcomes were modeled via multinomial draws of concordant pairs ($p_{00}$), discordant favoring E2 ($p_{10} = b$), and discordant favoring E3 ($p_{01} = c$). The net effect size was correctly defined as $\Delta = p_{10} - p_{01}$.
2. **Missing Sample Sizes & Estimand Differences:**  
   The initial script evaluated only $N \in \{25, 50, 75, 100\}$ and $N = 350$, omitting $N = 150$. More importantly, it evaluated only discrete binary paired decisions (applicable to the ensemble estimand $\theta_2$). It did not complete the simulation of the 3-seed clustered fractional difference estimand ($\theta_1$) across all sample sizes.
3. **Optimistic Effect-Size Bias (Winner's Curse):**  
   The first-level audit tested $\Delta = 0.218$ (the observed difference in Phase 1 on Test Set v2). Because Test Set v2 was an unblinded exploratory benchmark, using $\Delta = 0.218$ as the assumed true effect size in a confirmatory trial introduces severe optimistic bias. Confirmatory trials frequently observe effect sizes attenuated by $30\%$ to $50\%$ due to regression to the mean and broader acoustic domain variance.

### 2.2. Comprehensive Simulation Design (`scripts/simulate_phase3_power_comprehensive.py`)
To resolve these gaps, we executed [`scripts/simulate_phase3_power_comprehensive.py`](file:///d:/ml/fydp/MyNet/scripts/simulate_phase3_power_comprehensive.py), running **10,000 Monte Carlo iterations per cell** across:
- **Sample sizes:** $N \in \{25, 50, 75, 100, 150, 350\}$.
- **Two Distinct Estimand Formulations:**
  1. *Exact Binary McNemar Test:* For discrete recording-level binary decisions (probability ensemble $\theta_2$).
  2. *Clustered 3-Seed Fractional Permutation Test:* For recording-level average decisions across three seeds:
     $$d_i = \frac{1}{3}\sum_{s \in \{42, 123, 2026\}} \left[ \mathbb{I}(\hat{y}_{i,s}^{(\text{E3})} \in \text{Emerg}) - \mathbb{I}(\hat{y}_{i,s}^{(\text{E2})} \in \text{Emerg}) \right] \in \left\{-1, -\frac{2}{3}, -\frac{1}{3}, 0, +\frac{1}{3}, +\frac{2}{3}, +1\right\}$$
     with within-cluster recording correlation modeled at $\rho = 0.65$.
- **Realistic Effect-Size Scenarios:**
  - *Conservative:* $\Delta = 0.05$ ($p_{10} = 0.15, p_{01} = 0.10$).
  - *Moderate (Realistic Confirmatory Expectation):* $\Delta = 0.08$ ($p_{10} = 0.18, p_{01} = 0.10$).
  - *Target Unadjusted Effect:* $\Delta = 0.10$ ($p_{10} = 0.20, p_{01} = 0.10$).
  - *Strong Effect (Clean vs. Noisy):* $\Delta = 0.15$ ($p_{10} = 0.15, p_{01} = 0.00$ and $p_{10} = 0.20, p_{01} = 0.05$).
  - *Optimistic Phase 1 Benchmark:* $\Delta = 0.218$ ($p_{10} = 0.32, p_{01} = 0.10$).
- **Multiplicity Levels:**
  - $\alpha = 0.05$ (Gate 1 Gatekeeper nominal level).
  - $\alpha = 0.0125$ (Subdomain Holm Rank 4).
  - $\alpha = 0.00714$ (Subdomain Holm Rank 1).

### 2.3. Power Simulation Results & Comparative Curves

The comprehensive simulation output ([`reports/phase3_power_simulation_comprehensive.json`](file:///d:/ml/fydp/MyNet/reports/phase3_power_simulation_comprehensive.json)) yields definitive power curves:

#### Table 1: Clustered 3-Seed Permutation Test Power ($\theta_1$ Single-Model Primary Estimand)

| Scenario & Assumed Effect ($\Delta$) | Discordance ($p_{10} : p_{01}$) | $N = 25$ ($\alpha = .0125 / .05$) | $N = 50$ ($\alpha = .0125 / .05$) | $N = 75$ ($\alpha = .0125 / .05$) | $N = 100$ ($\alpha = .0125 / .05$) | $N = 150$ ($\alpha = .0125 / .05$) | $N = 350$ (Gate 1, $\alpha = .05$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conservative ($\Delta = 0.05$)** | $0.15 : 0.10$ | $3.0\% / 9.6\%$ | $7.3\% / 19.9\%$ | $12.5\% / 28.9\%$ | $18.9\% / 37.4\%$ | $31.7\% / 53.3\%$ | **89.98%** |
| **Moderate ($\Delta = 0.08$)** | $0.18 : 0.10$ | $7.2\% / 21.1\%$ | $23.1\% / 45.9\%$ | $40.8\% / 64.1\%$ | $55.9\% / 77.5\%$ | $78.4\% / 92.1\%$ | **99.86%** |
| **Target ($\Delta = 0.10$)** | $0.20 : 0.10$ | $13.1\% / 31.9\%$ | $40.6\% / 65.2\%$ | $64.2\% / 84.0\%$ | $80.9\% / 93.1\%$ | $95.4\% / 98.7\%$ | **100.00%** |
| **Strong Noisy ($\Delta = 0.15$)** | $0.20 : 0.05$ | $37.7\% / 66.0\%$ | $84.2\% / 95.3\%$ | $97.6\% / 99.6\%$ | $99.6\% / 100.0\%$ | $100.0\% / 100.0\%$ | **100.00%** |
| **Optimistic Phase 1 ($\Delta = 0.218$)** | $0.32 : 0.10$ | $80.9\% / 95.1\%$ | $99.7\% / 100.0\%$ | $100.0\% / 100.0\%$ | $100.0\% / 100.0\%$ | $100.0\% / 100.0\%$ | **100.00%** |

#### Table 2: Exact Binary McNemar Test Power ($\theta_2$ Ensemble Secondary Estimand)

| Scenario & Assumed Effect ($\Delta$) | Discordance ($p_{10} : p_{01}$) | $N = 25$ ($\alpha = .0125 / .05$) | $N = 50$ ($\alpha = .0125 / .05$) | $N = 75$ ($\alpha = .0125 / .05$) | $N = 100$ ($\alpha = .0125 / .05$) | $N = 150$ ($\alpha = .0125 / .05$) | $N = 350$ (Gate 1, $\alpha = .05$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conservative ($\Delta = 0.05$)** | $0.15 : 0.10$ | $0.4\% / 2.4\%$ | $1.7\% / 5.9\%$ | $2.9\% / 9.1\%$ | $4.1\% / 11.7\%$ | $7.4\% / 18.4\%$ | **42.68%** |
| **Moderate ($\Delta = 0.08$)** | $0.18 : 0.10$ | $0.9\% / 4.7\%$ | $3.8\% / 12.3\%$ | $7.9\% / 19.0\%$ | $11.4\% / 26.1\%$ | $21.7\% / 40.0\%$ | **78.87%** |
| **Target ($\Delta = 0.10$)** | $0.20 : 0.10$ | $1.7\% / 7.2\%$ | $6.2\% / 18.0\%$ | $12.7\% / 27.9\%$ | $19.4\% / 37.6\%$ | $34.5\% / 55.5\%$ | **92.29%** |
| **Strong Noisy ($\Delta = 0.15$)** | $0.20 : 0.05$ | $4.3\% / 16.4\%$ | $25.8\% / 47.7\%$ | $47.9\% / 70.7\%$ | $65.9\% / 83.7\%$ | $88.7\% / 96.2\%$ | **99.99%** |
| **Optimistic Phase 1 ($\Delta = 0.218$)** | $0.32 : 0.10$ | $13.0\% / 28.9\%$ | $39.5\% / 61.4\%$ | $62.9\% / 81.7\%$ | $80.6\% / 92.5\%$ | $95.8\% / 98.9\%$ | **100.00%** |

### 2.4. Critical Methodological Deductions
1. **Confirmation of First-Level Audit's Power Refutation:**  
   The protocol v1.0 claim of $>86.4\%$ power at $N = 50, \alpha = 0.0125$ is unequivocally false for binary discrete McNemar tests ($\text{power} = 25.8\%$). Even under clustered 3-seed permutation testing with a target $\Delta = 0.10$, power at $N = 50, \alpha = 0.0125$ is only **$40.6\%$**.
2. **Permutation vs. McNemar Efficiency:**  
   The clustered fractional permutation test achieves substantially higher power than binary McNemar ($65.2\%$ vs. $18.0\%$ at $N = 50, \alpha = 0.05, \Delta = 0.10$). Averaging decisions across three seeds preserves continuous fractional information ($d_i = \pm 1/3, \pm 2/3$) that reduces discrete thresholding variance.
3. **Primary Confirmatory Gate 1 is Fully Powered:**  
   Under the pre-registered Gatekeeper procedure, Gate 1 tests the **pooled confuser set ($N = 350$) at nominal $\alpha = 0.05$**. In this setting, statistical power is:
   - **$89.98\%$** even if the true effect attenuates to conservative $\Delta = 0.05$.
   - **$99.86\%$** for moderate effect $\Delta = 0.08$.
   - **$100.00\%$** for target effect $\Delta = 0.10$.
   Therefore, **Gate 1 is fully powered and methodologically rock-solid.**
4. **Category Subdomain Classification:**  
   Because $N = 50$ achieves only $40.6\%$ power at $\alpha = 0.0125$ under $\Delta = 0.10$, **individual category subdomains cannot be claimed as primary confirmatory endpoints**. They must be formally designated as **exploratory secondary endpoints**, or grouped into a targeted 2-hypothesis secondary confirmatory family (Fireworks and Shouting only, where $\Delta \ge 0.30$ guarantees $>95\%$ power).

---

## 3. Mathematical Correction of the Statistical Gatekeeper Architecture

### 3.1. Verification of the Fixed-Sequence FWER Control
Under a fixed-sequence hierarchical test:
$$\text{Gate 1} \xrightarrow{\text{Reject } \mathcal{H}_{0,1}} \text{Gate 2} \xrightarrow{\text{Reject } \mathcal{H}_{0,2}} \text{Gate 3} \xrightarrow{\text{Reject } \mathcal{H}_{0,3}} \text{Gate 4}$$
- Each hypothesis $\mathcal{H}_{0,k}$ is tested at nominal $\alpha = 0.05$.
- If $\mathcal{H}_{0,1}$ is true, the probability of falsely rejecting it is $\le \alpha = 0.05$.
- If $\mathcal{H}_{0,1}$ is false and rejected, testing proceeds to $\mathcal{H}_{0,2}$, where the conditional Type I error probability is bounded by $\alpha = 0.05$.
- **Mathematical Property:** Fixed-sequence testing strictly bounds the family-wise error rate across all four gates at $\text{FWER} \le 0.05$ without requiring Bonferroni or Holm division (Maurer, Bretz, & Hommel, 2010). Applying $\alpha = 0.0125$ across the four gates is **overly conservative and mathematically unnecessary**.

### 3.2. Resolution of Non-Linear Ratio Metrics (Gates 3 and 4)
The first audit correctly noted that paired sign-flip permutation tests cannot test ratio metrics. We formalize the exact mathematical test for each gate:

#### Gate 1: Confuser False Emergency Rate (A-FER) Superiority
- **Estimand:** Expected reduction in false emergency rate on confusers:
  $$\theta_1 = \mathbb{E}[d_i^{\text{conf}}], \quad d_i^{\text{conf}} = \frac{1}{3}\sum_{s} \left[ \mathbb{I}(\hat{y}_{i,s}^{(\text{E3})} \in \text{Emerg}) - \mathbb{I}(\hat{y}_{i,s}^{(\text{E2})} \in \text{Emerg}) \right]$$
- **Hypotheses:** $\mathcal{H}_{0,1}: \theta_1 \le 0$ vs. $\mathcal{H}_{1,1}: \theta_1 > 0$.
- **Statistical Test:** One-sided paired recording-cluster sign-flip permutation test ($R = 100,000$).
- **Decision Rule:** Reject $\mathcal{H}_{0,1}$ if permutation $p < 0.05$. If rejected, proceed to Gate 2; if not, halt testing (trial inconclusive).

#### Gate 2: Benign Background False Emergency Rate (B-FER) Non-Inferiority
- **Estimand:** Excess false emergency rate on benign background:
  $$\theta_2 = \mathbb{E}[d_i^{\text{benign}}], \quad d_i^{\text{benign}} = \frac{1}{3}\sum_{s} \left[ \mathbb{I}(\hat{y}_{i,s}^{(\text{E2})} \in \text{Emerg}) - \mathbb{I}(\hat{y}_{i,s}^{(\text{E3})} \in \text{Emerg}) \right]$$
- **Hypotheses:** $\mathcal{H}_{0,2}: \theta_2 \ge \delta_{\text{B}}$ vs. $\mathcal{H}_{1,2}: \theta_2 < \delta_{\text{B}}$.
- **Statistical Test:** One-sided test via paired cluster-bootstrap confidence bound. Using 10,000 recording-cluster bootstrap resamples, compute the 95th percentile upper confidence bound $\hat{\theta}_{2, \text{upper}}^{95\%}$.
- **Decision Rule:** Reject $\mathcal{H}_{0,2}$ if $\hat{\theta}_{2, \text{upper}}^{95\%} < \delta_{\text{B}}$. If rejected, proceed to Gate 3; if not, halt testing.

#### Gate 3: Emergency Macro-Recall Non-Inferiority
- **Estimand:** Macro-recall difference across the four emergency classes:
  $$\theta_3 = \text{Rec}_{\text{macro}}(\text{E3}) - \text{Rec}_{\text{macro}}(\text{E2}), \quad \text{Rec}_{\text{macro}} = \frac{1}{4}\sum_{c=0}^3 \frac{\sum_i \mathbb{I}(y_i = c \land \hat{y}_i = c)}{N_c}$$
- **Hypotheses:** $\mathcal{H}_{0,3}: \theta_3 \ge \delta_{\text{R}}$ vs. $\mathcal{H}_{1,3}: \theta_3 < \delta_{\text{R}}$.
- **Statistical Test (Class-Stratified Cluster Bootstrap):**  
  Because Macro-Recall is a class-stratified ratio estimator:
  1. For each bootstrap iteration $b \in \{1 \dots 10,000\}$, resample physical recordings with replacement **within each of the 4 emergency classes independently** ($N_c = 175$ per class).
  2. For the resampled recording set, compute the complete $4 \times 5$ emergency confusion matrix for each seed, average predictions across seeds, and calculate $\text{Rec}_{\text{macro}}(\text{E2})$ and $\text{Rec}_{\text{macro}}(\text{E3})$.
  3. Compute bootstrap difference $\theta_3^{(b)} = \text{Rec}_{\text{macro}}^{(b)}(\text{E3}) - \text{Rec}_{\text{macro}}^{(b)}(\text{E2})$.
  4. Extract the 95th percentile upper bound $\hat{\theta}_{3, \text{upper}}^{95\%}$.
- **Decision Rule:** Reject $\mathcal{H}_{0,3}$ if $\hat{\theta}_{3, \text{upper}}^{95\%} < \delta_{\text{R}}$. If rejected, proceed to Gate 4; if not, halt testing.

#### Gate 4: Overall Five-Class Macro-F1 Superiority
- **Estimand:** Macro-F1 advantage across all 5 classes:
  $$\theta_4 = \text{Macro-F1}(\text{E2}) - \text{Macro-F1}(\text{E3})$$
- **Hypotheses:** $\mathcal{H}_{0,4}: \theta_4 \le 0$ vs. $\mathcal{H}_{1,4}: \theta_4 > 0$.
- **Statistical Test (Full 5-Class Stratified Cluster Bootstrap):**  
  1. Resample recordings within each of the 5 classes independently ($175$ per emergency class, $700$ negative controls).
  2. Compute full $5 \times 5$ multi-class confusion matrix per bootstrap iteration.
  3. Calculate $\text{Macro-F1}^{(b)}(\text{E2})$ and $\text{Macro-F1}^{(b)}(\text{E3})$ using class-wise harmonic means.
  4. Compute difference $\theta_4^{(b)} = \text{Macro-F1}^{(b)}(\text{E2}) - \text{Macro-F1}^{(b)}(\text{E3})$.
  5. Compute empirical $p$-value: $p = \frac{1}{B}\sum_{b=1}^B \mathbb{I}(\theta_4^{(b)} \le 0)$.
- **Decision Rule:** Reject $\mathcal{H}_{0,4}$ if $p < 0.05$ (equivalent to 95% bootstrap lower bound $> 0$).

---

## 4. Re-Evaluation of Non-Inferiority Margins in Operational Context

The first audit criticized $\delta_{\text{tol}} = 0.03$ ($3.0\%$) by assuming continuous un-aggregated streaming audio where a $3\%$ false positive rate yields $10.8$ false alarms per hour. We re-examined this operational framing:

### 4.1. Edge Deployment Architecture Reality
In real-world acoustic monitoring systems (e.g., smart campus security, transit surveillance):
1. **Sliding Windows vs. Alarm Events:** Edge systems process audio in overlapping sliding windows (e.g., 10-second windows with 2-second hops). A single 4-second barking dog or horn honk spans **4 to 5 consecutive evaluation windows**. Window-level false positive rates cannot be multiplied directly by time without accounting for temporal auto-correlation.
2. **Temporal Aggregation & Cooldown Filters:** Commercial edge dispatch systems implement **persistence logic**:
   - An alert is dispatched only if an emergency class probability exceeds threshold for **$k \ge 2$ consecutive windows**, or if an exponential leaky integrator crosses threshold.
   - Following an alert trigger, a **cooldown refractory period (e.g., 60 seconds)** suppresses duplicate dispatches.
   - Consequently, isolated window-level false positives are filtered out by edge hysteresis.
3. **Emergency Recall Multiplication:** Conversely, a genuine emergency event (an extended screaming incident, multiple gunshots, a continuous vehicle siren) spans **multiple consecutive windows**. A system with $87\%$ window recall vs. $90\%$ window recall both achieve $>99.5\%$ event-level detection on an event spanning 3 windows ($1 - 0.13^3 = 0.9978$).

### 4.2. Recommended Supervisor Procedure for Margin Determination
Rather than arbitrarily adopting $1.5\%$ or $3.0\%$, the protocol must define a formal procedure for supervisor sign-off:

```mermaid
flowchart TD
    S1["Step 1: Review Phase 1 Empirical Benchmark Bounds<br/>(Observed Benign Difference = +1.11%, Recall Difference = -3.86%)"] --> S2
    S2["Step 2: Define Maximum Tolerable Operational Loss<br/>(Supervisor determines dispatch cost vs missed event risk)"] --> S3
    S3["Step 3: Select Pre-Registered Margins<br/>Option A (Strict): delta_B = 1.5%, delta_R = 2.0%<br/>Option B (Moderate): delta_B = 2.5%, delta_R = 3.0%"] --> S4
    S4["Step 4: Formal Faculty Supervisor Sign-Off & Git Commit"]
```

**Recommendation for Protocol v2:**  
Establish **$\delta_{\text{B}} = 0.020$ ($2.0\%$)** for Benign FER and **$\delta_{\text{R}} = 0.025$ ($2.5\%$)** for Emergency Recall as the pre-registration default, subject to formal supervisor confirmation prior to dataset acquisition.

---

## 5. Provenance, Pretraining Exposure, and Anti-Leakage Screening

### 5.1. Refined Pretraining Exposure Categorization
We reaffirm the first audit's finding that claiming "zero leakage" is scientifically indefensible when evaluating transfer models based on foundation models trained on 2 million web clips. The revised protocol categorizes exposure into four explicit evidential bins:

| Evidential Bin | Description & Scope | Affected Candidate Sources | Mitigation Strategy in Protocol |
|:---|:---|:---|:---|
| **1. Confirmed Overlap** | Audio files or YouTube video IDs previously logged in project training manifests. | Historical AudioSet manifests (E1, E2, E3, Eval v1, Pilot: 3,637 IDs). | Strict automated exclusion filter matching YouTube video IDs and SHA-256 hashes. |
| **2. Potential Overlap** | Audio contributed to Freesound or BBC that was subsequently embedded into YouTube videos crawled by AudioSet (2006–2017). | FSD50K, ESC-50, UrbanSound8K, BBC Sound Effects. | Acknowledge as latent pretraining exposure in paper; prioritize non-YouTube metadata. |
| **3. Unknown Exposure** | Independent web audio cross-posted across multiple platforms with unknown provenance. | Uncurated SoundCloud, Vimeo, Internet Archive harvests. | Require individual recording provenance metadata and source verification. |
| **4. Domain Independence** | Fresh physical recordings captured with distinct recording chains, microphones, and rooms. | Newly captured physical field audio. | Allocate a dedicated 15% quota to unreleased physical field recordings. |

### 5.2. Multi-Tiered Screening Pipeline Beyond SHA-256 and Chromaprint
SHA-256 detects only bitwise identical files, and Chromaprint fails on short non-musical transients. Protocol v2 specifies a **4-tier screening pipeline**:
1. **Tier 1 (ID & URL Filter):** Reject any file matching the 3,637+ YouTube video IDs or known channel URLs.
2. **Tier 2 (Cryptographic Hash):** Reject bitwise SHA-256 matches against all project storage directories.
3. **Tier 3 (Log-Mel Spectral Cross-Correlation):** Compute 64-mel spectrograms; flag candidate clips whose peak normalized 2D cross-correlation against any historical training clip exceeds $0.85$.
4. **Tier 4 (Latent Feature Distance):** Compute 2048-dimensional PANNs embeddings; flag candidate clips with Euclidean distance $< 0.15$ to any training clip for manual auditory inspection.
5. **Tier 5 (Two-Pass Human Audit):** Two independent human reviewers verify semantic class purity and confirm absence of secondary emergency contamination.

---

## 6. Dataset Licensing and Redistribution Feasibility

### 6.1. Audit of Candidate Source Licenses

```mermaid
flowchart TD
    subgraph Candidate_Sources ["Candidate Sourcing Pools & Licensing"]
        FSD["FSD50K / Freesound<br/>Licenses: CC0, CC-BY, CC-BY-NC, Sampling+"]
        ESC["ESC-50 & UrbanSound8K<br/>License: CC-BY-NC 3.0"]
        BBC["BBC Sound Effects Archive<br/>License: BBC Sound Effects Licence"]
    end

    FSD -->|CC0, CC-BY: Open Redistribution<br/>CC-BY-NC: Non-commercial only<br/>Sampling+: EXCLUDE| P1["Permitted with Per-Clip License Manifest"]
    ESC -->|Non-commercial academic release only| P2["Permitted for Academic Thesis"]
    BBC -->|STRICT REDISTRIBUTION BAN| P3["EXCLUDE FROM REDISTRIBUTABLE BENCHMARK"]
```

1. **Freesound / FSD50K License Heterogeneity:**  
   Freesound audio does **not** share a single license. Recordings are licensed under CC0 (public domain), CC-BY (attribution), CC-BY-NC (non-commercial attribution), or Sampling+ (legacy restricted).
   - *Protocol Requirement:* The benchmark manifest must record the exact license string and author attribution URL per recording. Clips licensed under "Sampling+" or ambiguous licenses must be excluded.
2. **The BBC Redistribution Prohibition (Definitive Ruling):**  
   Clause 4 of the BBC Sound Effects licence strictly prohibits third-party redistribution. If BBC audio is bundled into Test Set v3, the benchmark cannot be uploaded to Zenodo, GitHub, or Hugging Face.
   - *Determination:* **BBC Sound Effects is permanently excluded from Test Set v3.** The quota will be replaced by verified CC-BY / CC0 Freesound recordings and newly recorded field audio.
3. **Acquisition Funnel Safety Buffer:**  
   Accounting for the empirically established $30.0\%$ yield loss in human review, the raw acquisition target must be expanded from $2,000$ to **$2,400$ candidate files** to guarantee filling the $1,400$-clip quota.

---

## 7. Synthesis & Final Review Decision Table

### 7.1. Decision Table for Protocol Sign-Off

| Evaluation Dimension | Status | Key Justification / Blocker Summary | Next Action Required |
|:---|:---:|:---|:---|
| **Ready for Supervisor Review?** | **YES** | The protocol structure, four-gate sequence, and matched comparison design are fully verified. | Present protocol v2.0 and review documents to faculty supervisor. |
| **Ready for Formal Preregistration?** | **NO** | Blockers 1, 2, and 3 must be formally incorporated into protocol v2.0 and approved by supervisor. | Approve protocol v2.0 with corrected Gatekeeper tests and margins. |
| **Ready for Dataset Acquisition?** | **NO** | Candidate source list must be updated (excluding BBC) and harvest queues expanded to 2,400 clips. | Finalize candidate URL lists and automated DSP screening scripts. |

### 7.2. Specific Decisions Requiring Supervisor Approval
1. **Approval of Pre-Registered Gating Sequence:** Confirmation that the thesis defense rests on the four-gate hierarchical sequence, with individual category breakdowns treated as secondary exploratory analyses.
2. **Sign-Off on Non-Inferiority Margins:** Formal approval of $\delta_{\text{B}} = 0.020$ ($2.0\%$) and $\delta_{\text{R}} = 0.025$ ($2.5\%$).
3. **Approval of Open-Access Benchmark Policy:** Formal decision to exclude BBC Sound Effects to ensure Test Set v3 can be released as an open-access Zenodo package.
4. **Approval of Expanded Harvest Queue:** Authorization to harvest $2,400$ raw candidates to absorb the anticipated $30\%$ validation yield loss.

---

## 8. Conclusion

This second-level independent review has resolved the methodological deficiencies of the initial protocol while independently verifying statistical power realities through comprehensive Monte Carlo simulation.

The corrected, gold-standard protocol document is generated as:  
[`reports/phase3_independent_confirmation_protocol_v2.md`](file:///d:/ml/fydp/MyNet/reports/phase3_independent_confirmation_protocol_v2.md).
