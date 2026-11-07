# Critical Methodological Audit and Corrections for Phase 3 Protocols (v3.1)

**Document Status:** Formal Methodological Deliverable  
**Date:** November 2026  
**Operating Mode:** Strictly Read-Only Methodological Audit & Evidence Reconciliation  
**Target Documents Audited:**
- `reports/phase3_independent_audit_v3.md`
- `reports/phase3_publication_readiness_v3.md`
- `reports/phase3_document_conflict_matrix_v3.csv`
- `reports/phase3_independent_confirmation_protocol_v2_1.md` (specifically Section 5.1, Table 5.1)
- `scripts/simulate_phase3_power_comprehensive.py`
- `reports/phase3_power_simulation_comprehensive.json`

---

## 1. Executive Summary & Epistemic Verdict

This document provides a critical methodological review and formal correction of the newly generated Phase 3 audit deliverables and protocols. While Phase 3 Protocol v2.1 and Audit v3.0 established rigorous safeguards against historical methodological defects (such as pseudoreplication and unstratified ratio testing), this audit identifies five critical technical issues that require explicit correction before the protocol can be frozen for execution:

1. **Retraction of Speculative Sequence Power:** Audit v3.0 and Publication Readiness v3.0 reported an overall four-gate sequence power estimate of 75%–85%. This claim is methodologically unsupported. Monte Carlo simulations in `scripts/simulate_phase3_power_comprehensive.py` evaluated *only* Gate 1. Multiplying marginal powers assumes stochastic independence across gates, which is invalid because all four gates evaluate identical model checkpoints on overlapping slices of the same benchmark. **Overall four-gate sequence power is formally classified as strictly UNKNOWN.**
2. **Foundations of Paired Sign-Flip Permutation Testing:** Averaging multi-seed predictions across physical recording clusters resolves within-recording pseudoreplication, but does not inherently guarantee finite-sample exactness. The Gate 1 permutation test is exact under the *sharp null hypothesis of exchangeability* (reflection symmetry conditional on observed magnitudes), which is an assumption about model behavior under $\mathcal{H}_0$, not a property guaranteed by experimental design. Under error distribution skewness across recording clusters, the test is asymptotically valid rather than finite-sample exact.
3. **Disambiguation of Estimand Notation:** Estimand symbol $\theta_2$ suffers from a severe collision in Protocol v2.1: it denotes Gate 2 Benign B-FER non-inferiority in Table 5.1, but simultaneously denotes the discrete 3-model majority-vote ensemble secondary estimand in Section 6.2 and the Advisor Brief. Additionally, Gate 4 exhibits an internal contradiction between the estimand label ($\Delta \text{Macro-F1}$) and the decision rule symbol ($\hat{\theta}_{4,\text{lower}}^{95\%}$). A standardized, descriptive nomenclature ($\theta_{\text{A-FER}}, \theta_{\text{B-FER}}, \theta_{\text{Rec}}, \theta_{\text{F1}}, \theta_{\text{ens}}$) is established.
4. **Demarcation of Margins (Non-Inferiority vs. Equivalence):** The non-inferiority margins ($\delta_{\text{B}} = 0.020$, $\delta_{\text{R}} = 0.025$) and the proposed equivalence margin ($\Delta_{\text{equiv}} = 0.030$) represent operational risk policies rather than statistical conventions. In life-safety emergency classification, allowable false alarms and allowable missed events require explicit justification and written sign-off from the faculty advisor.
5. **Formalization of the E5 Control Arm:** The E2 vs. E3 comparison is fundamentally confounded because it alters category coverage (shouting/fireworks vs. ambient sounds) simultaneously with selection policy (error mining vs. random sampling). A fully matched E5 control arm (212 randomly selected confusers matching E2 category quotas, candidate pool, audio specifications, and seeds) is required to support any publishable claim regarding model-error guidance.

---

## 2. Issue 1: Four-Gate Statistical Power Audit & Retraction

### 2.1. Verification of Gate 1 Simulation Results
Forensic inspection of `scripts/simulate_phase3_power_comprehensive.py` and its ground-truth output `reports/phase3_power_simulation_comprehensive.json` confirms the exact Monte Carlo power estimates ($10,000$ iterations, $R = 500$ permutations per trial, within-cluster correlation $\rho = 0.65$, one-sided $\alpha = 0.05$) across sample sizes and true effect sizes:

| Scenario / Assumed Effect Size | Discordant Rates ($p_{10}, p_{01}$) | $N = 25$ | $N = 50$ | $N = 75$ | $N = 100$ | $N = 150$ | $N = 350$ (Gate 1 Confusers) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conservative ($\Delta = 0.05$)** | $p_{10}=0.15, p_{01}=0.10$ | 9.61% | 19.88% | 28.87% | 37.38% | 53.34% | **89.98%** |
| **Moderate ($\Delta = 0.08$)** | $p_{10}=0.18, p_{01}=0.10$ | 21.13% | 45.91% | 64.07% | 77.51% | 92.10% | **99.86%** |
| **Target ($\Delta = 0.10$)** | $p_{10}=0.20, p_{01}=0.10$ | 31.94% | 65.21% | 83.97% | 93.08% | 98.72% | **100.0%** |
| **Strong Clean ($\Delta = 0.15$)** | $p_{10}=0.15, p_{01}=0.00$ | 66.02% | 95.29% | 99.59% | 99.96% | 100.0% | **100.0%** |
| **Strong Noisy ($\Delta = 0.15$)** | $p_{10}=0.20, p_{01}=0.05$ | 66.02% | 95.29% | 99.59% | 99.96% | 100.0% | **100.0%** |
| **Optimistic Phase 1 ($\Delta = 0.218$)** | $p_{10}=0.32, p_{01}=0.10$ | 95.10% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |

### 2.2. Critical Nuances and Retraction of Sequence Power
Three methodological conclusions emerge from this audit:

1. **Gate 1 Power is Strictly Conditional:** The assertion in Protocol v2.1 (Section 1.2, line 39) that Gate 1 is "fully powered (>99%)" is true *only* if the true field effect size is $\Delta \ge 0.08$. If the true unexposed effect drops to $\Delta = 0.05$ due to acoustic domain shift on pristine field recordings, power drops to $89.98\%$. If the effect is $\Delta = 0.03$, power is only $52.40\%$. The protocol must qualify all power statements with their underlying effect size assumptions.
2. **Subdomain Hypotheses are Severely Underpowered:** For individual confuser subdomains ($N = 50$, such as fireworks or shouting alone), power at $\alpha = 0.05$ is only $19.88\%$ under the conservative scenario and $45.91\%$ under the moderate scenario. Under Bonferroni-Holm family-wise error control across the 7 subdomains ($\alpha_{\text{adjusted}} = 0.05 / 4 = 0.0125$), power collapses to $7.25\%$ and $23.10\%$, respectively. This mathematically confirms that **individual category hypotheses cannot serve as confirmatory endpoints and must remain strictly exploratory secondary analyses**.
3. **Formal Retraction of the 75%–85% Four-Gate Sequence Power Claim:** Audit v3.0 (Section 3.4) and Publication Readiness v3.0 (Section 4.1) stated that "overall sequence power cannot exceed ~76%–85%" based on the heuristic calculation $0.90 \times 0.85 \approx 0.765$. **This claim is methodologically flawed and must be formally retracted**:
   - **Absence of Simulation Code:** Inspection of `simulate_phase3_power_comprehensive.py` confirms that zero simulations were performed for Gate 2 (B-FER non-inferiority via cluster bootstrap), Gate 3 (macro-recall non-inferiority via class-stratified cluster bootstrap), or Gate 4 (five-class macro-F1 superiority via class-stratified cluster bootstrap).
   - **Violation of Independence:** Multiplying marginal probabilities $\mathbb{P}(\text{Pass } G_1 \cap \dots \cap G_4) = \prod_{k=1}^4 \mathbb{P}(\text{Pass } G_k)$ assumes stochastic independence between gate outcomes. This assumption is invalid. All four gates evaluate the *exact same model checkpoints* ($E2$ vs. $E3$, across 3 seeds) on the *same benchmark dataset* ($N = 1,400$). A model with poor decision boundaries on confusers is more likely to exhibit elevated false alarms on benign audio or degraded recall on emergency classes. Gate outcomes are positively correlated.
   - **Fréchet-Hoeffding and Bonferroni Bounds:** For dependent hypotheses, the joint sequence power is bounded by:
     $$\max\left(0, 1 - \sum_{k=1}^4 (1 - \mathbb{P}(\text{Pass } G_k))\right) \le \mathbb{P}\left(\bigcap_{k=1}^4 \text{Pass } G_k\right) \le \min_{k \in \{1,2,3,4\}} \mathbb{P}(\text{Pass } G_k)$$
     If marginal powers were hypothetically $P_1 = 0.9986, P_2 = 0.95, P_3 = 0.90, P_4 = 0.85$, the lower bound would be $1 - (0.0014 + 0.05 + 0.10 + 0.15) = 0.6986$ (69.9%), while the upper bound would be $0.85$ (85%). Because $P_2, P_3,$ and $P_4$ have never been simulated, **overall four-gate sequence power is formally UNKNOWN**.

---

## 3. Issue 2: Mathematical Foundations of Paired Sign-Flip Permutation Testing (Gate 1)

### 3.1. Formal Formulation
In Gate 1, the benchmark includes $N = 350$ targeted confuser audio clips. Each clip $i \in \{1, \dots, N\}$ is evaluated by $S = 3$ independently trained random seeds for both model E3 (general expansion) and model E2 (targeted expansion). The mean difference in binary false emergency alert indicators for clip $i$ is:
$$d_i = \frac{1}{3} \sum_{s=1}^3 \left[ I(\hat{y}_{i,s}^{\text{E3}} \in \mathcal{E}) - I(\hat{y}_{i,s}^{\text{E2}} \in \mathcal{E}) \right] \in \left\{ -1, -\frac{2}{3}, -\frac{1}{3}, 0, +\frac{1}{3}, +\frac{2}{3}, +1 \right\}$$
where $\mathcal{E} = \{\text{Gunshot}, \text{Scream}, \text{Siren}, \text{Glass break}\}$. The test statistic is the sum of differences across all clips:
$$T = \sum_{i=1}^N d_i$$
Under the permutation null, a Monte Carlo sample of $R = 100,000$ sign-flipped statistics is generated:
$$T^{(r)} = \sum_{i=1}^N \sigma_i^{(r)} d_i, \quad \sigma_i^{(r)} \in \{-1, +1\} \text{ with } \mathbb{P}(\sigma_i^{(r)} = +1) = 0.5$$
The one-sided $p$-value for superiority of E2 over E3 is:
$$p = \frac{1}{R + 1} \left( 1 + \sum_{r=1}^R I(T^{(r)} \ge T_{\text{obs}}) \right)$$

### 3.2. Intra-Cluster Collapsing vs. Permutation Exchangeability
Audit v3.0 (Section 3.2) asserted that this procedure "guarantees exact nominal Type I error control ($\alpha = 0.05$) regardless of intra-cluster correlation." A rigorous methodological audit clarifies this claim:

1. **What Seed Collapsing Solves:** Collapsing the 3 seed evaluations into the single recording-level mean $d_i$ *before* sign-flipping accounts for within-recording multi-seed correlation. It eliminates pseudoreplication, ensuring the test does not treat $3 \times 350 = 1,050$ inferences as independent degrees of freedom.
2. **What Permutation Exactness Requires:** A sign-flip permutation test is finite-sample exact if and only if, under the null hypothesis, the vector of differences $(d_1, \dots, d_N)$ possesses **conditional reflection symmetry around zero**:
   $$\mathbb{P}(d_i = +c \mid |d_i| = c) = \mathbb{P}(d_i = -c \mid |d_i| = c) = 0.5 \quad \forall i \in \{1, \dots, N\}$$
   This reflection symmetry requires that the paired outcomes $(y_i^{\text{E2}}, y_i^{\text{E3}})$ are **exchangeable** under $\mathcal{H}_0$:
   $$(y_i^{\text{E2}}, y_i^{\text{E3}}) \stackrel{d}{=} (y_i^{\text{E3}}, y_i^{\text{E2}})$$
3. **Observational / Benchmarking vs. Randomized Experimental Settings:**
   - In randomized clinical trials, exchangeability is guaranteed by physical randomization (patients are randomly assigned to Treatment A or B).
   - In machine learning benchmarking, both models are evaluated deterministically on the exact same audio benchmark. Exchangeability is **not an experimental design guarantee**; it is the *sharp null hypothesis* $\mathcal{H}_0^{\text{sharp}}$ that the two algorithms produce identical error distributions across every acoustic profile.
   - If the true null hypothesis is weak ($\mathbb{E}[d_i] = 0$) but the error distribution is asymmetric across recording types (e.g., E2 makes concentrated errors on specific complex sounds while E3 makes diffuse errors across all sounds), $d_i$ is skewed. Under skewness, the sign-flip permutation test is **asymptotically valid** via the central limit theorem on $T$, but not strictly finite-sample exact.
4. **Physical Recording Cluster Boundary:** If the 350 confuser clips contain multiple 10-second segments extracted from the same continuous upload or acoustic recording session, the assumption of independence across $i$ is violated. The protocol must enforce that each of the $N = 350$ confuser instances originates from an **acoustically and physically independent root recording (`recording_id`)**.

---

## 4. Issue 3: Disambiguation & Standardization of Estimand Notation

### 4.1. Audit of Symbol Collisions Across Documents
A systematic scan across `reports/phase3_independent_confirmation_protocol_v2_1.md`, Table 5.1 (as visualized in the live preview), and `FYDP_Advisor_Brief.docx` reveals critical notation collisions:

```
                                  NOTATION COLLISION MAP
   
   Protocol v2.1 Table 5.1 (Gate 2)             Protocol v2.1 Sec 6.2 / Advisor Brief Sec 7
   ┌───────────────────────────────────┐        ┌───────────────────────────────────┐
   │ Estimand: θ₂                      │        │ Estimand: θ₂                      │
   │ Metric: Excess Benign B-FER       │  VS    │ Metric: 3-Model Probability       │
   │ Formula: E[B-FER_E2 - B-FER_E3]   │        │         Majority-Vote Ensemble    │
   │ Test: Cluster Bootstrap Bound     │        │ Test: Exact McNemar Test          │
   └───────────────────────────────────┘        └───────────────────────────────────┘
```

1. **The $\theta_2$ Collision:**
   - In Protocol v2.1 Table 5.1, $\theta_2$ is defined as Gate 2 Benign False Emergency Rate degradation: $\theta_2 = \mathbb{E}[\text{B-FER}_{\text{E2}} - \text{B-FER}_{\text{E3}}]$.
   - In Protocol v2.1 Section 6.2 (item 2) and Advisor Brief Section 7 (para 131), $\theta_2$ is defined as the *3-Model Probability Ensemble Estimand* evaluated via paired McNemar tests!
   - This creates intolerable ambiguity during preregistration and artifact interpretation.
2. **The Gate 4 Symbol Contradiction:**
   - In Table 5.1 Estimand row, the estimand is named $\Delta \text{Macro-F1} = \text{Macro-F1}^{\text{E2}} - \text{Macro-F1}^{\text{E3}}$.
   - In Table 5.1 Decision Rule row, the test rule references $\mathbf{\hat{\theta}_{4,\text{lower}}^{95\%} > 0.000}$.
   - The symbol $\theta_4$ is never defined in the Estimand row, creating an internal syntax mismatch.
3. **Sign Convention Alternation:**
   - **Gate 1 (Superiority):** $\theta_1 = \text{A-FER}_{\text{E3}} - \text{A-FER}_{\text{E2}}$ $\implies$ **Positive means E2 is superior**.
   - **Gate 2 (Non-Inferiority):** $\theta_2 = \text{B-FER}_{\text{E2}} - \text{B-FER}_{\text{E3}}$ $\implies$ **Positive means E2 degrades B-FER (worse)**.
   - **Gate 3 (Non-Inferiority):** $\theta_3 = \text{Rec}_{\text{macro}}^{\text{E3}} - \text{Rec}_{\text{macro}}^{\text{E2}}$ $\implies$ **Positive means E2 loses recall (worse)**.
   - **Gate 4 (Superiority):** $\Delta \text{Macro-F1} = \text{Macro-F1}^{\text{E2}} - \text{Macro-F1}^{\text{E3}}$ $\implies$ **Positive means E2 is superior**.
   - Because the sign of $\theta$ flips between superiority and non-inferiority gates, using unadorned numeric subscripts ($\theta_1, \theta_2, \theta_3, \theta_4$) invites sign errors in code and reporting.

### 4.2. Proposed Standardized Estimand Nomenclature
To eliminate ambiguity, all Phase 3 protocols, codebases, and reports must adopt descriptive subscripts:

$$\begin{aligned}
\mathbf{\theta_{\text{A-FER}}} &\equiv \mathbb{E}[\text{A-FER}_{\text{E3}} - \text{A-FER}_{\text{E2}}] && \text{(Gate 1 Primary Confirmatory: Confuser A-FER Reduction)} \\
\mathbf{\theta_{\text{B-FER}}} &\equiv \mathbb{E}[\text{B-FER}_{\text{E2}} - \text{B-FER}_{\text{E3}}] && \text{(Gate 2 Safety Non-Inferiority: Excess Benign B-FER)} \\
\mathbf{\theta_{\text{Rec}}} &\equiv \text{Rec}_{\text{macro}}^{\text{E3}} - \text{Rec}_{\text{macro}}^{\text{E2}} && \text{(Gate 3 Sensitivity Non-Inferiority: Macro-Recall Loss)} \\
\mathbf{\theta_{\text{F1}}} &\equiv \text{Macro-F1}^{\text{E2}} - \text{Macro-F1}^{\text{E3}} && \text{(Gate 4 Utility Superiority: Five-Class Macro-F1 Gain)} \\
\mathbf{\theta_{\text{ens}}} &\equiv \mathbb{E}[\text{A-FER}_{\text{E3}}^{\text{ens}} - \text{A-FER}_{\text{E2}}^{\text{ens}}] && \text{(Secondary Exploratory: 3-Model Ensemble McNemar Estimand)}
\end{aligned}$$

### 4.3. Reconciled Specification of the Four Confirmatory Gates (Correcting Table 5.1)

| Dimension | Gate 1 (Primary Confirmatory) | Gate 2 (Safety Non-Inferiority) | Gate 3 (Sensitivity Non-Inferiority) | Gate 4 (Operational Utility) |
|:---|:---|:---|:---|:---|
| **Endpoint Metric** | Confuser False Emergency Rate ($\text{A-FER}$) | Benign False Emergency Rate ($\text{B-FER}$) | Emergency Macro-Recall ($\text{Rec}_{\text{macro}}$) | Overall Five-Class Macro-F1 |
| **Evaluated Slice** | $N = 350$ Targeted Confusers | $N = 350$ Benign Background Clips | $N = 700$ Emergency Targets ($175$ per class) | Full Benchmark ($N = 1,400$) |
| **Estimand ($\mathbf{\theta}$)** | $\theta_{\text{A-FER}} = \mathbb{E}[\text{A-FER}_{\text{E3}} - \text{A-FER}_{\text{E2}}]$ | $\theta_{\text{B-FER}} = \mathbb{E}[\text{B-FER}_{\text{E2}} - \text{B-FER}_{\text{E3}}]$ | $\theta_{\text{Rec}} = \text{Rec}_{\text{macro}}^{\text{E3}} - \text{Rec}_{\text{macro}}^{\text{E2}}$ | $\theta_{\text{F1}} = \text{Macro-F1}^{\text{E2}} - \text{Macro-F1}^{\text{E3}}$ |
| **Null Hypothesis ($\mathbf{\mathcal{H}_0}$)** | $\mathcal{H}_{0,1}: \theta_{\text{A-FER}} \le 0$ (E2 no better than E3) | $\mathcal{H}_{0,2}: \theta_{\text{B-FER}} \ge \delta_{\text{B}}$ (E2 degrades B-FER) | $\mathcal{H}_{0,3}: \theta_{\text{Rec}} \ge \delta_{\text{R}}$ (E2 misses recall) | $\mathcal{H}_{0,4}: \theta_{\text{F1}} \le 0$ (E2 F1 no better than E3) |
| **Alternative ($\mathbf{\mathcal{H}_1}$)** | $\mathcal{H}_{1,1}: \theta_{\text{A-FER}} > 0$ (E2 superior) | $\mathcal{H}_{1,2}: \theta_{\text{B-FER}} < \delta_{\text{B}}$ (E2 non-inferior) | $\mathcal{H}_{1,3}: \theta_{\text{Rec}} < \delta_{\text{R}}$ (E2 non-inferior) | $\mathcal{H}_{1,4}: \theta_{\text{F1}} > 0$ (E2 superior) |
| **Decision Rule** | One-sided sign-flip permutation $p < 0.05$ | $\hat{\theta}_{\text{B-FER},\text{upper}}^{95\%} < \delta_{\text{B}} = 0.020$ | $\hat{\theta}_{\text{Rec},\text{upper}}^{95\%} < \delta_{\text{R}} = 0.025$ | $\hat{\theta}_{\text{F1},\text{lower}}^{95\%} > 0.000$ (5th percentile $> 0$) |
| **Pre-Registered Margin** | Not applicable (Superiority) | $\delta_{\text{B}} = 0.020$ ($2.0\%$) | $\delta_{\text{R}} = 0.025$ ($2.5\%$) | Not applicable (Superiority) |
| **Unit of Analysis** | Physical recording cluster | Physical recording cluster | Class-stratified recording cluster | Class-stratified recording cluster |
| **Statistical Test** | Paired recording sign-flip permutation ($R = 100,000$) | Inversion of 10,000 cluster-bootstrap percentile bound | Inversion of 10,000 class-stratified cluster-bootstrap bound | Paired class-stratified cluster-bootstrap test ($B = 10,000$) |
| **Confidence Interval** | One-sided 95% Lower Bootstrap Bound (or 2-sided 90% CI) | One-sided 95% Upper Bootstrap Bound | One-sided 95% Upper Bootstrap Bound | One-sided 95% Lower Bootstrap Bound |

> [!NOTE]
> In the preview of Table 5.1, Gate 1 specified a "Two-sided 95% Cluster-CI". For a one-sided superiority test at $\alpha = 0.05$, the mathematically congruent interval is a **One-sided 95% Lower Confidence Bound** (or a two-sided 90% confidence interval). Reporting a two-sided 95% CI has a 2.5% tail probability in each direction, which creates a technical mismatch with an $\alpha = 0.05$ decision boundary.

---

## 5. Issue 4: Operational and Statistical Audit of Margins ($\delta_{\text{B}}, \delta_{\text{R}}, \Delta_{\text{equiv}}$)

### 5.1. Non-Inferiority vs. Equivalence
A critical distinction in biostatistical methodology must be preserved:

- **Non-Inferiority (One-Sided):** Tests whether a new intervention does not produce *unacceptable degradation* in a safety or secondary performance dimension relative to a standard comparator:
  $$\mathcal{H}_0: \theta \ge \delta \quad \text{vs.} \quad \mathcal{H}_1: \theta < \delta \quad (\delta > 0)$$
  Rejection of $\mathcal{H}_0$ demonstrates non-inferiority; it does *not* prove equivalence.
- **Equivalence (Two-Sided):** Tests whether the performance difference between two systems falls entirely within a zone of practical indifference $[-\Delta_{\text{equiv}}, +\Delta_{\text{equiv}}]$:
  $$\mathcal{H}_{0,-}: \theta \le -\Delta_{\text{equiv}} \quad \text{or} \quad \mathcal{H}_{0,+}: \theta \ge +\Delta_{\text{equiv}}$$
  Evaluated via Two One-Sided Tests (TOST) at level $\alpha = 0.05$ (equivalent to verifying that the two-sided $90\%$ confidence interval lies entirely within $[-\Delta_{\text{equiv}}, +\Delta_{\text{equiv}}]$).

### 5.2. Operational Evaluation of Margins

```
           BENIGN B-FER MARGIN (δ_B = 0.020)           EMERGENCY RECALL MARGIN (δ_R = 0.025)
     ┌───────────────────────────────────────────┐    ┌───────────────────────────────────────────┐
     │ Operational Impact:                       │    │ Operational Impact:                       │
     │ - Max allowable excess false alarms: 2.0% │    │ - Max allowable recall drop: 2.5%         │
     │ - In 350 benign clips: <= 7 false alarms  │    │ - In 700 emergency clips: <= 17.5 missed  │
     │ - Risk: Nuisance dispatches, alert fatigue│    │ - Risk: Life-safety missed true alarms    │
     └───────────────────────────────────────────┘    └───────────────────────────────────────────┘
```

1. **Gate 2 Margin ($\delta_{\text{B}} = 0.020$ / 2.0 pp):**
   - In 350 benign background clips, a 2.0 pp increase represents at most $350 \times 0.020 = 7.0$ additional false emergency classifications.
   - In emergency dispatch centers, false alarms on ambient noise (traffic, ventilation, footsteps) cause alert fatigue and wasteful resource allocation.
   - **Governance:** While $\delta_{\text{B}} = 0.020$ is technically plausible, it represents an operational tolerance policy that requires formal advisor sign-off.
2. **Gate 3 Margin ($\delta_{\text{R}} = 0.025$ / 2.5 pp):**
   - In 700 emergency target clips ($175$ per class), a 2.5 pp drop in macro-recall allows at most $700 \times 0.025 = 17.5$ additional missed emergencies across the 4 classes (~4.4 missed screams, gunshots, sirens, or glass breaks per class).
   - In life-safety critical systems, missing true emergencies is severely penalized relative to false alarms.
   - **Governance:** Setting $\delta_{\text{R}} = 0.025$ asserts that losing up to 2.5% emergency recall is an acceptable trade-off for reducing confuser false alarms. This is an ethical and domain-specific decision that **must be explicitly approved by the faculty advisor**.
3. **Equivalence Margin ($\Delta_{\text{equiv}} = 0.030$ / 3.0 pp):**
   - If Gate 1 superiority fails to reach statistical significance ($p \ge 0.05$), standard frequentist hypothesis testing yields an uninformative failure to reject ("absence of evidence is not evidence of absence").
   - By pre-registering $\Delta_{\text{equiv}} = 0.030$ (the Smallest Effect Size of Interest, SESOI), the research team can evaluate TOST. If the 90% confidence interval for $\theta_{\text{A-FER}}$ falls within $[-0.030, +0.030]$, the paper can publish a high-impact negative result: *"Model-error guidance produces performance equivalent within 3.0 pp to random background expansion."*
   - **Recommendation:** Pre-register $\Delta_{\text{equiv}} = 0.030$ as a formal secondary contingency analysis in Protocol v2.1.

---

## 6. Issue 5: Formal Specification of the E5 Random-Confuser Control Arm

### 6.1. The Causal Confound in E2 vs. E3
The central research hypothesis asks whether *model-error-guided hard-negative data selection* outperforms standard data augmentation. In Phase 1 and Protocol v2.1, the primary comparison is E2 vs. E3:
- **E2 (Targeted):** Baseline training clips ($N=422$) + 212 confusers mined from baseline model errors across 7 acoustic subdomains.
- **E3 (General):** Baseline training clips ($N=422$) + 212 randomly sampled general background clips (ambient, rain, speech).

**The Confound:** E2 and E3 differ in two factors simultaneously:
1. *Acoustic Category Coverage:* E2 contains shouting, fireworks, and dog barks; E3 contains ambient sound, rain, and speech.
2. *Selection Policy:* E2 uses model-error mining; E3 uses random sampling.

If E2 outperforms E3 on confuser false emergency rate, peer reviewers will correctly object: *"E2 did not win because of model-error guidance; it won simply because it saw shouting and fireworks during training, whereas E3 did not."*

### 6.2. The E5 Control Arm Specification
To isolate the pure causal effect of model-error guidance, the protocol must specify an **E5 Control Arm (Random Confuser-Category Negatives)**:
- Baseline training set ($N=422$) + 212 audio clips sampled **at random from the exact same 7 confuser categories**, without baseline model error guidance.

#### Strict 8-Dimension Matching Protocol (E2 vs. E5)

| Dimension | Matching Requirement | Control Rationale |
|:---|:---|:---|
| **1. Category Proportions** | Exactly matched clip quotas across the 7 subdomains (e.g. ~30 shouting, ~30 fireworks, ~30 dog barks, etc., identical to E2). | Prevents category prevalence bias from confounding selection method. |
| **2. Total Sample Size** | Exactly 212 added negative clips ($N = 634$ total training set). | Controls total training data volume. |
| **3. Source Repository Pool** | Sampled from the identical candidate pool (Freesound / FSD50K candidate pool) from which E2 candidates were screened. | Eliminates repository and acoustic environment bias. |
| **4. Technical Audio Specs** | 16.0 kHz mono, 10.0-second length, matched RMS amplitude range, and matched SNR profile. | Eliminates audio recording quality differences. |
| **5. Model Architecture** | Frozen PANNs CNN14 backbone with 10,245-parameter linear probe classification head. | Controls feature extraction and parameter capacity. |
| **6. Hyperparameters** | Adam optimizer ($\text{lr} = 0.001$, $\beta_1 = 0.9, \beta_2 = 0.999$, $\mu = 1.0$), batch size 16, 20 epochs. | Controls convergence trajectory and regularization. |
| **7. Random Training Seeds** | Identical 3 random seeds ($42, 123, 2026$) with identical weight initializations. | Eliminates initialization variance. |
| **8. Early Stopping Criterion** | Evaluated strictly on the invariant shared validation split ($N = 105$). | Prevents validation leakage. |

### 6.3. Blinding and Leakage Safeguards
To guarantee valid causal inference:
1. **Pre-Acquisition Freezing:** The E5 clip manifest must be sampled and cryptographically hashed (`SHA-256`) *prior* to downloading or acquiring Test Set v3.
2. **Zero Outcome Exposure:** E5 data selection must have zero interaction with or feedback from test set evaluation outcomes.
3. **Blinded Evaluation:** Model evaluation on Test Set v3 must be conducted automatically using frozen evaluation scripts under the same multi-seed harness.

---

## 7. Priority Checklist for Faculty Advisor Approval Prior to Protocol Freeze

Before the Phase 3 protocol is officially frozen for execution, the following six decisions must be formally submitted to and approved by the faculty advisor:

```
                          PRE-FREEZE DECISION FLOW
  
  [1. Sequence Power] ──> Classify as UNKNOWN; do not claim 75-85% or >99%
  [2. Estimand Naming] ──> Adopt θ_A-FER, θ_B-FER, θ_Rec, θ_F1, θ_ens
  [3. Margins δ_B, δ_R] ─> Formally justify 2.0% B-FER and 2.5% Recall tolerances
  [4. Equivalence TOST] ─> Pre-register Δ_equiv = 0.030 for interpretable nulls
  [5. E5 Control Arm] ───> Approve 8-dimension matched random-confuser design
  [6. Test Set v3] ──────> Enforce physical recording_id clustering & blinding
```

- [ ] **Approval 1: Retraction of Numerical Sequence Power Claim:** Approve classifying overall four-gate sequence power as strictly UNKNOWN until a joint multi-gate simulation is completed, while reporting Gate 1 power as conditional on $\Delta \ge 0.08$ ($99.86\%$) and $\Delta = 0.05$ ($89.98\%$).
- [ ] **Approval 2: Adoption of Standardized Estimand Nomenclature:** Approve replacing colliding symbols with unambiguous descriptive estimands: $\theta_{\text{A-FER}}, \theta_{\text{B-FER}}, \theta_{\text{Rec}}, \theta_{\text{F1}}$, and $\theta_{\text{ens}}$.
- [ ] **Approval 3: Ratification of Operational Non-Inferiority Margins:** Formally ratify $\delta_{\text{B}} = 0.020$ (at most 2.0% excess benign false alarms) and $\delta_{\text{R}} = 0.025$ (at most 2.5% loss in emergency macro-recall) based on deployment risk tolerance.
- [ ] **Approval 4: Pre-Registration of Equivalence Margin ($\Delta_{\text{equiv}} = 0.030$):** Pre-register Two One-Sided Tests (TOST) against $[-0.030, +0.030]$ for $\theta_{\text{A-FER}}$ to enable publishing an informative finding of operational equivalence if Gate 1 superiority is not rejected.
- [ ] **Approval 5: Formal Inclusion of E5 Control Arm:** Approve the 8-dimension matched E5 control arm ($N = 634$, 212 random confuser clips matching E2 category quotas) to eliminate the causal confound between category coverage and model-error guidance.
- [ ] **Approval 6: Enforcement of Root Recording Clustering:** Approve enforcing cluster grouping at the canonical physical recording level (`recording_id`) to ensure no two audio windows in Test Set v3 share an acoustic recording session or upload source.
