# Emergency Audio Classification: Hard-Negative Selection & Confirmatory Evaluation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: FYDP](https://img.shields.io/badge/Project-FYDP-brightgreen.svg)]()

This repository contains the complete research codebase, experimental manifests, evaluation outputs, audit reports, and confirmatory protocols for the **Final Year Design Project (FYDP)** on robust emergency audio event classification.

---

## 🎯 Research Overview

Automated acoustic emergency detection systems (monitoring for gunshots, screams, sirens, and breaking glass) are severely prone to high false-alarm rates when deployed in unconstrained ambient environments. This project investigates:

> **Core Research Inquiry:** *Does model-error-guided hard-negative data selection improve the reliability and false-alarm rejection of emergency audio classification more effectively than general background expansion, without degrading life-safety sensitivity?*

The project evaluates transfer learning using a frozen **PANNs CNN14** acoustic backbone with a specialized linear classification head across five target classes:
- **0: Scream**
- **1: Glass Breaking**
- **2: Gunshot**
- **3: Siren**
- **4: Other** (Benign Background / Ambient Non-Emergency)

---

## 📁 Repository Structure

```text
├── configs/            # Experiment configurations, ablation definitions, and class mappings
├── data/
│   └── processed/      # Ground-truth manifests, source-grouped splits, and candidate inventories
├── models/
│   ├── checkpoints/    # Model metadata, training configs, and training histories
│   └── experiments/    # Evaluation reports, prediction CSVs, and Phase 1/Phase 2 run metrics
├── reports/            # Research documentation, Phase 3 protocols, statistical audits, and briefs
│   └── advisor_submission_package/ # Consolidated deliverables for faculty review
├── research-work/      # Phase 0 baseline snapshots, evidence register, and decision logs
├── scripts/            # Reproducible pipelines: data downloading, evaluation, and power simulations
├── src/                # Core Python package: models, datasets, transforms, and training routines
├── tests/              # Automated unit tests, manifest integrity checks, and validation suites
├── AGENTS.md           # Repository instructions and graphify knowledge graph rules
├── master_experiment_registry_reconciled.csv # Audited master registry of all experimental runs
└── README.md           # Project documentation and reproduction guide
```

> **Note on Audio & Checkpoints:** Raw audio files (`.wav`) and heavy PyTorch binary weights (`.pt`, `.pth`) are excluded from Git version control to comply with repository size limits and data licensing. All evaluation predictions, training histories, and data manifests are fully committed for verifiable auditing.

---

## 🔬 Experimental Phases

### Phase 1: Baseline & Model Comparison
- Benchmarked compact CNN architectures against **PANNs CNN14** pretrained on AudioSet.
- Evaluated performance on baseline training ($N = 422$) vs. targeted expansion ($N = 634$) and general background expansion ($N = 634$).
- Demonstrated the performance superiority of pretrained audio representations.

### Phase 2: Category Ablation & Diagnostic Evaluation (E4)
- Investigated the individual impact of seven confuser subdomains (`human_shouting`, `fireworks_firecrackers`, `metal_impacts`, `door_slams`, `construction_tools`, `car_alarms_detectors`, `whistles_high_pitch`).
- Established recording-cluster-aware statistical testing to avoid multi-seed pseudoreplication.

### Phase 3: Independent Confirmatory Evaluation (Protocol v2.1 / v3.2)
- Formally pre-registered a **Hierarchical Fixed-Sequence Gatekeeper Testing Architecture** on an independent benchmark ($N = 1,400$):
  1. **Gate 1 (Primary Confirmatory):** Confuser False Emergency Rate (`theta_A_FER`) via paired sign-flip permutation ($R = 100,000$).
  2. **Gate 2 (Safety Non-Inferiority):** Benign False Emergency Rate (`theta_B_FER`) via cluster-bootstrap bound ($\delta_{\text{B}} = 0.020$).
  3. **Gate 3 (Sensitivity Non-Inferiority):** Emergency Macro-Recall (`theta_Rec`) via class-stratified cluster-bootstrap bound ($\delta_{\text{R}} = 0.025$).
  4. **Gate 4 (Operational Utility):** Five-class Macro-F1 (`theta_F1`) via class-stratified cluster bootstrap.
- Introduced the **E5 Random-Confuser Control Arm** ($N = 634$, 212 matched random confusers) to eliminate the causal confound between category coverage and model-error guidance.

---

## 🚀 Setup & Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.12/3.14)
- Git & Git LFS (if handling binary assets)
- CUDA-compatible GPU (optional, CPU execution supported)

### Environment Installation
```bash
# Clone the repository
git clone git@github.com:rajee99/fydp.git
cd fydp

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # or install torch, torchaudio, numpy, scipy, pandas
```

### Running Tests & Integrity Checks
```bash
# Run unit and manifest verification tests
pytest tests/
```

### Running Power Simulations
```bash
# Execute Phase 3 Monte Carlo statistical power simulation
python scripts/simulate_phase3_power_comprehensive.py
```

---

## 📊 Key Documentation & Reports

- **Master Experiment Registry:** [`master_experiment_registry_reconciled.csv`](master_experiment_registry_reconciled.csv)
- **Phase 3 Protocol (v2.1):** [`reports/phase3_independent_confirmation_protocol_v2_1.md`](reports/phase3_independent_confirmation_protocol_v2_1.md)
- **Pre-Freeze Consistency Audit (v3.2):** [`reports/phase3_protocol_v3_2_pre_freeze_audit.md`](reports/phase3_protocol_v3_2_pre_freeze_audit.md)
- **Pre-Freeze Approval Checklist:** [`reports/phase3_protocol_v3_2_approval_checklist.md`](reports/phase3_protocol_v3_2_approval_checklist.md)
- **Faculty Advisor Brief:** [`FYDP_Advisor_Brief.docx`](FYDP_Advisor_Brief.docx)

---

## 📄 License & Attribution

This project is developed as part of an academic Final Year Design Project (FYDP). Code is licensed under the MIT License. Audio dataset manifests reference samples from Freesound and AudioSet under their respective Creative Commons licenses.
