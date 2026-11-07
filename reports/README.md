# Project Reports & Documentation Directory

This directory centralizes all technical reports, experimental evaluations, dataset integration audits, and diagnostic studies conducted for the **FYDP Audio Emergency Classification Project** (`MyNet`).

---

## 📑 Report Index & Chronological Summary

| Report File | Topic / Scope | Key Findings & Content |
|---|---|---|
| [`targeted_dataset_expansion_report.md`](file:///d:/ml/fydp/MyNet/reports/targeted_dataset_expansion_report.md) | Baseline Error Analysis & Target Expansion | Analyzed baseline failure modes on screams, glass breaking, and background noise; drafted acoustic collection guidelines. |
| [`nigens_integration_report.md`](file:///d:/ml/fydp/MyNet/reports/nigens_integration_report.md) | NIGENS Dataset Audit & Ingestion | Audited 898 candidate NIGENS clips; approved 366 immediately, flagged 80 for manual review, and strictly isolated evaluation folds 7–8. |
| [`combined_training_dataset_report.md`](file:///d:/ml/fydp/MyNet/reports/combined_training_dataset_report.md) | AudioSet + NIGENS Integration | Merged 527 AudioSet clips with 419 approved NIGENS samples into the unified 946-sample 5-class training manifest. |
| [`combined_split_report.md`](file:///d:/ml/fydp/MyNet/reports/combined_split_report.md) | Source-Grouped Train/Val Split | Partitioned 946 clips into 756 train (80%) and 190 val (20%) with zero source-group leakage across YouTube video IDs and NIGENS recordings. |
| [`panns_cnn14_combined_training_report.md`](file:///d:/ml/fydp/MyNet/reports/panns_cnn14_combined_training_report.md) | PANNs CNN14 Combined Model Training | Trained transfer head on 946 combined clips for 21 epochs. Best checkpoint achieved 89.47% validation accuracy and 0.8874 macro F1. |
| [`model_comparison_report.md`](file:///d:/ml/fydp/MyNet/reports/model_comparison_report.md) | Baseline vs. Combined Independent Evaluation | Head-to-head benchmark on 519 independent evaluation clips. Proved AudioSet Baseline superior (84.78% acc, 0.8195 F1) vs. Combined (81.70% acc, 0.7837 F1). |
| [`nigens_domain_mismatch_report.md`](file:///d:/ml/fydp/MyNet/reports/nigens_domain_mismatch_report.md) | Root-Cause Acoustic Diagnostics | Extracted 10 acoustic parameters across 1,275 clips. Discovered 17.6 dB noise floor disparity, bandwidth collapse in `other`, and silence-padding artifacts. |
| [`train_audio_inventory_report.txt`](file:///d:/ml/fydp/MyNet/reports/train_audio_inventory_report.txt) | AudioSet In-The-Wild Inventory Audit | Quantitative status of raw AudioSet training clips, sample rates, durations, and channels. |
| [`train_download_retry_report.txt`](file:///d:/ml/fydp/MyNet/reports/train_download_retry_report.txt) | YouTube Download Pipeline Audit | Logs and tracking for download attempts, missing audio recoveries, and replacement clip validation. |
| [`targeted_expansion_summary.md`](file:///d:/ml/fydp/MyNet/reports/targeted_expansion_summary.md) | In-The-Wild Dataset Expansion Manifest | Vetted and sampled 672 fresh AudioSet candidates (435 P1 exact quota fulfillment) strictly isolated from evaluation and previous training sets. |
| [`targeted_expansion_readiness_report.md`](file:///d:/ml/fydp/MyNet/reports/targeted_expansion_readiness_report.md) | Pre-Download Execution Readiness Audit | Comprehensive audit of batch downloader compatibility, quota recalculations, backup controls, and dry-run verification (GO decision). |
| [`targeted_expansion_download_report.md`](file:///d:/ml/fydp/MyNet/reports/targeted_expansion_download_report.md) | In-The-Wild Expansion Download Audit | Execution audit of 542 total downloads (P1 + P2 replenishment): 400 verified audio files (92.0% yield), reconciled subdomains, and diagnostics. |
| [`targeted_expansion_audit_report.md`](file:///d:/ml/fydp/MyNet/reports/targeted_expansion_audit_report.md) | Expansion Verification & Subdomain Reconciliation Audit | In-depth audit resolving reporting subdomain collisions, 5-point inventory reconciliation, and semantic validation template. |
| [`AUDIO_REVIEW_README.md`](file:///d:/ml/fydp/MyNet/reports/AUDIO_REVIEW_README.md) | Audio Content Review Workflow & Guide | User guide for the local single-clip audio review tool: playback, verdict tracking, keyboard hotkeys, and ambiguity rules. |
| [`targeted_expansion_validation_results_report.md`](file:///d:/ml/fydp/MyNet/reports/targeted_expansion_validation_results_report.md) | In-The-Wild Dataset Expansion Human Validation Report | Empirical results from human semantic review of all 400 audio files: 376 accepted (94.0%), 4 reclassified, 19 rejected, and 0 emergency contamination. |
| [`NIGENS_README.md`](file:///d:/ml/fydp/MyNet/reports/NIGENS_README.md) | Third-Party Dataset Documentation | Original license, class inventory, and creator attributions for the NIGENS acoustic database. |

---

## 🎯 Primary Operational Conclusion

The empirical findings documented across [`model_comparison_report.md`](file:///d:/ml/fydp/MyNet/reports/model_comparison_report.md) and [`nigens_domain_mismatch_report.md`](file:///d:/ml/fydp/MyNet/reports/nigens_domain_mismatch_report.md) confirm that the **AudioSet-only baseline model** ([`models/experiments/panns_cnn14/best_model.pt`](file:///d:/ml/fydp/MyNet/models/experiments/panns_cnn14/best_model.pt)) remains the highest-performing and most noise-robust emergency classifier, and is recommended for primary deployment.
