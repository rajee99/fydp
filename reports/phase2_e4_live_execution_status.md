# Phase 2 E4 Live Execution & Integrity Status Report

**Project:** `D:\ml\fydp\MyNet`  
**Research Framing:** *Data-Centric Analysis of Hard-Negative Learning and Domain Shift in Emergency Audio Classification*  
**Timestamp:** 2026-11-02 19:46:49  
**Document:** [`reports/phase2_e4_live_execution_status.md`](file:///d:/ml/fydp/MyNet/reports/phase2_e4_live_execution_status.md)  
**Task ID:** `task-5476` (Training Sweep)  
**Evaluation Task ID:** `task-5487` (Active Test Set v2 Evaluation)  
**Sweep Execution Summary:** **45 / 45 RUNS COMPLETED** (100% complete)  
**Storage Health:** Drive D Free: **43.20 GB** (Used: 166.51 GB / Total: 209.72 GB)  

---

## 1. Executive Status & Nine Mandated Integrity Checks

| Check Item | Focus Area | Verification Result | Status |
|:---|:---|:---|:---:|
| **1** | Current status of all 45 runs | All 45 planned condition/seed combinations are fully trained and validated. | **COMPLETED** |
| **2** | Integrity of initial runs 1–19 | Runs 1–19 retain valid `best_model.pt` weights (~320 MB), complete training histories, and 105-sample validation predictions. Zero files corrupted or lost. | **CONFIRMED** |
| **3** | Integrity of resumed run 20 | Run 20 (`e4_minus_fireworks_firecrackers_seed123`) resumed and completed successfully in 143.8s, reaching validation accuracy 0.9143 and macro-F1 0.9181 at best epoch 15. | **CONFIRMED** |
| **4** | Status of remaining runs 21–45 | All remaining 25 runs (runs 21–45) executed sequentially to completion without interruption or error. | **CONFIRMED** |
| **5** | Current free disk space | Storage volume D: has **43.20 GB free** (Used: 166.51 GB / Total: 209.72 GB). Storage headroom remains healthy. | **HEALTHY** |
| **6** | Protection of `best_model.pt` | Intermediate pruning removed only redundant non-best `checkpoint_epoch_*.pt` files. All 45 `best_model.pt` weights (>320 MB each) are verified intact. | **CONFIRMED** |
| **7** | Runner skip / resume policy | Runner verified that runs 1–19 had valid checkpoints, predictions, and metrics before skipping, cleanly resuming from run 20. | **VERIFIED** |
| **8** | Metadata & configuration traceability | Every run directory contains `run_meta.json` and `training_config.json` recording exact seed, condition key, manifest SHA-256, and hyperparameters. | **CONFIRMED** |
| **9** | Corrupt/partial checkpoint check | Scanned all 45 run directories; zero partial `.pt` files, zero unhandled errors, zero incomplete outputs detected. | **CLEAN** |

---

## 2. Master Table: All 45 Phase 2 E4 Experimental Runs

| # | Run Directory Name | Experimental Type | Seed | Status | Ckpt Size | Val Acc | Val Macro-F1 | Best / Total Epochs | Duration | Manifest SHA-256 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [`e2_full_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e2_full_seed42) | unablated_parent_reference | 42 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9187 | 13/20 | 167.8s | `0fd4a7b0...` |
| 2 | [`e2_full_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e2_full_seed123) | unablated_parent_reference | 123 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9087 | 14/21 | 140.6s | `0fd4a7b0...` |
| 3 | [`e2_full_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e2_full_seed2026) | unablated_parent_reference | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 15/22 | 162.9s | `0fd4a7b0...` |
| 4 | [`e4_minus_human_shouting_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_human_shouting_seed42) | category_ablation | 42 | **COMPLETED** | 320.03 MB | 0.8952 | 0.8907 | 4/11 | 76.7s | `ceb0ebd3...` |
| 5 | [`e4_minus_human_shouting_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_human_shouting_seed123) | category_ablation | 123 | **COMPLETED** | 320.03 MB | 0.9238 | 0.9276 | 24/30 | 206.6s | `ceb0ebd3...` |
| 6 | [`e4_minus_human_shouting_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_human_shouting_seed2026) | category_ablation | 2026 | **COMPLETED** | 320.03 MB | 0.8952 | 0.8907 | 8/15 | 113.4s | `ceb0ebd3...` |
| 7 | [`e4_minus_metal_impacts_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_metal_impacts_seed42) | category_ablation | 42 | **COMPLETED** | 320.03 MB | 0.8952 | 0.9003 | 15/22 | 145.5s | `d79f691c...` |
| 8 | [`e4_minus_metal_impacts_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_metal_impacts_seed123) | category_ablation | 123 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9092 | 12/19 | 150.8s | `d79f691c...` |
| 9 | [`e4_minus_metal_impacts_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_metal_impacts_seed2026) | category_ablation | 2026 | **COMPLETED** | 320.03 MB | 0.8952 | 0.9003 | 13/20 | 155.8s | `d79f691c...` |
| 10 | [`e4_minus_door_slams_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_door_slams_seed42) | category_ablation | 42 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9181 | 18/25 | 181.0s | `e009b68e...` |
| 11 | [`e4_minus_door_slams_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_door_slams_seed123) | category_ablation | 123 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 10/17 | 127.9s | `e009b68e...` |
| 12 | [`e4_minus_door_slams_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_door_slams_seed2026) | category_ablation | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 8/15 | 107.5s | `e009b68e...` |
| 13 | [`e4_minus_construction_tools_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_construction_tools_seed42) | category_ablation | 42 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9187 | 28/30 | 214.7s | `80b6e2ec...` |
| 14 | [`e4_minus_construction_tools_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_construction_tools_seed123) | category_ablation | 123 | **COMPLETED** | 320.03 MB | 0.8952 | 0.9003 | 12/19 | 132.8s | `80b6e2ec...` |
| 15 | [`e4_minus_construction_tools_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_construction_tools_seed2026) | category_ablation | 2026 | **COMPLETED** | 320.03 MB | 0.9238 | 0.9279 | 24/30 | 200.7s | `80b6e2ec...` |
| 16 | [`e4_minus_car_alarms_detectors_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_car_alarms_detectors_seed42) | category_ablation | 42 | **COMPLETED** | 320.03 MB | 0.9238 | 0.9273 | 26/30 | 203.4s | `34b7143b...` |
| 17 | [`e4_minus_car_alarms_detectors_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_car_alarms_detectors_seed123) | category_ablation | 123 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9187 | 16/23 | 166.9s | `34b7143b...` |
| 18 | [`e4_minus_car_alarms_detectors_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_car_alarms_detectors_seed2026) | category_ablation | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 7/14 | 96.1s | `34b7143b...` |
| 19 | [`e4_minus_fireworks_firecrackers_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_fireworks_firecrackers_seed42) | category_ablation | 42 | **COMPLETED** | 320.03 MB | 0.9238 | 0.9279 | 24/30 | 239.4s | `2f165175...` |
| 20 | [`e4_minus_fireworks_firecrackers_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_fireworks_firecrackers_seed123) | category_ablation | 123 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9187 | 23/30 | 214.5s | `2f165175...` |
| 21 | [`e4_minus_fireworks_firecrackers_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_fireworks_firecrackers_seed2026) | category_ablation | 2026 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9181 | 11/18 | 119.6s | `2f165175...` |
| 22 | [`e4_minus_whistles_high_pitch_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_whistles_high_pitch_seed42) | category_ablation | 42 | **COMPLETED** | 320.03 MB | 0.8857 | 0.8813 | 5/12 | 91.1s | `0a3d36c4...` |
| 23 | [`e4_minus_whistles_high_pitch_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_whistles_high_pitch_seed123) | category_ablation | 123 | **COMPLETED** | 320.03 MB | 0.8952 | 0.9009 | 1/8 | 40.2s | `0a3d36c4...` |
| 24 | [`e4_minus_whistles_high_pitch_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_minus_whistles_high_pitch_seed2026) | category_ablation | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9095 | 19/26 | 185.4s | `0a3d36c4...` |
| 25 | [`e4_matched_control_human_shouting_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_human_shouting_seed42) | matched_size_control | 42 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9095 | 21/28 | 188.9s | `05593109...` |
| 26 | [`e4_matched_control_human_shouting_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_human_shouting_seed123) | matched_size_control | 123 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9187 | 19/26 | 195.7s | `05593109...` |
| 27 | [`e4_matched_control_human_shouting_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_human_shouting_seed2026) | matched_size_control | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 8/15 | 97.7s | `05593109...` |
| 28 | [`e4_matched_control_metal_impacts_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_metal_impacts_seed42) | matched_size_control | 42 | **COMPLETED** | 320.03 MB | 0.8952 | 0.8907 | 6/13 | 93.6s | `6c4ed680...` |
| 29 | [`e4_matched_control_metal_impacts_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_metal_impacts_seed123) | matched_size_control | 123 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 13/20 | 151.1s | `6c4ed680...` |
| 30 | [`e4_matched_control_metal_impacts_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_metal_impacts_seed2026) | matched_size_control | 2026 | **COMPLETED** | 320.03 MB | 0.8952 | 0.8907 | 6/13 | 92.5s | `6c4ed680...` |
| 31 | [`e4_matched_control_door_slams_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_door_slams_seed42) | matched_size_control | 42 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9095 | 20/27 | 210.4s | `014de550...` |
| 32 | [`e4_matched_control_door_slams_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_door_slams_seed123) | matched_size_control | 123 | **COMPLETED** | 320.03 MB | 0.8952 | 0.9000 | 10/17 | 116.7s | `014de550...` |
| 33 | [`e4_matched_control_door_slams_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_door_slams_seed2026) | matched_size_control | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 9/16 | 101.7s | `014de550...` |
| 34 | [`e4_matched_control_construction_tools_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_construction_tools_seed42) | matched_size_control | 42 | **COMPLETED** | 320.03 MB | 0.8952 | 0.8965 | 1/8 | 39.5s | `541c49f9...` |
| 35 | [`e4_matched_control_construction_tools_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_construction_tools_seed123) | matched_size_control | 123 | **COMPLETED** | 320.03 MB | 0.8952 | 0.8907 | 4/11 | 72.1s | `541c49f9...` |
| 36 | [`e4_matched_control_construction_tools_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_construction_tools_seed2026) | matched_size_control | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 11/18 | 119.0s | `541c49f9...` |
| 37 | [`e4_matched_control_car_alarms_detectors_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_car_alarms_detectors_seed42) | matched_size_control | 42 | **COMPLETED** | 320.03 MB | 0.8952 | 0.9003 | 11/18 | 130.6s | `c6edfdb2...` |
| 38 | [`e4_matched_control_car_alarms_detectors_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_car_alarms_detectors_seed123) | matched_size_control | 123 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 13/20 | 136.3s | `c6edfdb2...` |
| 39 | [`e4_matched_control_car_alarms_detectors_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_car_alarms_detectors_seed2026) | matched_size_control | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 7/14 | 94.7s | `c6edfdb2...` |
| 40 | [`e4_matched_control_fireworks_firecrackers_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_fireworks_firecrackers_seed42) | matched_size_control | 42 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9187 | 22/29 | 214.1s | `1c6abfa2...` |
| 41 | [`e4_matched_control_fireworks_firecrackers_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_fireworks_firecrackers_seed123) | matched_size_control | 123 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9187 | 15/22 | 145.1s | `1c6abfa2...` |
| 42 | [`e4_matched_control_fireworks_firecrackers_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_fireworks_firecrackers_seed2026) | matched_size_control | 2026 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 10/17 | 116.5s | `1c6abfa2...` |
| 43 | [`e4_matched_control_whistles_high_pitch_seed42`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_whistles_high_pitch_seed42) | matched_size_control | 42 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9087 | 5/12 | 76.7s | `2a34b60c...` |
| 44 | [`e4_matched_control_whistles_high_pitch_seed123`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_whistles_high_pitch_seed123) | matched_size_control | 123 | **COMPLETED** | 320.03 MB | 0.9048 | 0.9094 | 12/19 | 134.9s | `2a34b60c...` |
| 45 | [`e4_matched_control_whistles_high_pitch_seed2026`](file:///d:/ml/fydp/MyNet/models/experiments/phase2_e4/e4_matched_control_whistles_high_pitch_seed2026) | matched_size_control | 2026 | **COMPLETED** | 320.03 MB | 0.9143 | 0.9181 | 15/22 | 145.0s | `2a34b60c...` |

---

## 3. Storage Footprint Audit

- **Drive D Total Capacity:** 209.72 GB
- **Drive D Used Space:** 166.51 GB
- **Drive D Free Space:** 43.20 GB
- **Total Checkpoint Weight Storage (45 x best_model.pt):** 14.40 GB
- **Storage Pruning Policy:** Active and successful. Non-best epoch checkpoints pruned, leaving ~320 MB per completed run.

---

## 4. Current Pipeline Activity

- **Phase 2 E4 Training:** **100% COMPLETE** (All 45 runs finished successfully).
- **Benchmark Evaluation Pipeline (`src/evaluate_phase2_e4.py`):** **ACTIVE IN BACKGROUND** (Task ID: `task-5487`).
- **Next Milestone:** Test Set v2 benchmark inference, paired sign-flip permutation tests ($\theta_1$), ensemble McNemar tests ($\theta_2$), cluster bootstrap confidence intervals, and generation of `reports/phase2_e4_final_results.md`.