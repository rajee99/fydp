# Phase 1 repository and baseline audit

Date: 2026-11-07 (Asia/Dhaka). Owner: Person A. Scope: Work Plan v4.1 tasks 01.01–01.14, not the historical experiment called Phase 1.

Status: audit and available-input reproduction finished; full manifest regeneration remains blocked by missing original metadata/audio and audio dependencies. Do not certify Phase 1 fully complete. No training, inference or final-test evaluation was run.

## Repository and evidence

Audited commit: b737731fd633c6e68ea6a34162747fd9f05f29e2, branch master, origin https://github.com/rajee99/fydp.git. The inherited Phase 0 environment report describes the original D: drive host, not this computer. Local execution used bundled Python 3.12.14, with NumPy/pandas available and SoundFile/PyTorch/torchaudio unavailable. Windows Python launcher aliases were unusable.

`file_inventory.csv` lists 949 files under src, scripts, data, models and configs, with sizes and SHA-256 hashes. There is no root experiments directory; experiment artifacts live under models/experiments. There are no raw audio or model weight files in these directories, and data/raw is absent. These omissions match the repository .gitignore.

Graphify CLI was unavailable. Connected Graphify rejected this repository as unindexed for the connection. The committed graph.json supplied file locations, followed by inspection of current source. Graph results were navigation evidence, not proof of current behavior.

The required `graphify update .` was also attempted after adding the audit helper, but could not run because the CLI is absent. The committed graph remains unchanged. `git diff --check` passed for the tracked documentation updates.

## Task results

| Task | Status | Evidence or finding |
|---|---|---|
| 01.01 | Complete | file_inventory.csv; experiments are nested in models |
| 01.02 | Complete | src/prepare_manifest.py creates target_manifest.csv |
| 01.03 | Complete | src/prepare_eval_manifest.py creates eval_manifest.csv; scripts/prepare_test_set_v2_candidates.py creates the later candidate queue |
| 01.04 | Complete for static lineage | csv_lineage.csv and lineage below; original raw inputs absent, so content identity cannot be verified |
| 01.05 | Complete | src/batch_downloader.py, src/download_eval.py, src/collect_other_dataset.py; later scripts/download_test_set_v2.py and download_replenishments.py |
| 01.06 | Complete | batch_downloader.download_single_segment uses yt-dlp section extraction and FFmpeg 16 kHz mono conversion; dataset.load_audio_waveform resamples in memory |
| 01.07 | Complete for implemented training models | Compact CNN and PANNs frontends below; no runnable YAMNet training pipeline identified |
| 01.08 | Complete | src/train.py; scripts/run_phase1_experiments.py and run_phase2_experiments.py; saved training_config.json artifacts |
| 01.09 | Complete | Shared validation, checkpoint selection and isolation behavior below |
| 01.10 | Complete static search | metric_producers.csv; final evaluation paths src/evaluate_panns.py, compare_eval_models.py, evaluate_test_set_v2.py, evaluate_phase2_e4.py; scripts/run_independent_audit.py also runs test inference; recompute_phase1_metrics.py recomputes validation metrics |
| 01.11 | Complete | dataset_references.csv saves all matching Python lines with locations |
| 01.12 | Complete static review, limited assurance | evaluation_training_review.csv; reviewed training paths below; not a certification of historical experimental isolation |
| 01.13 | Partial | Seven isolated executions, two successful; see audit_summary.json and individual console logs |
| 01.14 | Complete for attempted runs | file hashes, counts, runtime, commands, return codes and console output saved |

## Manifest lineage

- target_manifest.csv: data/raw/yamnet_class_map.csv + balanced_train_segments.csv; label and exclusion rules from src/class_config.py. Emergency rows selected by label; Other sampled deterministically (seed 42, 35 per subdomain). CSV label filtering alone does not establish acoustic purity.
- eval_manifest.csv: same class map/configuration, with eval_segments.csv instead of balanced training. This is a separate historical evaluation manifest.
- Combined dataset: src/prepare_combined_dataset.py uses target_manifest.csv, other_manifest.csv, local AudioSet audio and NIGENS data; reads audio properties and hashes. src/split_combined_dataset.py splits combined_training_manifest.csv by source group, seed 42, 20% validation.
- Experimental E1/E2/E3: src/prepare_experimental_manifests.py consumes target_manifest.csv, train_split_manifest.csv, val_split_manifest.csv, eval_manifest.csv, pilot_manifest.csv, targeted_expansion_review_results.csv, balanced/unbalanced AudioSet lists and local audio. It preserves the baseline validation assignment. Its description of 224 E3 candidates is not the final 212-sample general-background addition count.
- Matched E2: src/build_matched_e2_manifest.py consumes e2_train_split_manifest.csv, e3_train_split_manifest.csv and shared_val_split_manifest.csv; removes 12 of 224 targeted additions with seed 42, preserving 422 baseline training rows and 105 validation rows.
- Test Set v2 candidate builder: scripts/prepare_test_set_v2_candidates.py reads class map and AudioSet eval/unbalanced training lists, and existing manifests for excluded IDs. It prioritizes eval candidates then supplements from unbalanced training. Its name does not prove independence from pretraining. Row-level source membership remains a Phase 2/6 investigation.

Full CSV references, including supplementary preparation scripts, are preserved in csv_lineage.csv.

## Features and training

src/dataset.py loads WAVs, averages channels, resamples to 16 kHz and pads short clips on the right or truncates long clips to 160,000 samples (10 seconds). CompactAudioCNN consumes natural-log power-mel features with FFT/window 400, hop 160, 64 mel bands, 125–7500 Hz and epsilon 1e-6.

PANNs receives raw waveforms through the dataset return_raw path. src/model.py uses an internal FFT/window 512, hop 160 frontend, 64 Slaney mel filters spanning 50–8000 Hz and a decibel transform. These feature pipelines must not be described as identical. LogmelFilterBank silently catches filter construction failures; runtime validation is needed once audio dependencies exist.

Historical primary runner: E1 422 training rows, matched E2 634 and E3 634, with shared 105-row validation and seeds 42/123/2026. The sampled saved E1 seed42 configuration specifies frozen CNN14, 5-class transfer head, Adam, learning rate 0.001, weight decay 0.0001, batch 16, maximum 30 epochs, class-weighted loss and validation macro-F1 checkpoint selection. Other saved configurations remain separate variants, not interchangeable defaults.

## Validation and evaluation-label review

src/train.py loads explicitly supplied train/validation manifests, checks class/source/hash/video metadata, and checks their video IDs against eval_manifest.csv. Its fallback creates a grouped train/validation split. Checkpoint selection uses validation macro-F1 for PANNs (or validation loss for the compact default). Both historical experiment runners pass shared_val_split_manifest.csv.

The inspected training code uses evaluation manifest video IDs for exclusion, not its labels as training targets or threshold-selection inputs. No explicit test-label-driven threshold tuning was identified in those paths. This static result does not prove that test observations never influenced historical decisions. The missing-evaluation-manifest branch only warns and skips the isolation check; that is a risk for future execution. Grouping and contamination validity require the later audits.

Final metric producers and all additional matching lines are saved in metric_producers.csv. No metrics were recomputed on audio and no new primary test was accessed.

## Counts and reproduction

Committed manifest rows (not counts of available audio): target 740; historical eval 744; combined training 946; Test Set v2 750. Historical E1 training 422, matched E2 training 634, E3 training 634, shared validation 105. Per-file/class counts are in manifest_counts.json; different manifest stages must not be conflated.

Original scripts and processed inputs were copied into reproduction_sandbox, preserving their relative directory structure. Scripts were executed unchanged there; outputs stayed in that ignored sandbox. Five generated CSVs matched committed bytes exactly: matched E2 train (634), validation (105), master (739), combined train (756) and combined validation (190). These reproduce transformations of existing manifests, not upstream provenance or audio validity.

prepare_manifest.py and prepare_eval_manifest.py failed because yamnet_class_map.csv is absent; their segment lists are also absent. prepare_combined_dataset.py and prepare_experimental_manifests.py failed importing SoundFile and also need omitted audio. prepare_test_set_v2_candidates.py failed on absent raw metadata. Full tracebacks are saved. Do not substitute newly downloaded lists without recording their versions and hashes.

SHA-256 verification found zero changes in the 949 audited baseline files. Existing research-work/baseline_snapshot copies were never used as output targets. Logs use .txt so the root *.log ignore rule does not hide evidence.

## Remaining Phase 1 work

Recover the versioned original raw CSV inputs and licensed audio used by the manifests; provision an isolated audio environment; repeat upstream generation and audio validation with those inputs. Preserve failures and compare generated outputs with this baseline. No claim of unseen data, full reproducibility, or completion of the Phase 9A gate follows from this audit.
