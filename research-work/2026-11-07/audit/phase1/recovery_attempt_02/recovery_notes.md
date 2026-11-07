# Phase 1 input recovery and execution evidence

Date: 2026-11-07 (Asia/Dhaka). Scope: Person A tasks 01.13 and 01.14 only. Status during recovery: incomplete. See the latest status report for final outcomes; this note records methods and caveats.

## Sources and input identity

Recovered official metadata:

- AudioSet v1 balanced training, evaluation and unbalanced training lists, linked from https://research.google.com/audioset/download.html and served by https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/ . The HTTP generation identifiers, last-modified times, sizes and SHA-256 values are in download_register.json.
- YAMNet class map from https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv . The downloaded response and hash are recorded; the moving upstream branch is not assumed to identify the original local file.
- Original AudioSet recordings are retrieved from their recorded YouTube IDs and segment times through the unchanged batch_downloader.download_single_segment function. Reviewed expansion inputs come from the existing review CSV; baseline input paths and source times come from combined_training_manifest.csv. No label decisions are newly certified.
- NIGENS v1: https://zenodo.org/records/2535878/files/NIGENS.zip?download=1 , record DOI 10.5281/zenodo.2535878. Range requests avoid downloading unrelated archive members. The central-directory inventory, selected 419 waveform names, fold lists and individual checksums are saved. ZIP CRC/length checks are applied before recovered files are written. The whole-archive MD5 is not verified because the whole archive is not downloaded.

These are upstream recoveries, not proof of byte identity with the unavailable historical local inputs. A successfully recovered AudioSet example differs from the file hash in the original combined manifest. Existing row-level review records are references tied to the historical recordings; this phase does not replace a later label-quality audit on recovered encodings.

## Execution environment

An isolated .venv under the Phase 1 audit folder was created from the bundled Python 3.12.14 interpreter. SoundFile, NumPy and pandas were installed to execute the existing scripts; yt-dlp was installed for original-source recovery. FFmpeg was already available. Exact resolved versions are in requirements_resolved.txt and environment.json; FFmpeg version details are in ffmpeg_version.txt.

Commands used from the repository root:

```powershell
& <bundled-python> -m venv research-work/2026-11-07/audit/phase1/.venv
& research-work/2026-11-07/audit/phase1/.venv/Scripts/python.exe -m pip install numpy pandas soundfile
& research-work/2026-11-07/audit/phase1/.venv/Scripts/python.exe -m pip install yt-dlp
& research-work/2026-11-07/audit/phase1/.venv/Scripts/python.exe research-work/2026-11-07/audit/phase1/recover_and_retry.py
& research-work/2026-11-07/audit/phase1/.venv/Scripts/python.exe research-work/2026-11-07/audit/phase1/recover_required_audio.py
& research-work/2026-11-07/audit/phase1/.venv/Scripts/python.exe research-work/2026-11-07/audit/phase1/recover_nigens.py
& research-work/2026-11-07/audit/phase1/.venv/Scripts/python.exe research-work/2026-11-07/audit/phase1/recover_nigens.py --reverse
```

Network/environment setup required sandbox escalation. No global Python packages were installed. The initial zero-attempt audio probe was an audit-helper error, corrected before actual recovery; audio_recovery_results.json contains the subsequent real one-attempt probe. One later FFmpeg failure succeeded on diagnostic retry, with full output preserved in audio_failure_4Iu-Em87lmA.txt. A truncated NIGENS range response was retried; previous failures remain in recovery registers. An optional README used unsupported compression and was excluded from recovery; all required waveform and fold entries remain eligible. This does not change either preparation script.

## Successful-script observations so far

The execution register records exit zero for prepare_manifest.py, prepare_eval_manifest.py and prepare_test_set_v2_candidates.py. A later successful prepare_experimental_manifests.py execution is saved in prepare_experimental_manifests_retry_03.txt; it generated experimental_manifests_summary.json and completed its checks. The copied script hashes are identical to the original scripts.

Target manifest: 775 generated rows versus 740 baseline. Evaluation manifest: 779 versus 744. All emergency segment/class keys match; Other increases from 175 to 210 and its selection differs. Current class_config.py defines six Other subdomains, each sampled at 35 by the existing generators. Their comments still describe five/175. No configuration was changed to force historical counts.

Candidate queue: 1,060 rows in both versions, with the same class totals but different IDs. collect_all_forbidden scans every processed CSV, including the existing candidate output and later manifests. The preserved baseline candidate IDs are therefore in the exclusion set during this rerun. Additionally, the sandbox target/eval manifests have already been regenerated. This run demonstrates current script execution, not reproduction of the earlier candidate set.

Experimental generation reused the fixed E1 split (422 train / 105 validation), generated full E2 (646 train / 105 validation) and an E3 candidate stage (646 train / 105 validation). That E3 candidate stage is not the final 634-row general-background training manifest; candidate generation is not training or a new experiment. See generated_output_register.json and generated_artifacts for exact outputs.

All successful output files are copied under generated_manifests or generated_artifacts. Raw metadata/audio remain within the ignored audit sandbox. No existing source scripts, baseline data, configs or research results have been overwritten. Baseline preservation is checked against the original 949-file inventory.
