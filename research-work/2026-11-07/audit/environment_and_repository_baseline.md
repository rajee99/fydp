# Phase 0 Environment and Repository Baseline

Recorded: 2026-11-07 (Asia/Dhaka)
Owner: Person B (as assigned in the two-person work plan)
Repository: `D:\ml\fydp\MyNet`

## Source control
- Branch: `master`
- HEAD commit: unavailable; repository reports no commits yet.
- Remote: none reported by `git remote -v`.
- Working tree: 17 top-level entries reported untracked; this baseline is not tied to a committed revision.

## Runtime and hardware
- Python: 3.11.9
- Virtual environment executable: `D:\ml\fydp\MyNet\.venv\Scripts\python.exe`
- PowerShell: 7.6.9
- PyTorch: 2.11.0+cu128
- torchaudio: 2.11.0+cu128
- CUDA build reported by package tag: 12.8; CUDA runtime availability was not verified because importing PyTorch did not return promptly.
- GPU: NVIDIA GeForce RTX 5060
- NVIDIA driver: 617.14
- GPU memory reported by `nvidia-smi`: 8151 MiB total, 6995 MiB free at capture time.
- Additional installed libraries: SoundFile 0.14.0, NumPy 2.4.6, SciPy 1.17.1, pandas 3.1.0.
- `librosa` and `scikit-learn` are not installed in this environment.

## Snapshot
Current `data/processed` CSV/JSON artifacts, files under `configs`, model `training_config.json` and `run_meta.json` artifacts, and the root reconciled experiment registry were copied into `baseline_snapshot/`. Copies are marked read-only. `audit/baseline_snapshot_sha256.csv` records the source path, snapshot path, byte size, and SHA-256 for each copied file.

## Open baseline limitations
There is no commit hash or Git remote to identify the source state. Keep the snapshot unchanged; once the team agrees a branch/commit convention, record it in the evidence register and decision log. CUDA execution should be verified separately before compute planning.
