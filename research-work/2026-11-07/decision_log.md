# Decision Log

| Date | Decision / fact recorded | Reason | Evidence | Responsible |
|---|---|---|---|---|
| 2026-11-07 | Use `research-work/2026-11-07/` as the dated Phase 0 workspace. | Follows Phase 0 task 00.01; no shared documentation location was specified. | `audit/environment_and_repository_baseline.md` | Person B |
| 2026-11-07 | Preserve the current manifests/configs as read-only copies and hash every copy. | Protect the existing experimental baseline and support reproducibility. | `audit/baseline_snapshot_sha256.csv` | Person B |
| 2026-11-07 | Record Git commit/repository identity as unresolved because this checkout has no commits or remote. | Do not invent a revision identifier. | `audit/environment_and_repository_baseline.md` | Person B |
