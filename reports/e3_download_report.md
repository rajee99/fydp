# E3 General Background Audio Download & Acquisition Report

**Project:** `D:\ml\fydp\MyNet`  
**Date:** October 30, 2026  
**Candidate Manifest:** [`data/processed/e3_general_train_candidates.csv`](file:///d:/ml/fydp/MyNet/data/processed/e3_general_train_candidates.csv)  
**Results Log:** [`data/processed/e3_download_results.csv`](file:///d:/ml/fydp/MyNet/data/processed/e3_download_results.csv)  
**Audio Destination:** [`data/raw/e3_general_background/other/`](file:///d:/ml/fydp/MyNet/data/raw/e3_general_background/other/)  
**Overall Yield:** 174/224 (77.7%)  

---

## 1. Executive Summary

This report documents the download and technical audio validation of the **224 AudioSet general background candidates** (E3) sampled to answer the core research question:

> *"Does model-error-guided hard-negative data selection improve emergency audio classification more effectively than simply increasing the amount of general background audio?"*

All candidate audio segments were extracted via `yt-dlp` using strict 16 kHz mono 16-bit PCM audio extraction, completely isolated in `data/raw/e3_general_background/other/`, with zero leakage into the shared 105-sample validation set.

---

## 2. Acquisition Yield by Subdomain

| Subdomain | Target Candidates | Successfully Downloaded | Failed / Unavailable | Yield Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **`animals`** | 37 | 28 | 9 | 75.7% |
| **`environmental`** | 38 | 34 | 4 | 89.5% |
| **`household`** | 37 | 25 | 12 | 67.6% |
| **`music`** | 37 | 27 | 10 | 73.0% |
| **`speech`** | 37 | 29 | 8 | 78.4% |
| **`traffic`** | 38 | 31 | 7 | 81.6% |
| **Total** | **224** | **174** | **50** | **77.7%** |

---

## 3. Failure Reason Analysis

All non-successful candidate attempts were classified according to error signatures:

| Failure Category | Occurrences | Share of Failures (%) | Description |
| :--- | :---: | :---: | :--- |
| Download / Extraction Error | 25 | 50.0% | YouTube source video unavailable, removed, or deleted |
| Video Deleted / Private / Removed | 20 | 40.0% | YouTube source video unavailable, removed, or deleted |
| Audio Validation Violation (format/duration) | 5 | 10.0% | YouTube source video unavailable, removed, or deleted |

---

## 4. Technical Audio Validation Audit

Every downloaded audio file in `data/raw/e3_general_background/` underwent automated inspection:
- **Sample Rate:** 100% verified 16,000 Hz.
- **Channel Layout:** 100% verified single-channel (mono).
- **Format & Subtype:** 100% verified 16-bit signed PCM WAV.
- **Duration Range:** Min: 9.45s, Max: 10.00s, Mean: 9.99s.
- **Corrupt / Empty Audio Files:** 0 occurrences.

---

## 5. Next Stage: Content Validation Protocol

Exactly **174 technically verified WAV files** are registered into [`data/processed/e3_content_validation.csv`](file:///d:/ml/fydp/MyNet/data/processed/e3_content_validation.csv).

Human review will be conducted via [`src/review_tool.py`](file:///d:/ml/fydp/MyNet/src/review_tool.py):
```bash
python src/review_tool.py --manifest data/processed/e3_content_validation.csv --results-file data/processed/e3_review_results.csv
```
Review criteria require confirming that each clip is a genuine general background sound and rejecting any emergency contamination (screams, glass breaks, gunshots, sirens) or misleading categories.