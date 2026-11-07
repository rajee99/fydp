"""Unit and integration tests for the NIGENS Dataset Integration Pipeline."""

import csv
import hashlib
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integrate_nigens import (
    ROOT,
    FYDP_CLASSES,
    REQUIRED_MANIFEST_COLUMNS,
    load_nigens_folds,
    scan_and_process_nigens,
    compute_file_sha256,
)

MANIFEST_PATH = ROOT / "data" / "processed" / "nigens_manifest.csv"
REPORT_PATH = ROOT / "reports" / "nigens_integration_report.md"
if not REPORT_PATH.exists():
    REPORT_PATH = ROOT / "data" / "processed" / "nigens_integration_report.md"
NIGENS_DIR = ROOT / "NIGENS"


@pytest.fixture(scope="module")
def manifest_rows():
    """Load and yield rows from nigens_manifest.csv."""
    assert MANIFEST_PATH.exists(), f"Manifest file not found: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def test_manifest_schema_and_columns():
    """Verify that nigens_manifest.csv has all required columns in exact order."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REQUIRED_MANIFEST_COLUMNS, (
            f"Columns mismatch: expected {REQUIRED_MANIFEST_COLUMNS}, got {reader.fieldnames}"
        )


def test_every_manifest_entry_points_to_existing_file(manifest_rows):
    """Verify that every entry in the manifest points to a real file on disk."""
    assert len(manifest_rows) > 0, "Manifest is empty"
    missing_files = []
    for r in manifest_rows:
        file_path = ROOT / r["file_path"]
        if not file_path.is_file():
            missing_files.append(r["file_path"])
    assert not missing_files, f"Found {len(missing_files)} missing files on disk: {missing_files[:5]}"


def test_no_duplicate_file_hashes_in_included_samples(manifest_rows):
    """Verify that included samples contain strictly unique file hashes."""
    included_rows = [r for r in manifest_rows if r["review_status"] == "INCLUDED"]
    assert len(included_rows) > 0, "No samples were included"

    seen_hashes = set()
    duplicate_hashes = []
    for r in included_rows:
        h = r["file_hash"]
        if h in seen_hashes:
            duplicate_hashes.append((r["file_path"], h))
        seen_hashes.add(h)

    assert not duplicate_hashes, f"Duplicate hashes in included samples: {duplicate_hashes}"


def test_no_evaluation_leakage(manifest_rows):
    """Verify that no sample from original evaluation folds (folds 7 and 8) is included."""
    eval_fold_rows = [r for r in manifest_rows if r["original_fold"] in ["fold7", "fold8"]]
    assert len(eval_fold_rows) == 217, f"Expected 217 evaluation fold samples, got {len(eval_fold_rows)}"

    leaked = [r for r in eval_fold_rows if r["review_status"] == "INCLUDED"]
    assert not leaked, f"Evaluation fold leakage detected! {len(leaked)} samples included: {[r['file_path'] for r in leaked[:5]]}"

    for r in eval_fold_rows:
        assert r["review_status"] == "EXCLUDED", f"Eval fold file {r['file_path']} has status {r['review_status']}"
        assert "evaluation_fold" in r["exclusion_reason"], (
            f"Expected 'evaluation_fold' in exclusion_reason, got '{r['exclusion_reason']}'"
        )


def test_no_cross_dataset_leakage_with_existing_project_data(manifest_rows):
    """Verify that no NIGENS file hash collides with existing train, eval, or pilot audio."""
    existing_hashes = {}
    for sub in ["train", "eval", "pilot"]:
        dir_path = ROOT / "data" / "raw" / sub
        if dir_path.exists():
            for p in dir_path.rglob("*.wav"):
                existing_hashes[compute_file_sha256(p)] = p.as_posix()

    colliding = []
    for r in manifest_rows:
        if r["file_hash"] in existing_hashes:
            colliding.append((r["file_path"], existing_hashes[r["file_hash"]]))

    assert not colliding, f"Cross-dataset collisions found: {colliding}"


def test_original_fold_information_retained(manifest_rows):
    """Verify fold information matches NIGENS_8-foldSplit flists."""
    fold_map, _ = load_nigens_folds(NIGENS_DIR)
    assert len(fold_map) == 897

    for r in manifest_rows:
        rel = r["file_path"].replace("NIGENS/", "")
        expected_fold_idx = fold_map.get(rel, None)
        if expected_fold_idx is not None:
            assert r["original_fold"] == f"fold{expected_fold_idx}", (
                f"Fold mismatch for {rel}: expected fold{expected_fold_idx}, got {r['original_fold']}"
            )
        else:
            assert r["original_fold"] == "unmapped"


def test_every_included_sample_has_valid_fydp_target_class(manifest_rows):
    """Verify that every included sample maps to a canonical FYDP class."""
    included_rows = [r for r in manifest_rows if r["review_status"] == "INCLUDED"]
    for r in included_rows:
        assert r["target_class"] in FYDP_CLASSES, (
            f"Invalid target class '{r['target_class']}' for included sample {r['file_path']}"
        )
        assert r["target_class"] in ["scream", "other"], (
            f"Unexpected class '{r['target_class']}' (NIGENS only provides scream and other)"
        )


def test_ambiguous_classes_require_manual_review(manifest_rows):
    """Verify crash, alarm, and general files are not included without manual review."""
    for r in manifest_rows:
        sc = r["source_class"]
        fold = r["original_fold"]
        # If in training folds, crash and alarm MUST be MANUAL_REVIEW_REQUIRED
        if fold in [f"fold{i}" for i in range(1, 7)]:
            if sc in ["crash", "alarm"]:
                assert r["review_status"] == "MANUAL_REVIEW_REQUIRED", (
                    f"Expected MANUAL_REVIEW_REQUIRED for {sc} sample {r['file_path']}, got {r['review_status']}"
                )
            elif sc == "general":
                assert r["review_status"] in ["MANUAL_REVIEW_REQUIRED", "EXCLUDED"], (
                    f"General file {r['file_path']} has invalid status {r['review_status']}"
                )


def test_audio_metadata_validity(manifest_rows):
    """Verify extracted acoustic metadata values are valid."""
    for r in manifest_rows:
        dur = float(r["duration_seconds"])
        sr = int(r["sample_rate"])
        ch = int(r["channels"])
        bd = int(r["bit_depth"])

        assert dur > 0.0, f"Duration <= 0 in {r['file_path']}"
        assert sr in [44100, 25000, 48000], f"Unexpected sample rate {sr} in {r['file_path']}"
        assert ch in [1, 2], f"Unexpected channels {ch} in {r['file_path']}"
        assert bd in [4, 8, 16, 24, 32, 64], f"Unexpected bit depth {bd} in {r['file_path']}"


def test_report_file_integrity():
    """Verify nigens_integration_report.md exists and contains all required sections."""
    assert REPORT_PATH.exists(), f"Report file not found: {REPORT_PATH}"
    text = REPORT_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "Total Audio Files Inspected",
        "Audio Integrity & Format Statistics",
        "Original Class Inventory & Mapping Breakdown",
        "Original 8-Fold Split Preservation",
        "Duplicate and Cross-Dataset Isolation Audit",
        "Detailed Breakdown of Manually Reviewable Samples",
        "Final Eligible Sample Summary",
        "Licensing and Dataset Use Restrictions",
    ]
    for snippet in required_snippets:
        assert snippet in text, f"Report missing section: '{snippet}'"
