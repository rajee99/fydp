"""Unit and integration tests for the Combined Training Dataset (AudioSet + NIGENS)."""

import csv
import hashlib
from pathlib import Path
import sys
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from prepare_combined_dataset import (
    ROOT,
    FYDP_CLASSES,
    COMBINED_MANIFEST_COLUMNS,
    compute_file_sha256,
)

COMBINED_MANIFEST = ROOT / "data" / "processed" / "combined_training_manifest.csv"
COMBINED_REPORT = ROOT / "reports" / "combined_training_dataset_report.md"
if not COMBINED_REPORT.exists():
    COMBINED_REPORT = ROOT / "data" / "processed" / "combined_training_dataset_report.md"


@pytest.fixture(scope="module")
def combined_rows():
    """Load rows from combined_training_manifest.csv."""
    assert COMBINED_MANIFEST.exists(), f"Combined manifest not found: {COMBINED_MANIFEST}"
    with open(COMBINED_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def test_combined_manifest_schema_and_columns():
    """Verify that combined_training_manifest.csv has all required base and extra columns."""
    with open(COMBINED_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COMBINED_MANIFEST_COLUMNS, (
            f"Columns mismatch: expected {COMBINED_MANIFEST_COLUMNS}, got {reader.fieldnames}"
        )


def test_every_included_manifest_path_exists(combined_rows):
    """Verify criterion 1: Every included manifest path exists."""
    assert len(combined_rows) == 946, f"Expected 946 combined rows, got {len(combined_rows)}"
    missing = []
    for r in combined_rows:
        p = ROOT / r["file_path"]
        if not p.is_file():
            missing.append(r["file_path"])
    assert not missing, f"Missing files on disk ({len(missing)}): {missing[:5]}"


def test_every_included_sample_is_readable(combined_rows):
    """Verify criterion 2: Every included sample is readable by soundfile."""
    unreadable = []
    for r in combined_rows:
        p = ROOT / r["file_path"]
        try:
            info = sf.info(p)
            assert info.duration > 0
            assert info.samplerate > 0
            assert info.channels in [1, 2]
        except Exception as e:
            unreadable.append((r["file_path"], str(e)))
    assert not unreadable, f"Unreadable files found: {unreadable[:5]}"


def test_every_included_sample_has_valid_target_class(combined_rows):
    """Verify criterion 3: Every included sample has a valid target class in the 5 FYDP classes."""
    for r in combined_rows:
        assert r["target_class"] in FYDP_CLASSES, (
            f"Invalid target class '{r['target_class']}' for sample {r['sample_id']}"
        )


def test_every_sample_id_is_unique(combined_rows):
    """Verify criterion 4: Every sample ID is unique."""
    seen_ids = set()
    duplicates = []
    for r in combined_rows:
        sid = r["sample_id"]
        if sid in seen_ids:
            duplicates.append(sid)
        seen_ids.add(sid)
    assert not duplicates, f"Duplicate sample IDs found: {duplicates}"
    assert len(seen_ids) == len(combined_rows)


def test_no_duplicate_hashes_among_included_samples(combined_rows):
    """Verify criterion 5: No duplicate hashes exist among included samples."""
    seen_hashes = set()
    duplicates = []
    for r in combined_rows:
        h = r["file_hash"]
        if h in seen_hashes:
            duplicates.append((r["sample_id"], r["file_path"], h))
        seen_hashes.add(h)
    assert not duplicates, f"Duplicate hashes in combined dataset: {duplicates}"
    assert len(seen_hashes) == 946


def test_no_evaluation_leakage_from_nigens_or_video_ids(combined_rows):
    """Verify criterion 6: No evaluation leakage exists."""
    # Check 1: No NIGENS sample collides with evaluation audio
    eval_dir = ROOT / "data" / "raw" / "eval"
    eval_hashes = {compute_file_sha256(p): p.as_posix() for p in eval_dir.rglob("*.wav")}

    nigens_eval_leaks = []
    for r in combined_rows:
        if r["source_dataset"] == "NIGENS" and r["file_hash"] in eval_hashes:
            nigens_eval_leaks.append((r["file_path"], eval_hashes[r["file_hash"]]))

    assert not nigens_eval_leaks, f"NIGENS evaluation leakage detected: {nigens_eval_leaks}"

    # Check 2: AudioSet video IDs do not overlap with eval manifest video IDs
    eval_manifest_path = ROOT / "data" / "processed" / "eval_manifest.csv"
    with open(eval_manifest_path, "r", encoding="utf-8") as f:
        eval_vids = set(row["video_id"].strip() for row in csv.DictReader(f))

    as_vids = set(r["video_id"].strip() for r in combined_rows if r["source_dataset"] == "AudioSet")
    vid_overlap = as_vids.intersection(eval_vids)
    assert not vid_overlap, f"Cross-split video ID overlap detected: {vid_overlap}"


def test_no_pilot_leakage_from_nigens(combined_rows):
    """Verify criterion 7: No pilot leakage exists from NIGENS."""
    pilot_dir = ROOT / "data" / "raw" / "pilot"
    pilot_hashes = {compute_file_sha256(p): p.as_posix() for p in pilot_dir.rglob("*.wav")}

    nigens_pilot_leaks = []
    for r in combined_rows:
        if r["source_dataset"] == "NIGENS" and r["file_hash"] in pilot_hashes:
            nigens_pilot_leaks.append((r["file_path"], pilot_hashes[r["file_hash"]]))

    assert not nigens_pilot_leaks, f"NIGENS pilot leakage detected: {nigens_pilot_leaks}"


def test_nigens_folds_7_and_8_are_excluded(combined_rows):
    """Verify criterion 8: NIGENS folds 7 and 8 are strictly excluded."""
    nigens_rows = [r for r in combined_rows if r["source_dataset"] == "NIGENS"]
    assert len(nigens_rows) == 419, f"Expected 419 NIGENS rows, got {len(nigens_rows)}"

    for r in nigens_rows:
        fold = r["original_fold"]
        assert fold in ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6"], (
            f"Evaluation fold sample leaked: {r['file_path']} with {fold}"
        )
        assert fold not in ["fold7", "fold8", "unmapped"], f"Invalid fold: {fold}"


def test_original_source_information_preserved(combined_rows):
    """Verify criterion 9: Original source information is preserved."""
    as_rows = [r for r in combined_rows if r["source_dataset"] == "AudioSet"]
    ng_rows = [r for r in combined_rows if r["source_dataset"] == "NIGENS"]

    assert len(as_rows) == 527
    assert len(ng_rows) == 419

    # AudioSet metadata checks
    for r in as_rows:
        assert r["video_id"], f"Missing video_id for {r['sample_id']}"
        assert r["source_id"] == r["video_id"]
        assert r["audio_set_mid"], f"Missing audio_set_mid for {r['sample_id']}"
        assert r["matched_label"], f"Missing matched_label for {r['sample_id']}"
        assert r["start_seconds"] != "", f"Missing start_seconds for {r['sample_id']}"
        assert r["end_seconds"] != "", f"Missing end_seconds for {r['sample_id']}"

    # NIGENS metadata checks
    for r in ng_rows:
        assert r["original_fold"].startswith("fold"), f"Missing original_fold for {r['sample_id']}"
        assert r["flist_membership"].endswith(".flist"), f"Missing flist_membership for {r['sample_id']}"
        assert r["source_class"], f"Missing source_class for {r['sample_id']}"
        assert r["source_id"].startswith("NIGENS_"), f"Invalid source_id for {r['sample_id']}"


def test_split_column_remains_empty(combined_rows):
    """Verify criterion 10: The split column remains empty for all rows."""
    non_empty_splits = [r for r in combined_rows if r["split"].strip() != ""]
    assert not non_empty_splits, f"Split column contains non-empty values: {non_empty_splits[:5]}"


def test_existing_manifests_and_checkpoints_unchanged():
    """Verify criterion 11: Existing manifests and checkpoints remain unchanged."""
    # Checkpoints
    ckpt_path = ROOT / "models" / "experiments" / "panns_cnn14" / "best_model.pt"
    assert ckpt_path.exists()
    assert ckpt_path.stat().st_size > 300_000_000

    # Manifests
    target_m = ROOT / "data" / "processed" / "target_manifest.csv"
    eval_m = ROOT / "data" / "processed" / "eval_manifest.csv"
    train_split_m = ROOT / "data" / "processed" / "train_split_manifest.csv"
    val_split_m = ROOT / "data" / "processed" / "val_split_manifest.csv"

    assert target_m.exists()
    assert eval_m.exists()
    assert train_split_m.exists()
    assert val_split_m.exists()

    with open(target_m, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 740
    with open(eval_m, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 744
    with open(train_split_m, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 422
    with open(val_split_m, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 105
