"""Unit tests for the source-grouped train/validation split of the combined dataset."""

import csv
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from split_combined_dataset import (
    ROOT,
    FYDP_CLASSES,
    extract_source_group_id,
    create_source_grouped_split,
    verify_split_integrity,
)

COMBINED_MANIFEST = ROOT / "data" / "processed" / "combined_training_manifest.csv"
TRAIN_SPLIT_MANIFEST = ROOT / "data" / "processed" / "combined_train_split_manifest.csv"
VAL_SPLIT_MANIFEST = ROOT / "data" / "processed" / "combined_val_split_manifest.csv"
REPORT_PATH = ROOT / "reports" / "combined_split_report.md"
if not REPORT_PATH.exists():
    REPORT_PATH = ROOT / "data" / "processed" / "combined_split_report.md"


@pytest.fixture(scope="module")
def manifests():
    """Load combined, train, and val split manifests."""
    assert COMBINED_MANIFEST.exists()
    assert TRAIN_SPLIT_MANIFEST.exists()
    assert VAL_SPLIT_MANIFEST.exists()

    with open(COMBINED_MANIFEST, "r", encoding="utf-8") as f:
        combined = list(csv.DictReader(f))
    with open(TRAIN_SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        train = list(csv.DictReader(f))
    with open(VAL_SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        val = list(csv.DictReader(f))

    return {"combined": combined, "train": train, "val": val}


def test_full_sample_coverage(manifests):
    """Verify that every approved sample in the combined manifest appears in exactly one split."""
    combined = manifests["combined"]
    train = manifests["train"]
    val = manifests["val"]

    assert len(combined) == 946
    assert len(train) == 756
    assert len(val) == 190
    assert len(train) + len(val) == len(combined)

    combined_ids = set(r["sample_id"] for r in combined)
    train_ids = set(r["sample_id"] for r in train)
    val_ids = set(r["sample_id"] for r in val)

    assert train_ids | val_ids == combined_ids
    assert not (train_ids & val_ids)


def test_no_duplicate_samples_across_splits(manifests):
    """Verify sample IDs are strictly disjoint between train and val."""
    train_ids = set(r["sample_id"] for r in manifests["train"])
    val_ids = set(r["sample_id"] for r in manifests["val"])
    assert len(train_ids & val_ids) == 0


def test_no_duplicate_hashes_across_splits(manifests):
    """Verify file hashes are strictly disjoint between train and val."""
    train_hashes = set(r["file_hash"] for r in manifests["train"])
    val_hashes = set(r["file_hash"] for r in manifests["val"])
    assert len(train_hashes & val_hashes) == 0


def test_source_group_isolation(manifests):
    """Verify that no source group is split across train and validation."""
    train_groups = set(extract_source_group_id(r) for r in manifests["train"])
    val_groups = set(extract_source_group_id(r) for r in manifests["val"])

    overlap = train_groups.intersection(val_groups)
    assert not overlap, f"Source group leakage detected across splits: {overlap}"


def test_audioset_video_id_isolation(manifests):
    """Verify that no AudioSet YouTube video ID appears in both splits."""
    train_vids = set(
        r["video_id"].strip() for r in manifests["train"] if r["source_dataset"] == "AudioSet"
    )
    val_vids = set(
        r["video_id"].strip() for r in manifests["val"] if r["source_dataset"] == "AudioSet"
    )

    vid_overlap = train_vids.intersection(val_vids)
    assert not vid_overlap, f"AudioSet video ID leakage: {vid_overlap}"


def test_valid_class_labels_and_representation(manifests):
    """Verify all 5 classes are represented in both train and validation."""
    for split_name in ["train", "val"]:
        rows = manifests[split_name]
        classes_present = set(r["target_class"] for r in rows)
        assert classes_present == set(FYDP_CLASSES), (
            f"Missing classes in {split_name}: {set(FYDP_CLASSES) - classes_present}"
        )
        for r in rows:
            assert r["target_class"] in FYDP_CLASSES


def test_correct_split_assignments(manifests):
    """Verify that the split column is populated with 'train' and 'validation' respectively."""
    for r in manifests["train"]:
        assert r["split"] == "train"
    for r in manifests["val"]:
        assert r["split"] == "validation"


def test_nigens_folds_7_and_8_absent(manifests):
    """Verify that NIGENS evaluation folds 7 and 8 are completely absent from both splits."""
    for split_name in ["train", "val"]:
        for r in manifests[split_name]:
            assert r.get("original_fold") not in ["fold7", "fold8"]


def test_reproducibility_with_seed_42(manifests):
    """Verify that running split with seed 42 produces identical train and val partitions."""
    train_1, val_1, _ = create_source_grouped_split(manifests["combined"], val_ratio=0.20, seed=42)
    train_2, val_2, _ = create_source_grouped_split(manifests["combined"], val_ratio=0.20, seed=42)

    ids_tr1 = [r["sample_id"] for r in train_1]
    ids_tr2 = [r["sample_id"] for r in train_2]
    ids_va1 = [r["sample_id"] for r in val_1]
    ids_va2 = [r["sample_id"] for r in val_2]

    assert ids_tr1 == ids_tr2
    assert ids_va1 == ids_va2

    # Matches files on disk
    disk_train_ids = [r["sample_id"] for r in manifests["train"]]
    disk_val_ids = [r["sample_id"] for r in manifests["val"]]

    assert ids_tr1 == disk_train_ids
    assert ids_va1 == disk_val_ids


def test_referenced_audio_files_exist(manifests):
    """Verify that every audio file in both split manifests physically exists on disk."""
    for split_name in ["train", "val"]:
        for r in manifests[split_name]:
            p = ROOT / r["file_path"]
            assert p.is_file(), f"File does not exist: {p}"
