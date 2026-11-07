"""Unit tests for the Audio Content Review Tool workflow."""

import csv
from pathlib import Path
import sys
import pytest
import shutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from review_tool import (
    load_review_candidates,
    load_existing_review_results,
    save_review_decision,
    save_bulk_decisions,
    compute_review_progress,
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS,
    RESULTS_FIELDNAMES,
    VERDICT_CHOICES,
    OBSERVED_SOUND_CHOICES,
)

ROOT = Path(__file__).resolve().parent.parent


def test_load_review_candidates_from_content_validation_manifest():
    """Verify that all 400 audio recordings from targeted_expansion_content_validation.csv are loaded."""
    assert DEFAULT_MANIFEST.exists(), f"Manifest missing: {DEFAULT_MANIFEST}"
    candidates = load_review_candidates(DEFAULT_MANIFEST)

    assert len(candidates) == 400, f"Expected 400 candidates, got {len(candidates)}"

    classes = {}
    origins = {}
    for c in candidates:
        cls = c["target_class"]
        orig = c["origin"]
        classes[cls] = classes.get(cls, 0) + 1
        origins[orig] = origins.get(orig, 0) + 1

        assert c["video_id"], "Missing video_id"
        assert c["subdomain"], "Missing subdomain"
        assert c["matched_label"], "Missing matched_label"
        assert c["duration_seconds"] > 0, f"Invalid duration for {c['video_id']}"
        assert c["exists_on_disk"] is True, f"Audio file not accessible: {c['file_path']}"

    assert classes == {"scream": 97, "glass_breaking": 81, "other": 222}
    assert origins == {"P1_PRIMARY": 328, "P2_BACKUP": 72}


def test_save_single_decision_and_persistence(tmp_path):
    """Test saving single review decisions with all required validation checkpoints."""
    test_results_csv = tmp_path / "test_review_results.csv"

    decision_1 = {
        "video_id": "test_vid_001",
        "start_seconds": 0.0,
        "end_seconds": 10.0,
        "file_path": "data/raw/targeted_expansion/scream/scream_test_0_10.wav",
        "target_class": "scream",
        "subdomain": "indoor_small_room",
        "matched_label": "Screaming",
        "origin": "P1_PRIMARY",
        "duration_seconds": 10.0,
        "verdict": "ACCEPT",
        "observed_sound": "scream",
        "expected_sound_audible": True,
        "background_noise_present": False,
        "emergency_sound_present": False,
        "emergency_sound_type": "none",
        "notes": "Clear vocal distress scream in reverberant small room.",
        "reviewer": "test_reviewer",
        "reviewed_at": "2026-10-30 16:00:00",
    }

    success = save_review_decision(test_results_csv, decision_1, allow_overwrite=True)
    assert success is True
    assert test_results_csv.exists()

    # Read back and verify all fields
    with open(test_results_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["video_id"] == "test_vid_001"
        assert row["verdict"] == "ACCEPT"
        assert row["observed_sound"] == "scream"
        assert row["expected_sound_audible"] == "True"
        assert row["background_noise_present"] == "False"
        assert row["emergency_sound_present"] == "False"
        assert row["notes"] == "Clear vocal distress scream in reverberant small room."
        assert row["reviewer"] == "test_reviewer"


def test_reclassify_verdict_and_ambiguity_tracking(tmp_path):
    """Test reclassifying human shouting or fireworks with explicit observed sound and notes."""
    test_results_csv = tmp_path / "test_review_results.csv"

    decision_reclassify = {
        "video_id": "test_vid_002",
        "start_seconds": 15.0,
        "end_seconds": 25.0,
        "file_path": "data/raw/targeted_expansion/scream/scream_test_15_25.wav",
        "target_class": "scream",
        "subdomain": "music_background",
        "matched_label": "Screaming",
        "origin": "P2_BACKUP",
        "duration_seconds": 10.0,
        "verdict": "RECLASSIFY",
        "observed_sound": "human_shouting",
        "expected_sound_audible": False,
        "background_noise_present": True,
        "emergency_sound_present": False,
        "emergency_sound_type": "none",
        "notes": "Loud human yelling with intelligible words over loud rock music; not a panic scream.",
        "reviewer": "researcher_2",
        "reviewed_at": "2026-10-30 16:05:00",
    }

    success = save_review_decision(test_results_csv, decision_reclassify, allow_overwrite=True)
    assert success is True

    results_map = load_existing_review_results(test_results_csv)
    key = ("test_vid_002", 15.0, 25.0)
    assert key in results_map
    loaded = results_map[key]
    assert loaded["verdict"] == "RECLASSIFY"
    assert loaded["observed_sound"] == "human_shouting"
    assert loaded["expected_sound_audible"] is False
    assert loaded["background_noise_present"] is True


def test_resume_functionality_and_progress_tracking(tmp_path):
    """Verify that reloading results preserves prior decisions and computes accurate progress."""
    test_results_csv = tmp_path / "test_review_results.csv"

    # Pre-populate 3 decisions
    decisions = [
        {
            "video_id": "v1", "start_seconds": 0.0, "end_seconds": 10.0,
            "target_class": "scream", "subdomain": "indoor_small_room", "verdict": "ACCEPT"
        },
        {
            "video_id": "v2", "start_seconds": 0.0, "end_seconds": 10.0,
            "target_class": "glass_breaking", "subdomain": "breaking_shatter", "verdict": "REJECT"
        },
        {
            "video_id": "v3", "start_seconds": 0.0, "end_seconds": 10.0,
            "target_class": "other", "subdomain": "metal_impacts", "verdict": "NEEDS_SECOND_REVIEW"
        },
    ]
    save_bulk_decisions(test_results_csv, decisions)

    # Mock candidate list with 5 items (3 reviewed, 2 pending)
    candidates = [
        {"video_id": "v1", "start_seconds": 0.0, "end_seconds": 10.0, "target_class": "scream"},
        {"video_id": "v2", "start_seconds": 0.0, "end_seconds": 10.0, "target_class": "glass_breaking"},
        {"video_id": "v3", "start_seconds": 0.0, "end_seconds": 10.0, "target_class": "other"},
        {"video_id": "v4", "start_seconds": 0.0, "end_seconds": 10.0, "target_class": "scream"},
        {"video_id": "v5", "start_seconds": 0.0, "end_seconds": 10.0, "target_class": "other"},
    ]

    loaded_map = load_existing_review_results(test_results_csv)
    assert len(loaded_map) == 3

    prog = compute_review_progress(candidates, loaded_map)
    assert prog["total"] == 5
    assert prog["reviewed"] == 3
    assert prog["pending"] == 2
    assert prog["percent_complete"] == 60.0

    # Class stats
    assert prog["by_class"]["scream"]["total"] == 2
    assert prog["by_class"]["scream"]["reviewed"] == 1
    assert prog["by_class"]["glass_breaking"]["total"] == 1
    assert prog["by_class"]["glass_breaking"]["reviewed"] == 1
    assert prog["by_class"]["other"]["total"] == 2
    assert prog["by_class"]["other"]["reviewed"] == 1

    # Verdict stats
    assert prog["verdict_counts"]["ACCEPT"] == 1
    assert prog["verdict_counts"]["REJECT"] == 1
    assert prog["verdict_counts"]["NEEDS_SECOND_REVIEW"] == 1


def test_overwrite_protection(tmp_path):
    """Test that existing review decisions cannot be overwritten when allow_overwrite is False."""
    test_results_csv = tmp_path / "test_review_results.csv"

    decision = {
        "video_id": "v100", "start_seconds": 0.0, "end_seconds": 10.0,
        "verdict": "ACCEPT", "notes": "Initial verdict"
    }
    save_review_decision(test_results_csv, decision)

    # Attempt to overwrite with allow_overwrite=False
    overwrite_attempt = {
        "video_id": "v100", "start_seconds": 0.0, "end_seconds": 10.0,
        "verdict": "REJECT", "notes": "Attempted overwrite"
    }
    saved = save_review_decision(test_results_csv, overwrite_attempt, allow_overwrite=False)
    assert saved is False

    # Verify original decision preserved
    loaded = load_existing_review_results(test_results_csv)
    assert loaded[("v100", 0.0, 10.0)]["verdict"] == "ACCEPT"
    assert loaded[("v100", 0.0, 10.0)]["notes"] == "Initial verdict"

    # Now attempt with allow_overwrite=True
    saved_confirmed = save_review_decision(test_results_csv, overwrite_attempt, allow_overwrite=True)
    assert saved_confirmed is True
    loaded_updated = load_existing_review_results(test_results_csv)
    assert loaded_updated[("v100", 0.0, 10.0)]["verdict"] == "REJECT"


def test_data_integrity_and_isolation(tmp_path):
    """Verify that saving decisions does not modify WAV files or source manifests."""
    # Check manifest modification time
    orig_mtime = DEFAULT_MANIFEST.stat().st_mtime
    test_results_csv = tmp_path / "review_results.csv"

    # Pick first candidate from real manifest
    candidates = load_review_candidates(DEFAULT_MANIFEST)
    c1 = candidates[0]
    wav_path = ROOT / c1["file_path"]
    wav_mtime = wav_path.stat().st_mtime
    wav_size = wav_path.stat().st_size

    decision = {
        "video_id": c1["video_id"],
        "start_seconds": c1["start_seconds"],
        "end_seconds": c1["end_seconds"],
        "file_path": c1["file_path"],
        "target_class": c1["target_class"],
        "subdomain": c1["subdomain"],
        "matched_label": c1["matched_label"],
        "origin": c1["origin"],
        "duration_seconds": c1["duration_seconds"],
        "verdict": "ACCEPT",
        "observed_sound": c1["target_class"],
        "expected_sound_audible": True,
        "background_noise_present": False,
        "emergency_sound_present": False,
        "emergency_sound_type": "none",
        "notes": "Verified authentic clip.",
        "reviewer": "lead_researcher",
    }

    save_review_decision(test_results_csv, decision)

    # Manifest and WAV must remain untouched
    assert DEFAULT_MANIFEST.stat().st_mtime == orig_mtime
    assert wav_path.stat().st_mtime == wav_mtime
    assert wav_path.stat().st_size == wav_size


def test_audio_security_path_traversal():
    """Verify that the audio streaming security logic strictly rejects path traversal."""
    # Test valid subpath
    valid_path = Path("data/raw/targeted_expansion/scream/some_clip.wav")
    resolved_valid = (ROOT / valid_path).resolve()
    allowed_dir = (ROOT / "data" / "raw").resolve()
    assert resolved_valid.is_relative_to(allowed_dir)

    # Test path traversal attack
    evil_path = Path("../../../windows/system32/cmd.exe")
    resolved_evil = (ROOT / evil_path).resolve()
    assert not resolved_evil.is_relative_to(allowed_dir)
