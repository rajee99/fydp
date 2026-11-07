"""Unit tests for the 'other' dataset collection workflow."""

import csv
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from class_config import OTHER_SUBDOMAINS, get_all_exclusion_mids
from collect_other_dataset import (
    load_excluded_video_ids,
    load_candidates,
)


def test_load_excluded_video_ids():
    """Verify that exclusion set loads from train, eval, and pilot manifests."""
    excluded = load_excluded_video_ids()
    assert len(excluded) >= 1400

    # Check known IDs from train and eval
    eval_m = ROOT / "data" / "processed" / "eval_manifest.csv"
    with open(eval_m, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        eval_vids = [r["video_id"].strip() for r in reader if r.get("video_id")]

    for vid in eval_vids[:10]:
        assert vid in excluded


def test_load_candidates_filters_emergency_and_confusers(tmp_path):
    """Verify that any row containing emergency or ambiguous MIDs is excluded."""
    dummy_segments = tmp_path / "dummy_segments.csv"
    dummy_segments.write_text(
        "# Test comments\n"
        "vid_clean, 0.0, 10.0, \"/m/09x0r\"\n"  # Speech
        "vid_emergency, 0.0, 10.0, \"/m/09x0r,/m/03qc9zr\"\n"  # Speech + Screaming
        "vid_confuser, 0.0, 10.0, \"/m/09x0r,/m/07p6fty\"\n"  # Speech + Shout
        "vid_music, 0.0, 10.0, \"/m/04rlf\"\n"  # Music
    )

    excluded_vids = set()
    exclusion_mids = get_all_exclusion_mids()

    candidates = load_candidates(
        dummy_segments,
        excluded_vids=excluded_vids,
        exclusion_mids=exclusion_mids,
        subdomains_config=OTHER_SUBDOMAINS,
    )

    speech_vids = [c["video_id"] for c in candidates["speech"]]
    assert "vid_clean" in speech_vids
    assert "vid_emergency" not in speech_vids
    assert "vid_confuser" not in speech_vids
    music_vids = [c["video_id"] for c in candidates["music"]]
    assert "vid_music" in music_vids


def test_load_candidates_filters_multi_subdomain_overlap(tmp_path):
    """Verify that a candidate matching multiple subdomains is excluded to avoid ambiguity."""
    dummy_segments = tmp_path / "dummy_multi.csv"
    dummy_segments.write_text(
        "vid_multi, 0.0, 10.0, \"/m/09x0r,/m/04rlf\"\n"  # Speech + Music
        "vid_pure_traffic, 0.0, 10.0, \"/m/0btp2\"\n"  # Traffic noise
    )

    excluded_vids = set()
    exclusion_mids = get_all_exclusion_mids()

    candidates = load_candidates(
        dummy_segments,
        excluded_vids=excluded_vids,
        exclusion_mids=exclusion_mids,
        subdomains_config=OTHER_SUBDOMAINS,
    )

    all_loaded_vids = [c["video_id"] for sub in candidates.values() for c in sub]
    assert "vid_multi" not in all_loaded_vids
    assert "vid_pure_traffic" in all_loaded_vids
