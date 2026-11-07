"""Unit tests for the Targeted In-The-Wild Dataset Expansion Manifest."""

import csv
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "processed" / "targeted_expansion_candidates.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "targeted_expansion_summary.md"

REQUIRED_COLUMNS = [
    "video_id",
    "start_seconds",
    "end_seconds",
    "target_class",
    "subdomain",
    "audio_set_mid",
    "matched_label",
    "source_catalog",
    "candidate_priority",
    "manual_verification_status",
    "flag_reason",
    "overlapping_labels",
    "all_labels",
]

EMERGENCY_MIDS = {
    "/m/03qc9zr": "Screaming",
    "/m/07rn7sz": "Shatter",
    "/m/039jq": "Glass",
    "/m/032s66": "Gunshot, gunfire",
    "/m/04zjc": "Machine gun",
    "/m/02z32qm": "Fusillade",
    "/m/0_1c": "Artillery fire",
    "/m/073cg4": "Cap gun",
    "/m/012n7d": "Ambulance (siren)",
    "/m/03kmc9": "Siren",
    "/m/04qvtq": "Police car (siren)",
    "/m/012ndj": "Fire engine, fire truck (siren)",
    "/m/0dgbq": "Civil defense siren",
}


@pytest.fixture(scope="module")
def candidates():
    """Load rows from targeted_expansion_candidates.csv."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_manifest_schema_and_columns(candidates):
    """Verify that all required columns are present in exact order."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REQUIRED_COLUMNS


def test_unique_video_ids(candidates):
    """Verify criterion: every candidate comes from a unique video ID."""
    vids = [r["video_id"] for r in candidates]
    assert len(vids) == len(set(vids)), f"Duplicate video IDs found: {len(vids) - len(set(vids))}"


def test_source_isolation_against_all_past_manifests(candidates):
    """Verify strict source isolation against training, eval, and pilot datasets."""
    forbidden_vids = set()
    manifest_files = [
        'data/processed/target_manifest.csv',
        'data/processed/train_split_manifest.csv',
        'data/processed/val_split_manifest.csv',
        'data/processed/eval_manifest.csv',
        'data/processed/combined_training_manifest.csv',
        'data/processed/combined_train_split_manifest.csv',
        'data/processed/combined_val_split_manifest.csv',
        'data/processed/pilot_manifest.csv',
        'data/processed/pilot_accessible.csv',
    ]
    for mf in manifest_files:
        p = ROOT / mf
        if p.exists():
            with open(p, encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    v = r.get('video_id') or r.get('youtube_id')
                    if v:
                        forbidden_vids.add(v.strip())

    with open(ROOT / 'data' / 'raw' / 'eval_segments.csv', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                forbidden_vids.add(line.split(',')[0].strip())

    cand_vids = {r["video_id"] for r in candidates}
    leakage = cand_vids.intersection(forbidden_vids)
    assert not leakage, f"Source leakage detected with {len(leakage)} forbidden IDs: {list(leakage)[:5]}"


def test_primary_quotas_fulfilled_exactly(candidates):
    """Verify that P1_PRIMARY candidates hit the requested quotas exactly."""
    p1 = [r for r in candidates if r["candidate_priority"] == "P1_PRIMARY"]
    assert len(p1) == 435

    scream_p1 = [r for r in p1 if r["target_class"] == "scream"]
    assert len(scream_p1) == 110

    glass_p1 = [r for r in p1 if r["target_class"] == "glass_breaking"]
    assert len(glass_p1) == 85

    other_p1 = [r for r in p1 if r["target_class"] == "other"]
    assert len(other_p1) == 240

    # Subdomain quotas for other
    sub_counts = {}
    for r in other_p1:
        s = r["subdomain"]
        sub_counts[s] = sub_counts.get(s, 0) + 1

    assert sub_counts.get("metal_impacts") == 45
    assert sub_counts.get("human_shouting") == 45
    assert sub_counts.get("door_slams") == 35
    assert sub_counts.get("car_alarms_detectors") == 35
    assert sub_counts.get("fireworks_firecrackers") == 30
    assert sub_counts.get("construction_tools") == 30
    assert sub_counts.get("whistles_high_pitch") == 20


def test_other_candidates_free_of_emergency_mids(candidates):
    """Verify that no P1/P2 other candidates contain any emergency MIDs."""
    other_rows = [r for r in candidates if r["target_class"] == "other"]
    for r in other_rows:
        assert r["audio_set_mid"] not in EMERGENCY_MIDS
        assert r["overlapping_labels"] == "None"


def test_glass_smash_crack_candidates_flagged_for_review(candidates):
    """Verify that Smash, crash and Crack are isolated into review status."""
    cand_review = [r for r in candidates if r["subdomain"] == "candidate_smash_crack"]
    assert len(cand_review) == 30
    for r in cand_review:
        assert r["manual_verification_status"] == "NEEDS_AUDIO_VERIFICATION"
        assert r["candidate_priority"] == "P2_REVIEW"


def test_summary_markdown_file_integrity():
    """Verify that targeted_expansion_summary.md exists and contains all required sections."""
    assert SUMMARY_PATH.exists(), f"Summary missing: {SUMMARY_PATH}"
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    required_sections = [
        "Executive Summary & Quota Fulfillment",
        "Source-Level Isolation Guarantee",
        "Subdomain Breakdown & Candidate Selection Details",
        "Recommended Download Priorities",
        "Verification Checklist & Compliance",
    ]
    for sec in required_sections:
        assert sec in text, f"Missing section in summary: '{sec}'"


def test_replacements_manifest_and_map_integrity():
    """Verify that the 107 P2 replenishment manifest and mapping log are intact and strictly isolated."""
    rep_csv = ROOT / "data" / "processed" / "targeted_expansion_replacements.csv"
    map_csv = ROOT / "data" / "processed" / "targeted_expansion_replacements_map.csv"
    eval_csv = ROOT / "data" / "processed" / "eval_manifest.csv"

    assert rep_csv.exists(), f"Replacements manifest missing: {rep_csv}"
    assert map_csv.exists(), f"Replacements map missing: {map_csv}"

    with open(rep_csv, "r", encoding="utf-8") as f:
        rep_rows = list(csv.DictReader(f))
    with open(map_csv, "r", encoding="utf-8") as f:
        map_rows = list(csv.DictReader(f))
    with open(eval_csv, "r", encoding="utf-8") as f:
        eval_vids = set(r["video_id"].strip() for r in csv.DictReader(f))

    assert len(rep_rows) == 107, f"Expected 107 replacement rows, got {len(rep_rows)}"
    assert len(map_rows) == 107, f"Expected 107 mapping rows, got {len(map_rows)}"

    rep_vids = set(r["video_id"].strip() for r in rep_rows)
    assert len(rep_vids) == 107, "Replacement video IDs must be unique"
    assert not rep_vids.intersection(eval_vids), "Replacements must have zero overlap with eval video IDs"

    for r in rep_rows:
        assert r["candidate_priority"] == "P2_BACKUP"

    for m in map_rows:
        assert m["replacement_status"] in ["SUCCESS", "FAILED", "VALIDATION_FAILED"]


def test_downloaded_audio_inventory_and_specs():
    """Verify that all 400 downloaded audio files on disk meet strict 16 kHz mono 16-bit PCM specs and past datasets are untouched."""
    import wave
    dest_dir = ROOT / "data" / "raw" / "targeted_expansion"
    assert dest_dir.exists(), f"Targeted expansion directory missing: {dest_dir}"

    wav_files = list(dest_dir.glob("*/*.wav"))
    assert len(wav_files) == 400, f"Expected 400 audio files on disk, found {len(wav_files)}"

    class_counts = {}
    for p in wav_files:
        c = p.parent.name
        class_counts[c] = class_counts.get(c, 0) + 1
        with wave.open(str(p), "rb") as wf:
            assert wf.getnchannels() == 1, f"Expected mono: {p.name}"
            assert wf.getframerate() == 16000, f"Expected 16kHz: {p.name}"
            assert wf.getsampwidth() == 2, f"Expected 16-bit: {p.name}"
            assert wf.getnframes() > 0, f"Empty frames: {p.name}"

    assert class_counts.get("scream") == 97
    assert class_counts.get("glass_breaking") == 81
    assert class_counts.get("other") == 222

    # Verify existing datasets are untouched
    train_dir = ROOT / "data" / "raw" / "train"
    eval_dir = ROOT / "data" / "raw" / "eval"
    pilot_dir = ROOT / "data" / "raw" / "pilot"

    assert len(list(train_dir.glob("*/*.wav"))) == 527
    assert len(list(eval_dir.glob("*/*.wav"))) == 519
    assert len(list(pilot_dir.glob("*.wav"))) == 15


def test_content_validation_template_schema_and_reconciliation():
    """Verify that the content review CSV exists, has 400 rows, and matches disk files exactly."""
    val_csv = ROOT / "data" / "processed" / "targeted_expansion_content_validation.csv"
    assert val_csv.exists(), f"Content validation CSV missing: {val_csv}"

    with open(val_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 400, f"Expected 400 review rows, got {len(rows)}"

    expected_cols = [
        "file_path", "target_class", "subdomain", "video_id",
        "start_seconds", "end_seconds", "matched_label", "origin",
        "duration_seconds", "file_size_bytes", "validation_status",
        "reviewer_verdict", "audited_class", "acoustic_notes"
    ]
    with open(val_csv, "r", encoding="utf-8") as f:
        assert csv.DictReader(f).fieldnames == expected_cols

    origins = {}
    classes = {}
    for r in rows:
        origins[r["origin"]] = origins.get(r["origin"], 0) + 1
        classes[r["target_class"]] = classes.get(r["target_class"], 0) + 1
        assert r["validation_status"] in ["PENDING", "COMPLETED"]
        assert r["reviewer_verdict"] in ["ACCEPT", "RECLASSIFY", "REJECT", "NEEDS_SECOND_REVIEW", ""]
        p = ROOT / r["file_path"]
        assert p.exists(), f"Audio file missing from disk: {p}"

    assert origins == {"P1_PRIMARY": 328, "P2_BACKUP": 72}
    assert classes == {"scream": 97, "glass_breaking": 81, "other": 222}


