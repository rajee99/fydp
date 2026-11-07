"""Comprehensive unit tests for experimental manifests, shared validation design, and E3 quality audit."""

import json
from pathlib import Path
import sys
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from class_config import CLASSES, CLASS_TO_ID, get_all_exclusion_mids, OTHER_SUBDOMAINS
from dataset import load_split_manifest
from prepare_experimental_manifests import (
    resolve_review_status,
    build_e1_baseline_dataset,
    build_e2_targeted_dataset,
    build_e3_general_dataset,
    build_e4_ablations,
    verify_all_isolation,
)


@pytest.fixture(scope="module")
def review_resolution():
    usable_other, audit_meta = resolve_review_status()
    return usable_other, audit_meta


@pytest.fixture(scope="module")
def manifests_dir():
    return ROOT / "data" / "processed"


def test_review_resolution_and_exclusions(review_resolution):
    usable_other, audit_meta = review_resolution

    # Verify count
    assert len(usable_other) == 224
    assert audit_meta["usable_targeted_other_count"] == 224

    # Verify NEEDS_SECOND_REVIEW exclusion
    assert "ja-Qxq5NAoE" in audit_meta["needs_second_review"]
    usable_vids = {r["video_id"] for r in usable_other}
    assert "ja-Qxq5NAoE" not in usable_vids

    # Verify RECLASSIFY mappings
    reclass_vids = set(audit_meta["reclassified"].keys())
    assert reclass_vids == {"ad-8X58aIEM", "Pdau8nsBqxs", "d4v3_z0ISrM", "I1TWWYbjvJ8"}
    for r in usable_other:
        if r["video_id"] in reclass_vids:
            assert r["target_class"] == "other"
            assert r["class_id"] == 4

    # Verify REJECT count
    assert audit_meta["rejected_count"] == 19

    # Verify all usable samples have target_class == 'other'
    for r in usable_other:
        assert r["target_class"] == "other"
        assert r["class_id"] == 4


def test_usable_other_audio_validity(review_resolution):
    usable_other, _ = review_resolution
    for r in usable_other:
        p = ROOT / r["relative_path"]
        assert p.exists(), f"File missing: {p}"
        info = sf.info(str(p))
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert info.duration > 0.0


def test_shared_validation_manifest_identical_across_all_experiments(manifests_dir):
    """Verify that all experiments and ablations use the EXACT identical 105 validation samples."""
    shared_val_path = manifests_dir / "shared_val_split_manifest.csv"
    assert shared_val_path.exists()
    df_shared = pd.read_csv(shared_val_path)
    assert len(df_shared) == 105

    shared_vids = df_shared["video_id"].tolist()

    val_manifests_to_check = [
        manifests_dir / "e1_val_split_manifest.csv",
        manifests_dir / "e2_val_split_manifest.csv",
        manifests_dir / "e3_general_val_candidates.csv",
    ]

    # Include all 14 E4 ablations & matched controls
    e4_dir = manifests_dir / "e4_ablations"
    val_manifests_to_check.extend(list(e4_dir.glob("*_val.csv")))

    assert len(val_manifests_to_check) == 17  # E1, E2, E3 + 7 E4 ablations + 7 matched controls

    for vpath in val_manifests_to_check:
        assert vpath.exists()
        df = pd.read_csv(vpath)
        assert len(df) == 105, f"{vpath.name} length is {len(df)}, expected 105"
        assert df["video_id"].tolist() == shared_vids, f"{vpath.name} does not match shared validation video IDs!"


def test_e1_baseline_manifest_integrity(manifests_dir):
    master_path = manifests_dir / "e1_baseline_manifest.csv"
    train_path = manifests_dir / "e1_train_split_manifest.csv"
    val_path = manifests_dir / "e1_val_split_manifest.csv"

    assert master_path.exists()
    assert train_path.exists()
    assert val_path.exists()

    df_master = pd.read_csv(master_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    assert len(df_master) == 527
    assert len(df_train) == 422
    assert len(df_val) == 105
    assert len(df_train) + len(df_val) == len(df_master)

    # Check zero video_id overlap
    train_vids = set(df_train["video_id"])
    val_vids = set(df_val["video_id"])
    assert len(train_vids.intersection(val_vids)) == 0

    # Check classes
    assert set(df_master["target_class"].unique()) == set(CLASSES)


def test_e2_targeted_manifest_integrity(manifests_dir):
    master_path = manifests_dir / "e2_targeted_manifest.csv"
    train_path = manifests_dir / "e2_train_split_manifest.csv"
    val_path = manifests_dir / "e2_val_split_manifest.csv"

    assert master_path.exists()
    assert train_path.exists()
    assert val_path.exists()

    df_master = pd.read_csv(master_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Shared validation design: 646 train, 105 val, 751 total
    assert len(df_master) == 751
    assert len(df_train) == 646
    assert len(df_val) == 105
    assert len(df_train) + len(df_val) == 751

    # Check zero video_id overlap
    train_vids = set(df_train["video_id"])
    val_vids = set(df_val["video_id"])
    assert len(train_vids.intersection(val_vids)) == 0

    # Check that ja-Qxq5NAoE is excluded
    assert "ja-Qxq5NAoE" not in set(df_master["video_id"])

    # Check that all 751 audio files exist on disk
    for _, row in df_master.iterrows():
        p = ROOT / row["relative_path"]
        assert p.exists(), f"E2 audio file missing: {p}"

    # Verify baseline training preservation
    df_e1_train = pd.read_csv(manifests_dir / "e1_train_split_manifest.csv")
    e1_train_vids = set(df_e1_train["video_id"])
    e2_train_vids = set(df_train["video_id"])
    assert e1_train_vids.issubset(e2_train_vids)

    # Subdomain distribution for 'other' in train
    other_train = df_train[df_train["target_class"] == "other"]
    assert len(other_train) == 320  # 96 baseline + 224 targeted
    sub_counts = other_train["subdomain"].value_counts().to_dict()
    assert sub_counts["human_shouting"] == 44
    assert sub_counts["metal_impacts"] == 40
    assert sub_counts["door_slams"] == 33
    assert sub_counts["construction_tools"] == 30
    assert sub_counts["car_alarms_detectors"] == 29
    assert sub_counts["fireworks_firecrackers"] == 28
    assert sub_counts["whistles_high_pitch"] == 19


def test_e3_candidate_quality_and_labels(manifests_dir):
    """Deep quality audit of the 224 E3 candidate AudioSet samples."""
    master_path = manifests_dir / "e3_general_candidate_manifest.csv"
    train_path = manifests_dir / "e3_general_train_candidates.csv"
    val_path = manifests_dir / "e3_general_val_candidates.csv"

    assert master_path.exists()
    assert train_path.exists()
    assert val_path.exists()

    df_master = pd.read_csv(master_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Shared validation design: 646 train, 105 val, 751 total
    assert len(df_master) == 751
    assert len(df_train) == 646
    assert len(df_val) == 105

    # Zero train/val overlap
    assert len(set(df_train["video_id"]).intersection(set(df_val["video_id"]))) == 0

    # Extract 224 candidate rows
    cands_df = df_train[df_train["source_dataset"] == "audioset_general_control"]
    assert len(cands_df) == 224
    assert cands_df["video_id"].nunique() == 224

    # Documented subdomain quotas
    sub_counts = cands_df["subdomain"].value_counts().to_dict()
    assert sub_counts["traffic"] == 38
    assert sub_counts["environmental"] == 38
    assert sub_counts["speech"] == 37
    assert sub_counts["music"] == 37
    assert sub_counts["animals"] == 37
    assert sub_counts["household"] == 37
    assert sum(sub_counts.values()) == 224

    # Verify zero emergency MIDs/labels in balanced segments
    excl_mids = get_all_exclusion_mids()
    cands_vids = set(cands_df["video_id"])

    bal_segments_path = ROOT / "data" / "raw" / "balanced_train_segments.csv"
    with open(bal_segments_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) >= 4 and parts[0] in cands_vids:
                labels = set([p.strip().strip('"') for p in parts[3:]])
                # Assert no exclusion MIDs
                overlap = labels.intersection(excl_mids)
                assert len(overlap) == 0, f"Candidate {parts[0]} has forbidden MIDs: {overlap}"


def test_e4_ablation_manifests_integrity(manifests_dir):
    config_path = ROOT / "configs" / "e4_ablation_config.json"
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    experiments = config["experiments"]
    assert len(experiments) == 14  # 7 ablations + 7 matched controls

    e4_dir = manifests_dir / "e4_ablations"
    assert e4_dir.exists()

    df_e1_train = pd.read_csv(manifests_dir / "e1_train_split_manifest.csv")
    e1_train_vids = set(df_e1_train["video_id"])

    expected_exclusions = {
        "human_shouting": (44, 602),
        "metal_impacts": (40, 606),
        "door_slams": (33, 613),
        "construction_tools": (30, 616),
        "car_alarms_detectors": (29, 617),
        "fireworks_firecrackers": (28, 618),
        "whistles_high_pitch": (19, 627),
    }

    for exp_name, exp_info in experiments.items():
        tr_p = ROOT / exp_info["train_manifest"]
        va_p = ROOT / exp_info["val_manifest"]
        assert tr_p.exists()
        assert va_p.exists()

        df_tr = pd.read_csv(tr_p)
        df_va = pd.read_csv(va_p)

        # Shared validation: exactly 105 samples
        assert len(df_va) == 105

        # Zero train/val overlap
        assert len(set(df_tr["video_id"]).intersection(set(df_va["video_id"]))) == 0

        # Exact train count
        sub = exp_info["target_subdomain"]
        num_excl, expected_tr_count = expected_exclusions[sub]
        assert len(df_tr) == expected_tr_count, f"{exp_name} train count is {len(df_tr)}, expected {expected_tr_count}"
        assert len(df_tr) + len(df_va) == expected_tr_count + 105

        # Baseline train preservation
        assert e1_train_vids.issubset(set(df_tr["video_id"]))

        # For specific category ablation, excluded subdomain is absent from targeted additions
        if exp_info["type"] == "category_ablation":
            targeted_additions = df_tr[df_tr["source_dataset"] == "audioset_targeted_expansion"]
            assert sub not in set(targeted_additions["subdomain"].unique())


def test_eval_isolation_across_all_manifests(manifests_dir):
    eval_df = pd.read_csv(manifests_dir / "eval_manifest.csv")
    eval_vids = set(eval_df["video_id"].dropna().str.strip())

    pilot_df = pd.read_csv(manifests_dir / "pilot_manifest.csv")
    pilot_vids = set(pilot_df["video_id"].dropna().str.strip())

    # Check E1, E2, E3
    for name in ["e1_baseline_manifest.csv", "e2_targeted_manifest.csv", "e3_general_candidate_manifest.csv"]:
        df = pd.read_csv(manifests_dir / name)
        vids = set(df["video_id"])
        eval_overlap = vids.intersection(eval_vids)
        assert len(eval_overlap) == 0, f"{name} overlaps with eval: {eval_overlap}"

    # Check additions vs pilot
    e1_vids = set(pd.read_csv(manifests_dir / "e1_baseline_manifest.csv")["video_id"])
    e2_df = pd.read_csv(manifests_dir / "e2_targeted_manifest.csv")
    e2_additions = set(e2_df["video_id"]) - e1_vids
    assert len(e2_additions.intersection(pilot_vids)) == 0

    e3_df = pd.read_csv(manifests_dir / "e3_general_candidate_manifest.csv")
    e3_additions = set(e3_df["video_id"]) - e1_vids
    assert len(e3_additions.intersection(pilot_vids)) == 0


def test_dataset_loader_compatibility(manifests_dir):
    """Verify that load_split_manifest from dataset.py successfully loads updated splits."""
    e1_train_samples = load_split_manifest(manifests_dir / "e1_train_split_manifest.csv")
    assert len(e1_train_samples) == 422
    for fpath, cid, cname in e1_train_samples:
        assert fpath.exists()
        assert cname in CLASSES
        assert cid == CLASS_TO_ID[cname]

    e1_val_samples = load_split_manifest(manifests_dir / "e1_val_split_manifest.csv")
    assert len(e1_val_samples) == 105

    e2_train_samples = load_split_manifest(manifests_dir / "e2_train_split_manifest.csv")
    assert len(e2_train_samples) == 646
    for fpath, cid, cname in e2_train_samples:
        assert fpath.exists()
        assert cname in CLASSES
        assert cid == CLASS_TO_ID[cname]

    e2_val_samples = load_split_manifest(manifests_dir / "e2_val_split_manifest.csv")
    assert len(e2_val_samples) == 105

    e3_train_samples = load_split_manifest(manifests_dir / "e3_train_split_manifest.csv")
    assert len(e3_train_samples) == 634
    for fpath, cid, cname in e3_train_samples:
        assert fpath.exists()
        assert cname in CLASSES
        assert cid == CLASS_TO_ID[cname]

    e3_val_samples = load_split_manifest(manifests_dir / "e3_val_split_manifest.csv")
    assert len(e3_val_samples) == 105


def test_e3_content_validation_and_readiness(manifests_dir):
    """Verify E3 content reviews, replacement handling, and final manifest integrity."""
    init_review_p = manifests_dir / "e3_review_results.csv"
    repl_review_p = manifests_dir / "e3_replacement_review_results.csv"
    train_manifest_p = manifests_dir / "e3_train_split_manifest.csv"
    val_manifest_p = manifests_dir / "e3_val_split_manifest.csv"

    assert init_review_p.exists()
    assert repl_review_p.exists()
    assert train_manifest_p.exists()
    assert val_manifest_p.exists()

    df_init = pd.read_csv(init_review_p)
    df_repl = pd.read_csv(repl_review_p)

    assert len(df_init) == 174
    assert len(df_repl) == 40

    acc_init = df_init[df_init["verdict"] == "ACCEPT"]
    rej_init = df_init[df_init["verdict"] == "REJECT"]
    sec_init = df_init[df_init["verdict"] == "NEEDS_SECOND_REVIEW"]

    assert len(acc_init) == 172
    assert len(rej_init) == 2
    assert len(sec_init) == 0

    acc_repl = df_repl[df_repl["verdict"] == "ACCEPT"]
    assert len(acc_repl) == 40

    # Total usable additions: 172 + 40 = 212
    df_train = pd.read_csv(train_manifest_p)
    assert len(df_train) == 634  # 422 baseline + 212 accepted E3
    assert df_train["video_id"].nunique() == 634

    # All 212 additions must be other class
    new_additions = df_train[df_train["candidate_priority"] == "E3_ACCEPTED"]
    assert len(new_additions) == 212
    assert (new_additions["target_class"] == "other").all()


def test_e2_matched_manifest_integrity(manifests_dir):
    """Verify E2 matched dataset integrity, size matching with E3, and isolation."""
    matched_train_p = manifests_dir / "e2_matched_train_split_manifest.csv"
    matched_val_p = manifests_dir / "e2_matched_val_split_manifest.csv"
    matched_master_p = manifests_dir / "e2_matched_manifest.csv"
    e3_train_p = manifests_dir / "e3_train_split_manifest.csv"
    shared_val_p = manifests_dir / "shared_val_split_manifest.csv"
    eval_manifest_p = manifests_dir / "eval_manifest.csv"

    assert matched_train_p.exists()
    assert matched_val_p.exists()
    assert matched_master_p.exists()
    assert e3_train_p.exists()

    df_e2_m_tr = pd.read_csv(matched_train_p)
    df_e2_m_va = pd.read_csv(matched_val_p)
    df_e2_m_ma = pd.read_csv(matched_master_p)
    df_e3_tr = pd.read_csv(e3_train_p)
    df_shared_va = pd.read_csv(shared_val_p)
    df_eval = pd.read_csv(eval_manifest_p)

    # 1. Size checks
    assert len(df_e2_m_tr) == 634
    assert len(df_e2_m_va) == 105
    assert len(df_e2_m_ma) == 739
    assert len(df_e2_m_tr) == len(df_e3_tr) == 634

    # 2. Shared baseline train checks
    e2_base_vids = set(df_e2_m_tr[df_e2_m_tr["source_dataset"] == "audioset"]["video_id"])
    e3_base_vids = set(df_e3_tr[df_e3_tr["source_dataset"] == "audioset"]["video_id"])
    assert len(e2_base_vids) == 422
    assert e2_base_vids == e3_base_vids

    # 3. Additions count
    targeted_additions = df_e2_m_tr[df_e2_m_tr["source_dataset"] == "audioset_targeted_expansion"]
    assert len(targeted_additions) == 212

    # 4. Duplicate checks
    assert df_e2_m_tr["video_id"].nunique() == 634
    assert df_e2_m_va["video_id"].nunique() == 105

    # 5. Isolation checks
    train_vids = set(df_e2_m_tr["video_id"])
    val_vids = set(df_e2_m_va["video_id"])
    eval_vids = set(df_eval["video_id"])

    assert len(train_vids.intersection(val_vids)) == 0
    assert len(train_vids.intersection(eval_vids)) == 0
    assert set(df_e2_m_va["video_id"]) == set(df_shared_va["video_id"])


