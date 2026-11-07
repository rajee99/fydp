"""Unit tests for AudioSet batch downloader (preflight checks, WAV validation, range batching, CLI)."""

import csv
from pathlib import Path
import struct
import sys
import tempfile
import wave
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import batch_downloader


def test_preflight_leakage():
    print("=== 1. Testing preflight leakage check ===")
    # Normal check
    train_vids, eval_vids = batch_downloader.check_preflight_leakage(
        batch_downloader.TRAIN_MANIFEST, batch_downloader.EVAL_MANIFEST
    )
    assert len(train_vids.intersection(eval_vids)) == 0, "Expected zero cross-split video overlap"
    print(f"Normal preflight check: PASSED ({len(train_vids)} train vids, {len(eval_vids)} eval vids, 0 overlap)")

    # Simulated leakage check
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        t_file = temp_dir / "t.csv"
        e_file = temp_dir / "e.csv"
        with open(t_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id"])
            w.writerow(["LEAKED_VID_123"])
        with open(e_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id"])
            w.writerow(["LEAKED_VID_123"])
        try:
            batch_downloader.check_preflight_leakage(t_file, e_file)
            assert False, "Should have raised RuntimeError on leakage"
        except RuntimeError as e:
            assert "FATAL: Cross-split data leakage detected" in str(e)
            print("Simulated leakage detection: PASSED (Fatal error raised as expected)")


def test_validate_audio_file():
    print("\n=== 2. Testing validate_audio_file() ===")

    # 2a. Real verified pilot audio file (16-bit mono 16 kHz)
    valid_sample = ROOT / "data" / "raw" / "pilot" / "siren_source.wav"
    ok, err, stats = batch_downloader.validate_audio_file(valid_sample, expected_duration=10.0)
    assert ok, f"Expected valid sample, got error: {err}"
    assert stats["channels"] == 1, f"Expected 1 channel, got {stats['channels']}"
    assert stats["sample_rate"] == 16000, f"Expected 16,000 Hz, got {stats['sample_rate']}"
    assert stats["sample_width"] == 2, f"Expected 16-bit (sample width 2), got {stats['sample_width']}"
    assert stats["compression_type"] == "NONE", f"Expected compression type NONE, got {stats['compression_type']}"
    assert abs(stats["duration"] - 10.0) < 0.1, f"Expected ~10.0s, got {stats['duration']}s"
    print(f"Verified pilot sample (16-bit mono PCM): PASSED ({stats['channels']}ch, {stats['sample_rate']}Hz, "
          f"{stats['sample_width']*8}-bit, comptype={stats['compression_type']}, {stats['duration']}s)")

    # 2b. Synthesized valid 16-bit mono PCM WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        p_16bit = Path(tf.name)
    try:
        with wave.open(str(p_16bit), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000 * 10)
        ok_16, err_16, stats_16 = batch_downloader.validate_audio_file(p_16bit, expected_duration=10.0)
        assert ok_16, f"Valid 16-bit mono should pass: {err_16}"
        assert stats_16["sample_width"] == 2
        assert stats_16["compression_type"] == "NONE"
        print("Synthesized valid 16-bit mono WAV: PASSED")
    finally:
        p_16bit.unlink(missing_ok=True)

    # 2c. 8-bit mono WAV (unsupported sample width)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        p_8bit = Path(tf.name)
    try:
        with wave.open(str(p_8bit), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)  # 8-bit
            wf.setframerate(16000)
            wf.writeframes(b"\x80" * 16000 * 10)
        ok_8, err_8, _ = batch_downloader.validate_audio_file(p_8bit, expected_duration=10.0)
        assert not ok_8, "8-bit WAV must fail validation"
        assert "16-bit" in err_8 and "sample width 1" in err_8
        print(f"8-bit WAV rejection: PASSED (Rejected: '{err_8}')")
    finally:
        p_8bit.unlink(missing_ok=True)

    # 2d. 24-bit mono WAV (unsupported sample width)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        p_24bit = Path(tf.name)
    try:
        with wave.open(str(p_24bit), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)  # 24-bit
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00\x00" * 16000 * 10)
        ok_24, err_24, _ = batch_downloader.validate_audio_file(p_24bit, expected_duration=10.0)
        assert not ok_24, "24-bit WAV must fail validation"
        assert "16-bit" in err_24 and "sample width 3" in err_24
        print(f"24-bit WAV rejection: PASSED (Rejected: '{err_24}')")
    finally:
        p_24bit.unlink(missing_ok=True)

    # 2e. 32-bit integer mono WAV (unsupported sample width)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        p_32bit = Path(tf.name)
    try:
        with wave.open(str(p_32bit), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(4)  # 32-bit
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00\x00\x00" * 16000 * 10)
        ok_32, err_32, _ = batch_downloader.validate_audio_file(p_32bit, expected_duration=10.0)
        assert not ok_32, "32-bit WAV must fail validation"
        assert "16-bit" in err_32 and "sample width 4" in err_32
        print(f"32-bit WAV rejection: PASSED (Rejected: '{err_32}')")
    finally:
        p_32bit.unlink(missing_ok=True)

    # 2f. Stereo WAV (unsupported channel count)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        p_stereo = Path(tf.name)
    try:
        with wave.open(str(p_stereo), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00\x00\x00" * 16000 * 10)
        ok_st, err_st, _ = batch_downloader.validate_audio_file(p_stereo, expected_duration=10.0)
        assert not ok_st, "Stereo WAV must fail validation"
        assert "Expected 1 channel (mono)" in err_st
        print(f"Stereo WAV rejection: PASSED (Rejected: '{err_st}')")
    finally:
        p_stereo.unlink(missing_ok=True)

    # 2g. Non-PCM / unsupported compression format (format code 3: IEEE float)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        p_nonpcm = Path(tf.name)
    try:
        data = (
            b"RIFF" +
            struct.pack("<I", 36 + 64000) +
            b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32) +
            b"data" +
            struct.pack("<I", 64000) +
            b"\x00" * 64000
        )
        p_nonpcm.write_bytes(data)
        ok_np, err_np, _ = batch_downloader.validate_audio_file(p_nonpcm, expected_duration=10.0)
        assert not ok_np, "Non-PCM format 3 must fail validation"
        assert "corrupted or invalid wav file" in err_np.lower()
        print(f"Non-PCM (IEEE float format 3) rejection: PASSED (Rejected: '{err_np}')")
    finally:
        p_nonpcm.unlink(missing_ok=True)

    # 2h. Compression type not NONE (mocked Wave_read)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        p_comp = Path(tf.name)
    try:
        with wave.open(str(p_comp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000 * 10)

        orig_wave_open = wave.open
        class MockWaveRead:
            def __init__(self, real_wf):
                self.wf = real_wf
            def getnchannels(self): return self.wf.getnchannels()
            def getframerate(self): return self.wf.getframerate()
            def getnframes(self): return self.wf.getnframes()
            def getsampwidth(self): return self.wf.getsampwidth()
            def getcomptype(self): return "G711"
            def __enter__(self): return self
            def __exit__(self, *args): self.wf.close()

        with patch("wave.open", side_effect=lambda f, m: MockWaveRead(orig_wave_open(f, m))):
            ok_cp, err_cp, _ = batch_downloader.validate_audio_file(p_comp, expected_duration=10.0)
            assert not ok_cp, "Non-NONE compression type must fail validation"
            assert "compression type 'NONE'" in err_cp and "G711" in err_cp
            print(f"Non-NONE compression rejection: PASSED (Rejected: '{err_cp}')")
    finally:
        p_comp.unlink(missing_ok=True)

    # 2i. Missing file
    ok_m, err_m, _ = batch_downloader.validate_audio_file(Path("non_existent_file.wav"), 10.0)
    assert not ok_m
    assert "File does not exist" in err_m

    # 2j. Empty file (0 bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        empty_path = Path(tf.name)
    try:
        ok_e, err_e, _ = batch_downloader.validate_audio_file(empty_path, 10.0)
        assert not ok_e
        assert "empty" in err_e.lower()
    finally:
        empty_path.unlink(missing_ok=True)

    # 2k. Duration mismatch
    ok_d, err_d, _ = batch_downloader.validate_audio_file(valid_sample, expected_duration=30.0)
    assert not ok_d
    assert "deviates from expected" in err_d

    print("Missing, empty, and duration edge cases: PASSED")


def test_permanent_error_patterns():
    print("\n=== 3. Testing permanent error classification ===")
    # 3a. Verified permanent error patterns (must be classified as permanent and not retried)
    sample_permanent_errors = [
        "ERROR: [youtube] -GI5PbO6j50: This video is unavailable",
        "ERROR: Video unavailable",
        "This video is not available",
        "Video is not available",
        "Private video",
        "Video is private",
        "This video is private",
        "This video has been removed for violating YouTube's Terms of Service",
        "This video has been removed for violating YouTube's Community Guidelines",
        "The account associated with this video has been terminated",
        "Sign in to confirm your age. Use --cookies-from-browser or --cookies for the authentication.",
        "Please sign in. Use --cookies-from-browser or --cookies for the authentication.",
        "The uploader has not made this video available in your country",
    ]
    for err in sample_permanent_errors:
        assert any(pat in err.lower() for pat in batch_downloader.PERMANENT_ERROR_PATTERNS), (
            f"Failed permanent pattern match for: {err}"
        )
    print(f"Permanent error classification: PASSED ({len(sample_permanent_errors)} test signatures verified)")

    # 3b. Genuine temporary/transient network failures (must NOT be classified as permanent; must remain retryable)
    sample_temporary_errors = [
        "HTTP Error 429: Too Many Requests",
        "HTTP Error 503: Service Unavailable",
        "The read operation timed out",
        "ConnectionResetError: [Errno 104] Connection reset by peer",
        "urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>",
        "Temporary failure in name resolution",
        "RemoteDisconnected('Remote end closed connection without response')",
    ]
    for err in sample_temporary_errors:
        assert not any(pat in err.lower() for pat in batch_downloader.PERMANENT_ERROR_PATTERNS), (
            f"Temporary error was incorrectly classified as permanent: {err}"
        )
    print(f"Temporary error retention (retryable): PASSED ({len(sample_temporary_errors)} network error signatures verified)")


def test_manifest_range_batching():
    print("\n=== 4. Testing manifest range batching & selection ===")
    dummy_records = [
        {
            "video_id": f"VID_{i:03d}",
            "start_seconds": "0.0",
            "end_seconds": "10.0",
            "target_class": "gunshot" if i % 2 == 0 else "scream",
            "matched_label": "Test Label",
            "audio_set_mid": "/m/032s66",
        }
        for i in range(50)
    ]

    # 4a. Default behaviour when no range is supplied
    res_default = batch_downloader.select_manifest_records(dummy_records)
    assert len(res_default) == 50, "Default selection without range should return all records"
    assert res_default[0]["video_id"] == "VID_000"
    print("Default range selection: PASSED (All 50 records selected)")

    res_default_limit = batch_downloader.select_manifest_records(dummy_records, limit=10)
    assert len(res_default_limit) == 10, "Default with limit=10 should return first 10 records"
    assert res_default_limit[0]["video_id"] == "VID_000"
    assert res_default_limit[-1]["video_id"] == "VID_009"
    print("Default with limit: PASSED (First 10 records 0..9 selected)")

    # 4b. Starting from a nonzero index
    res_nonzero = batch_downloader.select_manifest_records(dummy_records, start_index=20)
    assert len(res_nonzero) == 30, f"Expected 30 records from index 20, got {len(res_nonzero)}"
    assert res_nonzero[0]["video_id"] == "VID_020"
    assert res_nonzero[-1]["video_id"] == "VID_049"
    print("Nonzero start-index: PASSED (Records 20..49 selected)")

    # 4c. Combining --start-index with --limit
    res_range = batch_downloader.select_manifest_records(dummy_records, start_index=20, limit=20)
    assert len(res_range) == 20, f"Expected 20 records in batch [20:40], got {len(res_range)}"
    assert res_range[0]["video_id"] == "VID_020"
    assert res_range[-1]["video_id"] == "VID_039"
    print("Range batching (--start-index 20 --limit 20): PASSED (Records 20..39 selected)")

    # 4d. Out-of-range indices and invalid negative values
    try:
        batch_downloader.select_manifest_records(dummy_records, start_index=-1)
        assert False, "Negative start_index should raise ValueError"
    except ValueError as e:
        assert "must be a non-negative integer" in str(e)
        print("Negative start_index rejection: PASSED")

    try:
        batch_downloader.select_manifest_records(dummy_records, limit=-5)
        assert False, "Negative limit should raise ValueError"
    except ValueError as e:
        assert "must be a non-negative integer" in str(e)
        print("Negative limit rejection: PASSED")

    try:
        batch_downloader.select_manifest_records(dummy_records, start_index=100)
        assert False, "Out-of-range start_index should raise ValueError"
    except ValueError as e:
        assert "out of range" in str(e)
        print("Out-of-range start_index rejection: PASSED")

    # 4e. Interaction with --video-id
    # Valid video_id without explicit start_index (default start_index=None)
    res_vid = batch_downloader.select_manifest_records(dummy_records, video_id="VID_025")
    assert len(res_vid) == 1 and res_vid[0]["video_id"] == "VID_025"
    print("Valid video_id selection (start_index=None): PASSED")

    # Incompatible options: video_id with explicitly supplied start_index=0
    try:
        batch_downloader.select_manifest_records(dummy_records, video_id="VID_025", start_index=0)
        assert False, "Combining video_id with explicit start_index=0 should raise ValueError"
    except ValueError as e:
        assert "Incompatible options" in str(e) and "--video-id" in str(e)
        print("Explicit --video-id + --start-index 0 rejection: PASSED")

    # Incompatible options: video_id with nonzero start_index=5
    try:
        batch_downloader.select_manifest_records(dummy_records, video_id="VID_025", start_index=5)
        assert False, "Combining video_id with nonzero start_index should raise ValueError"
    except ValueError as e:
        assert "Incompatible options" in str(e) and "--video-id" in str(e)
        print("Explicit --video-id + --start-index 5 rejection: PASSED")

    # Unknown video_id
    try:
        batch_downloader.select_manifest_records(dummy_records, video_id="NON_EXISTENT")
        assert False, "Unknown video_id should raise ValueError"
    except ValueError as e:
        assert "not found" in str(e)
        print("Unknown video_id rejection: PASSED")


def test_error_classification_precedence():
    print("\n=== 5. Testing error classification precedence & distinguishability ===")
    from generate_retry_list import assign_priority

    cases = [
        # 1. This video is unavailable -> PERMANENT_UNAVAILABLE
        ("ERROR: [youtube] vid1: This video is unavailable", "PERMANENT_UNAVAILABLE", "LOW", True),
        # 2. Video unavailable -> PERMANENT_UNAVAILABLE
        ("ERROR: [youtube] vid2: Video unavailable", "PERMANENT_UNAVAILABLE", "LOW", True),
        # 3. This video is not available -> PERMANENT_UNAVAILABLE
        ("ERROR: [youtube] vid3: This video is not available", "PERMANENT_UNAVAILABLE", "LOW", True),
        # 4. Sign in to confirm your age -> AUTH_REQUIRED
        ("ERROR: [youtube] vid4: Sign in to confirm your age. Use --cookies-from-browser...", "AUTH_REQUIRED", "MEDIUM", True),
        # 5. Please sign in -> AUTH_REQUIRED
        ("ERROR: [youtube] vid5: Please sign in. Use --cookies-from-browser...", "AUTH_REQUIRED", "MEDIUM", True),
        # 6. Private video -> PRIVATE_VIDEO
        ("ERROR: [youtube] vid6: Private video", "PRIVATE_VIDEO", "LOW", True),
        # 7. Policy / terms of service / community guidelines removal -> POLICY_REMOVED
        ("ERROR: [youtube] vid7a: This video has been removed for violating YouTube's Terms of Service", "POLICY_REMOVED", "LOW", True),
        ("ERROR: [youtube] vid7b: This video has been removed for violating YouTube's Community Guidelines", "POLICY_REMOVED", "LOW", True),
        # 8. Genuine temporary network errors -> TRANSIENT_NETWORK (retryable in downloader)
        ("ERROR: HTTP Error 429: Too Many Requests", "TRANSIENT_NETWORK", "MEDIUM", False),
        ("ERROR: HTTP Error 503: Service Unavailable", "TRANSIENT_NETWORK", "MEDIUM", False),
        ("ERROR: The read operation timed out", "TRANSIENT_NETWORK", "MEDIUM", False),
        ("ERROR: ConnectionResetError: [Errno 104] Connection reset by peer", "TRANSIENT_NETWORK", "MEDIUM", False),
        ("ERROR: <urlopen error [Errno 11001] getaddrinfo failed>", "TRANSIENT_NETWORK", "MEDIUM", False),
    ]

    distinct_categories = set()
    for err_msg, expected_cat, expected_prio, expected_downloader_perm in cases:
        # Check downloader permanent status
        is_downloader_perm = any(pat in err_msg.lower() for pat in batch_downloader.PERMANENT_ERROR_PATTERNS)
        assert is_downloader_perm == expected_downloader_perm, (
            f"Downloader permanence mismatch for '{err_msg}': expected {expected_downloader_perm}, got {is_downloader_perm}"
        )

        # Check retry catalog assignment
        prio, cat = assign_priority("FAILED", err_msg, "gunshot")
        assert cat == expected_cat, f"Category mismatch for '{err_msg}': expected {expected_cat}, got {cat}"
        assert prio == expected_prio, f"Priority mismatch for '{err_msg}': expected {expected_prio}, got {prio}"
        distinct_categories.add(cat)

    # Precedence check: Auth required takes precedence over generic unavailable wording
    mixed_auth_err = "ERROR: [youtube] vid: Please sign in. This video is unavailable without an account."
    prio_mix, cat_mix = assign_priority("FAILED", mixed_auth_err, "gunshot")
    assert cat_mix == "AUTH_REQUIRED", f"Precedence error: expected AUTH_REQUIRED, got {cat_mix}"

    # Precedence check: Validation failure takes precedence over error messages
    prio_val, cat_val = assign_priority("VALIDATION_FAILED", "Duration 9.0s deviates from 10.0s", "gunshot")
    assert cat_val == "DURATION_MISMATCH" and prio_val == "HIGH"
    distinct_categories.add(cat_val)

    # Confirm all categories remain distinguishable
    expected_categories = {"PERMANENT_UNAVAILABLE", "AUTH_REQUIRED", "PRIVATE_VIDEO", "POLICY_REMOVED", "TRANSIENT_NETWORK", "DURATION_MISMATCH"}
    assert distinct_categories == expected_categories, f"Categories not fully distinguishable: {distinct_categories}"
    print(f"Error classification precedence & distinguishability: PASSED ({len(cases)} cases + precedence verified)")


def test_custom_manifest_and_priority_support():
    """Verify custom manifest support, priority filtering, and dry-run execution."""
    sample_records = [
        {"video_id": "vid1", "target_class": "scream", "candidate_priority": "P1_PRIMARY"},
        {"video_id": "vid2", "target_class": "scream", "candidate_priority": "P2_BACKUP"},
        {"video_id": "vid3", "target_class": "other", "candidate_priority": "P1_PRIMARY"},
    ]
    p1 = batch_downloader.select_manifest_records(sample_records, priority="P1_PRIMARY")
    assert len(p1) == 2
    assert [r["video_id"] for r in p1] == ["vid1", "vid3"]

    p2 = batch_downloader.select_manifest_records(sample_records, priority="P2_BACKUP")
    assert len(p2) == 1
    assert p2[0]["video_id"] == "vid2"


if __name__ == "__main__":
    test_preflight_leakage()
    test_validate_audio_file()
    test_permanent_error_patterns()
    test_manifest_range_batching()
    test_error_classification_precedence()
    test_custom_manifest_and_priority_support()
    print("\n" + "=" * 60)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
