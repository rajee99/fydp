"""Audio Preprocessing Smoke Test for Verified Pilot Audio Samples.

This module inspects verified WAV files in `data/processed/pilot_accessible.csv`
and tests conversion to the project's model input format: 16 kHz mono log-mel
spectrograms.

-------------------------------------------------------------------------------
DOCUMENTATION OF PROVISIONAL MODEL INPUT FORMAT:
-------------------------------------------------------------------------------
Note: This configuration is a documented PROVISIONAL baseline for the project.
While inspired by AudioSet/YAMNet feature extraction conventions, it does not
claim to be a bit-for-bit reproduction of TensorFlow YAMNet feature extractors.

- Target Sample Rate: 16,000 Hz
- Channels: 1 (Mono)
- Target Duration: 10.0 seconds (160,000 samples)
- Window Length (win_length): 400 samples (25.0 ms, Hann window)
- Hop Length (hop_length): 160 samples (10.0 ms)
- FFT Size (n_fft): 400 samples
- Mel Filter Banks (n_mels): 64 bins
- Frequency Range: 125.0 Hz to 7,500.0 Hz
- Log Transform: log(mel_spec + 1e-6)
- Expected Output Tensor Shape: torch.Size([1, 64, 1001])
  representing (channel, n_mels, time_steps).

Short-Clip Safety:
Any clip with fewer than 160,000 samples is zero-padded on the right.
Any clip exceeding 160,000 samples is truncated to 160,000 samples.
-------------------------------------------------------------------------------
"""

import csv
from pathlib import Path
import sys
import wave

import numpy as np
import torch
import torchaudio.transforms as T

# Configure UTF-8 stdout for Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "processed" / "pilot_accessible.csv"

# Preprocessing parameters (documented provisional baseline)
SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION_SEC)  # 160,000
WIN_LENGTH = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms
N_FFT = 400
N_MELS = 64
F_MIN = 125.0
F_MAX = 7500.0
EPSILON = 1e-6


def load_audio_file(file_path: Path):
    """Safely load a WAV file into a (1, num_samples) float32 PyTorch tensor at 16 kHz."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(file_path), dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.t()
    except Exception:
        # Fallback to standard library wave module
        with wave.open(str(file_path), "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if n_ch > 1:
                samples = samples.reshape(-1, n_ch).mean(axis=1)
            waveform = torch.from_numpy(samples).unsqueeze(0)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform, sr


def preprocess_audio(waveform: torch.Tensor, mel_transform: T.MelSpectrogram) -> torch.Tensor:
    """Pad/truncate waveform to TARGET_SAMPLES and compute log-mel spectrogram.

    Returns:
        Tensor of shape (1, N_MELS, 1001) containing log-mel energies.
    """
    num_samples = waveform.shape[-1]

    # Safe padding for short clips / safe truncation for long clips
    if num_samples < TARGET_SAMPLES:
        pad_amount = TARGET_SAMPLES - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount), mode="constant", value=0.0)
    elif num_samples > TARGET_SAMPLES:
        waveform = waveform[..., :TARGET_SAMPLES]

    # Mel spectrogram transform
    mel_spec = mel_transform(waveform)

    # Log energy transformation
    log_mel = torch.log(mel_spec + EPSILON)

    return log_mel


def run_smoke_test():
    print("=" * 80)
    print("  AUDIO PREPROCESSING SMOKE TEST: VERIFIED PILOT SAMPLES")
    print("=" * 80)
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz | Target Samples: {TARGET_SAMPLES} ({TARGET_DURATION_SEC}s)")
    print(f"Window: {WIN_LENGTH} samples (25ms) | Hop: {HOP_LENGTH} samples (10ms)")
    print(f"Mel Bins: {N_MELS} | Freq Range: {F_MIN}Hz – {F_MAX}Hz")
    print(f"Expected Log-Mel Shape: (1, {N_MELS}, 1001)")
    print("-" * 80)

    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Requirement 1: Load only rows whose local WAV exists, is readable, and is verified
    verified_candidates = []
    for r in all_rows:
        is_verified = (r.get("audio_verified", "").strip().lower() == "true")
        rel_path = r.get("local_audio_path", "").strip()

        if is_verified and rel_path:
            abs_path = ROOT / rel_path
            if abs_path.exists() and abs_path.stat().st_size > 0:
                verified_candidates.append((r, abs_path))

    print(f"Found {len(verified_candidates)} verified, existing audio files in manifest.\n")
    if len(verified_candidates) == 0:
        print("[ERROR] No verified audio files found to test.", file=sys.stderr)
        sys.exit(1)

    # Initialize shared MelSpectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        center=True,
        power=2.0,
    )

    results = []
    failures = []
    expected_shape = torch.Size([1, N_MELS, 1001])

    for i, (meta, wav_path) in enumerate(verified_candidates, 1):
        target_class = meta.get("target_class", "unknown")
        sub_domain = meta.get("sub_domain") or "(emergency)"
        rel_path = meta.get("local_audio_path")

        try:
            waveform, sr = load_audio_file(wav_path)
            raw_samples = waveform.shape[-1]
            raw_dur = raw_samples / float(sr)

            # Preprocess (handles padding for short clips safely)
            log_mel = preprocess_audio(waveform, mel_transform)

            # Verification checks
            shape_match = (log_mel.shape == expected_shape)
            is_finite = torch.isfinite(log_mel).all().item()
            min_val = log_mel.min().item()
            max_val = log_mel.max().item()
            mean_val = log_mel.mean().item()

            if not shape_match:
                raise ValueError(f"Shape mismatch: expected {expected_shape}, got {log_mel.shape}")
            if not is_finite:
                raise ValueError("Tensor contains NaN or Inf values")

            status = "PASS"
            results.append({
                "target_class": target_class,
                "sub_domain": sub_domain,
                "filename": wav_path.name,
                "raw_dur": f"{raw_dur:.2f}s",
                "raw_samples": raw_samples,
                "shape": list(log_mel.shape),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "mean": round(mean_val, 2),
            })

            short_note = " [PADDED]" if raw_samples < TARGET_SAMPLES else ""
            print(f"[{i:02d}/14] {status} | {target_class:<15} | {sub_domain:<13} | {raw_dur:5.2f}s{short_note:<9} -> {log_mel.shape} (min={min_val:6.2f}, max={max_val:5.2f}) | {wav_path.name}")

        except Exception as e:
            status = "FAIL"
            failures.append((wav_path.name, str(e)))
            print(f"[{i:02d}/14] {status} | {target_class:<15} | {wav_path.name}: {e}", file=sys.stderr)

    print("-" * 80)
    print(f"SMOKE TEST SUMMARY: {len(results)}/{len(verified_candidates)} PASSED, {len(failures)} FAILED.")
    print("-" * 80)

    if failures:
        print("\nFailures detected:")
        for fn, err in failures:
            print(f"  - {fn}: {err}")
        sys.exit(1)
    else:
        print("\nAll 14 verified WAV samples successfully converted to log-mel spectrograms.")
        print(f"Every tensor is finite (0 NaNs, 0 Infs) with shape {expected_shape}.")


if __name__ == "__main__":
    run_smoke_test()
