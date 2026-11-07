"""PyTorch Dataset and Preprocessing Pipeline for 5-Class Audio Classification.

Target classes (canonical order, strict 0-indexed):
  0: scream
  1: glass_breaking
  2: gunshot
  3: siren
  4: other

Preprocessing Specification:
- Sample rate: 16,000 Hz
- Channels: 1 (mono)
- Duration: 10.0 seconds (160,000 samples)
- Padding/Truncation: short clips zero-padded on right, long clips truncated
- Mel Spectrogram: 64 mel filterbanks, 125 Hz to 7500 Hz
- Window: 400 samples (25 ms Hann window), Hop: 160 samples (10 ms), FFT: 400
- Energy transform: log(mel + 1e-6)
- Expected log-mel tensor shape: (1, 64, 1001) representing (channel, n_mels, time_steps)
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union
import wave

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchaudio.transforms as T

# Configure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

try:
    from class_config import (
        CLASSES,
        CLASS_TO_ID,
        ID_TO_CLASS,
        NUM_CLASSES,
    )
except ImportError:
    from src.class_config import (
        CLASSES,
        CLASS_TO_ID,
        ID_TO_CLASS,
        NUM_CLASSES,
    )

# Acoustic Feature Hyperparameters
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


def load_audio_waveform(file_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """Safely load a WAV file into a 1D or 2D float32 PyTorch tensor and sample rate."""
    file_path = Path(file_path)
    try:
        import soundfile as sf
        data, sr = sf.read(str(file_path), dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.t()
    except Exception:
        # Fallback using standard library wave module
        with wave.open(str(file_path), "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            raw_bytes = wf.readframes(n_frames)

            if sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if n_ch > 1:
                samples = samples.reshape(-1, n_ch).mean(axis=1)
            waveform = torch.from_numpy(samples).unsqueeze(0)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if sample rate differs from target
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = SAMPLE_RATE

    return waveform, sr


class MelSpectrogramExtractor:
    """Computes log-mel spectrograms from audio waveforms."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        win_length: int = WIN_LENGTH,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        target_samples: int = TARGET_SAMPLES,
        epsilon: float = EPSILON,
    ):
        self.target_samples = target_samples
        self.epsilon = epsilon
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or truncate waveform to target_samples and return log-mel tensor of shape (1, n_mels, 1001)."""
        num_samples = waveform.shape[-1]
        if num_samples < self.target_samples:
            pad_amount = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount), mode="constant", value=0.0)
        elif num_samples > self.target_samples:
            waveform = waveform[..., : self.target_samples]

        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + self.epsilon)
        return log_mel


def extract_video_id_from_path(file_path: Union[str, Path]) -> str:
    """Extract YouTube video ID from audio file name.

    Handles conventions:
    - glass_breaking_{vid}_{start}_{end}.wav
    - other_{subdomain}_{vid}_{start}_{end}.wav
    - other_{vid}_{start}_{end}.wav
    - {class}_{vid}_{start}_{end}.wav
    """
    stem = Path(file_path).stem
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return stem
    prefix = parts[0]
    if prefix.startswith("glass_breaking_"):
        vid = prefix[len("glass_breaking_"):]
    elif prefix.startswith("other_"):
        rem = prefix[len("other_"):]
        for sub in ("animals_", "environmental_", "household_", "tools_"):
            if rem.startswith(sub):
                rem = rem[len(sub):]
                break
        vid = rem
    else:
        p_parts = prefix.split("_", 1)
        vid = p_parts[1] if len(p_parts) > 1 else p_parts[0]
    return vid


class AudioSetDataset(Dataset):
    """PyTorch Dataset loading emergency and background audio from class directories."""

    def __init__(
        self,
        audio_dir: Union[str, Path],
        classes: Optional[List[str]] = None,
        class_to_id: Optional[Dict[str, int]] = None,
        transform=None,
        samples: Optional[List[Tuple[Path, int, str]]] = None,
        return_raw: bool = False,
    ):
        self.audio_dir = Path(audio_dir)
        self.classes = classes if classes is not None else list(CLASSES)
        self.class_to_id = class_to_id if class_to_id is not None else dict(CLASS_TO_ID)
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        self.transform = transform
        self.return_raw = return_raw
        self.extractor = MelSpectrogramExtractor()

        if samples is not None:
            self.samples = list(samples)
        else:
            # Discover all valid samples
            self.samples: List[Tuple[Path, int, str]] = []
            for class_name in self.classes:
                class_folder = self.audio_dir / class_name
                if not class_folder.exists() or not class_folder.is_dir():
                    continue
                class_id = self.class_to_id[class_name]
                for wav_path in sorted(class_folder.glob("*.wav")):
                    if wav_path.is_file() and wav_path.stat().st_size > 0:
                        self.samples.append((wav_path, class_id, class_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        wav_path, class_id, class_name = self.samples[idx]
        waveform, _ = load_audio_waveform(wav_path)

        if self.return_raw:
            num_samples = waveform.shape[-1]
            if num_samples < TARGET_SAMPLES:
                pad_amount = TARGET_SAMPLES - num_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount), mode="constant", value=0.0)
            elif num_samples > TARGET_SAMPLES:
                waveform = waveform[..., : TARGET_SAMPLES]
            feat = waveform.squeeze(0)
        else:
            feat = self.extractor(waveform)

        if self.transform is not None:
            feat = self.transform(feat)

        return feat, class_id, str(wav_path)

    def get_class_counts(self) -> Dict[str, int]:
        """Return the count of valid samples per class."""
        counts = {c: 0 for c in self.classes}
        for _, _, cname in self.samples:
            counts[cname] = counts.get(cname, 0) + 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse-frequency class weights for CrossEntropyLoss.

        Formula: weight_c = total_samples / (num_classes * count_c)
        Normalized so mean weight equals 1.0.
        """
        counts = self.get_class_counts()
        total_samples = len(self.samples)
        num_classes = len(self.classes)

        weights = []
        for c in self.classes:
            cnt = counts[c]
            if cnt > 0:
                w = total_samples / (num_classes * float(cnt))
            else:
                w = 1.0
            weights.append(w)

        tensor_weights = torch.tensor(weights, dtype=torch.float32)
        # Normalize so mean is 1.0
        tensor_weights = tensor_weights / tensor_weights.mean()
        return tensor_weights

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler giving each class equal representation per epoch."""
        counts = self.get_class_counts()
        class_weights = {
            c: (1.0 / counts[c]) if counts[c] > 0 else 0.0 for c in self.classes
        }
        sample_weights = [class_weights[cname] for _, _, cname in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.samples),
            replacement=True,
        )


def run_preflight_dataset_check(
    audio_dir: Union[str, Path],
    classes: Optional[List[str]] = None,
    class_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[bool, List[str], Dict]:
    """Inspect dataset readiness, verifying all classes exist, have valid WAVs, and match formats.

    Returns:
        (is_ready: bool, issues: list of error strings, stats: dict of metrics)
    """
    audio_dir = Path(audio_dir)
    target_classes = classes if classes is not None else list(CLASSES)
    c_to_id = class_to_id if class_to_id is not None else dict(CLASS_TO_ID)

    issues = []
    stats = {
        "classes_found": {},
        "total_valid_files": 0,
        "format_errors": 0,
        "empty_classes": [],
        "sample_rate_violations": 0,
        "channel_violations": 0,
    }

    if not audio_dir.exists():
        issues.append(f"Training directory does not exist: {audio_dir}")
        return False, issues, stats

    for cname in target_classes:
        class_dir = audio_dir / cname
        if not class_dir.exists() or not class_dir.is_dir():
            issues.append(f"Missing class directory for '{cname}': {class_dir}")
            stats["empty_classes"].append(cname)
            continue

        wav_files = list(class_dir.glob("*.wav"))
        valid_in_class = 0
        for fpath in wav_files:
            try:
                with wave.open(str(fpath), "rb") as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    frames = wf.getnframes()
                    if sr != 16000:
                        stats["sample_rate_violations"] += 1
                        issues.append(f"Sample rate mismatch ({sr} Hz) in {fpath.name}")
                    elif ch != 1:
                        stats["channel_violations"] += 1
                        issues.append(f"Channel mismatch ({ch} ch) in {fpath.name}")
                    elif frames <= 0 or sw != 2:
                        stats["format_errors"] += 1
                        issues.append(f"Corrupt or non-16-bit PCM WAV in {fpath.name}")
                    else:
                        valid_in_class += 1
            except Exception as e:
                stats["format_errors"] += 1
                issues.append(f"Unreadable WAV file {fpath.name}: {str(e)}")

        stats["classes_found"][cname] = valid_in_class
        stats["total_valid_files"] += valid_in_class

        if valid_in_class == 0:
            stats["empty_classes"].append(cname)
            issues.append(f"Class '{cname}' has 0 valid audio files.")

    is_ready = len(issues) == 0 and len(stats["empty_classes"]) == 0
    return is_ready, issues, stats


def create_stratified_split(
    dataset: AudioSetDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int, str]], List[Tuple[Path, int, str]]]:
    """Create reproducible source/video-ID-grouped stratified train/val splits.

    Guarantees:
    - Segments from the same source/video ID are never split across train and val.
    - Stratified class proportions across both splits.
    - Zero overlap in video IDs between train and val.
    """
    import random
    from collections import defaultdict

    class_to_vids = defaultdict(lambda: defaultdict(list))
    for fpath, class_id, class_name in dataset.samples:
        vid = extract_video_id_from_path(fpath)
        class_to_vids[class_name][vid].append((fpath, class_id, class_name))

    rng = random.Random(seed)
    train_samples: List[Tuple[Path, int, str]] = []
    val_samples: List[Tuple[Path, int, str]] = []

    for cname in dataset.classes:
        vids_dict = class_to_vids[cname]
        vids = sorted(vids_dict.keys())
        rng.shuffle(vids)

        n_val = max(1, round(len(vids) * val_ratio))
        val_vids = set(vids[:n_val])
        train_vids = set(vids[n_val:])

        for vid in val_vids:
            val_samples.extend(vids_dict[vid])
        for vid in train_vids:
            train_samples.extend(vids_dict[vid])

    # Sort deterministically by path
    train_samples.sort(key=lambda x: str(x[0]))
    val_samples.sort(key=lambda x: str(x[0]))

    return train_samples, val_samples


def save_split_manifest(
    samples: List[Tuple[Path, int, str]],
    output_csv_path: Union[str, Path],
    root_dir: Optional[Path] = None,
):
    """Write split manifest CSV with video ID, relative path, class, and class ID."""
    import csv

    out_p = Path(output_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    base_dir = root_dir if root_dir is not None else ROOT

    fieldnames = ["video_id", "target_class", "class_id", "relative_path", "filename"]
    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fpath, class_id, class_name in samples:
            try:
                rel_path = fpath.relative_to(base_dir).as_posix()
            except ValueError:
                rel_path = str(fpath)
            vid = extract_video_id_from_path(fpath)
            writer.writerow({
                "video_id": vid,
                "target_class": class_name,
                "class_id": class_id,
                "relative_path": rel_path,
                "filename": fpath.name,
            })


def load_split_manifest(
    manifest_csv_path: Union[str, Path],
    root_dir: Optional[Path] = None,
) -> List[Tuple[Path, int, str]]:
    """Load samples from a split manifest CSV.

    Returns:
        List of (file_path: Path, class_id: int, class_name: str)
    """
    import csv

    manifest_path = Path(manifest_csv_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    base_dir = root_dir if root_dir is not None else ROOT
    samples = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_p = row.get("relative_path") or row.get("file_path")
            if not rel_p:
                continue
            fpath = (base_dir / rel_p).resolve()
            cname = row.get("target_class")
            if "class_id" in row and row["class_id"] != "":
                cid = int(row["class_id"])
            elif cname in CLASS_TO_ID:
                cid = CLASS_TO_ID[cname]
            else:
                raise ValueError(f"Cannot determine class_id for row: {row}")
            samples.append((fpath, cid, cname))

    return samples

