"""Neural Network Architectures for 5-Class Emergency and Ambient Sound Classification.

Models provided:
1. CompactAudioCNN: Lightweight, efficient 4-layer CNN operating on (1, 64, 1001) log-mel spectrograms.
2. PannsCNN14Transfer: Transfer learning adapter for Pretrained Audio Neural Networks (PANNs CNN14).

Documentation of Pretrained PANNs CNN14 Acquisition:
--------------------------------------------------------------------------------
PANNs (Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition,
Kong et al., IEEE/ACM TASLP 2020) provides state-of-the-art representations trained
on AudioSet (2.1 million audio clips).

Target Weights:
  - Checkpoint: Cnn14_16k_mAP=0.438.pth (~325 MB)
  - Hosted at: https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth
  - Alternatively accessible via `panns_inference` library.

Transfer Learning Strategy:
  - The feature extraction backbone produces a 2048-dimensional global acoustic embedding.
  - The final linear classification layer (originally 527 AudioSet classes) is replaced
    by an MLP head mapping 2048 -> 256 -> 5 canonical emergency classes.
  - Per project guidelines, large checkpoints are NOT downloaded automatically without
    explicit user authorization. The PannsCNN14Transfer adapter provides the exact interface
    and weight-loading hooks once the checkpoint is acquired.
--------------------------------------------------------------------------------
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from class_config import NUM_CLASSES
except ImportError:
    from src.class_config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: Tuple[int, int] = (2, 2),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CompactAudioCNN(nn.Module):
    """4-Layer Convolutional Neural Network for 5-class audio classification.

    Input shape:  (batch, 1, 64, 1001) log-mel spectrogram
    Output shape: (batch, 5) raw logits for CrossEntropyLoss
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        n_mels: int = 64,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels

        # Convolutional feature extraction backbone
        self.block1 = ConvBlock(1, 32, pool_size=(2, 2), dropout=0.15)
        self.block2 = ConvBlock(32, 64, pool_size=(2, 2), dropout=0.20)
        self.block3 = ConvBlock(64, 128, pool_size=(2, 2), dropout=0.25)
        self.block4 = ConvBlock(128, embedding_dim, pool_size=(2, 2), dropout=dropout)

        # Global average pooling across time and frequency
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense classification head
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn_fc = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop_fc = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global pooling embedding vector (batch, embedding_dim)."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits of shape (batch, num_classes)."""
        feat = self.extract_features(x)
        h = self.fc1(feat)
        if h.shape[0] > 1:
            h = self.bn_fc(h)
        h = self.relu(h)
        h = self.drop_fc(h)
        logits = self.classifier(h)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized softmax probabilities across the 5 classes."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class PannsConvBlock(nn.Module):
    """Convolution block for PANNs CNN14: Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU -> Pool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        pool_size: Tuple[int, int] = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(
                x, kernel_size=pool_size
            )
        return x


class DFTSTFT(nn.Module):
    """Discrete Fourier Transform Short-Time Fourier Transform implemented via Conv1d."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160, win_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.conv_real = nn.Conv1d(
            1, n_fft // 2 + 1, kernel_size=win_length, stride=hop_length, bias=False
        )
        self.conv_imag = nn.Conv1d(
            1, n_fft // 2 + 1, kernel_size=win_length, stride=hop_length, bias=False
        )

        # Initialize analytical DFT basis
        window = np.hanning(win_length)
        dft = np.fft.rfft(np.eye(win_length), n=n_fft, axis=1).T
        self.conv_real.weight.data = torch.from_numpy(
            (dft.real * window)[:, None, :]
        ).float()
        self.conv_imag.weight.data = torch.from_numpy(
            (-dft.imag * window)[:, None, :]
        ).float()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        return real, imag


class Spectrogram(nn.Module):
    """Computes power spectrogram using DFTSTFT."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160, win_length: int = 512):
        super().__init__()
        self.stft = DFTSTFT(n_fft, hop_length, win_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(x)
        return (real ** 2 + imag ** 2).transpose(1, 2).unsqueeze(1)


class LogmelFilterBank(nn.Module):
    """Log-mel filterbank transformation matching official PANNs configuration."""

    def __init__(
        self,
        n_fft: int = 512,
        n_mels: int = 64,
        sample_rate: int = 16000,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        ref: float = 1.0,
        amin: float = 1e-10,
    ):
        super().__init__()
        self.ref = ref
        self.amin = amin
        self.melW = nn.Parameter(
            torch.zeros(n_fft // 2 + 1, n_mels), requires_grad=False
        )
        try:
            import torchaudio.functional as aF

            fb = aF.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm="slaney",
                mel_scale="slaney",
            )
            self.melW.data.copy_(fb)
        except Exception:
            pass

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(spec, self.melW)
        logmel = 10.0 * torch.log10(torch.clamp(mel, min=self.amin, max=float("inf")))
        logmel = logmel - 10.0 * np.log10(max(self.amin, self.ref))
        return logmel


class PannsCNN14(nn.Module):
    """PANNs CNN14 model adapted for 5-class emergency sound classification.

    Paper: Kong et al., 'PANNs: Large-Scale Pretrained Audio Neural Networks
    for Audio Pattern Recognition', IEEE/ACM TASLP 2020.

    Backbone features:
      - 6 convolutional blocks (2 Conv2d layers each with BN and ReLU)
      - Max/Mean global pooling across time
      - 2048-dimensional dense feature representation (fc1)
    Transfer Head:
      - Replaces 527-way AudioSet head (`fc_audioset`) with `fc_transfer` (2048 -> num_classes)
      - Preserves canonical 5-class ordering:
        0: scream, 1: glass_breaking, 2: gunshot, 3: siren, 4: other
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = 2048

        # Frontend acoustic feature extraction
        self.spectrogram_extractor = Spectrogram(512, 160, 512)
        self.logmel_extractor = LogmelFilterBank(512, 64)
        self.bn0 = nn.BatchNorm2d(64)

        # 6 ConvBlocks matching official PANNs CNN14
        self.conv_block1 = PannsConvBlock(1, 64)
        self.conv_block2 = PannsConvBlock(64, 128)
        self.conv_block3 = PannsConvBlock(128, 256)
        self.conv_block4 = PannsConvBlock(256, 512)
        self.conv_block5 = PannsConvBlock(512, 1024)
        self.conv_block6 = PannsConvBlock(1024, 2048)

        # Bottleneck dense representation
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.dropout = nn.Dropout(p=dropout)

        # 5-Class Transfer Classifier Head (replaces AudioSet 527-way head)
        self.fc_transfer = nn.Linear(2048, num_classes, bias=True)

        if freeze_backbone:
            self.freeze_backbone()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dimensional global embedding vector."""
        if x.ndim == 2 or (x.ndim == 3 and x.shape[1] == 1):
            if x.ndim == 3:
                x = x.squeeze(1)
            spec = self.spectrogram_extractor(x)
            x = self.logmel_extractor(spec)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        # Ensure (batch, 1, time=1001, mel=64)
        if x.ndim == 4 and x.shape[2] == 64 and x.shape[3] != 64:
            x = x.transpose(2, 3)

        # Apply bn0 over mel bins
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        # Global average + max pooling over time
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = self.dropout(x)
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits of shape (batch, num_classes)."""
        feat = self.extract_features(x)
        logits = self.fc_transfer(feat)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized softmax probabilities across the classes."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def freeze_backbone(self):
        """Freeze all backbone parameters, keeping only fc_transfer trainable."""
        for name, param in self.named_parameters():
            if "fc_transfer" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_backbone(self, num_blocks: Optional[int] = None):
        """Unfreeze backbone parameters.

        Args:
            num_blocks: If None, unfreeze entire backbone.
                        If int (1-6), unfreeze fc_transfer, fc1, and the top N conv blocks
                        (e.g., num_blocks=1 unfreezes conv_block6, num_blocks=2 unfreezes conv_block6 and conv_block5).
        """
        if num_blocks is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for param in self.fc_transfer.parameters():
                param.requires_grad = True
            for param in self.fc1.parameters():
                param.requires_grad = True
            target_blocks = [f"conv_block{7 - i}" for i in range(1, num_blocks + 1)]
            for name, param in self.named_parameters():
                if any(tb in name for tb in target_blocks):
                    param.requires_grad = True

    def load_from_pretrain(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load pretrained PANNs checkpoint weights into backbone.

        Reports matched keys, missing keys, and unexpected keys explicitly.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found at: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = state["model"] if "model" in state else state

        own_state = self.state_dict()
        matched_keys = []
        unexpected_keys = []
        missing_keys = []

        for k, v in model_state.items():
            if k in own_state:
                if own_state[k].shape == v.shape:
                    own_state[k].copy_(v)
                    matched_keys.append(k)
                else:
                    unexpected_keys.append(f"{k} (shape mismatch: model {own_state[k].shape} vs ckpt {v.shape})")
            else:
                unexpected_keys.append(k)

        for k in own_state:
            if k not in model_state:
                missing_keys.append(k)

        loaded_param_count = sum(
            p.numel() for name, p in self.named_parameters() if name in matched_keys
        )

        return {
            "checkpoint_path": str(checkpoint_path),
            "matched_keys_count": len(matched_keys),
            "matched_keys": matched_keys,
            "unexpected_keys": unexpected_keys,
            "missing_keys": missing_keys,
            "loaded_param_count": loaded_param_count,
        }


# Alias for backward compatibility
PannsCNN14Transfer = PannsCNN14


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
