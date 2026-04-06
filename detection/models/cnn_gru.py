"""
CNN-GRU Sinkage Detector — Phase 1

Architecture:
  Conv1D front-end (local per-wheel pattern extraction)
  → GRU (temporal dependencies across the 5-second window)
  → FC classifier → [normal, sinking, entrapped]

Input:  (batch, seq_len=50, n_features=11)
        features = wheel_vel(4) + wheel_torque(4) + imu_acc(3)
Output: (batch, 3) logits
"""

import torch
import torch.nn as nn


class SinkageDetector(nn.Module):
    """CNN-GRU hybrid for wheel entrapment state classification."""

    def __init__(
        self,
        n_features:  int = 11,
        seq_len:     int = 50,
        hidden_size: int = 128,
        num_layers:  int = 2,
        n_classes:   int = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_features  = n_features
        self.seq_len     = seq_len
        self.hidden_size = hidden_size

        # ── CNN front-end: local pattern extraction per time window ───────
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool1d(2),        # T → T//2
        )

        # ── GRU: sequential dynamics across the compressed sequence ───────
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) — batch of proprioceptive sequences

        Returns:
            logits: (B, n_classes)
        """
        # CNN expects (B, F, T)
        x = x.permute(0, 2, 1)     # → (B, F, T)
        x = self.cnn(x)             # → (B, 64, T//2)
        x = x.permute(0, 2, 1)     # → (B, T//2, 64)

        # GRU — take last layer's final hidden state
        _, h = self.gru(x)          # h: (num_layers, B, hidden)
        x = h[-1]                   # (B, hidden)

        return self.classifier(x)   # (B, n_classes)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices. x: (B, T, F) or (T, F)."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return self.forward(x).argmax(dim=-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities. x: (B, T, F) or (T, F)."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return torch.softmax(self.forward(x), dim=-1)


# ── Label constants ────────────────────────────────────────────────────────
LABEL_NORMAL   = 0
LABEL_SINKING  = 1
LABEL_ENTRAPPED = 2
LABEL_NAMES    = ["normal", "sinking", "entrapped"]
