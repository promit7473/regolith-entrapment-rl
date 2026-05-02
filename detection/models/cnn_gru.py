import torch
import torch.nn as nn


class SinkageDetector(nn.Module):

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


        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool1d(2),
        )


        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )


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

        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)


        _, h = self.gru(x)
        x = h[-1]

        return self.classifier(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return self.forward(x).argmax(dim=-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return torch.softmax(self.forward(x), dim=-1)


LABEL_NORMAL   = 0
LABEL_SINKING  = 1
LABEL_ENTRAPPED = 2
LABEL_NAMES    = ["normal", "sinking", "entrapped"]
