import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = self.relu(x)

        return x


class SpectrumEncoder1D(nn.Module):
    """
    Reusable 1D-CNN spectrum encoder.

    Important:
    We do NOT use global average pooling to a single value,
    because absolute spectral position is chemically meaningful.

    Adaptive pooling to multiple bins preserves coarse
    positional information while allowing different input lengths.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 32, stride=1),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128, stride=2),
        )

        # Preserve coarse position information.
        self.pool = nn.AdaptiveAvgPool1d(16)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        z = self.projection(x)

        return z


class IRStructureModel(nn.Module):
    """
    Temporary structure-feature training head.

    The encoder is the important reusable component.
    The 14-label head is used to supervise structural learning.
    """

    def __init__(
        self,
        num_labels=14,
        embedding_dim=256,
    ):
        super().__init__()

        self.encoder = SpectrumEncoder1D(
            embedding_dim=embedding_dim
        )

        self.feature_head = nn.Linear(
            embedding_dim,
            num_labels
        )

    def forward(self, x):
        z = self.encoder(x)
        logits = self.feature_head(z)

        return {
            "embedding": z,
            "logits": logits,
        }


if __name__ == "__main__":
    model = IRStructureModel()

    x = torch.randn(8, 1, 3501)

    out = model(x)

    print("Input:", x.shape)
    print("Embedding:", out["embedding"].shape)
    print("Logits:", out["logits"].shape)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", f"{n_params:,}")
