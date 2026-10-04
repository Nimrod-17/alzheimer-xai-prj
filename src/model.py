import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """
    Basic residual block: two 3x3x3 convolutions with GroupNorm.
    GroupNorm is used instead of BatchNorm because 3D volumes force very small batches.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        # Non-inplace activations keep every intermediate tensor available to gradient-based XAI methods
        self.relu = nn.ReLU(inplace=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet3DClassifier(nn.Module):
    """
    Lightweight 3D ResNet for Alzheimer's risk classification from MNI-aligned T1w volumes.

    For a 128 x 160 x 128 input the spatial resolution goes 64x80x64 (stem) -> 32x40x32 -> 16x20x16 -> 8x10x8.
    The head is global average pooling + a single linear layer, so the class score is a linear
    combination of the last feature maps: this keeps Grad-CAM maps faithful to the decision.
    `target_layer` exposes the last convolutional stage for XAI tools.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 widths: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, widths[0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, widths[0]),
            nn.ReLU(inplace=False),
        )

        stages = []
        for stage_in, stage_out in zip(widths[:-1], widths[1:]):
            stages.append(nn.Sequential(
                ResidualBlock3D(stage_in, stage_out, stride=2),
                ResidualBlock3D(stage_out, stage_out),
            ))
        self.stages = nn.Sequential(*stages)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(widths[-1], num_classes)

    @property
    def target_layer(self) -> nn.Module:
        """Last convolutional stage, the standard Grad-CAM target."""
        return self.stages[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = torch.flatten(self.global_pool(x), start_dim=1)
        return self.classifier(self.dropout(x))

