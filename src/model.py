import torch
import torch.nn as nn


class Alzheimer3DCNN(nn.Module):
    """
    3D Convolutional Neural Network for Alzheimer's disease classification from MRI scans.
    """

    def __init__(self):
        super(Alzheimer3DCNN, self).__init__()

        # --- FEATURE EXTRACTION ---
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Reduces each feature map to a fixed (2, 2, 2) spatial size before flattening
        self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        # --- CLASSIFICATION HEAD ---
        # 32 channels * 2 * 2 * 2 = 256 input features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to 1D vector, keep batch dimension
        x = self.classifier(x)
        return x
    