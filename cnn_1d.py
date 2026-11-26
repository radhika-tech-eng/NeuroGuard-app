import torch
import torch.nn as nn

class EEG_CNN(nn.Module):
    def __init__(self, n_channels=23):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

if __name__ == "__main__":
    model = EEG_CNN()
    sample = torch.randn(4, 23, 1280)
    output = model(sample)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("1D CNN architecture working!")
