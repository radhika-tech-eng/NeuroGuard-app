
import torch
import torch.nn as nn


class EEG_CNN_LSTM(nn.Module):
    def __init__(self, n_channels=23, hidden_size=64, num_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)


if __name__ == "__main__":
    model = EEG_CNN_LSTM(n_channels=23)
    sample = torch.randn(4, 23, 1280)
    prediction = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", prediction.shape)
    print("CNN-LSTM architecture working!")
