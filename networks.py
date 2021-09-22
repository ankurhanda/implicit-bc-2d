import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.cnn = None
        self.mlp = None 
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x):
        out = self.cnn(x)

        # do global average pooling 
        out = torch.mean(out, [2, 3])

        # do MLP 
        out = self.mlp(out)

        return out

    def infer_output_size(self, input_size):

        if isinstance(input_size, tuple):
            h, w = input_size[0], input_size[1]
            sample_input = torch.zeros(1, self.in_channels, h, w)
        else:
            sample_input = torch.zeros(1, self.in_channels, input_size, input_size)

        with torch.no_grad():
            output = self.cnn(sample_input)

        if isinstance(input_size, tuple):
            return (output.shape[-2], output.shape[-1])
        return output.shape[-1]

