import torch
import torch.nn as nn

from dataclasses import dataclass, field 

@dataclass(eq=False) #PyTorch raises error if eq is set to True. You cannot compare two classes with = operator anymore now.
class spatial_softmax_2d(nn.Module):

    heatmap_width: int 
    heatmap_height: int 
    epsilon: float = field(default=1e-8)
    
    def __post_init__(self):
        super().__init__()
        
        r_y = torch.arange(0, self.heatmap_height, 1.0) / (self.heatmap_height - 1) * 2 - 1
        r_x = torch.arange(0, self.heatmap_width, 1.0) / (self.heatmap_width - 1) * 2 - 1
        
        rany, ranx = torch.meshgrid(-r_y, r_x)   # ranx left -1, right 1, rany top 1, bottom -1
        
        self.register_buffer("ranx", torch.FloatTensor(ranx).clone())
        self.register_buffer("rany", torch.FloatTensor(rany).clone())
        
    def forward(self, heatmap_logits):

        heatmap = nn.functional.softmax(heatmap_logits.reshape(-1, self.heatmap_width * self.heatmap_height), dim=1)
        heatmap = heatmap.reshape(-1, self.heatmap_height, self.heatmap_width)

        heatmap = heatmap / (torch.sum(heatmap, dim=(1, 2), keepdim=True) + self.epsilon)
        sx = torch.sum(heatmap * self.ranx[None], dim=(1, 2))
        sy = torch.sum(heatmap * self.rany[None], dim=(1, 2))
        xy_normalized = torch.stack([sx, sy], 1)
        return xy_normalized

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

        self.spatial_softmax = spatial_softmax_2d(heatmap_width=16, heatmap_height=16)

        self.mlp = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.cnn(x)

        out = self.spatial_softmax(out)
        out = out.view(batch_size, -1)

        # do global average pooling 
        # out = torch.mean(out, [2, 3])

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

class ImplicitCNN(BaseCNN):

    def __init__(self, in_channels):
        super(ImplicitCNN, self).__init__(in_channels)

        self.build_model()

    def build_model(self):

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
        )

        self.spatial_softmax = spatial_softmax_2d(heatmap_width=16, heatmap_height=16)

        self.mlp = nn.Sequential(
            nn.Linear(2*32+2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, coords):
        batch_size = x.shape[0]
        coords_batch_size = coords.shape[0]

        out = self.cnn(x)

        out = self.spatial_softmax(out)
        out = out.view(batch_size, -1)

        out_tile = out.tile((coords_batch_size, 1))
        out = torch.cat((out_tile, coords), dim=1)

        out = self.mlp(out)
        out = out.view(1, -1)

        return out