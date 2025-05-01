from collections import defaultdict, OrderedDict
import torch
import torch.nn as nn


class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self):
        super().__init__()
        layers = []
        counts = defaultdict(int)
        in_channels = 3

        def add(name: str, layer: nn.Module):
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        for x in self.ARCH:
            if x != "M":
                add(
                    "conv",
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                )
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                add("pool", nn.MaxPool2d(kernel_size=2))
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32) -> (B, 512, 2, 2)
        x = self.backbone(x)
        # (B, 512, 2, 2) -> (B, 512)
        x = x.mean([2, 3])
        # (B, 512) -> (B, 10)
        x = self.classifier(x)
        return x
