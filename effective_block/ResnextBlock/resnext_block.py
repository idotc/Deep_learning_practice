import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cardinality = 32, depth = 4, basewidth = 64):
        super().__init__()

        C = cardinality #How many groups a feature map was splitted into
	
        D = int(depth * out_channels / basewidth) #number of channels per group
        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, C * D, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * D, C * D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * D, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x))
