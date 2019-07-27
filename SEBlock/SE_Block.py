import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()
	midplanes = out_channels // 2
        self.residual = nn.Sequential(
	    nn.Conv2d(in_channels, midplanes, 1, 1, padding=0),
            nn.BatchNorm2d(midplanes),
            nn.ReLU(inplace=True)
		
            nn.Conv2d(midplanes, midplanes, 3, stride=stride, padding=1),
            nn.BatchNorm2d(midplanes),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(midplanes, out_channels, 1, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels// r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // r, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return x
