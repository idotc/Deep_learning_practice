import torch
import torch.nn as nn

class mobilenet_block(nn.Module):
    def __init__(self, inp_channels, out_channels, stride):
        super(mobilenet_block, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn( inp_channels,  inp_channels, stride), 
            conv_dw( inp_channels,  out_channels, stride),
        )

    def forward(self, x):
        out = self.model(x)
        return out
