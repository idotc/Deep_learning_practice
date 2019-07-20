import torch.nn as nn


class ResnetBlock(nn.Module):
    '''Resnet block '''
    def __init__(self, inplanes, outplanes, stride=1):
        super(HgResBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        expand_ratio=2;

        self.conv2 = nn.Conv2d(inplanes , inplanes , 3, stride, 1, groups=inplanes , bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes )
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(inplanes , outplanes, 1, 1, 0, bias=False)  # bias=False
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu3= nn.ReLU6(inplace=True)
        if inplanes != outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, 1, 1)

    def forward(self, x):
        residual = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out
