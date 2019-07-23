import torch.nn as nn


class ResnetBlock(nn.Module):
    '''Resnet block '''
    def __init__(self, inplanes, outplanes, stride=1):
        super(ResnetBlock, self).__init__()
        midplanes = outplanes // 2
        self.conv1 = nn.Conv2d(inplanes, midplanes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(midplanes , midplanes , 3, stride, 1, groups=midplanes , bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(midplanes , outplanes, 1, 1, 0, bias=False)  # bias=False
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu3= nn.ReLU(inplace=True)
        if inplanes != outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, 1, 1)

    def forward(self, x):
        residual = x
	out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out
