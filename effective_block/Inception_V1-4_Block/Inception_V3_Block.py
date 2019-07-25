import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os

def layer_init(m):
    classname = m.__class__.__name__
    classname = classname.lower()
    if classname.find('conv') != -1 or classname.find('linear') != -1:
        gain = nn.init.calculate_gain(classname)
        nn.init.xavier_uniform(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('batchnorm') != -1:
        nn.init.constant(m.weight, 1)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('embedding') != -1:
        num_columns = m.weight.size(1)
        sigma = 1/(num_columns**0.5)
        m.weight.data.normal_(0, sigma).clamp_(-3*sigma, 3*sigma)

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

#add Batchnorm
class Inception_v2(nn.Module):

    def __init__(self, depth_dim, input_size, config):
        super(Inception_v2, self).__init__()

        self.depth_dim = depth_dim

        self.branch1x1 = BasicConv2d(input_channels, out_channels=config[0][0], kernel_size=1)

	self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, out_channels=config[2][0], kernel_size=1),
            BasicConv2d(config[2][0], config[2][1], kernel_size=(1, 3), padding=(0, 1)),
	    BasicConv2d(config[2][1], config[2][2], kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, out_channels=config[1][0], kernel_size=1),
	    BasicConv2d(config[1][0], config[1][1], kernel_size=(1, 3), padding=(0, 1)),
	    BasicConv2d(config[1][1], config[1][2], kernel_size=(3, 1), padding=(1, 0)),
	    BasicConv2d(config[1][2], config[1][3], kernel_size=(1, 3), padding=(0, 1)),
	    BasicConv2d(config[1][3], config[1][4], kernel_size=(3, 1), padding=(1, 0))
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=config[3][0], stride=1, padding=1),
            BasicConv2d(input_channels, out_channels=config[3][1], kernel_size=3, padding=1)
        )

    	self.apply(layer_init)

    def forward(self, x):
        
        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

	#x -> 1x1 -> 3x3(1x3 -> 3x1)
        branch3x3 = self.branch3x3(x)

        #x -> 1x1 -> 5x5(1x3 -> 3x1 -> 1x3 -> 3x1)
        branch5x5 = self.branch5x5(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)
