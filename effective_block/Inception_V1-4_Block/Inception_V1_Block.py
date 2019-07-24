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


class Inception_base(nn.Module):
    def __init__(self, depth_dim, input_size, config):
        super(Inception_base, self).__init__()

        self.depth_dim = depth_dim

        #mixed 'name'_1x1
        self.conv1 = nn.Conv2d(input_size, out_channels=config[0][0], kernel_size=1, stride=1, padding=0)

        #mixed 'name'_3x3_bottleneck
        self.conv3_1 = nn.Conv2d(input_size, out_channels=config[1][0], kernel_size=1, stride=1, padding=0)
        #mixed 'name'_3x3
        self.conv3_3 = nn.Conv2d(config[1][0], config[1][1], kernel_size=3, stride=1, padding=1)

        # mixed 'name'_5x5_bottleneck
        self.conv5_1 = nn.Conv2d(input_size, out_channels=config[2][0], kernel_size=1, stride=1, padding=0)
        # mixed 'name'_5x5
        self.conv5_5 = nn.Conv2d(config[2][0], config[2][1], kernel_size=5, stride=1, padding=2)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)
        #mixed 'name'_pool_reduce
        self.conv_max_1 = nn.Conv2d(input_size, out_channels=config[3][1], kernel_size=1, stride=1, padding=0)

        self.apply(layer_init)

    def forward(self, input):

        output1 = F.relu(self.conv1(input))

        output2 = F.relu(self.conv3_1(input))
        output2 = F.relu(self.conv3_3(output2))

        output3 = F.relu(self.conv5_1(input))
        output3 = F.relu(self.conv5_5(output3))

        output4 = F.relu(self.conv_max_1(self.max_pool_1(input)))

        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)
