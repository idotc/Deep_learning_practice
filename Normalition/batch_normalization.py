import torch
from torch import nn
import numpy

bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)

## new test data
x = torch.rand(10, 3, 5, 5)*10000

## official batch normalization
official_bn = bn(x)

## ours batch normalization
x1 = x.permute(1, 0, 2, 3).contiguous().view(3, -1)
mu = x1.mean(dim=1).view(1,3,1,1)
std = x1.std(dim=1, unbiased=False).view(1,3,1,1)

my_bn = (x-mu)/std

## show the diff. of sum value 
diff = (official_bn-my_bn).sum()
print ('diff={}'.format(diff.item()))
