import torch
from torch import nn
import numpy

# num_features is the number of channels
In = nn.InstanceNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)

## new test data
x = torch.rand(10, 3, 5, 5)*10000

## official batch normalization
official_In = In(x)

## ours batch normalization
x1 = x.view(30, -1)
mu = x1.mean(dim=1).view(10,3,1,1)
std = x1.std(dim=1, unbiased=False).view(10,3,1,1)

my_In = (x-mu)/std

## show the diff. of sum value 
diff = (official_In-my_In).sum()
print ('diff={}'.format(diff.item()))
