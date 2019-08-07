import torch
from torch import nn
import numpy

# num_features is the number of channels
gn = nn.GroupNorm(num_groups=4, num_channels=20, eps=0, affine=False)

## new test data
x = torch.rand(10, 20, 5, 5)*10000

## official batch normalization
official_gn = gn(x)

## ours batch normalization
x1 = x.view(10, 4, -1)
mu = x1.mean(dim=2).view(10,4,-1)
std = x1.std(dim=2, unbiased=False).view(10,4,-1)

my_gn = (x1-mu)/std
my_gn = my_gn.reshape(10, 20, 5, 5)

## show the diff. of sum value 
diff = (official_gn-my_gn).sum()
print ('diff={}'.format(diff.item()))
