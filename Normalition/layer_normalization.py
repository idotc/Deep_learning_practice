import torch
from torch import nn
import numpy

ln = nn.LayerNorm(normalized_shape=[3, 5, 5], eps=0, elementwise_affine=False)

## new test data
x = torch.rand(10, 3, 5, 5)*10000

## official batch normalization
official_ln = ln(x)

## ours batch normalization
x1 = x.view(10, -1)
mu = x1.mean(dim=1).view(10,1,1,1)
std = x1.std(dim=1, unbiased=False).view(10,1,1,1)

my_ln = (x-mu)/std

## show the diff. of sum value 
diff = (official_ln-my_ln).sum()
print ('diff={}'.format(diff.item()))
