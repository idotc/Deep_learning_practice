#softmax loss 的实现

import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  用循环实现softmax损失函数
  D,C,N分别表示数据维度，标签种类个数和数据批大小
  Inputs:
  - W (D, C)：weights.
  - X (N, D)：data.
  - y (N,)： labels
  - reg: (float) regularization strength

  Returns :
  - loss 
  - gradient 
  """

  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    scores=np.dot(X[i],W)
    shift_scores=scores-max(scores)
    dom=np.log(np.sum(np.exp(shift_scores)))
    loss_i=-shift_scores[y[i]]+dom
    loss+=loss_i
    for j in range(num_classes): 
      softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
      if j == y[i]: # 类似one-hot的操作
        dW[:,j] += (-1 + softmax_output) *X[i].T 
      else: 
        dW[:,j] += softmax_output *X[i].T
  loss /= num_train 
  loss += reg * np.sum(W * W)
  dW = dW/num_train + 2*reg* W
  

  return loss, dW