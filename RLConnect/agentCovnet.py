import gym
import numpy as np
from torch import nn
import torch
import copy, time

class QNet(nn.Module):

  def __init__(self):
    # map state into each of 7 Q values
    super(QNet, self).__init__() # 2 hidden layers of 8 neurons
    self.feature_stack = nn.Sequential(nn.Conv2d(2,32,4), #32 4 by 4 filters with stride 1
                                      nn.ReLU(),
                                      nn.Conv2d(32,16,(3,2)), #8 2 by 2 filters
                                      nn.ReLU(),
                                      nn.Flatten(start_dim=1, end_dim=- 1))
    self.linear_stack = nn.Sequential(nn.Linear(48+1,20),
                                      nn.ReLU(),
                                      nn.Linear(20,7))
  def forward(self, x):
    # x is of shape (n,85), each input contains 85 0-1 numbers
    # position is 6 by 7 with 2 channels containing you and your opponent's stones
    turn = x[:,-1].unsqueeze(1)
    features = self.feature_stack(torch.reshape(x[:,:-1],(x.shape[0],2,6,7)))
    return self.linear_stack(torch.cat((features,turn),1))


x=QNet()
#print(x((torch.zeros((1,2,6,7),torch.zeros((1))))))
res = x.feature_stack(torch.zeros((1,2,6,7)))
print(res)
print('feature shape:', res.shape)
print(sum(dict((p.data_ptr(), p.numel()) for p in x.parameters()).values()))

test = torch.arange(5*85).reshape(5,85).float()
print(x(test))
