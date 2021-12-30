from torch import nn
from torch.distributions import Categorical
from connect4 import ConnectFour
import numpy as np
import torch

"""
This file contains all models used in the agent design
plus some utility functions
"""

def get(mask, r, c):
        # return if there is stone at (r,c) in mask
        # r,c 0 indexed
        if mask&(1<<(7*c+r)):
            return True
        else:
            return False
        
def inputify(state, unsqueeze=True):
    # state is player 1 mask, all stones mask, turn in that order 
    # Converts state into Neural Net input format
    """
    Unit Test:
    A = inputify([2097280, 2146432, 1])
    torch.reshape(A[:-1],(2,6,7))
    """
    res = [0]*85
    for r in range(6):
        for c in range(7):
            if get(state[0], r,c):
                res[7*r+c] = 1
            elif get(state[0]^state[1],r,c):
                res[42+7*r+c] = 1
    res[84] = state[2] == 1
    if unsqueeze:
        return torch.Tensor(res).unsqueeze(0).float()
    else:
        return torch.Tensor(res).float()

def value(model, state, detach=True):
    # evaluate state with model
    if detach:
        return model(inputify(state)).detach().item()
    else:
        #print("DEBUG", model(inputify(state)).squeeze(0).shape)
        return model(inputify(state)).squeeze(0) # return scalar tensor

def get_probabilities(model, state, detach = True):
    # evaluate policy model at state
    # invalid moves are masked (probabilities set to -1000)
    # returns probability vector
    probs = model(inputify(state))[0]
    if detach:
        probs.detach()
    for i in range(7):
        if get(state[1], 5, i):
            probs[i] = 0
    return nn.Softmax(dim=-1)(probs)

def sample_action(model, state):
    # sample action from model
    # invalid moves are masked (probabilities set to -1000)
    # remaining values softmaxed
    probs = Categorical(get_probabilities(model, state))
    return probs.sample().item()
"""
Pytorch setup:
two similar networks for valuing positions and determining turn policy

Two convolutional followed by two dense layers
The input is of shape (n,85)
0 ... 41 is player 1 mask
42 ... 83 is player 2 mask
84 = 1 if player 1 turn and 0 else
"""

class ValueNet(nn.Module):

  def __init__(self):
    # map state into position value (higher means good for player 1)
    super(ValueNet, self).__init__() # 2 hidden layers of 8 neurons
    self.feature_stack = nn.Sequential(nn.Conv2d(2,32,4), #32 4 by 4 filters with stride 1
                                      nn.ReLU(),
                                      nn.Conv2d(32,16,(3,2)), #8 2 by 2 filters
                                      nn.ReLU(),
                                      nn.Flatten(start_dim=1, end_dim=- 1))
    self.linear_stack = nn.Sequential(nn.Linear(48+1,20),
                                      nn.ReLU(),
                                      nn.Linear(20,1)) # must have some nonlinearity for turn 'input'
  def forward(self, x):
    # x is of shape (n,85), each input contains 85 0-1 numbers (84 is 1 iff player 1 turn)
    # position is 6 by 7 with 2 channels containing you and your opponent's stones
    turn = x[:,-1].unsqueeze(1)
    features = self.feature_stack(torch.reshape(x[:,:-1],(x.shape[0],2,6,7)))
    return self.linear_stack(torch.cat((features,turn),1))
  
class PolicyNet(nn.Module):

  def __init__(self):
    # map state into move probabilities for turn player 
    super(PolicyNet, self).__init__() # 2 hidden layers of 8 neurons
    self.feature_stack = nn.Sequential(nn.Conv2d(2,32,4), #32 4 by 4 filters with stride 1
                                      nn.ReLU(),
                                      nn.Conv2d(32,16,(3,2)), #8 2 by 2 filters
                                      nn.ReLU(),
                                      nn.Flatten(start_dim=1, end_dim=- 1))
    self.linear_stack = nn.Sequential(nn.Linear(48+1,20),
                                      nn.ReLU(),
                                      nn.Linear(20,7)) #we'll handle softmax after
  def forward(self, x):
    # x is of shape (n,85), each input contains 85 0-1 numbers (84 is 1 iff player 1 turn)
    # position is 6 by 7 with 2 channels containing you and your opponent's stones
    turn = x[:,-1].unsqueeze(1)
    features = self.feature_stack(torch.reshape(x[:,:-1],(x.shape[0],2,6,7)))
    return self.linear_stack(torch.cat((features,turn),1))
