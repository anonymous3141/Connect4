import numpy as np
from torch import nn
from torch.distributions import Categorical
import torch
from connect4 import ConnectFour
import torch
import copy, time
"""
This file implements the training procedure for our agent.
We use Policy Gradient with a value baseline. This comprises
two networks with similar architecture, a value network and a
policy network

If things dont work out well I'll just take the value networks
and switch to TD instead

W.R.T network the second convolution layer is pretty small
we'll see if 16 is enough
"""
env = ConnectFour()
np.random.seed(0)
torch.manual_seed(0)
EPISODES = 100000 # number of self plays
PERIOD = 2000 # Every 2000 games cache agents
ACTION_COUNT = 7
LEARNING_RATE = 0.001
USE_VALUE_BASELINE = True
DISCOUNT_FACTOR = 0.99

def get(mask, r, c):
        # return if stone at (r,c) or empty
        # r,c 0 indexed
        if mask&(1<<(7*c+r)):
            return 1
        else:
            return 0
        
def inputify(state, unsqueeze=True):
    # state is player 1 mask, all stones mask, turn in that order 
    # torch says need double but actually turning it into float works instead
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
    if detach:
        return model(inputify(state)).detach().item()
    else:
        #print("DEBUG", model(inputify(state)).squeeze(0).shape)
        return model(inputify(state)).squeeze(0) # return scalar tensor

def get_probabilities(model, state, detach = True):
    # invalid moves are masked (probabilities set to -1000)
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
    return probs.sample()
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

model = PolicyNet()
valueModel = ValueNet()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
valueOptimizer = torch.optim.Adam(valueModel.parameters(), lr = LEARNING_RATE)

# learning loop
candidate_agents = [copy.deepcopy(model)]

for ep in range(0, EPISODES):
    state = env.reset()
    done = False
    stateList = []

    # random but weighted selection of opponents, preferring more modern agents
    SKEW = 3
    agent_number = int(np.random.randint(len(candidate_agents)**SKEW)**(1/SKEW))
    opponent = candidate_agents[agent_number]

    if np.random.randint(2) == 1:
        # flip coin to go first or second
        state,_,_ = env.play(sample_action(opponent, state))
    
    while not done:

        # decide action by strictly following policy (no eps greedy)
        move = sample_action(model, state)
        # make action
        tmp_state, reward, done = env.play(move)


        # opponent makes action too
        
        if not done:
            state2, reward2, done = env.play(sample_action(opponent, tmp_state))
        else:
            reward2 = 0
        stateList.append([state, move, reward + reward2])
        state2 = state
        
    g = 0
    weightedProb = torch.Tensor([0])

    # We utilise N-step returns as the gradient returns are sparse
    for t in range(len(stateList)-1, -1, -1):
        curReward = stateList[t][2]
        curAction = stateList[t][1]
        curState = stateList[t][0]
        g = g*DISCOUNT_FACTOR + curReward # G(t)
        if USE_VALUE_BASELINE:
          delta = g - value(valueModel, curState)
        else:
          delta = g

        # - DELTA as we're doing gradient ascent not descent
        weightedProb += -delta*(DISCOUNT_FACTOR**t)*torch.log(get_probabilities(model, curState, detach=False)[curAction]) #log(Pr(Action))

    optimizer.zero_grad()
    weightedProb.backward()
    optimizer.step()

    if USE_VALUE_BASELINE:
        valuePredictions = [] # DIY calculation of MSELoss
        actualValues = []
        for t in range(len(stateList)-1, -1, -1):
            curReward = stateList[t][2]
            curAction = stateList[t][1]
            curState = stateList[t][0]
            g = g*DISCOUNT_FACTOR + curReward # G(t)
            valuePredictions.append(value(valueModel, curState, False))
            actualValues.append(g)
        #print(len(valuePredictions))
        valueOptimizer.zero_grad()
        valueLoss = nn.MSELoss()(torch.cat(valuePredictions), torch.Tensor(actualValues))
        valueLoss.backward()
        valueOptimizer.step()
      
    #update values
    if (ep+1)%25 == 0:
      print(f"Episode {ep+1}")
    
torch.save(model.state_dict(), "connect4PolicyVer1.pth")
torch.save(valueModel.state_dict(), "connect4ValueVer1.pth")
