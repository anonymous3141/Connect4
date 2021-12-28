import gym
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import copy, time
# This program implements the REINFORCE algorithm with or without
# a state value baseline for episodic enviroments, in this case Cartpole
# I would be pleased to know if experience replay is useful here
# Learning rate really does matter!
# This is implemented with 2 different networks. Works quite well
# REINFORCE w/ baseline is different to the A2C "advantage actor critic" algorithm: see below
# https://stats.stackexchange.com/questions/340987/how-can-i-understand-reinforce-with-baseline-is-not-a-actor-critic-algorithm
env = gym.make("CartPole-v1")
np.random.seed(0)
torch.manual_seed(0)
EPISODES = 1200 # AI exhibits exponential learning growth 
ACTION_COUNT = env.action_space.n
INPUT_DIM = env.observation_space._shape[0]
DISCOUNT_FACTOR = 1 # no discount
LEARNING_RATE = 0.001
USE_VALUE_BASELINE = True

def normalise_state(x):
  """
  avg = (env.observation_space.high + env.observation_space.low)/2
  spreadSize = (env.observation_space.high - env.observation_space.low)
  return (x - avg) / spreadSize
  """
  return x

def normalise_reward(x):
    #return x/env._max_episode_steps
    return x

def inputify(state):
    # torch says need double but actually turning it into float works instead
    return torch.from_numpy(state) # value network only

def value(model, state):
    return model(inputify(state)).detach().item()

#pytorch setup
class BasicDataset(Dataset):
  def __init__(self, dat):
    self.dat = dat

  def __len__(self):
    return len(self.dat)

  def __getitem__(self,idx):
    # is torch tensor needed?
    # probably not
    return self.dat[idx][0], self.dat[idx][1]

class ValueNet(nn.Module):

  def __init__(self):
    # map state to state value
    # min squared error
    # lets see if we can multihead this
    super(ValueNet, self).__init__() # 2 hidden layers of 8 neurons
    self.linear_stack = nn.Sequential(nn.Linear(INPUT_DIM,8),
                                      nn.ReLU(),
                                      nn.Linear(8,8),
                                      nn.ReLU())
    self.value_output = nn.Linear(8,1)

  def forward(self, x):
    # bug where x is a batch of inputs
    y = self.linear_stack(x)
    return self.value_output(y)
  
class PolicyNet(nn.Module):

  def __init__(self):
    # map state to state value
    # min squared error
    # lets see if we can multihead this
    super(PolicyNet, self).__init__() # 2 hidden layers of 8 neurons
    self.linear_stack = nn.Sequential(nn.Linear(INPUT_DIM,8),
                                      nn.ReLU(),
                                      nn.Linear(8,8),
                                      nn.ReLU())
    self.policy_output = nn.Sequential(nn.Linear(8,2),
                                       nn.Softmax(dim=-1))

  def forward(self, x):
    # bug where x is a batch of inputs
    y = self.linear_stack(x)
    return self.policy_output(y)

model = PolicyNet()
valueModel = ValueNet()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
valueOptimizer = torch.optim.Adam(valueModel.parameters(), lr = LEARNING_RATE)
max_seen = 0
# learning loop

for ep in range(0, EPISODES):
    state = env.reset()
    done = False
    stateList = []

    while not done:

        # decide action by strictly following policy (no eps greedy)
        probabilities = model(inputify(state)).detach()
        rng = np.random.random()
        best_action = 0
        while probabilities[best_action].item() < rng:
            rng -= probabilities[best_action].item()
            best_action += 1
        # make action
        state2, reward, done, info = env.step(best_action)

        stateList.append([state, best_action, reward])

        # Update Value
        
        state = state2

    # add terminal state
    stateList.append([state,0,0])
    max_seen = max(max_seen, len(stateList))
    g = 0
    weightedProb = torch.Tensor([0])

    # Important: only update after simulation complete
    # Must make sure the gradients are updated at once, otherwise previous gradients broken
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
        weightedProb += -delta*(DISCOUNT_FACTOR**t)*torch.log(model(inputify(curState))[curAction]) #log(Pr(Action))

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
            valuePredictions.append(valueModel(inputify(curState)))
            actualValues.append(g)
        
        valueOptimizer.zero_grad()
        valueLoss = nn.MSELoss()(torch.cat(valuePredictions), torch.Tensor(actualValues))
        valueLoss.backward()
        valueOptimizer.step()
      
    #update values
    if (ep+1)%25 == 0:
      print(f"Episode {ep+1}, max={max_seen}")
    
# test model out
# It is important to apply stochastic selection
state = env.reset()
done = False
survival_time = 0

while not done:
    env.render()
    survival_time += 1
    best_action = 0
    probabilities = model(inputify(state))
    rng = np.random.random()
    while probabilities[best_action].item() < rng:
            rng -= probabilities[best_action].item()
            best_action += 1
    time.sleep(0.01)
    state, _, done, _ = env.step(best_action)

env.close()    
print(survival_time)
