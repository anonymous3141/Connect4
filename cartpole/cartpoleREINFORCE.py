import gym
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import copy, time
# This program implements the REINFORCE algorithm with or without
# a state value baseline for episodic enviroments, in this case Cartpole
# It is different to the A2C "advantage actor critic" algorithm
# I would be pleased to know if experience replay is useful here
# For some reason baseline is ridiculously slow and doesnt work well
# Maybe having multiheaded network is bad idea QaQ
env = gym.make("CartPole-v1")
np.random.seed(0)
torch.manual_seed(0)
EPISODES = 200
ACTION_COUNT = env.action_space.n
INPUT_DIM = env.observation_space._shape[0]
DISCOUNT_FACTOR = 1 # no discount
LEARNING_RATE = 0.01
USE_VALUE_BASELINE = False

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
    return model(inputify(state))[-1].detach().item()

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


class ValuePolicyNet(nn.Module):

  def __init__(self):
    # map state to state value
    # min squared error
    # lets see if we can multihead this
    super(ValuePolicyNet, self).__init__() # 2 hidden layers of 8 neurons
    self.linear_stack = nn.Sequential(nn.Linear(INPUT_DIM,8),
                                      nn.ReLU(),
                                      nn.Linear(8,8),
                                      nn.ReLU())
    self.policy_output = nn.Sequential(nn.Linear(8,2),
                                       nn.Softmax(dim=-1))
    self.value_output = nn.Linear(8,1)

  def forward(self, x):
    # bug where x is a batch of inputs
    y = self.linear_stack(x)
    return torch.cat((self.policy_output(y), self.value_output(y)), dim=-1)

model = ValuePolicyNet()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
# learning loop

for ep in range(0, EPISODES):
    state = env.reset()
    done = False
    stateList = []

    while not done:

        # decide action by strictly following policy (no eps greedy)
        probabilities = model(inputify(state))[:-1].detach()
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
    g = 0
    weightedProb = torch.Tensor([0])
    # Must make sure the gradients are updated at once, otherwise previous gradients broken
    for t in range(len(stateList)-1, -1, -1):
        curReward = stateList[t][2]
        curAction = stateList[t][1]
        curState = stateList[t][0]
        g = g*DISCOUNT_FACTOR + curReward # G(t)
        if USE_VALUE_BASELINE:
          delta = g - value(model, curState)
        else:
          delta = g

        weightedProb += -delta*torch.log(model(inputify(curState))[curAction]) #log(Pr(Action))

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
            valuePredictions.append(model(inputify(curState))[-1].unsqueeze(0))
            actualValues.append(g)

        valueLoss = nn.MSELoss()(torch.cat(valuePredictions), torch.Tensor(actualValues))
        optimizer.zero_grad()
        valueLoss.backward()
        optimizer.step()
      
    #update values
    if (ep+1)%25 == 0:
      print(f"Episode {ep+1}")
    
# test model out
state = env.reset()
duplicate_state = copy.deepcopy(env)
done = False
survival_time = 0
while not done:
    env.render()
    survival_time += 1
    best_action = torch.argmax(model(inputify(state))[:-1]).item()
    time.sleep(0.2)
    state, _, done, _ = env.step(best_action)

env.close()    
print(survival_time)
