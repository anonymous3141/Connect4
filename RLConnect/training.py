from torch import nn
from torch.distributions import Categorical
from connect4 import ConnectFour
from AgentModels import get, inputify, value, get_probabilities, sample_action
from AgentModels import PolicyNet, ValueNet
import torch
import numpy as np
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

# This should take about 2-3 hours to train
env = ConnectFour()
np.random.seed(0)
torch.manual_seed(0)
EPISODES = 100000 # number of self plays
PERIOD = 500 # every PERIOD games cache agent and keep as possible opponent
ACTION_COUNT = 7
LEARNING_RATE = 0.001
USE_VALUE_BASELINE = True
DISCOUNT_FACTOR = 0.5 #idk maybe I should impl n step reward instead
SKEW = 2 # controls distribution for picking opponents
model = PolicyNet()
valueModel = ValueNet()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
valueOptimizer = torch.optim.Adam(valueModel.parameters(), lr = LEARNING_RATE)

# learning loop
candidate_agents = [copy.deepcopy(model)]
prev_time = time.time()
for ep in range(0, EPISODES):
    state = env.reset()
    done = False
    stateList = []

    # random but weighted selection of opponents, preferring more modern agents
    agent_number = int(np.random.randint(len(candidate_agents)**SKEW)**(1/SKEW))
    opponent = candidate_agents[agent_number]

    if np.random.randint(2) == 1:
        # flip coin to go first or second
        state,_,_ = env.play(sample_action(opponent, state))
    
    while not done:

        # decide action by strictly following policy (no eps greedy)
        move = sample_action(model, state)
        
        # make action
        #env.displayBoard()
        tmp_state, reward, done = env.play(move)


        # opponent makes action too
        
        if not done:
            state2, reward2, done = env.play(sample_action(opponent, tmp_state))
        else:
            reward2 = 0
        stateList.append([state, move, reward + reward2])
        state = state2
        
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
      print(f"Episode {ep+1}, took {time.time()-prev_time} seconds")
      prev_time = time.time()

    if (ep+1)%PERIOD == 0:
        candidate_agents.append(copy.deepcopy(model))
    
torch.save(model.state_dict(), "connect4PolicyVer1.pth")
torch.save(valueModel.state_dict(), "connect4ValueVer1.pth")
