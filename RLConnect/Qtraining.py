from torch import nn
from torch.distributions import Categorical
from connect4 import ConnectFour
from AgentModels import get, inputify, Q_action
from AgentModels import QNet
import torch
import numpy as np
import copy, time
"""
This file implements an alternative training procedure for our agent.
- use DQN with same architecture as before
- invalid move masking and experience replay.

the position losses and rewards are in absolute terms
going second means agent choices w/ epsilon greedy choices that minimise reward
going first means maximising reward
"""
# This should take about 2-3 hours to train
env = ConnectFour()
np.random.seed(0)
torch.manual_seed(0)
USE_PRETRAINED = False
PRETAINED_Q_DIR = "connect4QVer3.pth"
EPISODES = 10000 # number of self plays
PERIOD = 100 # every PERIOD games cache agent and keep as possible opponent
ACTION_COUNT = 7
LEARNING_RATE = 0.0005
REG = 0
EPS = 0.9
OPPONENT_EPS = 0.1 # add some stochasticity to opponent
DISCOUNT_FACTOR = 1
BUFFER_SIZE = 10000
REPLAY_BATCH_SIZE = 30
SKEW = 1 # controls distribution for picking opponents
model = QNet()
target_model = copy.deepcopy(model)

def random_valid():
    valid = []
    for i in range(7):
        if env.canPlay(i):
            valid.append(i)
    return np.random.choice(valid)

if USE_PRETRAINED:
    model.load_state_dict(torch.load(PRETAINED_Q_DIR))

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=REG)

# learning loop
prev_time = time.time()
opponents = [copy.deepcopy(model)]
playback_samples = []
for t in range(0, EPISODES):
    state = env.reset()
    cur_eps = 0.2+(EPISODES-t) * (EPS-0.2)/EPISODES # linearly decrease to 0.1
    done = False
    state = env.reset()
    stateList = []

    # random but weighted selection of opponents, preferring more modern agents
    agent_number = int(np.random.randint(len(opponents)**SKEW)**(1/SKEW))
    opponent = opponents[agent_number]
    if np.random.randint(2) == 1:
        # flip coin to go first or second
        state,_,_ = env.play(Q_action(opponent, state)[0])
    while not done:

        # decide action by epsilon greedy for current policy
        if np.random.random() < cur_eps:
            move = random_valid()
        else:
            move , _ = Q_action(model, state)
        
        # make action
        tmp_state, reward, done = env.play(move)
        
        # opponent makes action too, we'll add some stochasticity as well
                
        if not done:
            if np.random.random() < OPPONENT_EPS:
                reply = random_valid()
            else:
                reply, _ = Q_action(opponent, tmp_state)
            state2, reward2, done = env.play(reply)
            _, q_best = Q_action(target_model, state2)
        else:
            reward2 = 0
            q_best = 0
        
        playback_samples.append([inputify(state), move, reward+reward2+DISCOUNT_FACTOR*q_best])
        state = state2

    # off policy means being able to learn on the go
    # to play more games we'll just sample 3 times after each game
    for its in range(3):
        batch = []
        for i in range(REPLAY_BATCH_SIZE):
            batch.append(playback_samples[-np.random.randint(1, 1+min(len(playback_samples), BUFFER_SIZE))])

        # calculate losses
        expected_value = torch.cat([torch.Tensor([c[2]]) for c in batch])
        our_prediction = torch.cat([model(c[0])[0][c[1]].unsqueeze(0) for c in batch])
        loss = nn.MSELoss()(expected_value, our_prediction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # every 250 episodes update target model

    if (t+1)%250 == 0:
        target_model = copy.deepcopy(model)
        
    # shave off size of target model if needed
    if len(playback_samples) > 10*BUFFER_SIZE:
        # amortization
        playback_samples = playback_samples[-BUFFER_SIZE:]

    #update values
    if (t+1)%250 == 0:
      print(f"Episode {t+1}, took {time.time()-prev_time} seconds")
      prev_time = time.time()
      torch.save(model.state_dict(), "connect4QVer7.pth")
    
torch.save(model.state_dict(), "connect4QVer7.pth")
