import gym
import numpy as np
from torch import nn
from collections import deque
from torch.utils.data import Dataset, DataLoader
import torch
import copy, time

env = gym.make("CartPole-v1")
np.random.seed(0)
torch.manual_seed(0)
EPISODES = 400
MEMORY_LENGTH = 1000
ACTION_COUNT = env.action_space.n
INPUT_DIM = env.observation_space._shape[0]
DISCOUNT_FACTOR = 1
EPS = 0.5 # for exploration

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

def inputify(state, a):
    # torch says need double but actually turning it into float works instead
    one_hot = np.array([1.0 if i == a else 0.0 for i in range(ACTION_COUNT)])
    return torch.from_numpy(np.concatenate((normalise_state(state),one_hot))).float()
def value(model, state, a):
    return model(inputify(state,a))
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
    # hopefully map x -> (val(x)+200)/20
    # min squared error
    super(ValueNet, self).__init__() # 2 hidden layers of 8 neurons
    self.linear_stack = nn.Sequential(nn.Linear(INPUT_DIM + ACTION_COUNT,8),
                                      nn.ReLU(),
                                      nn.Linear(8,8),
                                      nn.ReLU(),
                                      nn.Linear(8,1)) # no final layer RELU

  def forward(self, x):
    return self.linear_stack(x)

model = ValueNet()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
loss_fn = nn.MSELoss()
target_model = copy.deepcopy(model)
playback_samples = []
#Q learning loop


for t in range(0, EPISODES):
    state = env.reset()
    cur_eps = (EPISODES-t) * EPS/EPISODES # linearly decrease to 0
    done = False
    while not done:

        # decide action
        if np.random.random() < cur_eps:
            # choose random
            best_action = np.random.randint(0, ACTION_COUNT)
        else:
            best_action = 0
            best_value = value(model, state, 0)
            for i in range(1, ACTION_COUNT):
                val = value(model, state, i)
                if val > best_value:
                    best_value = val
                    best_action = i

        # make action
        state2, reward, done, info = env.step(best_action)

        # bootstrap and add to data list
        if not done:
            q_best = -10**9
            for a in range(ACTION_COUNT):
                q_best = max(q_best, value(target_model, state2, a))
        else:
            q_best = 0
        
        playback_samples.append([inputify(state, best_action), torch.Tensor([normalise_reward(reward+DISCOUNT_FACTOR*q_best)])])
        state = state2

    # every iterations do some improvement from the 500 most recent samples
    loader = DataLoader(BasicDataset(playback_samples[-min(500, len(playback_samples)):]),batch_size=20, shuffle=True)

    for batchnum, (X,y) in enumerate(loader):
                #print(X,y)
                pred = model(X)
                #print(model(X))
                loss = loss_fn(pred,y)
                #print(loss) #bugged
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        

    # every 25 episodes update target model

    if (t+1)%25 == 0:
        target_model = copy.deepcopy(model)

    
# test model out
state = env.reset()
done = False
survival_time = 0
while not done:
    env.render()
    survival_time += 1
    best_val = -10**9
    best_action = -1
    for i in range(ACTION_COUNT):
        thisValue = value(model, state, i) 
        if thisValue > best_val:
            best_action = i
            best_val = thisValue
    time.sleep(0.1)
    state, _, done, _ = env.step(best_action)

env.close()    
print(survival_time)
