import numpy as np
from torch import nn
from torch.distributions import Categorical
from connect4 import ConnectFour
from AgentModels import get, inputify, value, get_probabilities, sample_action
from AgentModels import PolicyNet, ValueNet
import torch
import copy, time
"""
This file enables benchmarking of agent.
We get the agent to play an alphaBeta agent (todo)
or ourselves
"""

# This should take about 2-3 hours to train
np.random.seed(0)
torch.manual_seed(0)



def play_input(policyModel_dir, ai_goes_first = True):
    # get human to play model
    model = PolicyNet()
    model.load_state_dict(torch.load(policyModel_dir))
    env = ConnectFour()
    state = env.reset()

    done = False

    if ai_goes_first:
        # cant be done on first move
        move = sample_action(model, state)
        state, _, _ = env.play(move)

    while not done:
        env.displayBoard()
        move = -1
        while True:
            move = int(input("Make a move (cols 0-6): "))
            if env.canPlay(move):
                tmp_state, reward, done = env.play(move)
                break
            else:
                print("Invalid move. Try again")

        if not done:
            response = sample_action(model, tmp_state)
            print(response)
            state, reward, done = env.play(response)

        if reward == 1:
            print("Player 1 Wins")
        elif reward == -1:
            print("Player 2 Wins")

play_input("connect4PolicyVer1.pth")
    
