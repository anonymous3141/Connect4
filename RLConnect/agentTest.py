import numpy as np
from torch import nn
from torch.distributions import Categorical
from connect4 import ConnectFour
from AgentModels import get, inputify, value, get_probabilities, sample_action, Q_action
from AgentModels import PolicyNet, ValueNet, QNet
from Negamax import Negamax
import torch
import copy, time
"""
This file enables benchmarking of agent by manually playing against it
"""

np.random.seed(0)
torch.manual_seed(0)

def get_move(model_type, model, state):
    if model_type == "policy":
        return sample_action(model, state)
    elif model_type == "Q":
        return Q_action(model, state, True)[0]
    elif model_type == "negamax":
        A = Negamax()
        return A.negamax(ConnectFour(state), 5)[1]

def play_input(model_type, model_dir, ai_goes_first = True):
    # get human to play model
    if model_type == "policy":
        model = PolicyNet()
        model.load_state_dict(torch.load(model_dir))
    elif model_type == "Q":
        model = QNet()
        model.load_state_dict(torch.load(model_dir))
    else:
        model = None

    env = ConnectFour()
    state = env.reset()

    done = False

    if ai_goes_first:
        # cant be done on first move
        move = get_move(model_type, model, state)
        state, _, _ = env.play(move)

    while not done:
        env.displayBoard()
        move = -1
        while True:
            #print("AI Suggested move:")
            #print(get_move(model_type, model, state))
            move = input("Make a move (cols 0-6): ")

            try:
                move = int(move)
            except:
                print("Invalid move. Try again")
                continue
            if env.canPlay(move):
                tmp_state, reward, done = env.play(move)
                break
            else:
                print("Invalid move. Try again")

        if not done:
            response = get_move(model_type, model, tmp_state)
            print(response)
            state, reward, done = env.play(response)
        
        if reward == 1:
            print("Player 1 Wins")
        elif reward == -1:
            print("Player 2 Wins")

#play_input("policy", "connect4PolicyVer4.pth")
#play_input("Q", "connect4QVer7.pth")
play_input("negamax", None)
