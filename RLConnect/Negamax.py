import torch
from AgentModels import QNet, Q_action

class Negamax:

    def __init__(self):
        self.Q_MODEL_DIR = "connect4QVer6.pth"
        self.model = QNet()
        self.model.load_state_dict(torch.load(self.Q_MODEL_DIR))
        self.trans_table = {}
        
    def negamax(self, env, depth):
        if env.terminal():
            mult = 1 if env.turn == 1 else -1
            return env.getResult()*1000*mult, -1
        elif self.trans_table.get(env.hash(), "N/A") != "N/A":
            #print(self.trans_table[env.hash()])
            return self.trans_table[env.hash()]
        elif depth == 0:
            self.trans_table[env.hash()] = Q_action(self.model, env.getState())
            return self.trans_table[env.hash()]
        else:
            best_value = -1001
            best_move = -1
            for i in range(7):
                if not env.canPlay(i): continue
                succ = env.duplicate()
                succ.play(i)
                res, _ = self.negamax(succ, depth-1) 
                if -res > best_value:
                    best_value = -res
                    best_move = i
            self.trans_table[env.hash()] = (best_value, best_move)
            return best_value, best_move


