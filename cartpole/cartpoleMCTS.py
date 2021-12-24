import numpy as np
import gym, copy

# demonstration of Monte Carlo Tree search in deterministic enviroment
# epsilon greedy tree policy with random rollout
# It seems that MCTS is most suited to such enviroments
env = gym.make("CartPole-v1")
np.random.seed(0)
SIMULATIONS_PER_MOVE = 30
ACTION_COUNT = env.action_space.n
INPUT_DIM = env.observation_space._shape[0]
DISCOUNT_FACTOR = 1
EPS = 0.5 # for exploration
        
class Node:

    def __init__(self, state, parent, isTerminal):
        self.state = state
        self.Q = [0]*ACTION_COUNT # computed Q value for each move
        self.N = [0]*ACTION_COUNT # number visits
        self.R = [0]*ACTION_COUNT # cached rewards for backup (deterministic)
        self.successors = [0]*ACTION_COUNT # successor nodes
        self.parent = parent # parent nodes
        self.isTerminal = isTerminal

    def bestAction(self, eps = EPS):
        # Can slot in value estimation here too

        # epsilon greedy
        if self.isTerminal:
            raise ValueError('tried to expand terminal state')
        
        if np.random.random() < eps:
            return np.random.randint(0,ACTION_COUNT)
        
        bestCandidates = [0]
        for a in range(1,ACTION_COUNT):
            if self.Q[a] > self.Q[bestCandidates[-1]]:
                bestCandidates = [a]
            elif self.Q[a] == self.Q[bestCandidates[-1]]:
                bestCandidates.append(a)
        
        return bestCandidates[np.random.randint(0,len(bestCandidates))]

    def updAction(self, a, g):
        # should I bootstrap here
        self.N[a] += 1
        self.Q[a] += (g - self.Q[a])/self.N[a]

    def expandChild(self, a):

        if self.successors[a] != 0 or self.isTerminal:
            raise ValueError('tried to expand to already created node')
        newState = copy.deepcopy(self.state)
        _, reward, done, _ = newState.step(a)
        self.successors[a] = Node(newState, self, done)
        self.R[a] = reward
        return self.successors[a], reward

def MCTS(node):
    start_node = node
    # One MTCS pass
    # Tree policy
    actions = []
    rewards = []
    while not node.isTerminal:
        a = node.bestAction()
        actions.append(a)
        if node.successors[a] == 0:
            node, reward = node.expandChild(a)
            rewards.append(reward)
            break
        else:
            rewards.append(node.R[a])
            node = node.successors[a]
            

    # Random Rollout
    result = 0

    if not node.isTerminal:
        curState = copy.deepcopy(node.state)
        done = False
        while not done:
            _, reward, done, _ = curState.step(np.random.randint(0,ACTION_COUNT))
            result += reward
    
    # dont back up new expanded node or terminal
    node = node.parent 

    # Backup
    while True:
        result += rewards[-1]
        node.updAction(actions[-1], result)
        if node == start_node:
            break
        node = node.parent
        rewards.pop()
        actions.pop()

env.reset()            
root = Node(env, None, False)
dispEnv = copy.deepcopy(env)
res_actions = []

while not root.isTerminal:
    for i in range(SIMULATIONS_PER_MOVE):
        MCTS(root)
    curAction = root.bestAction(0)
    print(curAction, root.Q, root.N)
    res_actions.append(curAction)
    root = root.successors[curAction]


for a in res_actions:
    dispEnv.render()
    dispEnv.step(a)

dispEnv.close()
