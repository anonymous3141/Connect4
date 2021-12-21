import numpy as np
from connect4 import ConnectFourBase
np.random.seed(0)

class Node:

    def __init__(self, state, parent):
        self.state = state
        self.Q = [0]*7
        self.N = [0]*7
        self.successors = [0]*7
        self.parent = parent
        self.isTerminal = 0
        if self.state.hasGameEnded():
            self.isTerminal = self.state.checkWin() + 1

    def initialiseActions(self):
        # to be implemented to initialise value functions for actions
        # i.e initialise confidence levels N (in basic case) and guesses for Q
        pass

    def bestAction(self):
        # Can slot in value estimation here too
        # how do you handle terminal nodes
        """
        # UCT algorithm
        bestVal = -10**9
        C = 1.4
        n = sum(self.N)+1 # how to avoid logging by 0? Is adding 1 okay
        optimal = []
        for a in range(7):
            thisVal = self.Q[a] + C*np.sqrt(np.log(n)/(self.N[a]+1))
            if thisVal > bestVal:
                optimal.append(a)
                bestVal = thisVal
        return optimal[np.random.randint(0,len(optimal))]
        """

        # epsilon greedy
        eps = 0.1
        a_list = self.state.listValidMoves()
        if np.random.random() < eps:
            return a_list[np.random.randint(0,len(a_list))]
        
        bestMove = 0
        canMove = [0]*7
        for i in a_list:
            canMove[i] = 1

        for a in range(7):
            if canMove[a] and self.Q[a] > self.Q[bestMove]:
                bestMove = a
        return bestMove

    def updAction(self, a, newVal):
        self.N[a] += 1
        self.Q[a] += (newVal - self.Q[a])/self.N[a]

    def expandChild(self, a):
        if self.successors[a] == 0 and not self.isTerminal:
            newState = self.state.duplicate()
            if newState.play(a) != newState.R:
                #print("Successfully expanded",a)
                self.successors[a] = Node(newState, self)
            else:
                self.successors[a] = -1
                self.Q[a] = -1
                self.N[a] = 1

def MTCS(node):
    # One MTCS pass
    # Tree policy
    startNumMoves = node.state.numMoves
    actions = []
    while True:
        a = node.bestAction()
        actions.append(a)
        if node.successors[a] == 0:
            node.expandChild(a)
            break
        elif node.successors[a] == -1:
            return
        else:
            node = node.successors[a]

    #node.state.displayBoard()
    #print()
    # Rollout
    result = 0

    if node.isTerminal:
        result = [0,1,-1][node.isTerminal]
    else:
        state = node.successors[a].state.duplicate()
        while not state.hasGameEnded():
            # random strat
            #print()
            #state.displayBoard()
            #print()
            state.play(np.random.randint(0,7))
            if state.hasGameEnded():
                #print('Terminal simulation state')
                #state.displayBoard()
                res = state.checkWin()
                if res != 0:
                    result = 1 if res == 1 else -1
                    break

    # Backup
    while node and node.state.numMoves != startNumMoves-1:
        #print(type(node))
        #print(node.state)
        v = result
        if node.state.turn == 2: v *= -1
        node.updAction(actions[-1], v)
        node = node.parent
        actions.pop()
    
# Unit Tests
seq = "44444222242575737753377556666251611611133"

x = ConnectFourBase()
for i in range(10):
    x.play(int(seq[i])-1)

A = Node(x, None)
for i in range(10):
    #print("Iteration", i)
    MTCS(A)

print(A.Q, A.N)
