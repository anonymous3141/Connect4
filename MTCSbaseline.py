import numpy as np
from connect4 import ConnectFourBase
import time
np.random.seed(0)

class Node:

    def __init__(self, state, parent):
        self.state = state
        self.W = [0]*7 #wins
        self.N = [0]*7
        self.successors = [0]*7
        self.parent = parent
        self.isTerminal = 0
        if self.state.hasGameEnded():
            self.isTerminal = self.state.checkWin() # + 1

    def initialiseActions(self):
        # to be implemented to initialise value functions for actions
        # i.e initialise confidence levels N (in basic case) and guesses for Q
        pass

    def selectRandomAction(self):
        a_list = self.state.listValidMoves()
        

    def selectAction(self):
        # Can slot in value estimation here too
        # how do you handle terminal nodes
        if self.isTerminal != 0:
            return -1
        # UCT algorithm
        bestVal = -10**9
        C = 1.4
        #Visit a unvisited node
        a_list = self.state.listValidMoves()
        for i in a_list:
            if self.N[i] == 0:
                return i
        #Then no need to add 1

        n = sum(self.N) # how to avoid logging by 0? Is adding 1 okay
        optimal = []
        for a in a_list:
            thisVal = self.W[a] / self.N[a] + C*np.sqrt(np.log(n)/self.N[a])
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
        """

    def updAction(self, a, result):
        self.N[a] += 1
        self.W[a] += result

    def expandChild(self, a):
        if self.successors[a] == 0 and not self.isTerminal:
            newState = self.state.duplicate()
            if newState.play(a) != newState.R:
                #print("Successfully expanded",a)
                self.successors[a] = Node(newState, self)
            else:
                self.successors[a] = -1
                self.W[a] = -1
                self.N[a] = 1

    def decide(self, time_limit):
        start = time.time()
        while (time.time() - start < time_limit or np.count_nonzero(self.N) != len(self.N)):
            MTCS(self)
        # Choose best action
        return np.argmax(np.divide(self.W, self.N)) #Lookout for zero division


def MTCS(node):
    root = Node(node.state.duplicate(), None)
    var = False
    # One MTCS pass
    # Tree policy
    startNumMoves = node.state.numMoves
    actions = []
    while True:
        if node.isTerminal:
            break
        a = node.selectAction()
        actions.append(a)
        if node.successors[a] == 0:
            node.expandChild(a)
            break
        elif node.successors[a] == -1:
            return
        else:
            node = node.successors[a]
            var = True

    #node.state.displayBoard()
    #print()
    # Rollout
    result = 0
    # So node is the furthest 'old' node in the tree.

    if node.isTerminal:
        result = [0,1,-1][node.isTerminal]
        node = node.parent # Go back one since you want the last node where you took an action
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
                if res == 1:
                    result = 1
                elif res == 2:
                    result = -1
                        
                        
    # Backup
    #if actions == [0]:
    #    print(actions, result)
    while True: #node and node.state.numMoves != startNumMoves-1:
        #print(type(node))
        #if var: node.state.displayBoard()

        v = result
        if node.state.turn != root.state.turn: v *= -1

        node.updAction(actions[-1], v)
        if (node.state.board == root.state.board):
            break
        node = node.parent
        actions.pop()

'''  
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
'''
