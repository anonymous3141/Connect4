from enviromentBase import EnviromentBase
import numpy as np 
import copy

np.random.seed(0)

class ConnectFourBase:

    def __init__(self):
        self.R = 6
        self.C = 7
        self.N = 4
        self.board = [[0]*self.C for r in range(self.R)]
        self.turn = 1
        self.numMoves = 0
    
    def newGame(self):
        self.board = [[0]*self.C for r in range(self.R)]
        self.turn = 1

    def checkWin(self):
        dy = [1,0,-1,0,1,1,-1,-1]
        dx = [0,1,0,-1,1,-1,1,-1]

        for r in range(self.R):
            for c in range(self.C):
                for i in range(8):
                    cr = r 
                    cc = c 
                    for n in range(self.N):
                        if min(cr,cc) < 0 or cr >= self.R or cc >= self.C or\
                        self.board[cr][cc] == 0 or self.board[cr][cc] != self.board[r][c]:
                            break
                        if n+1 == self.N:
                            return self.board[cr][cc]
                        cr += dy[i]
                        cc += dx[i]
        return 0 # no win

    def isDraw(self):
        return self.numMoves == self.R * self.C

    def hasGameEnded(self):
        return self.isDraw() or self.checkWin()

    def listValidMoves(self):
        a = []
        for i in range(self.C):
            if self.board[self.R-1][i] == 0:
                a.append(i)
        return a
    
    def play(self,x):
        for i in range(self.R):
            if self.board[i][x] == 0:
                self.board[i][x] = self.turn
                self.numMoves += 1
                self.turn = 3 - self.turn # 1 <-> 2
                return i # return height of stone fell to
        return self.R # invalid move

    def displayBoard(self):
        for r in range(self.R-1,-1,-1):
            for c in range(self.C):
                print(self.board[r][c],end="")
            print()

    def duplicate(self):
        return copy.deepcopy(self)

x = ConnectFourBase()

