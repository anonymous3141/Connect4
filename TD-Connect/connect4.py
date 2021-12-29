import numpy as np 
import copy

"""
Algorithm Credits to Pascal Pons, very cute code!
http://blog.gamesolver.org/solving-connect-four/06-bitboard/
"""

np.random.seed(0)

class ConnectFour:
    """ The board is encoded as 2 bitmasks
    .  .  .  .  .  .  .
    5 12 19 26 33 40 47
    4 11 18 25 32 39 46
    3 10 17 24 31 38 45
    2  9 16 23 30 37 44
    1  8 15 22 29 36 43
    0  7 14 21 28 35 42

    Bits 6,13 ... are always 0
    """
    def __init__(self):
        self.mask1 = 0 #player 1's stones
        self.mask2 = 0 #everyone's stones
        self.turn = 1
        self.numMoves = 0
    
    def reset(self):
        self.mask1 = 0
        self.mask2 = 0
        self.turn = 1
        self.numMoves = 0
    
    def checkAlignment(self, x):
        """
        Determine if in position x there is 4 in a row
        Our top row buffer comes in handy and we
        only check upwards so invalid lines always
        intersect the top row.

        The rough idea is that x&(x<<1)&(x<<2)&(x<<3) being nonzero
        implies there is a vertical 4 in a row and so on
        """

        # check vertical:
        if x&(x<<1)&(x<<2)&(x<<3):
            return True

        #check horizontal
        if x&(x<<7)&(x<<14)&(x<<21):
            return True

        #check diagonal going in upper right direction
        if x&(x<<8)&(x<<16)&(x<<24):
            return True

        #check diagonal going in upper left direction
        if x&(x>>6)&(x>>12)&(x>>18):
            return True

        return False
        

    def get(self, r, c):
        # return player whose stone at (r,c) or empty
        # r,c 0 indexed
        if self.mask1&(1<<(7*c+r)):
            return 1
        elif self.mask2&(1<<(7*c+r)):
            return 2
        else:
            return 0

    def terminal(self):
        # check if game over
        topRow = (1<<5)+(1<<12)+(1<<19)+(1<<26)+(1<<33)+(1<<40)+(1<<47)
        return (topRow == (self.mask2&topRow)) or (self.getResult() != 0)
    
    def getResult(self):
        # return -1 if player 2 won
        # return 0 if draw or not over yet
        # return 1 if player 1 won
        if self.checkAlignment(self.mask1):
            return 1
        elif self.checkAlignment(self.mask1^self.mask2):
            return -1
        else:
            return 0

    def canPlay(self, col):
        # return if can play at col
        return self.get(5, col) == 0
    
    def addStone(self,pos, col):
        # add stone to column if column empty
        # if column nonempty take column, shift it up by 1
        # and OR with column
        # does not check for invalid plays
        if pos&(1<<(7*col)) == 0:
            return (pos|(1<<(7*col)))
        pos |= (pos&((1<<(7*col+6))-(1<<(7*col))))<<1
        return pos

    def play(self, col):
        # turn player puts stone in col
        # actions after game is ended is undefined behaviour
        if not self.canPlay(col):
            raise ValueError(f'invalid move {col}')
        
        move = self.mask2 ^ self.addStone(self.mask2, col)
        self.mask2 |= move
        if self.turn == 1:
            self.mask1 |= move

        self.numMoves += 1
        self.turn = 3 - self.turn
        return self.getResult()

    def displayBoard(self):
        # display board
        for r in range(5,-1,-1):
            for c in range(7):
                print(self.get(r,c),end="")
            print()
        print('-----------')

    def duplicate(self):
        # duplicate state
        return copy.deepcopy(self)


