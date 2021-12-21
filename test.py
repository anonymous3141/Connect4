from MTCSbaseline import Node
from connect4 import ConnectFourBase
'''
# Test 1: should just make the move to win
game = ConnectFourBase()
game.play(0)
game.play(2)
game.play(0)
game.play(2)
game.play(0)
game.play(2)
game.displayBoard()
act = Node(game, None)
print(act.decide(10))
print(act.W)
print(act.N)
'''

'''
# Test 2: should stop the opponent from winning
game2 = ConnectFourBase()
game2.play(0)
game2.play(3)
game2.play(0)
game2.play(3)
game2.play(1)
game2.play(3)
game2.displayBoard()
act2 = Node(game2, None)
print(act2.decide(10))
print(act2.W)
print(act2.N)
'''

