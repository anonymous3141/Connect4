from connect4 import ConnectFour
"""
Unit Tests
"""

def testSequence(sequence,display=True):
    print('-----------')
    print('Running sequence', sequence)
    x = ConnectFour()
    for i in range(len(sequence)):
        x.play(sequence[i])
        if i != len(sequence)-1:
            assert(x.getResult() == 0 and x.terminal() == 0)
    if display: x.displayBoard()
    print(x.terminal(), x.getResult())
    print(x.numMoves, x.turn, x.mask1, x.mask2)

# basic functionality
testSequence([1,1,1,1,1,1])
try:
    testSequence([1,1,1,1,1,1,1]) # should error
except:
    print('error spotted')
testSequence([0,1,1,2,2,2,3,3,3,3])

# check win

testSequence([0,1,0,1,0,1,0]) # player 1 vertical win
testSequence([0,1,2,1,0,1,2,1]) # player 2 vertical win
testSequence([0,6,1,6,2,6,3]) # player 1 horizontal win

long_test = '44444135555354137672222221571771661333766'
seq = [int(c)-1 for c in long_test]
testSequence(seq)

diagonal_test = '12233434414'
seq2 = [int(c)-1 for c in diagonal_test]
testSequence(seq2)
diagonal_test_yellow = '4534332222'
seq3 = [int(c)-1 for c in diagonal_test_yellow]
testSequence(seq3)
