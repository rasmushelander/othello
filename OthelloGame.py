import numpy as np
import itertools 
#This class represents an instance of the board game othello. The rules are implemented as well as a simple command
# line interface
class Othello:
    def __init__(self, boardsize):
        import numpy as np
        self.boardsize = boardsize
        self.board = self.createboard(size = self.boardsize)
        self.dirs = list(itertools.product([-1, 0, 1], repeat = 2))
        self.dirs.remove((0,0))
        self.dirs = np.array(self.dirs)
        self.indexes = list(itertools.product(list(range(self.boardsize)), repeat = 2)) 
        self.indexes = np.array(self.indexes)
        return
    
    def createboard(self, size):
        if (size % 2) == 1: 
            print('Size must be an even number')
            return
        board = np.zeros([size, size])
        board[size//2-1][size//2-1] = -1
        board[size//2][size//2] = -1
        board[size//2][size//2-1] = 1
        board[size//2-1][size//2] = 1
        return board 
    
    def printboard(self):
        return print(self.board)
    
    def board_to_hash(self, board = None):
        if board is None: 
            board = self.board
        Hash = str(board.reshape(self.boardsize*self.boardsize)).replace('[', '').replace(']','').replace('.','').replace(' ', '')
        return Hash
    
    def isflippable(self, index, direction, color): 
        index = index + direction
        #index = (index[0], index[1])
        if self.checkindexoob(index) == False:
            nbrflips = 0
            while (self.board[index[0],index[1]]*color == -1.0):
                nbrflips += 1
                newindex = index + direction
                if self.checkindexoob(newindex):
                    break
                else:
                    index = newindex
            if self.board[index[0],index[1]]*color == 1 and nbrflips > 0:
                return(True, nbrflips)
        return (False, 0)
    
    def checkindexoob(self, index): 
        if index[0] >= 0 and index[0] < self.boardsize and index[1] >= 0 and index[1] < self.boardsize:
            return False
        return True 
    
    def possiblesteps(self, color, board = None):
        if board is None:
            board = self.board
        possibilities = [idx for idx in self.indexes if board[idx[0],idx[1]] == 0 and any(self.isflippable(idx, direction, color)[0] for direction in self.dirs)]
        #possibilities = []
        #for row in range(self.boardsize):
        #   for col in range(self.boardsize):
        #        if self.board[row][col] == 0:
        #            for direction in self.dirs:
        #                idx = np.array((row, col))
        #                if self.isflippable(idx, direction, color)[0] == True:
        #                    possibilities += [idx]
        #                    break
        return possibilities
        possibilities = [np.array(idx) for idx in self.indexes if idx == 0 and self.isflippable(np.array(idx), np.array(direction), color)[0] == True]
        
    def turn(self, index, color, keep_board = False, board = None):
        if self.checkindexoob(index):
            print('Index out of bounds')
            return False
        elif self.board[index[0],index[1]] != 0:
            print('Cannot play at ', index)
            return False

        flipped = False
        if board is None:
            if keep_board: 
                board = np.copy(self.board)
            else:
                board = self.board
        else:
            board = np.copy(board)
            
        for direction in self.dirs: 
            (flippable, flips) = self.isflippable(index, direction, color)
            if flippable: 
                board[index[0],index[1]] = color
                for i in range(flips): 
                    next = index + (i+1)*direction
                    board[next[0]][next[1]] *= -1
                    flipped = True
        if flipped: 
            board[index[0],index[1]] = color
            return board
        else: 
            print('Cannot play at ', index)
            return False

    
    def gamefinished(self):
        return (0 not in self.board) or (len(self.possiblesteps(1)) == 0 and len(self.possiblesteps(-1)) == 0)
    
    def winner(self):
        if self.gamefinished():
            return np.sign(np.sum(self.board))
        print('Game is not yet finished')
        return
    
    def dif(self):
        return(np.sum(self.board))
        
    
    def restart(self):
        self.board = self.createboard(size = self.boardsize)
        return
                
  

                
    def copy(self):
        copy = Othello(self.boardsize)
        copy.board = np.copy(self.board)
        return copy
                        



