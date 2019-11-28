#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import itertools 
import pickle
from random import choice, randrange, uniform, randint
#from keras import models, layers, optimizers


# # Player 
# This is the superclass used for all players. 

# In[2]:


#Superclass Player, 
class Player: 
    def __init__(self, game):
        self.game = game
        self.color = None
        return 
    
    def setcolor(self, color):
        self.color = color
        
    def move(self):
        if self.color is None:
            print('Must assign color')
            return 
        return
    
    def getgame(self):
        return self.game
    
    def playvshuman(self, startplayer = 'bot'):
        self.setcolor(1)
        humancolor = -1

        if startplayer == 'bot':
            self.move()
        while self.game.gamefinished() == False:
            self.game.printboard()
            if len(self.game.possiblesteps(humancolor)) != 0:
                r = int(input("Row index ")) 
                c = int(input('Column index'))
                
                while self.game.turn((r,c), self.color*-1) is False:
                    cont = input(('Can not pick', r,',', c,' press anywhere to enter new index'))
                    r = int(input("Row index ")) 
                    c = int(input('Column index'))
                
                self.game.printboard()
            else:
                print('No possible moves!')
            cont = input('Press somewhere')
            self.move()
        self.game.printboard()
        print('Game finished, winner: ', 'bot' if self.game.winner()== 1 else 'human')
        return 
 
 


# ## Randomplayer
# Randomly picks the next action from a list of all possible actions 

# In[24]:


class RandomPlayer(Player):       
    def move(self):
        super().move()
        posmoves = self.game.possiblesteps(self.color)
        if len(posmoves) != 0: 
            move = choice(posmoves)
            self.game.turn(move, self.color)
        return

    


# ## Probability functions
# Functions below are used to calculate probability of win given a certain state (assuming all actions are random). i.e. if all possible combinations of actions are taken from a certain state, how often do they result in a win. These methods cannot actually be executed right now because they rely on methods that I changed and put in the Othello class. 

# In[25]:


# The two functions below together calculates the probability that color will win, given that they make move. 
def probabilityofwin(board, move, color): 
    tempgame = np.copy(board) 
    turn(tempboard, move, color)
    if gamefinished(tempboard):
        return 1 if np.sign(np.sum(tempboard)) == np.sign(color) else 0
    return probwinoppturn(tempboard, color)



def probwinoppturn(board, owncolor):
    posmovesopponent = possiblesteps(board, owncolor*-1)
    winprobs = []
    
    if gamefinished(board):
        return 1 if np.sign(np.sum(board)) == np.sign(owncolor) else 0
    
    if len(posmovesopponent) == 0: 
        posmoves = possiblesteps(board, owncolor)
        for posmove in posmoves: 
            winprobs+= [probabilityofwin(board, posmove, owncolor)]
        
    else:
        for oppmove in posmovesopponent: 
            tempboard = np.copy(board)
            turn(tempboard, oppmove, owncolor*-1)
            if gamefinished(tempboard):
                winprobs += [1 if np.sign(np.sum(tempboard)) == np.sign(owncolor) else 0]
            
            else:
                posmoves = possiblesteps(tempboard, owncolor)
                if len(posmoves) == 0: 
                    winprobs += [probwinoppturn(tempboard, owncolor)]
                else:
                    for posmove in posmoves:
                        winprobs += [probabilityofwin(tempboard, posmove, owncolor)]
    return np.mean(winprobs)



#Generating all possible boards starting from board, with firstcolor making the first move 
def genpossibleboards(board, firstcolor = 1, firstcall = True):
    if gamefinished(board):
        return [], []
    posmoves = possiblesteps(board, firstcolor)
    posboards = [turn(np.copy(board), posmove, firstcolor) for posmove in posmoves]
    followingboards = []
    colors = []
    colors = np.append(colors, firstcolor*np.ones(len(posboards)))
    for posboard in posboards: 
        temp = genpossibleboards(np.copy(posboard), firstcolor*-1, firstcall = False)
        followingboards += temp[0]
        colors = np.append(colors, temp[1])
        if firstcall:
            print(len(followingboards + posboards))
    if len(posboards) == 0: 
        temp = genpossibleboards(np.copy(board), firstcolor*-1, firstcall = False)
        followingboards += temp[0]
        colors = np.append(colors, temp[1])
        
        if firstcall: 
            print(len(followingboards + posboards))
    
    return posboards + followingboards, colors



#Calculating list of win probabilities for the boards and colors list specified.  
def winprobs_for_boards(boards, colors): # the colors list specifies what color made the last move in a particular board
    probs = [probwinoppturn(b, c) for (b,c) in zip(boards, colors)]
    return probs

# Calculates win probabilities for all possible boards of size boardsize
def allpossibleboardsandprobs(boardsize):
    board = createboard(boardsize)
    boards, colors = genpossibleboards(board)
    probs = winprobs_for_boards(boards, colors)
    return boards, probs


# ## ProbabilityPlayer
# Uses brute force to calculate all possible outcomes, assign win probability for possible next actions and chooses next action to maximize win probability. Can use previously calculated win probabilities from file if path is specified. 

# In[34]:


class ProbabilityPlayer(Player):
    def __init__(self, game, probdict = None, probdict_file = None):
        super().__init__(game)
        if probdict != None:
            self.probdict = probdict
        elif probdict_file != None:
            f = open(probdict_file,"rb")
            self.probdict = pickle.load(f)
            f.close()
        else: 
            boards, probs = allpossibleboardsandprobs(boardsize = self.game.boardsize)
            self.probdict = {}
            for i in range(len(boards)): 
                board = boards[i]
                Hash = str(board.reshape(len(board)*len(board))).replace('[', '').replace(']','').replace('.','').replace(' ', '')
                self.probdict[Hash] = probs[i]
        return 
    
    def move(self):
        posmoves = self.game.possiblesteps(self.color)
        if len(posmoves) != 0:
            posboards = [self.game.turn(move, self.color, keep_board = True) for move in posmoves]
            posboards = [self.game.board_to_hash(board) for board in posboards]
            probabilities = [self.probdict[board] for board in posboards]
            index = np.argmax(probabilities)
            move = posmoves[index]
            self.game.turn(move, self.color)
        return
    
    
   


# ## RLPlayer
# Reinforcement learning player (epsilon-greedy). Keeps a table action_value_map to map string representations of boards to values (ranging between -win_value and win_value). Boards representing finished games will either have value win_value (if win), -win_value (if loose) or 0 (if draw). After a game is finished the values should be updating s.t. $$ V(a_{i})_{new} := V(a_{i})_{old} + lr*(V(a_{i+1})-V(a_{i})_{old}) $$ i.e. it will increase if it leads to higher ensuing values and decrease if it leads to lower ensuing values. Values are all initialized to starting_values. The Player will choose a random action a fraction of the time, specified by epsilon. 

# In[42]:


class RLPlayer(Player): 
    def __init__(self, game, epsilon = 0.1, starting_values = 0, win_value = 1, lr = 0.1):
        super().__init__(game)
        self.action_value_map = {self.game.board_to_hash() : starting_values}
        self.train_epsilon = epsilon
        self.epsilon = epsilon
        self.starting_values = starting_values
        self.win_value = win_value
        self.actions = []
        self.lr = lr
        return 
    
    def move(self):
        posmoves = self.game.possiblesteps(self.color)
        if len(posmoves) != 0: 
            pos_actions = [self.game.turn(move, self.color, keep_board = True) for move in posmoves]
            pos_actions = [self.game.board_to_hash(board) for board in pos_actions]
            
            values = []
            for action in pos_actions: 
                if action in self.action_value_map:
                    values += [self.action_value_map[action]]
                else:
                    self.action_value_map[action] = self.starting_values
                    values += [self.starting_values]
            if uniform(0,1) > self.epsilon:
                index = np.argmax(values)
            else:
                index = randrange(len(posmoves))
            move = posmoves[index]
            
            self.actions += [pos_actions[index]]
            self.game.turn(move, self.color)
        return 
    
    def update_values(self): 
        self.action_value_map[self.actions[-1]] = self.game.winner()*self.color*self.win_value 
        for index, action in reversed(list(enumerate(self.actions[0:-1]))):
            prev = self.actions[index+1]
            self.action_value_map[action] = self.action_value_map[action]*(1-self.lr) + self.lr*self.action_value_map[prev]
        self.actions = []
        
    def test_mode(self):
        self.epsilon = 0
        return
    def train_mode(self):
        self.epsilon = self.train_epsilon
        return
    
    def save_action_value_map(self, directory):
        import pickle
        with open(directory, 'wb') as f:
            pickle.dump(self.action_value_map, f)
        return 
    
    def load_action_value_map(self, directory):
        import pickle 
        with  open(directory, 'rb') as f:
            self.action_value_map = pickle.load(f)
        return 
    
    
   


# ## NNPlayer
# Similar to RLPlayer, but uses neural network to map boards (actions) to values ranging between -1 and 1. When a game is finished, all actions during the game will be assigned the highest value, and the network will perform one gradient descent step to fit these new action-value pairs. The idea is that the player will recognize a board even if it has not seen that specific board (because it has seen similar ones before), and have some perception of the value of the board. 

# In[7]:


class NNPlayer(Player):
    def __init__(self, game, epsilon = 0, lr = 0.0001, opt = 'adam'):
        super().__init__(game)
        from keras import models, layers, optimizers
        self.network = models.Sequential()
        self.network.add(layers.Conv2D(50,(2,2)))
        self.network.add(layers.Conv2D(50,(2,2), padding = 'same', activation = 'relu'))
        self.network.add(layers.Conv2D(50,(2,2), padding = 'same', activation = 'relu'))
        self.network.add(layers.Flatten())
        self.network.add(layers.Dense(1))
        if opt == 'adam':
            self.opt = optimizers.Adam(learning_rate = lr)
        elif opt == 'rmsprop':
            self.opt = keras.optimizers.RMSprop(learning_rate= lr)
        else:
            self.opt = opt

        self.network.compile(loss='mean_squared_error', optimizer = self.opt)
        self.actions = []
        self.epsilon = epsilon
        self.train_epsilon = epsilon
        self.values = []
        return 
    
    def move(self):
        posmoves = self.game.possiblesteps(self.color)
        
        if len(posmoves) != 0:
            pos_actions = [self.game.turn(move, self.color, keep_board = True) for move in posmoves]
            pos_actions = np.array(pos_actions).reshape((len(pos_actions), self.game.boardsize, self.game.boardsize, 1))*self.color
            
   
            if uniform(0,1) > self.epsilon:
                index = np.argmax(self.network.predict(pos_actions,  batch_size=len(pos_actions)))
            else:
                index = randint(0, len(posmoves)-1)
            move = posmoves[index]
            self.actions += [pos_actions[index]]
            self.game.turn(move, self.color)


        return 
        
        
    # Returns a list of same length as the number of actions taken, and value corresponding to who won the game and by how much
    def get_values_for_game(self):
        self.values += list(self.game.winner()*self.color*np.ones(len(self.actions)-len(self.values)))
        return
    
    def update_values(self):
        self.actions = np.array(self.actions).reshape((len(self.actions), self.game.boardsize, self.game.boardsize, 1))
        self.network.fit(self.actions, np.array(self.values), verbose = 0)
        self.actions = []
        self.values = []
        return
            
    def test_mode(self):
        self.epsilon = 0
        return
    
    def train_mode(self):
        self.epsilon = self.train_epsilon
        self.actions = []
        self.values = []
        return
    
    def change_opt(self, opt):
        self.opt = opt 
        self.network.compile(loss='mean_squared_error', optimizer = self.opt)
        return
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon 
        self.train_epsilon = epsilon
        return
    
        
    
    def save_network(self, directory):
        self.network.save(directory + '.h5')
        return
    
    def load_network(self, directory):
        self.network = models.load_model(directory + '.h5')
        return 
      


# ## NNPlayer_mod
# Like the above, but instead of choosing action in order to maximize value, it chooses action from probabability distribution proportional to the values. i.e when choosing between action A with value 1, and action B with value 0, 
# it will choose action A in 66% of the cases (as values range from -1 to 1)

# In[29]:


class NNPlayer_mod(Player):
    def __init__(self, game, epsilon = 0, lr = 0.0001, opt = 'adam'):
        super().__init__(game)
        self.network = models.Sequential()
        self.network.add(layers.Conv2D(50,(2,2)))
        self.network.add(layers.Conv2D(50,(2,2), padding = 'same', activation = 'relu'))
        self.network.add(layers.Conv2D(50,(2,2), padding = 'same', activation = 'relu'))
        self.network.add(layers.Flatten())
        self.network.add(layers.Dense(1))
        if opt == 'adam':
            self.opt = optimizers.Adam(learning_rate = lr)
        elif opt == 'rmsprop':
            self.opt = keras.optimizers.RMSprop(learning_rate= lr)
        else:
            self.opt = opt

        self.network.compile(loss='mean_squared_error', optimizer = self.opt)
        self.actions = []
        self.epsilon = epsilon
        self.train_epsilon = epsilon
        return 
    
    def move(self):
        posmoves = self.game.possiblesteps(self.color)
        if len(posmoves) != 0:
            pos_actions = [self.game.turn(move, self.color, keep_board = True) for move in posmoves]
            pos_actions = np.array(pos_actions).reshape((len(pos_actions), self.game.boardsize, self.game.boardsize, 1))
            if uniform(0,1) > self.epsilon:
                vals = self.network.predict(pos_actions)
                if np.min(vals) < 0: #if the worst action has negative value, the value vector must be shifted before normalizing
                    vals += -1*np.min(vals)*np.ones(len(vals))
                vals *= vals/np.sum(vals) # mapping values to probability distribution
                index = choices(np.arange(0, len(posmoves)), vals)[0]
            else:
                index = randint(0, len(posmoves)-1)
            move = posmoves[index]
            self.actions += [pos_actions[index]]
            self.game.turn(move, self.color)

        return 
    
    def update_values(self):
        values = self.game.winner()*self.color*np.ones(len(self.actions))
        self.actions = np.array(self.actions).reshape((len(self.actions), self.game.boardsize, self.game.boardsize, 1))
        self.network.fit(self.actions, values, verbose = 0)
        self.actions = []
        return
            
    def test_mode(self):
        self.epsilon = 0
        return
    
    def train_mode(self):
        self.epsilon = self.train_epsilon
        return
    
    def change_opt(self, opt):
        self.opt = opt 
        self.network.compile(loss='mean_squared_error', optimizer = self.opt)
        return
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon 
        self.train_epsilon = epsilon
        return
                
class NNPlayer_2(NNPlayer):
    def __init__(self, game, epsilon = 0, lr = 0.0001, opt = 'adam'):
        super().__init__(game, epsilon, lr, opt)
        return 
    
    def move(self):
        posmoves = self.game.possiblesteps(self.color)
        
        if len(posmoves) != 0:
            pos_actions = [self.game.turn(move, self.color, keep_board = True) for move in posmoves]
            pos_actions = np.array(pos_actions).reshape((len(pos_actions), self.game.boardsize, self.game.boardsize, 1))*self.color
            
   
            if uniform(0,1) > self.epsilon:
                index = np.argmax(self.network.predict(pos_actions,  batch_size=len(pos_actions)))
            else:
                index = randint(0, len(posmoves)-1)
            move = posmoves[index]
            self.actions += [pos_actions[index]]
            self.game.turn(move, self.color)


        return 
    
    # Returns a list of same length as the number of actions taken, and value corresponding to who won the game
    def get_values_for_game(self):
        self.values += list(np.sum(self.game.board())*self.color*np.ones(len(self.actions)-len(self.values)))
        return
    
    


class NNPlayer_deepernetwork(NNPlayer):
    def __init__(self, game, epsilon = 0, lr = 0.0001, opt = 'adam'):
        super(self, game, epsilon = 0, lr = 0.0001, opt = 'adam')
        from keras import models, layers, optimizers        
        self.network = models.Sequential()
        self.network.add(layers.Conv2D(50,(3,3)))
        self.network.add(layers(Conv2D(50, (3,3))))
        self.network.add(layers.Conv2D(50,(2,2), padding = 'same', activation = 'relu'))
        self.network.add(layers.Conv2D(50,(2,2), padding = 'same', activation = 'relu'))
        self.network.add(layers.Flatten())
        self.network.add(layers.Dense(1))
        return


class MiniMaxPlayer(Player):
    
    def __init__(self, game, n = 1):
        super().__init__( game)
        self.n = n
        return
    
    def move(self):
        posmoves = self.game.possiblesteps(self.color)
        if len(posmoves) != 0:
            pos_actions = [self.game.turn(move, self.color, keep_board = True) for move in posmoves]
            vals = [self.calc_value(action, self.n) for action in pos_actions]
            index = np.argmax(vals)
            move = posmoves[index]
            self.game.turn(move, self.color)        
        return
    
    
    def calc_value(self, board, n, selfturn = False):
        if self.game.gamefinished() or n == 0:
            return np.sum(board)
        n = n-1
        posmoves = self.game.possiblesteps(self.color if selfturn else self.color*-1,  board)
        if len(posmoves) == 0:
            return self.calc_value(board, n, selfturn = False if selfturn else True)
        pos_actions = [self.game.turn(move, self.color if selfturn else self.color*-1, keep_board = True, board = board) for move in posmoves]
        
        if selfturn:
            vals = [self.calc_value(action, n) for action in pos_actions]
            return np.max(vals)
        else:
            vals = [self.calc_value(action, n, selfturn = True) for action in pos_actions]
            return np.min(vals)
           
    
