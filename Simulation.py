

# # Simulation
# Class to train the learning agents, as well as test the performance of agents. 


class Simulation:
    def __init__(self, p1, p2, n):
        if p1.getgame() != p2.getgame():
            print('Both players must be playing the same game')
        self.game = p1.getgame()
        self.n = n #number of simulations
        self.p1 = p1
        self.p2 = p2
        return
    
    def simulate(self, n = None, confint = 95): 
        if n == None: 
            n = self.n
            
        from statsmodels.stats import proportion
        import time
        self.p1.setcolor(1)
        self.p2.setcolor(-1)
        startwins = 0
        start = time.time()
        for i in range(n):
            while self.game.gamefinished() == False:
                self.p1.move()
                self.p2.move()
            if self.game.winner() == 1:
                startwins += 1
            self.game.restart()    
            
        end = time.time()
        p = startwins/n
        lower, higher = proportion.proportion_confint(startwins, n, alpha = 1-confint/100)
        
        print('P1 (starting player) won ', p*100, '% of the time.')
        print(confint, '% confidence interval: ', (lower, higher) )
        print((end-start)/n, ' s per round')
        return p, lower, higher
    
    def train(self, learnerstart = True, plot_every_test = False, test_freq = 500, n = None, update_freq = 1):
        if n is None:
            n = self.n
        import matplotlib.pyplot as plt 
        import time
        self.p1.setcolor(1)
        self.p2.setcolor(-1)
        if learnerstart:
            learnercolor = 1
            p = self.p1
        else: 
            learnercolor = -1
            p = self.p2
        probs = []
        lowers = []
        highers = []
        x_values_for_graph = []
        start = time.time()
        
        for i in range(n):
            while(self.game.gamefinished() == False):
                self.p1.move()
                self.p2.move()
            p.get_values_for_game()
            if i % update_freq == 0 or i == (n-1):
                p.update_values()  
            self.game.restart()
            if i%test_freq == 0 and i != 0:
                p.test_mode()
                prob, low, high = self.simulate(n = 100)
                p.train_mode()
                probs += [prob]
                lowers += [low]
                highers += [high]
                x_values_for_graph += [i]
                if plot_every_test:
                    plt.plot(x_values_for_graph, probs, 'b', x_values_for_graph, lowers, 'g', x_values_for_graph, highers, 'g')
                    plt.show()
        end = time.time()
        plt.plot(x_values_for_graph, probs, 'b', x_values_for_graph, lowers, 'g', x_values_for_graph, highers, 'g')
        plt.show()
        print((end-start)/n, ' s per training round')
        return 
    
    def train_both(self, test_freq = 500, plot_every_test = False, n = None, testvsrandom = False):
        if n == None:
            n = self.n
        import matplotlib.pyplot as plt 
        import time
        self.p1.setcolor(1)
        self.p2.setcolor(-1)
        probs = []
        lowers = []
        highers = []
        if testvsrandom:
            probs2 = []
            lowers2 = []
            highers2 = []
        x_values_for_graph = []
        start = time.time()
        for i in range(n):
            while(self.game.gamefinished() == False):
                self.p1.move()
                self.p2.move()
            self.p1.update_values()
            self.p2.update_values()
            self.game.restart()
            if i%test_freq == 0:
                self.p1.test_mode()
                self.p2.test_mode()
                if testvsrandom: 
                    from ipynb.fs.full.Players import RandomPlayer
                    randomplayer = RandomPlayer(self.p1.game)
                    sim1 = Simulation(self.p1, randomplayer, n = 100)
                    print('Results for testing starting player vs randomr')
                    prob1, low1, high1 = sim1.simulate(n = 100)
                    sim2 = Simulation(randomplayer, self.p2, n = 100)
                    print('Results for testing non starting player vs random')
                    prob2, low2, high2 = sim2.simulate()
                    probs2 += [prob2]
                    lowers2 += [low2]
                    highers2 += [high2]
                else:
                    prob1, low1, high1 = self.simulate(n = 100)
                self.p1.train_mode()
                self.p2.train_mode()
                probs += [prob1]
                lowers += [low1]
                highers += [high1]
                
                
                x_values_for_graph += [i]
                if plot_every_test:
                    plt.plot(x_values_for_graph, probs1, 'b', x_values_for_graph, lowers1, 'g', x_values_for_graph, highers1, 'g')
                    plt.show()
                    if testvsrandom:
                        plt.plot(x_values_for_graph, probs2, 'b', x_values_for_graph, lowers2, 'g', x_values_for_graph, highers2, 'g')
                        plt.show()    
        
        end = time.time()
        plt.plot(x_values_for_graph, probs, 'b', x_values_for_graph, lowers, 'g', x_values_for_graph, highers, 'g')
        plt.show()
        if testvsrandom:
            plt.plot(x_values_for_graph, probs2, 'b', x_values_for_graph, lowers2, 'g', x_values_for_graph, highers2, 'g')
            plt.show()    
        
        print((end-start)/n, ' s per training round')
        return 
        

        

