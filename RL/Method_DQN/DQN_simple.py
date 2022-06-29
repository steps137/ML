from   collections import deque
import random
import gym
import numpy as np
import torch
import torch.nn as nn

class Memory:
    def __init__(self, capacity, nS):
        self.capacity = capacity  # memory capacity (number of examples)
        self.count    = 0         # number of examples added
        self.S0 = torch.empty( (capacity, nS), dtype=torch.float32)
        self.S1 = torch.empty( (capacity, nS), dtype=torch.float32)
        self.A0 = torch.empty( (capacity, 1),  dtype=torch.int64)        
        self.R1 = torch.empty( (capacity, 1),  dtype=torch.float32)
        self.Dn = torch.empty( (capacity, 1),  dtype=torch.float32)

    def add(self, s0, a0, s1, r1, done):
        """ Add to memory (s0,a0,s1,r1) """
        idx = self.count % self.capacity
        self.S0[idx] = torch.tensor(s0, dtype=torch.float32)
        self.S1[idx] = torch.tensor(s1, dtype=torch.float32)
        self.A0[idx] = a0;  self.R1[idx] = r1; self.Dn[idx] = done
        self.count += 1

    def get(self, count):
        """ Return count of examples for (s0,a0,s1,r1) """        
        high = min(self.count, self.capacity)
        num  = min(count, high)
        ids = torch.randint(high = high, size = (num,) )
        return self.S0[ids], self.A0[ids], self.S1[ids], self.R1[ids], self.Dn[ids]

class DQN:
    """ DQN method for for discrete actions """
    def __init__(self, env):
        self.env  = env                         # environment we work with
        self.low  = env.observation_space.low   # minimum observation values
        self.high = env.observation_space.high  # maximum observation values
        self.nA   = self.env.action_space.n     # number of discrete actions
        self.nS   = self.env.observation_space.shape[0] # number of state components

        self.params = {           # default parameters        
            'ticks'    : 200,       # length of episode
            'timeout'  : True,      # whether to consider reaching ticks as a terminal state
            'gamma'    : 0.99,      # discount factor
            'eps1'     : 1.0,       # initial value epsilon
            'eps2'     : 0.001,     # final value   epsilon
            'decays'   : 500,       # number of episodes to decay eps1 - > eps2
            'update'   : 100,       # target model update rate (in frames = time steps)             
            'batch'    : 100,       # batch size for training
            'capacity' : 100000,    # memory size
            'hiddens'  : [256,128], # hidden layers
            'scale'    : True,      # scale or not observe to [-1...1]            
            'lm'       : 0.001,     # learning rate           
        }
        
    #------------------------------------------------------------------------------------

    def create_model(self, sizes, hidden=nn.ReLU, output=nn.Identity):
        """ Create a neural network """
        layers = []        
        for i in range(len(sizes)-1):            
            activation = hidden if i < len(sizes)-2 else output
            layers += [ nn.Linear(sizes[i], sizes[i+1]), activation() ]        
        return nn.Sequential(*layers)

    #------------------------------------------------------------------------------------

    def init(self):
        """ Create a neural network and optimizer """
        self.device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        sizes = [self.nS] + self.params['hiddens'] + [self.nA]
        self.model  = self.create_model(sizes).to(self.device)  # current Q
        self.target = self.create_model(sizes).to(self.device)  # target  Q

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lm'])

        self.memo = Memory(self.params['capacity'], self.nS)
        self.maxQ = torch.zeros( (self.params['batch'], ),  dtype=torch.float32).to(self.device)
        
        self.epsilon    = self.params['eps1']        # start value in epsilon greedy strategy
        self.decay_rate = np.exp(np.log(self.params['eps2']/self.params['eps1'])/self.params['decays'])

    #------------------------------------------------------------------------------------

    def scale(self, obs):
        """ to [-1...1] """
        if self.params['scale']:
            return -1. + 2.*(obs - self.low)/(self.high-self.low)
        else:
            return obs

    #------------------------------------------------------------------------------------

    def policy(self, state):
        """ Return action according to epsilon greedy strategy """
        if np.random.random() < self.epsilon:        
            return np.random.randint(self.nA)    # random action

        x = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y = self.model(x).detach().cpu().numpy() 
        return np.argmax(y)                      # best action

    #------------------------------------------------------------------------------------

    def run_episode(self, ticks):
        """ Run one episode, keeping the environment model in memory """
        rew = 0                                  # total reward
        s0 = self.env.reset()                    # initial state
        s0 = self.scale (s0)                     # scale it
        a0 = self.policy(s0)                     # get action
        for t in range(1, ticks+1):
            s1, r1, done, _ = self.env.step(a0)
            s1 = self.scale (s1)
            a1 = self.policy(s1)

            dn = done and (self.params['timeout'] or t < ticks)
            self.memo.add( s0, a0, s1, r1, float(dn) )  
            
            if self.frame % self.params['update'] == 0:
                self.target.load_state_dict( self.model.state_dict() ) 

            if self.memo.count >= self.params['batch']:    
                self.learn_model()                         

            rew += r1
            self.frame += 1

            if done:
                break

            s0, a0 = s1, a1
        return rew

    #------------------------------------------------------------------------------------

    def learn(self, episodes = 100000, stat = 100):
        """ Repeat episodes episodes times """
        self.frame = 1
        rews, mean   = [], 0
        for episode in range(1, episodes+1):
            rew = self.run_episode( self.params['ticks'] )
            rews.append( rew )

            self.epsilon *= self.decay_rate                # epsilon-decay
            if self.epsilon < self.params['eps2']:
                self.epsilon = 0.

            if  episode % stat == 0:                               
                mean, std = np.mean(rews[-stat:]), np.std(rews[-stat:])                
                print(f"{episode:6d} rew: {mean:7.2f} ± {std/stat**0.5:4.2f}")

    #------------------------------------------------------------------------------------

    def learn_model(self):
        """ Model Training """
        s0, a0, s1, r1, dn = self.memo.get(self.params['batch'])
        s0 = s0.to(self.device); s1 = s1.to(self.device); a0 = a0.to(self.device)
        r1 = r1.to(self.device); dn = dn.to(self.device)

        with torch.no_grad():
            y = self.target(s1).detach()
        self.maxQ, _ = torch.max(y, 1)      # maximum Q values for S1

        q1 = self.maxQ.view(-1,1)
        yb = r1 + self.params['gamma']*q1*(1.-dn)

        y = self.model(s0)             # forward
        y = y.gather(1, a0)
        L = self.loss(y, yb)

        self.optimizer.zero_grad()     # reset the gradients
        L.backward()                   # calculate gradients
        self.optimizer.step()          # adjusting parameters
        
"""
#========================================================================================

env_name = "MountainCar-v0"
env = gym.make(env_name)

dqn = DQN( env )

dqn.params = {       
    'ticks'    : 200,       # length of episode
    'timeout'  : True,      # whether to consider reaching ticks as a terminal state
    'gamma'    : 0.99,      # discount factor
    'eps1'     : 1.0,       # initial value epsilon
    'eps2'     : 0.001,     # final value   epsilon
    'decays'   : 500,       # number of episodes to decay eps1 - > eps2
    'update'   : 100,       # target model update rate (in frames = time steps)             
    'batch'    : 100,       # batch size for training
    'capacity' : 100000,    # memory size
    'hiddens'  : [256,128], # hidden layers
    'scale'    : True,      # scale or not observe to [-1...1]
    'lm'       : 0.001,     # learning rate           
}

dqn.init()
print(dqn.params)
dqn.learn(episodes = 1000)
"""
#========================================================================================
"""
env_name = "CartPole-v0"
env = gym.make(env_name)

dqn = DQN( env )

dqn.params = {       
    'ticks'    : 200,       # length of episode
    'timeout'  : False,     # whether to consider reaching ticks as a terminal state
    'gamma'    : 0.99,      # discount factor
    'eps1'     : 1.0,       # initial value epsilon
    'eps2'     : 0.001,     # final value   epsilon
    'decays'   : 500,       # number of episodes to decay eps1 - > eps2
    'update'   : 100,       # target model update rate (in frames = time steps)             
    'batch'    : 100,       # batch size for training
    'capacity' : 1000,      # memory size
    'hiddens'  : [64,32],   # hidden layers
    'scale'    : False,     # scale or not observe to [-1...1]    
    'lm'       : 0.001,     # learning rate           
}

dqn.init()
print(dqn.params)
dqn.learn(episodes = 1000)
"""

env_name = "LunarLander-v2" # (nS=8, nA=4)
dqn = DQN( gym.make(env_name) )

dqn.params = {    
    'ticks'    : 500,
    'timeout'  : True,      # whether to consider reaching ticks as a terminal state
    'method'   : "DQN",     # kind of the method (DQN, DDQN)     
    'gamma'    : 0.99,      # discount factor
    'eps1'     : 1.0,       # initial value epsilon
    'eps2'     : 0.001,     # final value   epsilon
    'decays'   : 1000,      # number of episodes to decay eps1 - > eps2
    'update'   : 1000,      # target model update rate (in frames = time steps)             
    'batch'    : 100,       # batch size for training
    'capacity' : 100000,    # memory size
    'rewrite'  : 1,         # rewrite memory (if < 1 - sorted)
    'hiddens'  : [256,64],  # hidden layers
    'scale'    : False,     # scale or not observe to [-1...1]
    'lm'       : 0.0001,     # learning rate           
}

dqn.init()
print(dqn.params)
dqn.learn(episodes = 2000)