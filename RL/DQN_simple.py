from   collections import deque
import random
import gym
import numpy as np
import torch
import torch.nn as nn

class DQN:
    """ DQN method for for discrete actions """
    def __init__(self, env):
        self.env     = env                         # environment we work with
        self.obs_min = env.observation_space.low   # minimum observation values
        self.obs_max = env.observation_space.high  # maximum observation values
        self.nA      =  self.env.action_space.n    # number of discrete actions
        self.nS      =  self.env.observation_space.shape[0] # number of state components

        self.params = {           # default parameters            
            'gamma'    : 0.99,      # discount factor
            'eps1'     : 1.0,       # initial value epsilon
            'eps2'     : 0.001,     # final value   epsilon
            'decays'   : 1000,      # number of episodes to decay eps1 - > eps2
            'update'   : 10,        # target model update rate (in frames = time steps)         
            'batch'    : 100,       # batch size for training
            'capacity' : 100000,    # memory size
            'hiddens'  : [256,128], # hidden layers
            'scale'    : True,      # scale or not observe to [-1...1]
            'lm'       : 0.001,     # learning rate           
        }
        
    #------------------------------------------------------------------------------------

    def get_model(self, hiddens):
        """ Create a neural network """
        neurons, layers = [self.nS] + hiddens + [self.nA], []
        for i in range(len(neurons)-1):
            layers.append(nn.Linear(neurons[i], neurons[i+1]) )
            if i < len(neurons)-2:
                layers.append( nn.ReLU() )        
        return nn.Sequential(*layers)

    #------------------------------------------------------------------------------------

    def init(self):
        """ Create a neural network and optimizer """
        self.gpu =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        self.model  = self.get_model(self.params['hiddens']).to(self.gpu)      # current Q
        self.target = self.get_model(self.params['hiddens']).to(self.gpu)      # target  Q

        self.loss      = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params['lm'], momentum=0.8)

        self.memo = deque(maxlen=self.params['capacity'])
        self.maxQ = torch.zeros( (self.params['batch'], ),  dtype=torch.float32).to(self.gpu)
        
        self.epsilon     = self.params['eps1']        # start value in epsilon greedy strategy
        self.decay_rate  = np.exp(np.log(self.params['eps2']/self.params['eps1'])/self.params['decays'])

    #------------------------------------------------------------------------------------

    def scale(self, obs):
        """ to [-1...1] """
        if self.params['scale']:
            return -1. + 2.*(obs - self.obs_min)/(self.obs_max-self.obs_min)
        else:
            return obs
    #------------------------------------------------------------------------------------

    def policy(self, state):
        """ Return action according to epsilon greedy strategy """
        if np.random.random() < self.epsilon:        
            return np.random.randint(self.nA)    # random action

        x = torch.tensor(state, dtype=torch.float32).to(self.gpu)
        with torch.no_grad():
            y = self.model(x).detach().to('cpu').numpy() 
        return np.argmax(y)                      # best action

    #------------------------------------------------------------------------------------

    def run_episode(self, ticks = 200):
        """ Run one episode, keeping the environment model in memory """
        rew = 0                                  # total reward
        s0 = self.env.reset()                    # initial state
        s0 = self.scale (s0)                     # scale it
        a0 = self.policy(s0)                     # get action
        for t in range(ticks):
            s1, r1, done, _ = self.env.step(a0)
            s1 = self.scale (s1)
            a1 = self.policy(s1)

            self.memo.append( (s0, a0, s1, r1, 1 - float(done and t < ticks)) )
            
            if self.frame % self.params['update'] == 0:
                self.target.load_state_dict( self.model.state_dict() ) 

            if len(self.memo) >= self.params['batch']:    
                self.learn_model()                         

            rew += r1
            self.frame += 1

            if done:
                break

            s0, a0 = s1, a1
        return rew

    #------------------------------------------------------------------------------------

    def learn(self, episodes = 100000, ticks = 200, stat = 100, plots = 1000, env_name = ""):
        """ Repeat episodes episodes times """
        self.frame = 1
        rews, mean   = [], 0
        for episode in range(1, episodes+1):
            rew = self.run_episode(ticks)
            rews.append( rew )

            self.epsilon *= self.decay_rate                # epsilon-decay
            if self.epsilon < self.params['eps2']:
                self.epsilon = 0.

            if  episode % stat == 0:                               
                mean, std = np.mean(rews[-stat:]), np.std(rews[-stat:])                
                print(f"{episode:6d} rew: {mean:7.2f} ± {std/stat**0.5:4.2f}")

    #------------------------------------------------------------------------------------

    def get_batch(self, count):
        """ Get batch """
        S0 = torch.empty((count, self.nS), dtype=torch.float32)
        A0 = torch.empty((count, 1),       dtype=torch.int64)
        S1 = torch.empty((count, self.nS), dtype=torch.float32)
        R1 = torch.empty((count, 1),       dtype=torch.float32)
        Dn = torch.empty((count, 1),       dtype=torch.float32)
        
        batch = random.sample(self.memo, self.params['batch']) 
        for i, (s0, a0, s1, r1, dn) in enumerate(batch):
            S0[i] = torch.tensor(s0, dtype=torch.float32)
            A0[i] = torch.tensor(a0, dtype=torch.int64)
            S1[i] = torch.tensor(s1, dtype=torch.float32) 
            R1[i] = torch.tensor(r1, dtype=torch.float32) 
            Dn[i] = torch.tensor(dn, dtype=torch.float32)

        return S0.to(self.gpu), A0.to(self.gpu), S1.to(self.gpu), \
               R1.to(self.gpu), Dn.to(self.gpu)

    #------------------------------------------------------------------------------------

    def learn_model(self):
        """ Model Training """
        s0, a0, s1, r1, dn = self.get_batch(self.params['batch'])
        with torch.no_grad():
            y = self.target(s1).detach()
        self.maxQ, _ = torch.max(y, 1)      # maximum Q values for S1

        q1     = self.maxQ.view(-1,1)
        yb = r1 + self.params['gamma']*q1*dn

        y = self.model(s0)             # forward
        y = y.gather(1, a0)
        L = self.loss(y, yb)

        self.optimizer.zero_grad()     # reset the gradients
        L.backward()                   # calculate gradients
        self.optimizer.step()          # adjusting parameters
        
env_name = "MountainCar-v0"
env = gym.make(env_name)

dqn = DQN( env )

dqn.params = {
    'method'   : "DQN",     # kind of the method (DQN, DDQN)     
    'gamma'    : 0.99,      # discount factor
    'eps1'     : 1.0,       # initial value epsilon
    'eps2'     : 0.001,     # final value   epsilon
    'decays'   : 500,       # number of episodes to decay eps1 - > eps2
    'update'   : 100,       # target model update rate (in frames = time steps)             
    'batch'    : 100,       # batch size for training
    'capacity' : 100000,    # memory size
    'hiddens'  : [256,128], # hidden layers
    'scale'    : True,      # scale or not observe to [-1...1]
    'optimizer': 'sgd',     # optimizer (sgd, adam)
    'lm'       : 0.001,     # learning rate           
}

dqn.init()

print(dqn.params)
dqn.learn(episodes = 5000, ticks = 200, stat=100, plots = 1000, env_name = env_name)