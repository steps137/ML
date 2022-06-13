import math
import copy
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class MemoryBuffer:
    """ Затираемая память для хранения модели среды """
    def __init__(self, capacity, state_shape):
        """
        capacity    - ёмкость памяти (максимальное число сохраняемых элементов)
        state_shape - форма тензора состояния: (num_states, ) и т.п.
        """
        self.capacity = capacity
        self.count = 0              # сколько элементов уже сохранили

        self.S0 = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.A0 = torch.zeros( capacity,                dtype=torch.int64)
        self.S1 = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.R1 = torch.zeros( capacity,                dtype=torch.float32)
        self.Dn = torch.zeros( capacity,                dtype=torch.float32)

    def add(self, s0, a0, s1, r1, done):
        """ Добавить в пямять элементы """
        index = self.count % self.capacity
        self.count += 1

        self.S0[index] = torch.tensor(s0, dtype=torch.float32)
        self.A0[index] = a0
        self.S1[index] = torch.tensor(s1, dtype=torch.float32)
        self.R1[index] = r1
        self.Dn[index] = 1. - done

    def samples(self, count):
        """ Вернуть count случайных примеров из памяти """
        mem_max = min(self.count, self.capacity)
        indxs = np.random.choice(mem_max, count, replace=False)

        return self.S0[indxs],self.A0[indxs],self.S1[indxs],self.R1[indxs],self.Dn[indxs]

#========================================================================================

class DQN:
    """ DQN метод для дискретных действий """
    def __init__(self, env):
        self.env     = env                         # среда с которой мы работаем
        self.obs_min = env.observation_space.low   # минимальные значения наблюдений
        self.obs_max = env.observation_space.high  # максимальные значения наблюдений
        self.nA      =  self.env.action_space.n    # число дискретных действий
        self.shape_S =  self.env.observation_space.shape # форма тензора состояния

        self.params = {
            'gamma'   : 0.99,      # дисконтирующий множитель
            'eps1'    : 1.0,       # начальное значение epsilon
            'eps2'    : 0.001,     # конечное значение epsilon
            'decays'  : 5000,      # число эпизодов для распада eps1 - > eps2
            'update'  : 10000,     # частота обновления модели для max Q     
            'epochs'  : 1,         # число эпох обучения (после обновление сети)
            'batches' : 100,       # количество батчей для обучения
            'batch'   : 100,       # размер батча для обучения
            'capacity': 100000,    # величина памяти
            'hiddens' : [128, 32], # скрытые слои
            'lm'      : 0.001,     # скорость обучения            
        }
        self.last_loss = 0.        # последняя ошибка
        self.history = []

        print("obs_min:   ", self.obs_min)
        print("obs_max:   ", self.obs_max)
        print("obs_shape: ", self.obs_max)

        self.old_model = None
    #------------------------------------------------------------------------------------

    def init(self, model = None):
        """ Сформировать нейронную сеть c nH нейронами в скрытом слое """
        if model:
            self.model = model
        else:
            nH = self.params['hiddens']
            self.model = nn.Sequential(
                nn.Linear(self.shape_S[0], nH[0]),
                nn.ReLU(),
                nn.Linear(nH[0], nH[1]),
                nn.ReLU(),
                nn.Linear(nH[1], self.nA)
                )

        self.gpu =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", self.gpu)
        self.model.to(self.gpu)

        self.loss      = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params['lm'], momentum=0.8)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lm'])

        self.memo = MemoryBuffer(self.params['capacity'], self.shape_S)
        self.maxQ = torch.zeros( (self.params['batch']*self.params['batches'], ),  dtype=torch.float32).to(self.gpu)
        self.best_rew = -100000
        self.epsilon     = self.params['eps1']        # эпсилон-жадная стратегия
        self.decay_rate  = math.exp(math.log(self.params['eps2']/self.params['eps1'])/self.params['decays'])

        print(f"decay_rate: {self.decay_rate:.4f}")
    #------------------------------------------------------------------------------------

    def scale(self, obs):
        return -1. + 2.*(obs - self.obs_min)/(self.obs_max-self.obs_min)
        #return obs
    #------------------------------------------------------------------------------------

    def policy(self, state):
        """
        Вернуть action в соответствии с epsilon-жадной стратегией
        """
        if np.random.random() < self.epsilon:
            #return 2*int(state[1] > 0)
            return np.random.randint(self.nA)    # случайное действие

        x = torch.tensor(state, dtype=torch.float32).to(self.gpu)
        with torch.no_grad():
            y = self.model(x).detach().to('cpu').numpy()   # значения Q для всех действий
        return np.argmax(y)                      # лучшее действие

    #------------------------------------------------------------------------------------

    def run_episode(self, ticks = 200):
        """ Запускаем один эпизод, сохраняя модель среды в памяти """
        rew = 0                                  # суммарное вознаграждение
        s0 = self.env.reset()                    # начальное состояние
        s0 = self.scale (s0)                     # масштабируем его
        a0 = self.policy(s0)                     # получаем действие
        for t in range(ticks):
            s1, r1, done, _ = self.env.step(a0)
            s1 = self.scale (s1)
            a1 = self.policy(s1)

            self.memo.add(s0, a0, s1, r1, float(done and t < ticks) )
            rew += r1

            if done:
                break

            s0, a0 = s1, a1
        return rew

    #------------------------------------------------------------------------------------

    def learn(self, episodes = 100000, ticks = 200):
        """
        Повторяем эпизоды episodes раз
        """
        rews, beg   = [],  time.process_time()
        for episode in range(1, episodes+1):
            rews.append( self.run_episode(ticks) )

            if self.memo.count >= self.params['batch']*self.params['batches']:    
                self.learn_model()                         # буфер памяти набрали обучаем модель

            self.epsilon *= self.decay_rate                # epsilon-распад
            if self.epsilon < self.params['eps2']:
                self.epsilon = 0.

            if  episode % 100 == 0:
                rews = rews[-100:]
                mean, std = np.mean(rews), np.std(rews)
                self.history.append([episode, mean])
                if mean > self.best_rew:
                    self.best_rew = mean
                maxQ = self.maxQ.to('cpu')
                print(f"{episode:6d} rew: {mean:7.2f} ± {std/len(rews)**0.5:4.2f},  best: {self.best_rew:7.2f},  epsilon: {self.epsilon:.3f},  Q:{maxQ.mean():8.2f} ± {maxQ.std():7.3f}, loss:{self.last_loss:7.3f}, time:{(time.process_time() - beg):3.0f}s")
                beg = time.process_time()

            if episode % self.params['update'] == 0:
                self.old_model = copy.deepcopy(self.model)

    #------------------------------------------------------------------------------------

    def learn_model(self):
        """
        Обучение модели
        """
        batch, batches = self.params['batch'], self.params['batches']

        for _ in range(self.params['epochs']): 
            S0, A0, S1, R1, Dn = self.memo.samples(batch * batches)
            S0 = S0.to(self.gpu); A0 = A0.to(self.gpu)
            S1 = S1.to(self.gpu); R1 = R1.to(self.gpu);  Dn = Dn.to(self.gpu)

            if self.old_model != None:
                with torch.no_grad():
                    y = self.old_model(S1).detach()
                self.maxQ, _ = torch.max(y, 1)      # максимальные значения Q для S1

            sum_loss = 0
            for i in range(0, batches*batch, batch):
                s0, a0 = S0[i:i+batch], A0[i:i+batch].view(-1,1)
                r1, dn = R1[i:i+batch].view(-1,1), Dn[i:i+batch].view(-1,1)
                q1     = self.maxQ[i:i+batch].view(-1,1)

                yb = r1 + self.params['gamma']*q1*dn

                y = self.model(s0)              # прямое распространение
                y = y.gather(1, a0)
                L = self.loss(y, yb)

                self.optimizer.zero_grad()     # обнуляем градиенты
                L.backward()                   # вычисляем градиенты
                self.optimizer.step()          # подправляем параметры

                sum_loss += L.detach().item()
            self.last_loss = sum_loss/batches

    #------------------------------------------------------------------------------------

    def test(self, episodes = 1000, ticks = 1000):
        """
        Тестирование с неизменной Q-функцией
        """
        rews = []
        for _ in range(episodes):
            tot = 0
            obs =  self.env.reset()
            for _ in range(ticks):
                action = self.policy( self.scale(obs) )
                obs, rew, done, _ = self.env.step(action)
                tot += rew
                if done:
                    break
            rews.append(tot)

        print(f"Reward[{episodes},{ticks}]: {np.mean(rews):7.3f} ± {np.std(rews)/len(rews)**0.5:.3f}")

#========================================================================================

env = gym.make("MountainCar-v0")
#env = gym.make("CartPole-v1")
dqn = DQN( env )
"""
# dqn.obs_min = np.array([-4.8, -5, -0.418, -5])
# dqn.obs_max = np.array([ 4.8,  5,  0.418,  5])
"""

dqn.params = {
    'gamma'   : 0.99,      # дисконтирующий множитель
    'eps1'    : 1.0,       # начальное значение epsilon
    'eps2'    : 0.001,     # конечное значение epsilon
    'decays'  : 20000,     # число эпизодов для распада eps1 - > eps2
    'update'  : 10000,     # частота обновления модели для max Q     
    'epochs'  : 1,         # число эпох обучения (после обновление сети)
    'batches' : 200,       # количество батчей для обучения
    'batch'   : 50,        # размер батча для обучения
    'capacity': 100000,    # величина памяти
    'hiddens' : [128, 64], # скрытые слои
    'lm'      : 0.001       # скорость обучени    
}

dqn.init()
print(dqn.params)
dqn.learn(episodes = 100000, ticks = 200)
dqn.test (episodes = 1000)

#========================================================================================

history = np.array(dqn.history)
params = [ f"{k:8s}: {v}\n" for k,v in dqn.params.items()]
plt.figure(figsize=(12,8))
plt.title(f"MountainCar-v0 best: {dqn.best_rew:7.1f}", fontsize=20)
plt.plot(history[:,0], history[:,1])
plt.ylim(-200, -100);
plt.text(-1000, -200, "".join(params), {'fontsize':12, 'fontname':'monospace'})
plt.xlabel('episode',  fontsize=16); 
plt.ylabel('reward',   fontsize=16) 
plt.show()