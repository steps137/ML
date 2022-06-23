import os
from   game import Game
import numpy as np
import pygame
from   collections import deque

class Snake(Game):
    def __init__(self, dummy = True):
        self.grid  = [16, 16]
        self.cell  = [32, 32]

        self.color_field = (255, 255, 255)
        self.color_food  = (  0, 255,   0)
        self.color_snake = (  0,  0,  255)

        self.dummy = dummy
        if dummy:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.init()
        self.reset()

    #------------------------------------------------------------------------------------

    def init(self):
        """ Init pygame """
        pygame.init()
        self.screen = pygame.display.set_mode( ( self.grid[0]*self.cell[0]+1, 
                                                 self.grid[1]*self.cell[1]+1) )
        self.clock = pygame.time.Clock()        

    #------------------------------------------------------------------------------------

    def render(self):
        """ Show render window """
        if self.dummy:
            os.environ["SDL_VIDEODRIVER"] = None
            self.dummy = False
            self.init()
            self.plot()

    #------------------------------------------------------------------------------------

    def close(self):
        """ Close render window """
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.dummy = True
        self.init()
        self.plot()

    def pause(self, t):
        self.clock.tick(t)            

    #------------------------------------------------------------------------------------

    def reset(self):
        self.gaming = True
        self.tick  = 0
        self.pos   = np.array([0, self.grid[1] // 2])
        self.vel   = np.array([1, 0])
        self.snake = deque()
        self.snake.append(self.pos.copy())        
        self.food  = np.array([0,0])
        self.put_food()
        self.message()
        self.plot()
        return pygame.surfarray.array3d(self.screen)

    #------------------------------------------------------------------------------------

    def step(self, a):
        if a == 0:
            pass
        elif a== 1: self.key_left()
        elif a== 2: self.key_right()
        elif a== 3: self.key_up()
        elif a== 4: self.key_down()

        self.gaming = self.logic()
        self.plot()        
        self.tick += 1
        score = -100*(1-int(self.gaming)) - self.tick + len(self.snake)-1
        return pygame.surfarray.array3d(self.screen), score, 1-self.gaming, {}

    #------------------------------------------------------------------------------------

    def equal(self, p1, p2):
        return (p1==p2).sum() == 2

    #------------------------------------------------------------------------------------

    def plot(self):
            self.screen.fill( self.color_field )
            pygame.draw.rect(self.screen, (0,0,0), (0, 0, self.grid[0]*self.cell[0], self.grid[1]*self.cell[1] ), 2)
            for i,s in enumerate(self.snake):       
                color = self.color_snake
                if not self.gaming and i==0:
                    color = (255,0,0)
                pygame.draw.rect(self.screen,  color, 
                                [s[0]*self.cell[0]+1, s[1]*self.cell[1]+1 ] 
                              + [self.cell[0]-1,    self.cell[1]-1]  )
            pygame.draw.rect(self.screen,  self.color_food, 
                                 [self.food[0]*self.cell[0]+1, self.food[1]*self.cell[1]+1] 
                               + [self.cell[0]-1,            self.cell[1]-1])                        
            pygame.display.update()

    #------------------------------------------------------------------------------------

    def put_food(self):
        for i in range(1,1001):
            self.food[0] = np.random.randint(0, self.grid[0])
            self.food[1] = np.random.randint(0, self.grid[1])
            ok = True
            for s in self.snake:
                if  self.equal(s, self.food):
                    ok = False
                    break
            if ok:
                break

    #------------------------------------------------------------------------------------

    def reflect(self):
        for i in range(2):
            if self.pos[i] < 0:
                self.pos[i] = self.grid[i]
            elif self.pos[i] >= self.grid[i]:
                self.pos[i] = 0

    #------------------------------------------------------------------------------------

    def cross(self):
        for i, s in enumerate(self.snake):
            if i > 0 and self.equal(s, self.pos):
                return True
        return False

    #------------------------------------------------------------------------------------

    def out_field(self):
        return self.pos[0] < 0 or self.pos[0] >= self.grid[0] \
            or self.pos[1] < 0 or self.pos[1] >= self.grid[1]

    #------------------------------------------------------------------------------------

    def message(self):
        pygame.display.set_caption(f'Score: {len(self.snake)-1}  Esc - for exit, r - reset')            

    #------------------------------------------------------------------------------------

    def logic(self):
        self.pos += self.vel        

        if self.out_field() or self.cross():            
            return False

        if self.equal(self.pos, self.food):
            self.snake.appendleft(self.food.copy())
            self.put_food()
            self.message()            
        else:
            self.snake.pop()
            self.snake.appendleft(self.pos.copy())        

        return True

    #------------------------------------------------------------------------------------

    def key_left(self):
        self.vel[0] = -1;  self.vel[1] = 0

    def key_right(self):
        self.vel[0] =  1;  self.vel[1] = 0

    def key_up(self):
        self.vel[0] = 0; self.vel[1] = -1

    def key_down(self):
        self.vel[0] = 0; self.vel[1] =  1

    #------------------------------------------------------------------------------------

    def run(self):                        
        while True:            
            event = self.key_event()            
            if  event == 'quit':
                break
            elif event == 'reset':                
                self.reset()

            self.heuristic()

            if self.gaming:
                self.gaming = self.logic()            

            self.plot()
            self.clock.tick(50)            
        pygame.quit()

    #------------------------------------------------------------------------------------
    def can(self, v):        
        p = self.pos + np.array(v)
        for s in self.snake:
            if self.equal(s,p): 
                return False
        if p[0] < 0 or p[0] >= self.grid[0] or p[1] < 0 or p[1] >= self.grid[1]:
                return False
        return True

    def heuristic(self):
        if   self.pos[0] < self.food[0] and self.vel[0] != -1 and self.can([ 1, 0]):  self.key_right()
        elif self.pos[0] > self.food[0] and self.vel[0] !=  1 and self.can([-1, 0]):  self.key_left()
        elif self.pos[1] < self.food[1] and self.vel[1] != -1 and self.can([ 0, 1]):  self.key_down()
        elif self.pos[1] > self.food[1] and self.vel[1] !=  1 and self.can([ 0,-1]):  self.key_up()
        elif self.pos[0] + self.vel[0] >= self.grid[0]        and self.can([ 0,-1]):  self.key_up()
        elif self.pos[0] + self.vel[0] <  0                   and self.can([ 0,-1]):  self.key_up()
        elif self.pos[1] + self.vel[1] >= self.grid[1]        and self.can([-1,0]):  self.key_left()
        elif self.pos[1] + self.vel[1] <  0                   and self.can([-1,0]):  self.key_left()        
        elif not self.can(self.vel):
            if   self.vel[0] == 0: 
                if   self.can([-1, 0]):  self.key_left() 
                elif self.can([ 1, 0]):  self.key_right() 
            elif self.vel[1] == 0: 
                if   self.can([ 0, -1]):  self.key_up() 
                elif self.can([ 0,  1]):  self.key_down() 


env = Snake(False)
env.run()
"""
for _ in range(100):
    s = env.reset()
    for _ in range(10000):
        a = np.random.randint(5)
        s, r, done, _ = env.step(a)
        print(r, done)
        env.render()
        env.pause(10)
        if done:
            break
"""
    
 
