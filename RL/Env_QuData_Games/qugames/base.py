import numpy as np
import pygame

class Space:
    def __init__(self, shape, low, high):        
        if type(shape) == tuple:
            self.shape = shape        
            self.n     = np.prod( shape )
            self.low   = np.full((shape), low,  dtype=np.float32)
            self.high  = np.full((shape), high, dtype=np.float32)
            self.dtype = np.float32
        else:
            self.n    = high + 1
            self.low  = low
            self.high = high

    def sample(self):
        return np.random.randint(self.low, self.high+1)


class Base:

    def reset(self):
        pass

    def key_left(self):
        pass

    def key_right(self):
        pass

    def key_up(self):
        pass

    def key_down(self):
        pass

    def key_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                if event.key == pygame.K_r:
                    return 'reset'
                elif event.key == pygame.K_LEFT:
                    self.key_left()
                elif event.key == pygame.K_RIGHT:
                    self.key_right()
                elif event.key == pygame.K_UP:
                    self.key_up()
                elif event.key == pygame.K_DOWN:
                    self.key_down()
        return ''
