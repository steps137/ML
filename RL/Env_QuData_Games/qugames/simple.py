import pygame
import random
from   collections import deque

class Simple:
    def __init__(self) -> None:
        pygame.init()
        self.pos   = [20, 20]
        self.screen = pygame.display.set_mode( ( 800, 600) )
        self.clock = pygame.time.Clock()
        self.blue = (0, 0, 255)


    def plot(self):
            self.screen.fill( (255,255,255) )
            pygame.draw.rect(self.screen,  self.blue, self.pos + [50,50])
            #pygame.draw.circle(self.screen, self.blue, self.pos, 75)
            #pygame.display.flip()
            pygame.display.update()

    def logic(self):
        self.pos[0] += 1
        return True

    def event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_LEFT:
                    self.pos[0] -= 10                    
                elif event.key == pygame.K_RIGHT:
                    self.pos[0] += 10                                        
                elif event.key == pygame.K_UP:
                    self.pos[1] -= 10                                        
                elif event.key == pygame.K_DOWN:
                    self.pos[1] += 10                                        
        return True


    def run(self):                
        while True:            
            self.plot()
            if not self.event():
                break
            if not self.logic():
                break
            self.clock.tick(10)            
        pygame.quit()

game = Simple()
game.run()
    
 

