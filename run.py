import pygame
from pygame.locals import *

from assets.scripts.math_and_data.enviroment import *
pygame.init()

pygame.mixer.pre_init(48000, -16, 2, 4096)
pygame.mixer.init()
pygame.mixer.music.set_volume(0.1)
screen = pygame.display.set_mode(SIZE, DOUBLEBUF, 16)
clock = pygame.time.Clock()

from assets.scripts.learning import mlData
from assets.scripts.scenes.TitleScene import TitleScene
from assets.scripts.scenes.GameScene import GameScene

if not mlData.status:
    active_scene = TitleScene()
else:
    active_scene = GameScene()

ticksLastFrame = pygame.time.get_ticks()

delta_time = 1 / FPS

while active_scene is not None:
    #print(active_scene)
    active_scene.process_input(pygame.event.get())  #game select action
    active_scene.update(delta_time)                 #game take action 6
    #active_scene.render(screen, clock)              #game render changes
    active_scene = active_scene.next
    #pygame.display.flip()
    delta_time = clock.tick(FPS) / 1000
    #delta_time = 0.1
db_module.close()