import pygame
from pygame.locals import *

from assets.scripts.math_and_data.enviroment import *
pygame.init()

pygame.mixer.pre_init(48000, -16, 2, 4096)
pygame.mixer.init()
pygame.mixer.music.set_volume(0.1)
screen = pygame.display.set_mode(SIZE, DOUBLEBUF, 16)
clock = pygame.time.Clock()

from assets.scripts.learning.rlAgent import RLProcess, QNetwork
from assets.scripts.scenes.TitleScene import TitleScene
active_scene = TitleScene()

ticksLastFrame = pygame.time.get_ticks()

delta_time = 1 / FPS

qlnn = QNetwork(11) #size of hidden layer, whatever that means
rlCode = RLProcess(active_scene,qlnn)

while active_scene is not None:
    #print(active_scene)
    active_scene.process_input(pygame.event.get())  #game select action
    active_scene.update(delta_time)                 #game take action
    rlCode.reviewAction()                           #reinforcement learning voodoo
    active_scene.render(screen, clock)              #game render changes
    active_scene = active_scene.next
    rlCode.updateState(active_scene)           #reinforcement learning state = new state stuff, also updates game scenes
    #since gamescene is always rendered after something else, state is initialised here, right after the scene change instead of in front of the loop,
    #as the scene changes, the agent class does __init__() to initialise state automatically, probably
    pygame.display.flip()
    delta_time = clock.tick(FPS) / 1000

db_module.close()