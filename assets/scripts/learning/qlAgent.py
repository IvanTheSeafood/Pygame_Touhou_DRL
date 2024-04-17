from assets.scripts.classes.game_logic.Collider import Collider
from assets.scripts.math_and_data.Vector2 import Vector2
from assets.scripts.learning import mlData
#from assets.scripts.classes.game_logic.Player import Player
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

class agent:
    def __init__(self,player):
        self.switch = mlData.status
        self.player = player

        self.state = agentState(player)#player pos,enemy pos, bullet pos, 
        self.newState = agentState(player)
        self.action = [0, False] # 9 possible actions
        self.reward = 0
        self.terminal = False

        self.episode = mlData.episode
        self.q = mlData.Q
        self.ring = None
    
    def moveDirection(self, action=[4, False]):
 
        #Moving
        finalMove =[Vector2.up() + Vector2.left(),   Vector2.up(),   Vector2.up() + Vector2.right(),
                    Vector2.left(),                  Vector2.zero(), Vector2.right(),
                    Vector2.down() + Vector2.left(), Vector2.down(), Vector2.down() + Vector2.right()] 
        
        #Shooting
        if action[1] == True:
            self.player.shoot()
            
        return finalMove[action[0]]

class agentState:
    def __init__(self, player):
        #self.playerPos=player.position
        self.playerCoord = player.position.coords

        #self.enemyPos = []
        self.enemyCoord = [mlData.emptyCoord]*mlData.maxEnemies

        #self.bulletPos = []
        self.bulletCoord = [mlData.emptyCoord]*mlData.maxBullets

    def appendBullet(self, agent, bullet, i):
        hitbox= Collider(10,bullet.position)
        if agent.ring.check_collision(hitbox):
            if i<mlData.maxBullets:
                self.bulletCoord[i]= bullet.position.coords
            else:
                cellMax = 0
                cellPos = 50
                for j in range(50):
                    cellValue = self.checkDistance(self.bulletCoord[j])
                    if cellValue > cellMax:
                        cellMax = cellValue
                        cellPos = j
                if self.checkDistance(bullet.position.coords):
                    self.bulletCoord[cellPos] = bullet.position.coords

    def appendEnemy(self, enemy, i):
        if i < mlData.maxEnemies:
            self.enemyCoord[i] = enemy.position.coords
        else:
            print("max enemies on scree RN =", i)

    def checkDistance(self, a):
        b = self.playerCoord
        return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)

class QNetwork(nn.Module):

    def __init__(self, hidden_size, state_size= mlData.maxBullets + mlData.maxEnemies + 1, action_size=9):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Define layers
        self.fc1 = nn.Linear(state_size + action_size + state_size + 1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action, new_state, reward):
        # Concatenate inputs
        x = torch.cat((state, action, new_state, reward), dim=1)
        
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x