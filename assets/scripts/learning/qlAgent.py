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
        self.action = [0, True] # 9 possible actions
        self.reward = 0
        self.terminal = False

        self.episode = mlData.episode
        self.q = mlData.Q
        self.ring = None

    def qArr(self, qNN):
        q=[]
        for i in range(9):
            #r = self.returnR()
            q.append(qNN(self.state, i, self.newState, 1))
        return q
    
    def chooseAction(self, qNN):
        
        q = self.qArr(qNN)
        bestActions=[n for n, v in enumerate(q) if v ==max(q)]    

        if np.random.rand()< mlData.epsilon:                                                             
            return np.random.randint(0,8)
        elif len(bestActions)>1:                                                                
            return np.random.choice(bestActions)
        else:                                                                                   
            return np.argmax(q)
        
    def returnR(self):

        data=mlData
        r = 1   #survive

        if data.oldHp < data.hp:    #dead
            r-=100

        r += (data.points-data.oldPoints)/100   #points earned
        
        return r

    def moveDirection(self):
 
        #Moving
        finalMove =[Vector2.up() + Vector2.left(),   Vector2.up(),   Vector2.up() + Vector2.right(),
                    Vector2.left(),                  Vector2.zero(), Vector2.right(),
                    Vector2.down() + Vector2.left(), Vector2.down(), Vector2.down() + Vector2.right()] 
        
        #Shooting
        if self.action[1] == True:
            self.player.shoot()
            
        return finalMove[self.action[0]]

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
        # Adjust input size to match concatenated input
        self.fc1 = nn.Linear(mlData.maxBullets + mlData.maxEnemies + 1 , hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, stateClass, action, newStateClass, reward):
        # Concatenate inputs
        state = self.initState(stateClass)
        new_state = self.initState(newStateClass)
        # Concatenate action and reward (assuming action is a tensor)
        action_reward = torch.cat((action, reward.unsqueeze(1)), dim=1)
        x = torch.cat((state, action_reward, new_state), dim=1)  # Concatenate along dim=1
        
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def initState(self,stateClass):
        state =[stateClass.playerCoord]
        for i in range(mlData.maxEnemies):
            state.append(stateClass.enemyCoord[i])
        for i in range(mlData.maxBullets):
            state.append(stateClass.bulletCoord[i])
        print(state)
        return state