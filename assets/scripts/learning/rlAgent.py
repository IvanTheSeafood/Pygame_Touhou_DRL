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

        self.state = agentState(player) #player pos, enemy pos, bullet pos, 
        self.newState = agentState(player)
        self.action = [0, False] # 9 possible actions
        self.reward = 0
        self.terminal = False

        self.episode = mlData.episode
        self.q = mlData.Q
        self.ring = None
    
    def takeAction(self):
        self.action = [np.random.randint(0,8),True]
        return self.moveDirection(self.action)
    
    def moveDirection(self, action=[4, False]):
 
        #Moving
        finalMove =[Vector2.up() + Vector2.left(),   Vector2.up(),   Vector2.up() + Vector2.right(),
                    Vector2.left(),                  Vector2.zero(), Vector2.right(),
                    Vector2.down() + Vector2.left(), Vector2.down(), Vector2.down() + Vector2.right()] 
        
        #Shooting
        if action[1] == True:
            self.player.shoot()
            
        return finalMove[action[0]]
    
    def returnR(self,data=mlData):
        self.reward = 0
        if data.hp < data.oldHp:
            data.oldHp = data.hp
            self.reward -=10
        self.reward += (data.points-data.oldPoints)/100 +1
        return self.reward

class agentState:
    def __init__(self, player):
        #self.playerPos=player.position
        self.playerCoord = player.position.coords

        #self.enemyPos = []
        self.enemyCoord = [mlData.emptyCoord]*mlData.maxEnemies

        #self.bulletPos = []
        self.bulletCoord = [mlData.emptyCoord]*mlData.maxBullets

    def updateBullet(self, agent, bullet, i):
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

    def updateEnemy(self, enemy, i):
        if i < mlData.maxEnemies:
            self.enemyCoord[i] = enemy.position.coords
        else:
            print("max enemies on scree RN =", i)

    def checkDistance(self, a):
        b = self.playerCoord
        return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)

class RLProcess:
    def __init__(self,scene,rlnn):
        self.scene=scene
        self.rlnn = rlnn

    def checkScene(self):
        if self.scene.agent is not None and mlData.status == True:
            return True
        else: 
            return False
            
    def reviewAction(self, action = None):
            
        if self.checkScene() == True:
            if action is None:
                action = self.scene.agent.action[0]
        
            #self.scene.agent.r=self.scene.agent.returnR()
            #predictQ = self.rlnn(self.scene.agent.state, action,self.scene.agent.newState, self.scene.agent.reward)
            #targetQ = self.rlnn(self.scene.agent.state, action,self.scene.agent.newState, 5)
            #loss = nn.MSELoss()(predictQ,targetQ)
            #loss.backward()
            
    def updateState(self,scene):
        if self.checkScene() == True:
            self.scene.agent.state = self.scene.agent.newState
        self.scene =scene

class QNetwork(nn.Module):
    #search softMax
    def __init__(self, hidden_size, state_size= (mlData.maxBullets + mlData.maxEnemies + 1)*2, action_size=9):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Define layers
        self.fc1 = nn.Linear(state_size + action_size + state_size + 1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action, new_state, reward):
        state=self.fuseState(state)
        new_state = self.fuseState(new_state)
        state=torch.tensor(state)
        action = torch.tensor([action])
        new_state = torch.tensor(new_state)
        reward = torch.tensor([reward])
        print("Shape of state:", state.shape)
        print("Shape of action:", action.shape)
        print("Shape of new_state:", new_state.shape)
        print("Shape of reward:", reward.shape)

        #state=self.fuseState(stateArr)
        #new_state = self.fuseState(new_stateArr)
        # Concatenate inputs
        x = torch.cat((state, action, new_state, reward), dim=0)
        
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def fuseState(self, state,data=mlData):
        result = []
        result.append(state.playerCoord[0])
        result.append(state.playerCoord[1])
        for i in range(data.maxEnemies):
            result.append(state.enemyCoord[i][0])
            result.append(state.enemyCoord[i][1])
        for j in range(data.maxBullets):
            result.append(state.bulletCoord[i][0])
            result.append(state.bulletCoord[i][1])
        return result
