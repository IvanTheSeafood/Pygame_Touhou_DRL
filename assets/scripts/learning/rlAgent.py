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
        self.action = [0, False] # 9 possible actions, shoot or not
        self.reward = 0
        mlData.rewardTotal = 0
        mlData.oldHp = 4
        mlData.oldPoints = 0
        #print(mlData.rewardTotal)
        self.terminal = False

        self.episode = mlData.episode
        self.q = QNet
        #self.targetQ = 0
        self.predictQ = 0
        self.ring = None

    def selectAction(self):                     #Select action based on NN
        
        self.qList = self.q(self.state).detach().numpy()    #Do NN

        if np.random.rand()>mlData.epsilon:                   #Explore vs Exploit (or something)
            self.action[0]=np.argmax(self.qList)
            bestActions =[i for i, j in enumerate(self.qList) if j ==max(self.qList)]   #OG tiebreaker
            self.action[0]=np.random.choice(bestActions)
        else:
            self.action[0]=np.random.randint(0,8)
    
        self.action[1]=True
        self.pedictQ = self.qList[self.action[0]]
        print('episode: {}, position: [{}, {}], action: {}, reward: {}'.format(
            mlData.episode,int(self.player.position.coords[0]),
            int(self.player.position.coords[1]),self.action[0],
            round(mlData.rewardTotal,2)), end ='\r')

        return self.moveDirection(self.action)
    
    def moveDirection(self, action=[4, False]): #Translate action integers into pygame language of vectors
 
        #Moving
        finalMove =[Vector2.up() + Vector2.left(),   Vector2.up(),   Vector2.up() + Vector2.right(),
                    Vector2.left(),                  Vector2.zero(), Vector2.right(),
                    Vector2.down() + Vector2.left(), Vector2.down(), Vector2.down() + Vector2.right()] 
        
        #Shooting
        if action[1] == True:
            self.player.shoot()
            
        return finalMove[action[0]]
    
    def returnR(self,data=mlData):  #not working: dead and score counter
        self.reward = 0
        #print(self.player.hp)
        if self.player.hp ==4:
            data.oldHp = 4
            if data.rewardTotal <0:
                data.rewardTotal = 0

        if self.player.hp < data.oldHp:
            data.oldHp = self.player.hp
            self.reward -=1000
     
        #print(self.player.points,'-',data.oldPoints,')/100 * ',(self.player.power-1.4),' + 0.01 =',end='')
        self.reward += (self.player.points-data.oldPoints)/1000*(self.player.power - 1.4) + 0.01
        #print( self.reward, '(',self.player.hp,',', data.oldHp,')total = ',data.rewardTotal,'                                           ',end='\r')
        data.oldPoints = self.player.points
        data.rewardTotal += self.reward
        return self.reward
    
    def reviewAction(self):
        data = mlData
        targetQ = self.predictQ + data.alpha * (self.reward + data.gamma*np.max(self.q(self.newState).detach().numpy())-self.predictQ)

        targetQTensor=self.qList
        targetQTensor[self.action[0]]=targetQ
        targetQTensor=torch.tensor(targetQTensor, dtype=torch.float32)  #keep the outputshapes intact
        #predictQTensor=torch.tensor(self.qList, dtype=torch.float32)    #not to be mixed with self.predictQ

        #print(type(targetQTensor),targetQTensor)
        #print(type(predictQTensor),predictQTensor)

        #targetQTensor.requires_grad_(True)
        #predictQTensor.requires_grad_(True)

        loss = nn.MSELoss()(self.q(self.state), targetQTensor)
        loss.backward()

    
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

class QNetwork(nn.Module):
    #search softMax
    def __init__(self, hidden_size, state_size= (mlData.maxBullets + mlData.maxEnemies + 1)*2, maxAction = 9):
        super(QNetwork, self).__init__()
        self.inputSize = state_size
        self.outputSize = maxAction
        
        # Define layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size-1)
        self.fc3 = nn.Linear(hidden_size-1, hidden_size-3)
        self.fc4 = nn.Linear(hidden_size-3, hidden_size-5)
        self.fc5 = nn.Linear(hidden_size-5, self.outputSize)
    
    def forward(self, state):
        state=self.fuseState(state)
        state=torch.tensor(state, dtype=torch.float32)
  
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        #print(type(x),x)
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

QNet = QNetwork(11)
print('nn start')