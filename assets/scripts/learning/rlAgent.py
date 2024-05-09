from assets.scripts.classes.game_logic.Collider import Collider
from assets.scripts.math_and_data.Vector2 import Vector2
from assets.scripts.learning import mlData

import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnFunc
import pygame

class agent:
    def __init__(self,player):
        self.switch = mlData.status
        self.player = player

        self.initBool = True
        self.state = agentState(player) #player pos, enemy pos, bullet pos, 
        self.newState = agentState(player)
        self.action = [0, False] # 9 possible actions, shoot or not
        self.reward = 0
        self.terminal = False

        mlData.rewardTotal = 0
        mlData.oldHp = 4
        mlData.oldPoints = 0
        mlData.killTotal = 0
        mlData.survive = 0
        mlData.timeStep = 0
        mlData.terminal = False
        self.episode = mlData.episode

        self.time = 0
        self.timeDeath = -5
        self.timeDumb = 0
        self.q = QNet
        self.qTarget = QTrain
        self.ring = None

    def selectAction(self):                     #Select action based on NN
        
        if mlData.status:
            if self.state.enemyCoord[0][0] == mlData.emptyCoord[0] and self.state.enemyCoord[0][1] == mlData.emptyCoord[1]: #no enemies on map
                dynaTemp = 0.9
            else:
                dynaTemp = mlData.temperature
            #Softmax
            actionProb = nnFunc.softmax(self.q(self.state)/dynaTemp)
            self.action[0]=torch.multinomial(actionProb, num_samples=1).item()
            self.action[1]=True
        else:
            self.action = self.getKeyPress()

        print('Episode: {}, Stage:{}, Time:{}, Position: [{}, {}], Action: {}, Kill Count: {}, Reward: {}           '.format(mlData.episode, mlData.difficulty, round(self.time,2), int(self.player.position.coords[0]), int(self.player.position.coords[1]), self.action[0], mlData.killTotal, round(mlData.rewardTotal,2)), end ='\r')

        return self.moveDirection(self.action)
    
    def getKeyPress(self):
            action = 4
            if pygame.key.get_pressed()[pygame.K_UP]:
                action -= 3
            if pygame.key.get_pressed()[pygame.K_DOWN]:
                action += 3
            if pygame.key.get_pressed()[pygame.K_LEFT]:
                action -= 1
            if pygame.key.get_pressed()[pygame.K_RIGHT]:
                action += 1
            return action,True
    
    def moveDirection(self, action=[4,True]): #Translate action integers into pygame language of vectors
 
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
        if self.player.hp ==4:  #init
            data.oldHp = 4
            if data.rewardTotal <0:
                data.rewardTotal = 0

        if self.player.hp < data.oldHp: #damaged
            data.oldHp = self.player.hp
            self.reward -=10
            self.timeDeath = self.time
        elif self.player.hp > data.oldHp:   #healitem
            data.oldHp = self.player.hp

        if data.kill > 2:   #Balance out enemies that die of heart attack
            data.kill = 0
        data.killTotal += data.kill

        if self.state.playerCoord[1]> mlData.enemyLine:     #agent keeps going top left corner thinking it's a good strategy.  (SPOILER: it's not)
            mlData.enemyLineColor = (0,125,255)
            self.timeDumb=self.time
        else:
            mlData.enemyLineColor = (255,255,0)
            self.timeDeath = self.time

        survivalBonus = self.time-self.timeDeath
        
        if self.state.enemyCoord[0][0] == mlData.emptyCoord[0] and self.state.enemyCoord[0][1] == mlData.emptyCoord[1]: #no enemies on map
            waveDetect =  -0.0005
        else:
            waveDetect = -0.01

        if not mlData.terminal:
            mlData.terminalPoints = 0

        self.reward += data.kill*5 + waveDetect*survivalBonus + (self.player.points - data.oldPoints)/10000 + mlData.terminalPoints

        data.kill = 0
        #print( self.reward, '(',self.player.hp,',', data.oldHp,')total = ',data.rewardTotal,'                                           ',end='\r')
        data.oldPoints = self.player.points
        data.rewardTotal += self.reward
        return self.reward

    def addPrioritizedReplay(self):
        transition = self.state,self.newState,self.action,self.reward,self.terminal
        max_priority = np.max(mlData.priorities) if mlData.buffer else 1.0
        mlData.buffer.append(transition)
        if len(mlData.buffer)>mlData.replayMax:
            mlData.buffer.pop(0)
        mlData.priorities[len(mlData.buffer) - 1] = max_priority

    def samplePrioritizedBatch(self):
        priorities = mlData.priorities[:len(mlData.buffer)]
        probs = priorities ** mlData.beta / np.sum(priorities ** mlData.beta)
        indices = np.random.choice(len(mlData.buffer), mlData.batchTotal, p=probs)
        batch = [mlData.buffer[idx] for idx in indices]
        weights = (len(mlData.buffer) * probs[indices]) ** (-mlData.beta)
        weights /= np.max(weights)
        return batch, indices, weights
    
    def updatePriorities(self, indices, tdError):
        for idx, error in zip(indices, tdError):
            mlData.priorities[idx] = (error + 1e-5) ** mlData.alpha

    def updateQtarget(self):
        self.qTarget.load_state_dict(self.q.state_dict())
        torch.save(self.qTarget.state_dict(),('QNetwork_'+mlData.version+'_Target.pth'))

    def reviewAction(self):
        tdError=[]
        batch, indices, weights = self.samplePrioritizedBatch()
        for i in range (len(batch)):
            pState, pNewState, pAction, pReward,pTerminal = batch[i]
            targetQTensor=self.q(pState).detach().numpy()
            predictQ=targetQTensor[pAction[0]]
            tdError.append(pReward + mlData.gamma*np.max(self.qTarget(pNewState).detach().numpy()) - predictQ) 
        
        self.updatePriorities(indices,tdError)

        loss = nn.MSELoss()(sum(tdError)/float(len(tdError)) ** 2 * sum(weights)/float(len(weights)))  #mean square error or something
        loss.backward()

        torch.save(self.q.state_dict(),('QNetwork_'+mlData.version+'_Online.pth'))
            
class agentState:
    def __init__(self, player):
        #self.playerPos=player.position
        self.time = 0
        self.playerCoord = player.position.coords
        self.playerHp =player.hp

        self.enemyCoord = [mlData.emptyCoord]*mlData.maxEnemies

        self.bulletCoord = [mlData.emptyCoord]*mlData.maxBullets

        self.bulletAngle = [0]*mlData.maxBullets

    def updateTime(self,time):
        mlData.time = time
        self.time = time

    def updateBullet(self, agent, bullet, i):
        hitbox= Collider(10,bullet.position)
        if agent.ring.check_collision(hitbox):
            if i<mlData.maxBullets:
                self.bulletCoord[i]= bullet.position.coords
                self.bulletAngle[i]= bullet.angle
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
                    self.bulletAngle[cellPos] = bullet.angle

    def updateEnemy(self, enemy, i):
        if i < mlData.maxEnemies:
            self.enemyCoord[i] = enemy.position.coords
            if mlData.enemyLine< enemy.position.coords[1] and enemy.position.coords[1]<500:
                mlData.enemyLine= enemy.position.coords[1]
        else:
            pass

    def checkDistance(self, a):
        b = self.playerCoord
        return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)

class QNetwork(nn.Module):
    #search softMax
    def __init__(self, hidden_size, state_size= (mlData.maxBullets)*3 + (mlData.maxEnemies + 1)*2 +2, maxAction = 9):
        super(QNetwork, self).__init__()
        self.inputSize = state_size
        self.outputSize = maxAction
        
        # Define layers
        self.fc1 = nn.Linear(state_size, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, self.outputSize)
    
    def forward(self, state):
        state=self.fuseState(state)
        state=torch.tensor(state, dtype=torch.float32)
  
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        #x = self.fc5(x)
        #x = self.fc6(x)
        #x = self.fc7(x)

        return x
     
    def fuseState(self, state,data=mlData):
        result = []
        result.append(state.time)
        result.append(state.playerCoord[0])
        result.append(state.playerCoord[1])
        result.append(state.playerHp)
        for i in range(data.maxEnemies):
            result.append(state.enemyCoord[i][0])
            result.append(state.enemyCoord[i][1])
        for j in range(data.maxBullets):
            result.append(state.bulletCoord[j][0])
            result.append(state.bulletCoord[j][1])
            result.append(state.bulletAngle[j])

        return result
    
QNet = QNetwork(350)
QTrain = QNetwork(350)

print('NN Start, Current Version:', mlData.version)

if mlData.trainedNN:
    version = 0
    versionString = ''
    try:
        version = int(mlData.version[4])*10 + int(mlData.version[5])

    except:
        version = int(mlData.version[4]) -1

    for i in range(4):
        versionString+=mlData.version[i]
   
    versionStringOld = versionString
    versionString += str(version)
    versionStringOld += str(version-1)
    print('loading NN version','QNetwork_'+versionString+'_Target.pth',':', end = ' ')

    try:
        QNet.load_state_dict(torch.load('QNetwork_'+versionString+'_Online.pth'))
        QTrain.load_state_dict(torch.load('QNetwork_'+versionString+'_Target.pth'))
        print('NN Located')
    except FileNotFoundError:
        try:
            QNet.load_state_dict(torch.load('QNetwork_'+versionStringOld+'_Online.pth'))
            QTrain.load_state_dict(torch.load('QNetwork_'+versionStringOld+'_Target.pth'))
            print('Old NN Found')
        except FileNotFoundError:
            print('NN Failed')
