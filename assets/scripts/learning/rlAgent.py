from assets.scripts.classes.game_logic.Collider import Collider
from assets.scripts.math_and_data.Vector2 import Vector2
from assets.scripts.learning import mlData

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchviz import make_dot
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

        mlData.rewardArray=[]
        mlData.terminal = False
        mlData.rewardTotal = 0
        mlData.oldHp = 4
        mlData.oldPoints = 0
        mlData.killTotal = 0
        mlData.survive = 0
        mlData.timeStep = 0
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
        self.terminal = mlData.terminal

        if not self.terminal:
            mlData.terminalPoints = 0
        else:
            if self.player.hp<0:
                mlData.terminalPoints = -25
            else:
                mlData.terminalPoints = 10

            if len(mlData.finalScoreArray)>0:
                if self.player.points>np.max(mlData.finalScoreArray) and self.player.hp>=0:
                    mlData.terminalPoints = 20

            mlData.finalScoreArray.append(self.player.points)

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

        self.reward += data.kill*5 + waveDetect*survivalBonus + (self.player.points - data.oldPoints)/10000 + mlData.terminalPoints
        
        mlData.kill = 0
        #print( self.reward, '(',self.player.hp,',', data.oldHp,')total = ',data.rewardTotal,'                                           ',end='\r')
        mlData.oldPoints = self.player.points
        mlData.rewardTotal += self.reward
        mlData.rewardArray.append(mlData.rewardTotal)
        return self.reward

    def addReplay(self):

        mlData.replay.append((self.state,self.newState,self.action,self.reward,self.terminal))  #Store exp into replay
        if len(mlData.replay)>mlData.replayMax:     #If too long, delete oldest replay
            mlData.replay.pop(0)

        if len(mlData.replay)>=mlData.batchTotal:
                mlData.batch = random.sample(mlData.replay, mlData.batchTotal)

    def addTransition(self):  #9-10
        max_priority = max(max(mlData.priorities), 1.0)  # Ensure priorities are non-zero
        priority = max_priority
        mlData.bufferP.append((self.state,self.newState,self.action,self.reward,self.terminal,priority))
        if len(mlData.bufferP)>mlData.replayMax:     #If too long, delete oldest replay
            mlData.bufferP.pop(0)

    def sampleMiniBatch(self):
    # Extract priorities from the replay memory
        priorities = np.array(mlData.priorities[:len(mlData.bufferP)])
    
    # Calculate probabilities for sampling based on priorities
        probs = priorities ** mlData.alphaP
        probs /= np.sum(probs)
 
        indices = np.random.choice(len(mlData.bufferP), size=mlData.batchPSize, p=probs)
        minibatch = [mlData.bufferP[idx] for idx in indices]
    
        return minibatch, indices
    
    def updatePriorities(self, batch, indexArray): #11-12
        index = 0
        losses=[]
        for transition in batch:
            qState = self.q(transition[0]).detach().numpy()
            qAction = qState[transition[2][0]]
            if not transition[4]:
                tdError = abs(transition[3] +mlData.gamma*np.max(self.qTarget(transition[1]).detach().numpy())-qAction)
            else:
                tdError = abs(transition[3])

            mlData.priorities[indexArray[index]]= (tdError + mlData.temperature) ** mlData.alphaP

            weightP = ((mlData.batchPSize*mlData.priorities[indexArray[index]])/(np.sum(mlData.priorities)))**mlData.betaP 

            losses.append((transition[3] +mlData.gamma*np.max(self.qTarget(transition[1]).detach().numpy())-qAction)**2*weightP)
            loss= torch.tensor(np.mean(losses),dtype=torch.float32, requires_grad = True)
            loss.backward()
            index +=1 
            #self.updatePrioritizedQnet()
        self.updateQtarget()
        mlData.betaP =min(1.0, mlData.betaP + mlData.betaPIncrement)

    def updateQtarget(self):
        self.qTarget.load_state_dict(self.q.state_dict())
        try:
            torch.save(self.qTarget.state_dict(),('QNetwork_'+mlData.version+'_Target.pth'))
        except:
            pass

    def reviewAction(self):
        data = mlData
        targetQTensor=self.q(self.state).detach().numpy()
        predictQ=targetQTensor[self.action[0]]
        if data.mode == 'DQN':
            targetQ = predictQ + data.alpha * (self.reward + data.gamma*np.max(self.q(self.newState).detach().numpy())-predictQ)
        else:
            targetQ = predictQ + data.alpha * (self.reward + data.gamma*np.max(self.qTarget(self.newState).detach().numpy())-predictQ)

        targetQTensor[self.action[0]]=targetQ
        targetQTensor=torch.tensor(targetQTensor, dtype=torch.float32)  #keep the outputshapes intact

        loss = nn.MSELoss()(self.q(self.state), targetQTensor)
        loss.backward()
        try:
            torch.save(self.q.state_dict(),('QNetwork_'+mlData.version+'_Online.pth'))
        except:
            pass
    def printNN(self):
        output = self.q(self.state)
        graph = make_dot(output, params=dict(self.q.named_parameters()))
        graph.render("model_graph", format="png")

    def expReplay(self):
        for experience in mlData.batch:
                
            expState, expNewState, expAction, expR, expTerminal = experience
            expArray = self.q(expState).detach().numpy()
            expNewArray = self.q(expNewState).detach().numpy()

            if not expTerminal:
                target = expR +  mlData.gamma * np.max(expNewArray)
            else:
                target = expR
            expArray[expAction[0]] +=  mlData.alpha * (target - expArray[expAction[0]])

            targetQTensor=torch.tensor(expArray, dtype=torch.float32)

            loss = nn.MSELoss()(self.q(expState), targetQTensor)
            loss.backward()
            
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
    def __init__(self, hidden_size, state_size= (mlData.maxBullets)*3 + (mlData.maxEnemies + 1)*2 +1, maxAction = 9):
        super(QNetwork, self).__init__()
        self.inputSize = state_size+1
        self.outputSize = maxAction
        
        # Define layers
        self.fc1 = nn.Linear(state_size+1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), self.outputSize)
    
    def forward(self, state):
        state=self.fuseState(state)
        stateTensor=torch.tensor(state, dtype=torch.float32)
  
        x = torch.relu(self.fc1(stateTensor))
        x = self.fc2(x)
        x = self.fc3(x)

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
    
QNet = QNetwork(20)
QTrain = QNetwork(20)

# Save the graph as an image

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

