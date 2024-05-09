from assets.scripts.math_and_data.Vector2 import Vector2
import numpy as np

version = "1.2.14"
mode = 'DQN'          #DQN, DDQN, EDDQN
epMax = 100     #max no of episodes before the code ends

alpha = 0.01
temperature =0.5        #the softmax version of epsilon
gamma = 0.9

status = True
hitBoxStatus = True
trainedNN = True        #Loads a previously trained NN

terminal = False
episode = 1             #currnt episode (init)
rewardTotal = 0

position : Vector2 = Vector2(0,0)
emptyVector = np.array([-1,-1,0])
emptyCoord = np.array([-1,-1])
maxBullets = 70
maxEnemies = 11
difficulty = [True, 70]
hp = 4
proxyRange = 400

oldPoints = 0
points = 0
oldHp = 4
kill = 0
survive = 0
killTotal = 0
time = 0
enemyLine = -1
enemyLineColor = (0,125,255)
timeStep = 0

replayMax = 10000
replay = []
batchTotal = 30
batch = []

deathCoord = []

rewardArray = []
finalScoreArray = []
QTargetStep = 1000

terminalPoints = 0

alphaP = 0.6
betaP = 0.4
betaPIncrement = 0.01
priorities=[1.0]*replayMax
bufferP =[]
bufferPSize = replayMax
batchPSize= batchTotal
'''
- game doesnt count kills during invincibility it seems, though its fair
Scores:
kill = 10 000
point item = 30 000 + some number that i cant be asked to understand, see item.py

wave 1: 07
wave 2: 17
wave 3: 25
'''
