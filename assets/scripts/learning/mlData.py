from assets.scripts.math_and_data.Vector2 import Vector2
import numpy as np

version = "1.2.11"
alpha = 0.01
temperature =0.5        #the softmax version of epsilon
gamma = 0.9

status = True
hitBoxStatus = True
trainedNN = True        #Loads a previously trained NN

terminal = False
episode = 0
rewardTotal = 0

position : Vector2 = Vector2(0,0)
emptyVector = np.array([-1,-1,0])
emptyCoord = np.array([-1,-1])
maxBullets = 70
maxEnemies = 11
difficulty = [True, 10]
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

dumbFuckPenalty = 0
deathCoord = []

finalScoreArray = []
QTargetStep = 1000

terminalPoints = 0
terminalLive=10
terminalHighScore = 20
terminalDeath = -25
'''
- game doesnt count kills during invincibility it seems, though its fair
Scores:
kill = 10 000
point item = 30 000 + some number that i cant be asked to understand, see item.py

wave 1: 07
wave 2: 17
wave 3: 25
'''
