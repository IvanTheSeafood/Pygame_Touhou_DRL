from assets.scripts.math_and_data.Vector2 import Vector2
import numpy as np

alpha = 0.2
epsilon =0.1
gamma = 0.9

status = True
hitBoxStatus = True
playSpeed = 1
renderStatus =  True            #I think this variable is useless atm

terminal = False
episode = 0
rewardTotal = 0

position : Vector2 = Vector2(0,0)
emptyVector = np.array([-1,-1,0])
emptyCoord = np.array([-1,-1])
maxBullets = 70
maxEnemies = 11

oldPoints = 0
points = 0
oldHp = 4
hp = 4
proxyRange = 400
kill = 0
survive = 0
killTotal = 0
waveTime = 7
time = 0
enemyLine = -1
enemyLineColor = (0,125,255)

replayMax = 10000
replay = []
batchTotal = 30
batch = []

dumbFuckPenalty = 0
deathCoord = []

finalScoreArray = []
'''
- game doesnt count kills during invincibility it seems, though its fair
Scores:
kill = 10 000
point item = 30 000 + some number that i cant be asked to understand, see item.py

wave 1: 07
wave 2: 17
wave 3: 25
'''
