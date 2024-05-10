from assets.scripts.math_and_data.Vector2 import Vector2
import numpy as np

version = "1.2.15"
mode = 'PDDQN'          #DQN, DDQN, EDDQN
epMax = 100     #max no of episodes before the code ends

alpha = 0.01
temperature =0.5        #the softmax version of epsilon
gamma = 0.9

status = True
hitBoxStatus = True
trainedNN =False       #Loads a previously trained NN

terminal = False
episode = 0

position : Vector2 = Vector2(0,0)
emptyCoord = np.array([-1,-1])
maxBullets = 50
maxEnemies = 10
oldPoints = 0
points = 0
oldHp = 4
hp = 4
proxyRange = 100

Q={}
