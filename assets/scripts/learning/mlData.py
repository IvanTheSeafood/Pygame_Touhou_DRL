from assets.scripts.math_and_data.Vector2 import Vector2
import numpy as np

alpha = 0.2
epsilon =0.01
gamma = 0.9

status = True
hitBoxStatus = True

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
