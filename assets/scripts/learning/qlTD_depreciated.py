
from assets.scripts.math_and_data.Vector2 import Vector2
from assets.scripts.learning import mlData
#from assets.scripts.learning.dodgeRing import proxyRing
from assets.scripts.classes.game_logic.Collider import Collider
import numpy as np

class agent:
    def __init__(self,player):
        self.switch = mlData.status
        self.moveDirection = Vector2.zero()
        self.action = [0, False] # 9 possible actions
        self.shoot = False
        self.state = [[player.position.coords],[],[]]#player pos,enemy pos, bullet pos, 
        self.newState = [[player.position.coords],[],[]]
        self.terminal = False
        self.episode = 0
        self.q = mlData.Q

    def updateQ(self):
        mlData.Q = self.q
        return None
    
    def returnR(self,data):
        reward = 0

        if data.oldHp > data.hp:            #dead
            data.oldHp = data.hp
            reward = -10
        elif data.points > data.oldPoints:  #survive + kill
            data.oldPoints = data.points
            reward += +11
        else:                               #survive
            reward += +1        
        return reward
    
    def chooseAction(self,state):
        
        stateTuple = tuple(state)
        if stateTuple not in self.q:                                                                    #Check if state has been visited
            self.q[stateTuple] = [0]*9

        bestActions=[n for n, v in enumerate(self.q[stateTuple]) if v ==max(self.q[stateTuple])]

        if np.random.rand()< mlData.epsilon:                                                                   #Explore
            action = np.random.choice(range(8))
        elif len(bestActions)>1:                                                                        #Tie breaker
            action = np.random.choice(bestActions)
        else:                                                                                           #Greedy
            action = np.argmax(self.q[stateTuple])

        return [action,True]
    
    def takeAction(self, action=[4, False]):
 
        #Moving
        if action[0] == 0:
            self.moveDirection = Vector2.up() + Vector2.left()
        elif action[0] == 1:
            self.moveDirection = Vector2.up()
        elif action[0] == 2:
            self.moveDirection = Vector2.up() + Vector2.right()
        elif action[0] == 3:
            self.moveDirection = Vector2.left()
        elif action[0] == 4:
            self.moveDirection = Vector2.zero()
        elif action[0] == 5:
            self.moveDirection = Vector2.right()
        elif action[0] == 6:
            self.moveDirection = Vector2.down() + Vector2.left()
        elif action[0] == 7:
            self.moveDirection = Vector2.down()
        elif action[0] == 8:
            self.moveDirection = Vector2.down() + Vector2.right()

        #Shooting
        if action[1] == True:
            self.shoot = True
        else:
            self.shoot = False 
            
        return None

    def reviewAction(self,player):
        mlData.points = player.points
        mlData.position = player.position
        mlData.hp = player.hp

        self.terminal = mlData.terminal
        self.newState = tuple([int(mlData.position.coords[0]), int(mlData.position.coords[1])])
        reward = self.returnR(mlData)

        return reward