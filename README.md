# Touhou Pygame AI
* A project to train an AI to play a pygame version of the bullet hell shooter game: Touhou, with deep reinforcement learning
* An individual project for fun turned into a school group project as the author`s teamates couldn`t figure out Pyboy but their RL project deadlines closing in

## Requirements
Python 3.12, any version lower than that may cause errors (from personal experience)  

## Getting Started
- Install the whole folder
- Open the entire folder in VS Code (MAC users you`re on your own)
- In the terminal input the command `pip install -r requirements.txt`
- then also do `pip install -r pytorch` for neural network stuff.
- If no errors were reported you should be good to go
- To run the game, simply run the `run.py` file in VS Code. *(Depending on the version errors might appear as the Author is still fucking around with the god damn neural network which keeps crashing)*

## Gameplay
- Player moves up, down, left, right with arrow keys
- player shoots with `x` key
- Player skips boring UI stuff with `ENTER` key
- Enemy spawns in waves and shoots at player
- Player looses HP(max 4 per gameplay) and respawns at the bottom of the map if they get hit
- Lose all HP and you get sent to the scoreboard
- Player gets points if they shoot an enemy
- Power is an accurate gameplay mechanic to the real deal but does aboslutely nothing in the pygame version other than extra points *(the blue and red stuff dropped by enemies)*

## Code
- The game is split into 50 million smaller scripts.  
- Most of the important stuff are in `assets\scripts`, with most gameplay processing stuff in the `\scenes\GameScene.py` script, more on that later  

- `run.py` is basically an infinite loop where it:  
    - 1. Process user/agent input within a scene  
    - 2. Update whatevers supposed to happen within the scene  
    - 3. Render the changes  
    - 4. Change to the next scene if necessary  
- the OG code were the ones with `active_scene.something()`  

-  The gameplay is split into 3 different scenes:

### TitleScene
- The first thing you see when you open the game

### GameScene
- The actual place where the gameplay`s happening

### ScoreboardScene
- The scene that tells you that you died and you suck  

These scenes keep looping indefinitely, but if RL is on, the game automatically restarts upon agent death.

## Reinforcement Learning
The `learning` Folder is created for reinforcement learning stuff.  Most of the stuff related to RL (vairables, classes, functions) can be found here.

### mlData.py
This folder stores variables shared across multiple python scripts, which can be imported with:
`from assets.scripts.learning import mlData`
and its variables can be called and updated with:
`variable = mlData.[variable_name]` and `mlData.[variable_name]=value`

#### Switches
- `status`, `bool`: True=enable RL, False=play regularly
- `hitBoxStatus`, `bool`: True=shows circle around player where the 50 closest bullets are accounted for

#### Average RL parameters
- `alpha`, `int`: learning rate
- `epsilon`, `int`: exploration rate
- `gamma`, `int`: decay rate
- `terminal`, `bool`: terminal state, since the game terminates on its own when hp reaches 0, this bool is considered obsolete 
- `Q`, `empty`: IDK wt to do with Q values yet
*That said, at the time of writing I`m still figuring out all the RL stuff so these parameters are never found in actual execution of this version of code*

#### Initial Values
- hp = current player hp
- oldHp = player hp before update
- points = current player points
- oldPoints = player points before update
- position = Initial player position
- maxEnemies = number of enemies the `agent` should consider
- maxBullets =  number of bullets the `agent` should consider
- proxyRange = determines bullet detection ring around player
- all the initialising values and values we keep track of

### rlAgent.py
- Here is where all the functions and classes are stored for RL  
- If you spot a redundant variable or defined variables that never appear in the script, you`re probably right. IDK what i was doing most of the time and just added random stuff as I learn along.

#### class agent

##### __init__
- stores all the stuff needed for the agent:
    variables: states, new states, position, actions, rewards

- the `action` variable is a 1x2 list, where `action[0]` is an integer value (0-8) indicating direction just like the car in the racetrack coursework.  `action[1]` is a boolean controlling whether the player should shoot.  There are basically no drawbacks in holding the shoot button so most of the time I just kept the thing as True.

Note:
- A `player` is the in game character that handles what happens in the environment, where as an `agent` is the AI with brains n stuff.  You would see something similar to `player.agent` in GameScene.py when I want to update the agent`s states during gameplay, and `agent.player` in rlAgent.py because I just learnt the existence of classes in python and don`t entirely understand how they work.

- A `position` is a vector2 variable inside the class of player, while `coords` is a 1x2 numpy array storing the x and y coordinates within `position`, ie. `player.positon.coords`.

##### def takeAction():
Determines how the agent should move.  Right now its completely random.

##### def moveDirection():
converts action variable into actual controls understandable by pygame.

##### def returnR():
returns the reward by checking the following:
    - If hp decreases: -10
    - If is alive: + 1
    - If earn points: + amount of points earned/100 *for balancing*

#### class agentState
- the state of agent is its own seperate class due to the sheer amount of shit to be handled.
- state of agent stores:  
 * coordinates of agent (numpyarr, [float x, float y]) *(not to be confused with position, which is a Vector2 datatype)*
 * coordinates of maximum enemies on map (numpyarr, [[float x, float y]]*maxEnemies) *(list size kept constant for easier NN input,empty slots are registerd as [-1,-1])*
 * coordinates of 50 nearest bullets in player proxy range (numpyarr, [[float x, float y]]*maxEnemies) *(It was originally planned to use proxy circle to reduce the number of bulletrs to be processed by the agent, it is then found out that NNs with constant input sizes seem easier to be implemented, but i can`t be asked to remove the ring after all that work, so it stays)*

##### def update[state]():
- Updates the coordinate list based on info from GameScene.py  
- State.playerCoord is updated directly in GameScene.py

##### def checkDistance(a,b):
- Literally a distance formula function to compare distances of 2 game objects.

#### class RLProcess
- Unlike agent which is used in GameScene.py, RLProcess is used in run.py and is designed to deal with all the DRL voodoo.
- It is supposed to make the run.py loop resemble a regular RL loop whenever the actual game is running by adding extra lines of RL code inbetween.
- Note none of the stuff here is currently working.

##### def checkScene(scene):
- Since run.py did not import GameScene.py, I opted to check whether the game is currently running GameScene.py by checking whether the active scene has a valid agent class, as the class only exists in the game scene.a
- returns a boolean.

##### def reviewAction(scene,nn):
- The remaining process of doing Q value and stuff

##### def updateState(scene):
- updates the state of agent to its new state.

#### class Qnetwork():
- The neural network code written by non other than the GPT god itself, which uses the pytorch package.
-  I have 0 idea WTF is going on with this class, and this thing crashes everytime.  So all of its functions are commented out within the class RLProcess.