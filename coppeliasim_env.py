###############################################################################
# File Name: coppeliasim_env.py
# Purpose: Define a simulation environment to allow Reinforcement Learning
#          training in CoppeliaSim using Pytorch
# Creator: James May
# Date Created: 4/17/2023
###############################################################################

# Environment Levels
# 0 - Singular static block floating in space
# 1 - space junk dataset objects static floating in space
# 2 - space junk dataset objects with randomized colors static floating in space
# 3 - space junk dataset objects with randomized colors drifting in space with randomized angular and linear and velocity

import cbor
from zmqRemoteApi import RemoteAPIClient

class SpaceJunkEnv:
    def __init__(self):
        self.env_level = 0 #level
        self.sim = None
    
    def load(self): #connect to CoppeliaSim & load environment
        client = RemoteAPIClient('localhost',23000)
        self.sim = client.getObject('sim')
        self.sim.loadScene('/home/vlarko/rl-space-junk/space-sim.ttt')
        

