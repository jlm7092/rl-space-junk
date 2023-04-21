###############################################################################
# File Name: coppeliasim_env.py
# Purpose: Define a simulation environment to allow Reinforcement Learning
#          training in CoppeliaSim
# Creator: James May
# Date Created: 4/17/2023
###############################################################################


# Environment Levels
# 0 - Singular static cubesat floating in space
# 1 - space junk dataset objects static floating in space


import cbor
from zmqRemoteApi import RemoteAPIClient
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SpaceJunkEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    #metadata = {"render.modes": ["human"]}
    
    ###### CORE METHODS FOR RL ######
    def __init__(self,level=0):
        super().__init__()
        self.level = level #level
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,1), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255, shape=(360,640,3), dtype=np.uint8)
        self.sim = None
        self.load()

        
    def step(self, action): #this is the chonky one
        reward = 0 #default
        return observation, reward, done, info

    
    def reset(self):
        self.load()
        observation = getheadimage()
        return observation  # reward, done, info can't be included

    
    def close(self):
        self.sim.stopSimulation()
        return 0


    ###### HELPER METHODS ######
    def load(self): #connect to CoppeliaSim & load environment
        client = RemoteAPIClient('localhost',23000)
        self.sim = client.getObject('sim')
        self.sim.loadScene('/home/vlarko/rl-space-junk/space-sim.ttt')
        if self.level == 0:
            self.addcubesat([0.5,0,0.25])
        self.sim.startSimulation()
        return 0

        
    def addcubesat(self,position):
        sim = self.sim
        block = sim.createPrimitiveShape(self.sim.primitiveshape_cuboid,[0.05,0.05,0.15],0)
        sim.setObjectPosition(block,sim.handle_world,position)
        return 0
        
        
    def getheadimage(self):
        sim = self.sim
        cam=sim.getObject("/Sawyer/head_camera")
        sim.handleVisionSensor(cam)
        image,resolution=sim.getVisionSensorImg(cam)
        pixels = list(image)
        x = np.array(pixels, dtype=np.uint8)
        array = x.reshape(360,640,3)
        return array
    
    def getsim(self):
        return self.sim
 
