###############################################################################
# File Name: coppeliasim_env.py
# Purpose: Define a simulation environment to allow Reinforcement Learning
#          training in CoppeliaSim
# Creator: James May
# Date Created: 4/17/2023
###############################################################################


# Environment
# Singular static cubesat floating in space with slightly varied locations & randomized orientation


import cbor
import random
from time import sleep
from zmqRemoteApi import RemoteAPIClient
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SpaceJunkEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    
    ###### CORE METHODS FOR RL ######
    def __init__(self):
        super().__init__()
        
        #Actions
        # action[0] = "/Sawyer/joint"
        # action[1] = "/Sawyer/link/joint"
        # action[2] = "/Sawyer/link/joint/link/joint"
        # action[3] = "/Sawyer/link/joint/link/joint/link/joint"
        # action[4] = "/Sawyer/link/joint/link/joint/link/joint/link/joint"
        # action[5] = "/Sawyer/link/joint/link/joint/link/joint/link/joint/link/joint"
        # action[6] = "/Sawyer/link/joint/link/joint/link/joint/link/joint/link/joint/link/joint"
        # action[7] = "/Sawyer/BaxterGripper/centerJoint"
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,1), dtype=np.float32)
        
        self.observation_space = spaces.Dict({"headcamera": spaces.Box(low=0, high=255, shape=(360,640,3), dtype=np.uint8), "joints": spaces.Box(low=-1, high=1, shape=(8,1), dtype=np.float32)}, seed=42)
        
        self.sim = None
        self.load()
        
    def step(self, action): #this is the chonky one
        reward = 0 #default
        action[0]
        return observation, reward, done, info

    def reset(self):
        observation = spaces.Dict{}
        self.sim.stopSimulation()
        sleep(0.1)#give sim enough time to stop
        self.load()
        observation["headcamera"] = self.getheadimage()
        observation["joints"] = 
        return observation  # reward, done, info can't be included

    def close(self):
        self.sim.stopSimulation()
        return 0


    ###### HELPER METHODS ######
    def load(self): #connect to CoppeliaSim & load environment
        client = RemoteAPIClient('localhost',23000)
        self.sim = client.getObject('sim')
        self.sim.loadScene('/home/vlarko/rl-space-junk/space-sim.ttt')
        self.stochasticaddcubesat([0.5,0,0.5])
        self.sim.startSimulation()
        return 0

    def addcubesat(self,position):
        sim = self.sim
        block = sim.createPrimitiveShape(self.sim.primitiveshape_cuboid,[0.05,0.05,0.15],0)
        sim.setObjectPosition(block,sim.handle_world,position)
        return 0
    
    def stochasticaddcubesat(self,base_position):
        sim = self.sim
        position = []
        orientation = []
        for pos in base_position:
            position.append(pos + (random.random() - 0.5)/5) #-0.1 to 0.1
        block = sim.createPrimitiveShape(self.sim.primitiveshape_cuboid,[0.05,0.05,0.15],0)
        sim.setObjectPosition(block,sim.handle_world,position)
        for i in range(3):
            orientation.append(random.random()*2*3.14159265)
        sim.setObjectOrientation(block,block,orientation)
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
 