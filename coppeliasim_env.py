###############################################################################
# File Name: coppeliasim_env.py
# Purpose: Define a simulation environment to allow Reinforcement Learning
#          training in CoppeliaSim
# Creator: James May
# Date Created: 4/17/2023
###############################################################################

# Environment:
# Singular static cubesat floating in space with slightly varied locations & randomized orientation

import cbor
import random
from time import sleep
from typing import Dict
from zmqRemoteApi import RemoteAPIClient
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SpaceJunkEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    
    ###### CORE METHODS FOR RL ######
    def __init__(self,stepped=True,fast=True):
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
        
        self.stepped = stepped #determines whether or not simulation is stepped (yes for train, no for test)
        self.fast = fast #determines whether or not simulation is in fast mode (rendering)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,1), dtype=np.float32)
        
        self.observation_space = spaces.Dict({"headcamera": spaces.Box(low=0, high=255, shape=(360,640,3), dtype=np.uint8), "wristcamera": spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8),"joints": spaces.Box(low=-1, high=1, shape=(8,1), dtype=np.float32)}, seed=42)
        
        self.sim = None
        self.client = None
        self.load()
        
        
    def step(self, action): #this is the chonky one
        sim = self.sim
        #setup defaults
        #observation = spaces.Dict({"headcamera":np.zeros((360,640,3)),"joints":np.zeros((8,1))})
        observation = {}#spaces.Dict({"headcamera":None,"joints":None})
        reward = -0.01 #doing anything hurts
        done = False
        info = {}
        
        #tell Sawyer position goals
        joint0 = sim.getObject("/Sawyer/joint")
        sim.setJointTargetPosition(joint0,float(action[0])*3.14159265,[0.1,0.1,0.1])
        
        joint1 = sim.getObject("/Sawyer/link/joint")
        sim.setJointTargetPosition(joint1,float(action[1])*3.14159265,[0.1,0.1,0.1])
        
        joint2 = sim.getObject("/Sawyer/link/joint/link/joint")
        sim.setJointTargetPosition(joint2,float(action[2])*3.14159265,[0.1,0.1,0.1])
        
        joint3 = sim.getObject("/Sawyer/link/joint/link/joint/link/joint")
        sim.setJointTargetPosition(joint3,float(action[3])*3.14159265,[0.1,0.1,0.1])
        
        joint4 = sim.getObject("/Sawyer/link/joint/link/joint/link/joint/link/joint")
        sim.setJointTargetPosition(joint4,float(action[4])*3.14159265,[0.1,0.1,0.1])
        
        joint5 = sim.getObject("/Sawyer/link/joint/link/joint/link/joint/link/joint/link/joint")
        sim.setJointTargetPosition(joint5,float(action[5])*3.14159265,[0.1,0.1,0.1])
        
        joint6 = sim.getObject("/Sawyer/link/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
        sim.setJointTargetPosition(joint6,float(action[6])*3.14159265,[0.1,0.1,0.1])
        
        gripper = sim.getObject("/Sawyer/BaxterGripper/centerJoint")
        if float(action[7]) > 0:
            sim.setJointTargetVelocity(gripper,-0.005)
        else:
            sim.setJointTargetVelocity(gripper,0.005)
        
        
        #step environment
        self.client.step()
        
        #observe
        observation["headcamera"] = self.getheadimage()
        observation["wristcamera"] = self.getwristimage()
        observation["joints"] = self.getjointpositions()
        
        #calculate reward
        gripperHandle = sim.getObject("/Sawyer/BaxterGripper")
        leftPadHandle = sim.getObject("/Sawyer/BaxterGripper/leftPad")
        rightPadHandle = sim.getObject("/Sawyer/BaxterGripper/righPad")
        cubeSatHandle = sim.getObject("/Cuboid")
        
        leftPadCollide,collidingObjectHandles=sim.checkCollision(leftPadHandle,cubeSatHandle)
        rightPadCollide,collidingObjectHandles=sim.checkCollision(rightPadHandle,cubeSatHandle)
        gripperCollide,collidingObjectHandles=sim.checkCollision(gripperHandle,cubeSatHandle)
        
        if leftPadCollide == 1:
            reward += 10
            
        if rightPadCollide == 1:
            reward += 10
            
        if gripperCollide == 1:
            reward += 0.1
        
        #done if collide with floor
        floorHandle = sim.getObject("/Floor/box")
        
        floorCollide,collidingObjectHandles=sim.checkCollision(floorHandle,gripperHandle)
        
        if floorCollide == 1:
            reward -= 10
            done = True
        
        
        return observation, reward, done, {}, info

    
    def reset(self):
        sim = self.sim
        #observation = spaces.Dict({"headcamera":np.zeros((360,640,3)),"joints":np.zeros((8,1))})
        observation = {}#spaces.Dict({"headcamera":None,"joints":None})
        sim.stopSimulation()
        sleep(0.1)#give sim enough time to stop
        self.load()
        observation["headcamera"] = self.getheadimage()
        observation["wristcamera"] = self.getwristimage()
        observation["joints"] = self.getjointpositions()
        return observation, {} # reward, done, info can't be included

    
    def close(self):
        sim = self.sim
        sim.stopSimulation()
        return 0


    ###### HELPER METHODS ######
    def load(self): #connect to CoppeliaSim & load environment
        self.client = RemoteAPIClient('localhost',23000)
        self.sim = self.client.getObject('sim')
        if self.stepped == True:
            self.client.setStepping(True)
        self.sim.loadScene('/home/vlarko/rl-space-junk/space-sim.ttt')
        self.stochasticaddcubesat([0.5,0,0.5])
        self.sim.startSimulation()
        if self.fast == True:
            self.sim.setBoolParam(self.sim.boolparam_display_enabled,False)
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
        sim.setObjectInt32Param(block,sim.shapeintparam_respondable,1)
        sim.setObjectInt32Param(block,sim.shapeintparam_static,0)
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
    
    
    def getwristimage(self):
        sim = self.sim
        cam=sim.getObject("/Sawyer/wristCamera")
        sim.handleVisionSensor(cam)
        image,resolution=sim.getVisionSensorImg(cam)
        pixels = list(image)
        x = np.array(pixels, dtype=np.uint8)
        array = x.reshape(480,640,3)
        return array
    
    
    def getsim(self):
        sim = self.sim
        return sim
    
    
    def getjointpositions(self):
        sim = self.sim
        joint_positions = np.zeros((8,1))
        
        joint0 = sim.getObject("/Sawyer/joint")
        
        joint1 = sim.getObject("/Sawyer/link/joint")
        
        joint2 = sim.getObject("/Sawyer/link/joint/link/joint")
        
        joint3 = sim.getObject("/Sawyer/link/joint/link/joint/link/joint")
        
        joint4 = sim.getObject("/Sawyer/link/joint/link/joint/link/joint/link/joint")
        
        joint5 = sim.getObject("/Sawyer/link/joint/link/joint/link/joint/link/joint/link/joint")
        
        joint6 = sim.getObject("/Sawyer/link/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
        
        gripper = sim.getObject("/Sawyer/BaxterGripper/centerJoint")
        
        joint_positions[0]=sim.getJointPosition(joint0)
        joint_positions[1]=sim.getJointPosition(joint1)
        joint_positions[2]=sim.getJointPosition(joint2)
        joint_positions[3]=sim.getJointPosition(joint3)
        joint_positions[4]=sim.getJointPosition(joint4)
        joint_positions[5]=sim.getJointPosition(joint5)
        joint_positions[6]=sim.getJointPosition(joint6)
        joint_positions[7]=sim.getJointPosition(gripper)
        
        return joint_positions
 
