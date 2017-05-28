import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class ChainWalk():
    
    def __init__(self,num):
        self.start_state = 0
        self.state = 0
        self.num_states = num
#         0 = left, 1 = right
        self.action_space = [0 ,1]
        self.terminal_state = [0,self.num_states-1]
        
    def move(self,a):
        
        success = self.RandAction()
        print('old state:',self.state,'action:',a,'success:',success)
        if (a == 0):
            if (self.state == self.terminal_state[0] and success):
                self.state  = 0
            elif (success):
                self.state -= 1
            elif (self.state == self.terminal_state[1]):
                self.state = self.terminal_state[1]
            else:
                self.state +=1
        else:
            if (self.state == self.terminal_state[1] and success):
                self.state = self.terminal_state[1]
            elif (success):
                self.state += 1
            elif (self.state == self.terminal_state[0]):
                self.state = self.terminal_state[0]
            else:
                self.state -=1
        
        if (self.state == 1 or self.state == 2):
            reward =1
        else:
            reward =0
        
        print('new state:',self.state,'reward:',reward)
        
        return self.state,reward
            

    def RandAction(self):
        p = 0.9
        n = np.random.randint(100)
        if (n<0.9*100):
            s = True
        else:
            s = False
        return s
    
    def reset(self):
        self.state = 0
        print('start state:',self.state)
        
c = ChainWalk(4)
# c.reset()
count = 0
Qsample = np.zeros(25*3*50).reshape(50*3,25)
for j in range(50):
    c.reset()
    steps = []
    rew = []
    action = []
    steps.append(c.state)
    for i in range(24):
        a = np.random.randint(len(c.action_space))
        action.append(a)
        snew,r = c.move(a)
        steps.append(snew)
        rew.append(r)
    
    Qsample[count,:] = steps
    Qsample[count+1,0:24] = rew
    Qsample[count+2,0:24] = action
    count += 3
    
np.save(r'/home/jack/Documents/LiClipse Workspace/RL/ChainWalk_data/Qsample.npy',Qsample) 
print(Qsample)

        