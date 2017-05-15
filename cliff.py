import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from cProfile import label

class cliffWalking():
    def __init__(self,start,end):
#         action = [left up right down]
#         action = [0 1 2 3]
        self.x = 12
        self.y = 4
        self.states = np.arange(self.x*self.y).reshape(self.y,self.x)
        self.action_space = [0,1,2,3]
        self.size = self.x*self.y
        self.done = False
        self.state = start
        self.endstate = end
        self.Q = np.zeros(len(self.action_space)*self.size).reshape(self.size,len(self.action_space))
        
    def _step(self, action,reward):
        reward -= 1
#         left
        pos = np.where(self.states == self.state)
        if (action==0):
            if (pos[1] == 0):
                self.state = self.state
#                 reward += 1
            else:
                newPos = list(pos)
                if (newPos[0] < 0):
                    newPos[0] = 0  
                newPos = tuple(newPos)
                self.state = int(self.states[newPos])
                self.state = self.state - 1
                
#         right
        if (action==2):
            if (pos[1] == self.x-1):
                self.state = self.state
#                 reward += 1
            else:
                newPos = list(pos)
                if (newPos[0] < 0):
                    newPos[0] = 0  
                newPos = tuple(newPos)
                self.state = int(self.states[newPos])
                self.state = self.state + 1
#         up
        if (action==1):
            if (pos[0] == 0):
                self.state = self.state
#                 reward += 1
            else:
                newPos = list(pos)
                newPos[0] -=1
                if (newPos[0] < 0):
                    newPos[0] = 0  
                newPos = tuple(newPos)
                self.state = int(self.states[newPos])
#         down
        if (action==3):
            if (pos[0] == self.y-1):
                self.state = self.state
#                 reward += 1
            else:
                newPos = list(pos)
                newPos[0] +=1
                if (newPos[0] < 0):
                    newPos[0] = 0  
                newPos = tuple(newPos)
                self.state = int(self.states[newPos])
          
        if (self.state == self.endstate):
            reward = 0
            self.done = True
        
#         print('new state: ',self.state, ', reward: ', reward)
        return self.state, reward, self.done
    
    def _reset(self):
        self.state = start
        self.done = False
        print('starting state: ', self.state)
    
    def _render(self):
        print('current state: ', self.state)
        
            
    def eGreedy(self,epsilon):
        m = 100
        n = epsilon * 100
        p = np.random.randint(m)
        if (p>(m-n-1)):
            epsilon = 1
        else:
            epsilon = 0
        return epsilon
    
c = cliffWalking(36,47)
c._step(3,0)
print(c.state)