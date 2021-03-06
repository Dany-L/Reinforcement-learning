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
#         print('old state:',self.state,'action:',a,'success:',success)
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
        
#         print('new state:',self.state,'reward:',reward)
        
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
        self.state = np.random.randint(len(self.action_space))
#         self.state = 0
#         print('start state:',self.state)
        
c = ChainWalk(4)
# c.reset()
count = 0
D = np.zeros(25*3*50).reshape(50*3,25)
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
    
    D[count,:] = steps
    D[count+1,0:24] = rew
    D[count+2,0:24] = action
    count += 3
    
# np.save(r'/home/jack/Documents/LiClipse Workspace/RL/ChainWalk_data/D_sample.npy',D) 


def plotChain(w,k):
    plt.figure(k)
    l = []
    r = []
    
    for i in range(c.num_states):
        s = i
        phi = np.matrix([1,s,s**2,1,s,s**2])
        l.append(float(phi[0,0:3]*w[0:3,0]))
        plt.plot(i+1,phi[0,0:3]*w[0:3,0],'xr')
        r.append(float(phi[0,3:6]*w[3:6,0]))
        plt.plot(i+1,phi[0,3:6]*w[3:6,0],'ob')
    s = np.array([1,2,3,4])
    plt.plot(s,l,'r',label='left')
    plt.plot(s,r,'b',label='right')
    plt.legend()
    
    

def getPhi(s,a):
    
    if a ==0:
        phiNew = np.matrix([1,s,s**2,0,0,0])
    if a ==1:
        phiNew = np.matrix([0,0,0,1,s,s**2])
        
    return phiNew

def LSTDQ(D,k,phi,gamma,w):
    A = np.zeros(k*k).reshape(k,k)
    b = np.zeros(k)
    
    count = 0
    
    for ep in range(50):

        for st in range(24):
            
            s = D[count,st]
            r = D[count+1,st]
            a = int(D[count+2,st])
            snew = D[0,st+1]
            anew = np.argmax([getPhi(snew,0)[0,0:3]*w[0:3,0],getPhi(snew,1)[0,3:6]*w[3:6,0]])
            
            A = A+ np.transpose(getPhi(s,a))*(getPhi(s,a)-gamma*getPhi(snew,anew))
            b = b+ getPhi(s,a)*r
        
    count += 3
    w_pi = np.linalg.inv(A)*np.transpose(b)
    w = w_pi

    return w_pi

def LSPI(D,k,phi,gamma,epsilon,w_null):
    w_tick = w_null
    count = 1
    c2 = 0
    while True:
#     for i in range(7):
        print('iter:',count)
        w = w_tick
        w_tick = LSTDQ(D[0:3,:],k,phi,gamma,w)
        plotChain(w_tick,count)
        print(w_tick) 
        c2 +=3
        count +=1
        if (np.linalg.norm(w_tick-w)<epsilon):
            break
    
    plt.show()
    
    return w_tick
        
k = 6
w_null = np.matrix(np.zeros(k)).T
epsilon = 1e-3
gamma = 0.9
phi = np.matrix(np.zeros(k))
        
w = LSPI(D,k,phi,gamma,epsilon,w_null)

# wopt = np.concatenate((w[0:3,0].T,w[3:6,0].T),axis=0)


# print(D)
# print(w)
        