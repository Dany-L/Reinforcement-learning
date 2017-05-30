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
        
        if (self.state == self.terminal_state[0] or self.state == self.terminal_state[1]):
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
   
st = 20     
c = ChainWalk(st)
# c.reset()
step = 5000
count = 0
samp =5
D = np.zeros(step*3*samp).reshape(samp*3,step)
for j in range(samp):
    c.reset()
    steps = []
    rew = []
    action = []
    steps.append(c.state)
    for i in range(step-1):
        a = np.random.randint(len(c.action_space))
        action.append(a)
        snew,r = c.move(a)
        steps.append(snew)
        rew.append(r)
    
    D[count,:] = steps
    D[count+1,0:step-1] = rew
    D[count+2,0:step-1] = action
    count += 3
    
# np.save(r'/home/jack/Documents/LiClipse Workspace/RL/ChainWalk_data/D_sample.npy',D) 


def plotChain(w,k):
    plt.figure(k)
    l = []
    r = []
    
    for i in range(c.num_states):
        s = i
        phi = np.matrix([1,s,s**2,s**3,s**4,1,s,s**2,s**3,s**4])
        l.append(float(phi[0,0:k/2]*w[0:k/2,0]))
        plt.plot(i,phi[0,0:k/2]*w[0:k/2,0],'xr')
        r.append(float(phi[0,k/2:k]*w[k/2:k,0]))
        plt.plot(i,phi[0,k/2:k]*w[k/2:k,0],'ob')
    plt.plot(range(st),l,'r',label='left')
    plt.plot(range(st),r,'b',label='right')
    plt.legend()
    
    

def getPhi(s,a):
    
    if a ==0:
        phiNew = np.matrix([1,s,s**2,s**3,s**4,0,0,0,0,0])
    if a ==1:
        phiNew = np.matrix([0,0,0,0,0,1,s,s**2,s**3,s**4])
        
    return phiNew

def LSTDQ(D,k,phi,gamma,w):
    A = np.zeros(k*k).reshape(k,k)
    b = np.zeros(k)
    
    count = 0
    
    for ep in range(samp):

        for st in range(step-1):
            
            s = D[count,st]
            r = D[count+1,st]
            a = int(D[count+2,st])
            snew = D[0,st+1]
#             anew = 0
            anew = np.argmax([getPhi(snew,0)[0,0:k/2]*w[0:k/2,0],getPhi(snew,1)[0,k/2:k]*w[k/2:k,0]])
            
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
        w_tick = LSTDQ(D[0:k/2,:],k,phi,gamma,w)
        plotChain(w_tick,count)
        print(w_tick) 
        c2 +=3
        count +=1
        if (np.linalg.norm(w_tick-w)<epsilon):
            break
    
    plt.show()
    
    return w_tick
        
k = 10
w_null = np.matrix(np.zeros(k)).T
epsilon = 1e-3
gamma = 0.9
phi = np.matrix(np.zeros(k))
        
w = LSPI(D,k,phi,gamma,epsilon,w_null)

# wopt = np.concatenate((w[0:3,0].T,w[3:6,0].T),axis=0)


# print(D)
# print(w)
        