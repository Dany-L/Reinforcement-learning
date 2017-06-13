import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

env=gym.make('CartPole-v1')


episodes = 50
state_space = 4
sigma_squared = 1e-3
gamma = 1
max_ep = 1000
rew = []
alpha = 0.5
iter = 60

w = np.matrix(np.zeros(state_space))

def perform_rollout(w,perturbed):
    J = []
    Theta = np.matrix(np.zeros(state_space*episodes).reshape(episodes,state_space))
    for i in range(episodes):
        s  = env._reset()
        s = np.matrix(s)
        rew = []
        if perturbed:
            delta_w = np.random.uniform(-1,1,state_space)
            delta_w = np.matrix(delta_w)
            w = w + delta_w
            Theta[i,:] = delta_w
        while True:            
            x = w*s.T
            a = np.random.normal(x,sigma_squared)
            
            if a <0:
                a=0
            else:
                a = 1
            
            snew,r,done,_ = env._step(a)
            if done:
                r=-1   
#             env._render()
            snew = np.matrix(snew)
            rew.append(r)
            
            if done:
                J_w = 0
                for i in range(len(rew)):
                    J_w += gamma**i*rew[i]
                J.append(J_w)
                break
            s = np.matrix(snew)
            
    J = np.matrix(J).transpose()
#     print(J)
    return J,Theta


J_k = []

for i in range(iter):
    J_w_new,Theta = perform_rollout(w,True)
    J_w,_ = perform_rollout(w,False)
    
    Delta_J = J_w_new - J_w
    
    grad = np.linalg.inv(Theta.T*Theta)*Theta.T*Delta_J
    
    w = w + alpha * grad.T/np.linalg.norm(grad)
    
    J_k.append(np.mean(J_w))
    
s = env._reset()
s = np.matrix(s)
while True:
    x = w*s.T
    a = np.random.normal(x,sigma_squared)
    if a <0:
        a=0
    else:
        a = 1
    snew,r,done,_ = env._step(a)
    env._render()
    if done:
        break
    s = np.matrix(snew)

plt.figure(1)
plt.plot(range(iter),J_k,label = "mean of J(w_k)")
plt.legend()
plt.xlabel('iteration k')
plt.ylabel('J(w_k)')
plt.grid()
plt.show()
    
    
    
    
    