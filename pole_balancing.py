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
iter = 200

w = np.matrix(np.zeros(state_space))

def perform_rollout(w_init,perturbed):
    J = []
    Theta = np.matrix(np.zeros(state_space*episodes).reshape(episodes,state_space))
    for i in range(episodes):
        s  = env._reset()
        s = np.matrix(s)
        rew = []
        if perturbed:
            delta_w = np.random.uniform(-1,1,state_space)
            delta_w = np.matrix(delta_w)
            w = w_init + delta_w
            Theta[i,:] = delta_w
        else:
            w = w_init
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




def train_fix_alpha(w,alpha):
    J_k = []
    for i in range(iter):
        J_w_new,Theta = perform_rollout(w,True)
        J_w,_ = perform_rollout(w,False)
        
        Delta_J = J_w_new - J_w
        
        grad = np.linalg.inv(Theta.T*Theta)*Theta.T*Delta_J
        
        w = w + alpha * grad.T/np.linalg.norm(grad)
        J_k.append(np.mean(J_w))
        
    
    plt.figure(1)
    plt.plot(range(iter),J_k,label = "alpha = 0.5")
    plt.legend()
    plt.xlabel('iteration k')
    plt.ylabel('J(w_k)')
    plt.grid()
    plt.show()
    
def train_dyn_alpha(w):
    J_k = []
    for i in range(iter):
        J_w_new,Theta = perform_rollout(w,True)
        J_w,_ = perform_rollout(w,False)
        
        Delta_J = J_w_new - J_w
        
        grad = np.linalg.inv(Theta.T*Theta)*Theta.T*Delta_J
        alpha = 10.0/(i+1)
        w = w + alpha * grad.T/np.linalg.norm(grad)
        J_k.append(np.mean(J_w))
        
    
    plt.figure(1)
    plt.plot(range(iter),J_k,label = "alpha = 10/k")
    plt.legend()
    plt.xlabel('iteration k')
    plt.ylabel('J(w_k)')
    plt.grid()
    plt.show()
    
def train_adaptivly(w):
    J_k = []
    alpha = np.ones(state_space)*0.5
    gprev = np.zeros(state_space)
    alpha_min = 0.01
    alpha_max = 5
    for j in range(iter):
        J_w_new,Theta = perform_rollout(w,True)
        J_w,_ = perform_rollout(w,False)
        
        Delta_J = J_w_new - J_w
        
        grad = np.linalg.inv(Theta.T*Theta)*Theta.T*Delta_J
#         Do Rprop for calculating stepsize
        for i in range(state_space):
            if grad[i,0]*gprev[i]<0:
                alpha[i] = 1.2*alpha[i]
                w[0,i] = w[0,i]+alpha[i]*np.sign(grad[i,0])
                gprev[i] = grad[i]
            elif grad[i,0]*gprev[i]>0:
                alpha[i] = 0.5*alpha[i]
                w[0,i] = w[0,i]+alpha[i]*np.sign(grad[i,0])
                gprev[i] = 0
            else:
                w[0,i] = w[0,i]+alpha[i]*np.sign(grad[i,0])
                gprev[i] = grad[i]
            if alpha[i]<alpha_min:
                alpha[i]=alpha_min
            if alpha[i]>alpha_max:
                alpha[i] = alpha_max

        J_k.append(np.mean(J_w))
        
    
    plt.figure(1)
    plt.plot(range(iter),J_k,label = "alpha calulated by Rprop")
    plt.legend()
    plt.xlabel('iteration k')
    plt.ylabel('J(w_k)')
    plt.grid()
    plt.show()
# train_fix_alpha(w,alpha)
# train_dyn_alpha(w)
train_adaptivly(w)
    

    
    
    