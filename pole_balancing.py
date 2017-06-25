import gym
from gym.envs.registration import registry, register, make, spec
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# env=gym.make('CartPole-v1')

gym.envs.register(
        id='CartPole-v2',
        entry_point='env_mountain_car_noisy:GaussianNoiseCartPoleEnv',
        max_episode_steps=1000,
        reward_threshold=1000.0,
    )
  
env=gym.make('CartPole-v2')

episodes = 100
state_space = 4
sigma_squared = 1e-3
sigma = np.sqrt(sigma_squared)
gamma = 1
max_ep = 1000
rew = []
alpha = 0.5
iter = 200

w = np.matrix(np.zeros(state_space))

def perform_rollout(w_init,perturbed):
    J = []
    train_state = np.zeros(episodes*max_ep*state_space).reshape(episodes*state_space,max_ep)
    train_action = np.zeros(episodes*max_ep).reshape(episodes,max_ep)
    train_reward = np.zeros(episodes*max_ep).reshape(episodes,max_ep)
    train_iter = []
    Theta = np.matrix(np.zeros(state_space*episodes).reshape(episodes,state_space))
    count = 0
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
        for j in range(max_ep):
            train_state[count:count+state_space,j] = s      
            x = w*s.T
            a = np.random.normal(x,sigma)
            train_action[i,j] = a
            if a <0:
                a=0
            else:
                a = 1
            
            snew,r,done,_ = env._step(a)
            train_reward[i,j] = r
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
                train_iter.append(j)
                count+=4
                break
            s = np.matrix(snew)
        
            
    J = np.matrix(J).transpose()
#     print(J)
    return J,Theta,train_state,train_action,train_reward,train_iter


def train_vanilla_gradient(w,alpha):
    J_k = []
    delta_J_w = np.zeros(state_space)

    
    for j in range(iter):
        J_w,_,state,action,reward,train_iter = perform_rollout(w, False)
#         sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
        count = 0
        for m in range(episodes):
            sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
            grad_w = np.zeros(train_iter[m]*state_space).reshape(state_space,train_iter[m])
            
            for t in range(train_iter[m]):
                s = np.matrix(state[count:count+4,t])
                a = action[m,t]
                r = reward[m,t]
                
                grad_w[0:state_space,t] = (-w*s.T + a)*s/sigma_squared
                
            count += 4
            
            for k in range(state_space):
                sum_t[k,m] = sum(grad_w[k,:])*J_w[m,0]
        for k in range(state_space):
            delta_J_w[k] = 1/float(episodes)*sum(sum_t[k,:])
            
#         delta_J_w = np.matrix(delta_J_w)
        
        w = w + alpha*delta_J_w
        J_k.append(np.mean(J_w))
    
    
    
    return w,J_k

def train_vanilla_gradient_dyn(w):
    J_k = []
    delta_J_w = np.zeros(state_space)

    
    for j in range(iter):
        J_w,_,state,action,reward,train_iter = perform_rollout(w, False)
#         sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
        count = 0
        for m in range(episodes):
            sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
            grad_w = np.zeros(train_iter[m]*state_space).reshape(state_space,train_iter[m])
            
            for t in range(train_iter[m]):
                s = np.matrix(state[count:count+4,t])
                a = action[m,t]
                r = reward[m,t]
                
                grad_w[0:state_space,t] = (-w*s.T + a)*s/sigma_squared
                
            count += 4
            
            for k in range(state_space):
                sum_t[k,m] = sum(grad_w[k,:])*J_w[m,0]
        for k in range(state_space):
            delta_J_w[k] = 1/float(episodes)*sum(sum_t[k,:])
            
#         delta_J_w = np.matrix(delta_J_w)
        alpha = 10.0/(j+1)
        w = w + alpha*delta_J_w
        J_k.append(np.mean(J_w))
    
    
    return w,J_k

def train_vanilla_gradient_ad(w):
    J_k = []
    alpha = np.ones(state_space)*0.5
    wprev = np.zeros(state_space)
    alpha_min = 0.01
    alpha_max = 5
    delta_J_w = np.zeros(state_space)
    
    for j in range(iter):
        J_w,_,state,action,reward,train_iter = perform_rollout(w, False)
#         sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
        count = 0
        for m in range(episodes):
            sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
            grad_w = np.zeros(train_iter[m]*state_space).reshape(state_space,train_iter[m])
            
            for t in range(train_iter[m]):
                s = np.matrix(state[count:count+4,t])
                a = action[m,t]
                r = reward[m,t]
                
                grad_w[0:state_space,t] = (-w*s.T + a)*s/sigma_squared
                
            count += 4
            
            for k in range(state_space):
                sum_t[k,m] = sum(grad_w[k,:])*J_w[m,0]
        for k in range(state_space):
            delta_J_w[k] = 1/float(episodes)*sum(sum_t[k,:])
            
#         delta_J_w = np.matrix(delta_J_w)
#         Do Rprop for calculating stepsize
        for i in range(state_space):
            if w[0,i]*wprev[i]<0:
                alpha[i] = 1.2*alpha[i]
                w[0,i] = w[0,i]+alpha[i]*np.sign(w[0,i])
                wprev[i] = w[0,i]
            elif w[0,i]*wprev[i]>0:
                alpha[i] = 0.5*alpha[i]
                w[0,i] = w[0,i]+alpha[i]*np.sign(w[0,i])
                wprev[i] = 0
            else:
                w[0,i] = w[0,i]+alpha[i]*np.sign(w[0,i])
                wprev[i] = w[0,i]
            if alpha[i]<alpha_min:
                alpha[i]=alpha_min
            if alpha[i]>alpha_max:
                alpha[i] = alpha_max
                
        J_k.append(np.mean(J_w))
    
    
    return w,J_k
        
def train_REINFORCE_gradient(w,alpha):
    J_k = []
    delta_J_w = np.zeros(state_space)

    
    for j in range(iter):
        J_w,_,state,action,reward,train_iter = perform_rollout(w, False)
#         sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
        count = 0
        for m in range(episodes):
            sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
            grad_w = np.zeros(train_iter[m]*state_space).reshape(state_space,train_iter[m])
            
            for t in range(train_iter[m]):
                s = np.matrix(state[count:count+4,t])
                a = action[m,t]
                r = reward[m,t]
                
                grad_w[0:state_space,t] = (-w*s.T + a)*s/sigma_squared
                
            count += 4
            
            for k in range(state_space):
                b = sum(grad_w[k,0:train_iter[m]-1])**2*J_w[m,0]/sum(grad_w[k,0:train_iter[m]-1])
                sum_t[k,m] = sum(grad_w[k,:])*(J_w[m,0]-b)
        for k in range(state_space):
            delta_J_w[k] = 1/float(episodes)*sum(sum_t[k,:])
            
#         delta_J_w = np.matrix(delta_J_w)
        
        w = w + alpha*delta_J_w
        J_k.append(np.mean(J_w))
    
    
    return w,J_k

def train_REINFORCE_gradient_dyn(w):
    J_k = []
    delta_J_w = np.zeros(state_space)

    
    for j in range(iter):
        J_w,_,state,action,reward,train_iter = perform_rollout(w, False)
#         sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
        count = 0
        for m in range(episodes):
            
            grad_w = np.zeros(train_iter[m]*state_space).reshape(state_space,train_iter[m])
            sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
            for t in range(train_iter[m]):
                s = np.matrix(state[count:count+4,t])
                a = action[m,t]
                r = reward[m,t]
                
                grad_w[0:state_space,t] = (-w*s.T + a)*s/sigma_squared
                
            count += 4
            
            for k in range(state_space):
                b = sum(grad_w[k,0:train_iter[m]-1])**2*J_w[m,0]/sum(grad_w[k,0:train_iter[m]-1])
                sum_t[k,m] = sum(grad_w[k,:])*(J_w[m,0]-b)
        for k in range(state_space):
            delta_J_w[k] = 1/float(episodes)*sum(sum_t[k,:])
            
#         delta_J_w = np.matrix(delta_J_w)
        alpha = 10.0/(j+1)
        w = w + alpha*delta_J_w
        J_k.append(np.mean(J_w))
    
    
    return w,J_k

def train_REINFORCE_gradient_ad(w):
    J_k = []
    delta_J_w = np.zeros(state_space)
    alpha = np.ones(state_space)*0.5
    wprev = np.zeros(state_space)
    alpha_min = 0.01
    alpha_max = 5
    
    for j in range(iter):
        J_w,_,state,action,reward,train_iter = perform_rollout(w, False)
#         sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
        count = 0
        for m in range(episodes):
            sum_t = np.zeros(state_space*episodes).reshape(state_space,episodes)
            grad_w = np.zeros(train_iter[m]*state_space).reshape(state_space,train_iter[m])
            
            for t in range(train_iter[m]):
                s = np.matrix(state[count:count+4,t])
                a = action[m,t]
                r = reward[m,t]
                
                grad_w[0:state_space,t] = (-w*s.T + a)*s/sigma_squared
                
            count += 4
            
            for k in range(state_space):
                b = sum(grad_w[k,0:train_iter[m]-1])**2*J_w[m,0]/sum(grad_w[k,0:train_iter[m]-1])
                sum_t[k,m] = sum(grad_w[k,:])*(J_w[m,0]-b)
        for k in range(state_space):
            delta_J_w[k] = 1/float(episodes)*sum(sum_t[k,:])
            
#         delta_J_w = np.matrix(delta_J_w)
#         Do Rprop for calculating stepsize
        for i in range(state_space):
            if w[0,i]*wprev[i]<0:
                alpha[i] = 1.2*alpha[i]
                w[0,i] = w[0,i]+alpha[i]*np.sign(w[0,i])
                wprev[i] = w[0,i]
            elif w[0,i]*wprev[i]>0:
                alpha[i] = 0.5*alpha[i]
                w[0,i] = w[0,i]+alpha[i]*np.sign(w[0,i])
                wprev[i] = 0
            else:
                w[0,i] = w[0,i]+alpha[i]*np.sign(w[0,i])
                wprev[i] = w[0,i]
            if alpha[i]<alpha_min:
                alpha[i]=alpha_min
            if alpha[i]>alpha_max:
                alpha[i] = alpha_max
        J_k.append(np.mean(J_w))
    
    
    return w,J_k

def train_fix_alpha(w,alpha):
    J_k = []
    for i in range(iter):
        J_w_new,Theta = perform_rollout(w,True)
        J_w,_ = perform_rollout(w,False)
        
        Delta_J = J_w_new - J_w
        
        grad = np.linalg.inv(Theta.T*Theta)*Theta.T*Delta_J
        
        w = w + alpha * grad.T/np.linalg.norm(grad)
        J_k.append(np.mean(J_w))
        
        if np.mean(J_w)>=500:
            return w
        
    
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
    
    return w
    
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
    
# def run_environment(w):
#     s = env._reset()
#     s = np.matrix(s)
#     step = 0
#     while True:
#         x = w*s.T
#         a = np.random.normal(x,sigma)
#         
#         if a <0:
#             a=0
#         else:
#             a = 1
#         
#         snew,r,done,_ = env._step(a)
#         env._render()
#         
#         if done:
#             print('survived %i steps'%(step))
#             break
#         
#         s = np.matrix(snew)
#         step +=1
            
            
        
    
# wnew = train_fix_alpha(w,alpha)
# wnew = train_dyn_alpha(w)
# run_environment(wnew)
# train_adaptivly(w)


runs = 10
J_k_static_vanilla = np.zeros(runs*iter).reshape(runs,iter)
J_k_dyn_vanilla = np.zeros(runs*iter).reshape(runs,iter)
J_k_ad_vanilla = np.zeros(runs*iter).reshape(runs,iter)

J_k_static_REINFORCE = np.zeros(runs*iter).reshape(runs,iter)
J_k_dyn_REINFORCE = np.zeros(runs*iter).reshape(runs,iter)
J_k_ad_REINFORCE = np.zeros(runs*iter).reshape(runs,iter)

for k in range(runs):
    w = np.matrix(np.zeros(state_space))
    w_new_stat,J_k_static_vanilla_tmp = train_vanilla_gradient(w, 0.2)
    J_k_static_vanilla[k,:] = J_k_static_vanilla_tmp
    
    w_new_dyn,J_k_dyn_vanilla_tmp = train_vanilla_gradient_dyn(w)
    J_k_dyn_vanilla[k,:]=J_k_dyn_vanilla_tmp
    
    w_new_ad,J_k_ad_vanilla_tmp = train_vanilla_gradient_ad(w)
    J_k_ad_vanilla[k,:]=J_k_ad_vanilla_tmp
    
    w_new_stat_RF,J_k_static_REINFORCE_tmp=train_REINFORCE_gradient(w, 0.2)
    J_k_static_REINFORCE[k,:] = J_k_static_REINFORCE_tmp
    
    w_new_dyn_RF,J_k_dyn_REINFORCE_tmp=train_REINFORCE_gradient_dyn(w)
    J_k_dyn_REINFORCE[k,:] = J_k_dyn_REINFORCE_tmp

    w_new_ad_RF,J_k_ad_REINFORCE_tmp=train_REINFORCE_gradient_ad(w)
    J_k_ad_REINFORCE[k,:] = J_k_ad_REINFORCE_tmp

mean_J_static_vanilla = []
mean_J_dyn_vanilla = []
mean_J_ad_vanilla = []
mean_J_static_RF = []
mean_J_dyn_RF = []
mean_J_ad_RF = []

for i in range(iter):
    mean_J_static_vanilla.append(np.mean(J_k_static_vanilla[:,i]))
    mean_J_dyn_vanilla.append(np.mean(J_k_dyn_vanilla[:,i]))
    mean_J_ad_vanilla.append(np.mean(J_k_ad_vanilla[:,i]))
    
    mean_J_static_RF.append(np.mean(J_k_static_REINFORCE[:,i]))
    mean_J_dyn_RF.append(np.mean(J_k_dyn_REINFORCE[:,i]))
    mean_J_ad_RF.append(np.mean(J_k_ad_REINFORCE[:,i]))
    
    
plt.figure(1)
plt.plot(range(iter),mean_J_static_vanilla,label = "alpha ="+str(0.2))
plt.plot(range(iter),mean_J_dyn_vanilla,label = "alpha = 10/k")
plt.plot(range(iter),mean_J_ad_vanilla,label="alpha calc. by Rprop")
plt.title('Vanilla')
plt.legend()
plt.xlabel('iteration k')
plt.ylabel('mean over '+str(runs)+' iteration: J(w_k)')
plt.grid()

plt.figure(2)
plt.plot(range(iter),mean_J_static_RF,label = 'alpha='+str(0.2))
plt.plot(range(iter),mean_J_dyn_RF,label = "alpha = 10/k")
plt.plot(range(iter),mean_J_ad_RF,label="alpha calc. by Rprop")
plt.title('REINFORCE')
plt.legend()
plt.xlabel('iteration k')
plt.ylabel('mean over '+str(runs)+' iteration: J(w_k)')
plt.grid()
plt.show()

    
    
    