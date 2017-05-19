import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def eGreedy(epsilon):
    n = np.random.randint(100)
    ep = epsilon*100
    if (ep>n):
        e = 1
    else:
        e = 0
    return e

def findState(P,s):
    minpos = minDelta(s,P,0)
    minvel = minDelta(s,P,1)
    
    posPos = np.where(P[:,0] == minpos)[0]
    posVel = np.where(P[:,1] == minvel)[0]
    
    for i in range(len(posPos)):
        for j in range(len(posVel)):
            if (posPos[i] == posVel[j]):
                pos = posPos[i]
    return pos
    
def minDelta(p,P,state):
    delta = []
    for i in range(len(P[:,state])):
        delta.append(abs(p[state]-P[i,state]))
    pos = delta.index(min(delta))
    min_dist = P[pos,state]
    return min_dist

def rearangeQ(z):
    Z = np.zeros(d*d).reshape(d,d)
    row = 0
    column = 0
    for i in range(len(z)):
        if (i%d == 0 and i>0):
            row +=1
            column = 0
        Z[row,column] = abs(z[i])
        column += 1
    return Z 

def plotQ(Q,k):
    z=[]
    for i in range(len(Q)):
        m=max(Q[i,:])
        z.append(m)
    fig = plt.figure(k)
    ax = fig.gca(projection='3d')
    X = np.linspace(-1.2,0.5, d)
    Y = np.linspace(-0.07,0.07,d)
    X,Y = np.meshgrid(X,Y)
    Z = rearangeQ(z)
    
    ax.plot_surface(X,Y,Z)
    
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_zlabel('max Q value')
    
    plt.show()
    
# Q = np.arange(1200).reshape(400,3)
# k = 1
# plotQ(Q,k)
# plt.show()  
        

def TabularQ(Q,P,episode,steps,epsilon,gamma,alpha,lamb):
    step_til_end = []
    for ep in range(episode):
        print('#episode:',ep)
        print('___________________________________________________________________')
        
        E = np.zeros(d*d*env.action_space.n).reshape(20*20,env.action_space.n)
        
        s = env._reset()
        s_d = findState(P,s)
        
        e = eGreedy(epsilon)
        l = list(Q[s_d,:])
        
        a = e*np.random.randint(env.action_space.n) + (1-e)*l.index(max(l))
        
        for st in range(steps):
            if (st%100==0):
                print('oldpos:', s[0],'oldvel:',s[1],'action:',a)
            snew,rnew,done,_ = env._step(a)
            if (st%100==0):
                print('newpos:', snew[0],'newvel:',snew[1],'done?:',done)
            env._render()
            s_d_new = findState(P, snew)
            
            e = eGreedy(epsilon)
            l = list(Q[s_d_new,:])
            
            anew = e*np.random.randint(env.action_space.n) + (1-e)*l.index(max(l))
            
            astar = l.index(max(l))
            
            sigma = rnew + gamma*Q[s_d_new,astar] - Q[s_d,a]
            E[s_d,a] = E[s_d,a]+1
            
            for i in range(len(Q[:,0])):
                Q[s_d,a] = Q[s_d,a] + alpha * sigma *E[s_d,a]
                if (astar == anew):
                    E[s_d,a] = gamma*lamb *E[s_d,a]
                else:
                    E[s_d,a] = 0
            
            if (snew[0]>= 0.5):
                if (ep%10 == 0 and ep > 0):
                    plotQ(Q,ep)
                step_til_end.append(st)
                print('geschafft in', st,'steps')
                break
                
            s = snew
            s_d = s_d_new
            a = anew
    
    return step_til_end

        
env=gym.make('MountainCar-v0')
d = 20

p_d = np.linspace(-1.2,0.5, d)
p_dot_d = np.linspace(-0.07,0.07,d)

P = np.zeros(d*d*2).reshape(d*d,2)
Q = np.zeros(d*d*env.action_space.n).reshape(20*20,env.action_space.n)

count = 0
for i in range(len(p_dot_d)):
    for j in range(len(p_d)):
        P[count,:] = np.array([p_d[j],p_dot_d[i]])
        count +=1
    

episode = 100
steps = int(1e5)
epsilon = 0.0
gamma = 0.99
alpha = 0.1
lamb = 0.8

step_til_end = TabularQ(Q,P,episode, steps, epsilon, gamma, alpha, lamb)

plt.figure(2)
plt.plot(range(episode),step_til_end)
plt.xlabel('#steps unit goal is reached')
plt.ylabel('#episodes')
plt.grid()
plt.show()
# for ep in range(episode):
#     print('#episode:',ep)
#     print('___________________________________________________________________')
#     step_til_end = []
#     E = np.zeros(d*d*env.action_space.n).reshape(20*20,env.action_space.n)
#     
#     s = env._reset()
#     s_d = findState(P,s)
#     
#     e = eGreedy(epsilon)
#     l = list(Q[s_d,:])
#     
#     a = e*np.random.randint(env.action_space.n) + (1-e)*l.index(max(l))
#     
#     for st in range(steps):
#         if (st%100==0):
#             print('oldpos:', s[0],'oldvel:',s[1],'action:',a)
#         snew,rnew,done,_ = env._step(a)
#         if (st%100==0):
#             print('newpos:', snew[0],'newvel:',snew[1],'done?:',done)
#         env._render()
#         s_d_new = findState(P, snew)
#         
#         e = eGreedy(epsilon)
#         l = list(Q[s_d_new,:])
#         
#         anew = e*np.random.randint(env.action_space.n) + (1-e)*l.index(max(l))
#         
#         astar = l.index(max(l))
#         
#         sigma = rnew + gamma*Q[s_d_new,astar] - Q[s_d,a]
#         E[s_d,a] = E[s_d,a]+1
#         
#         for i in range(len(Q[:,0])):
#             Q[s_d,a] = Q[s_d,a] + alpha * sigma *E[s_d,a]
#             if (astar == anew):
#                 E[s_d,a] = gamma*lamb *E[s_d,a]
#             else:
#                 E[s_d,a] = 0
#         
#         if (snew[0]>= 0.5):
#             step_til_end.append(st)
#             print('geschafft')
#             break
#             
#         s = snew
#         s_d = s_d_new
#         a = anew

     
#     a = 2
#     a = np.random.randint(env.action_space.n)
#     print('action:',a)
#     s,r,done,_ = env._step(a)
#     print('state:',s)
#     print('reward:',r)
#     env._render()
# print('lower bound',)
