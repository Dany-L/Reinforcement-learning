import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

d = 20

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

def calcQ(Theta,phi):
    Q = np.zeros(d*d*env.action_space.n).reshape(d*d,env.action_space.n)
    p_d = np.linspace(-1.2,0.5, d)
    p_dot_d = np.linspace(-0.07,0.07,d)
    
    count = 0
    for i in range(d):
        for j in range(d):
            s = np.array([p_d[j],p_dot_d[i]])
            Q[count,:]=sum(calcPhi(s,phi,c,sigma_p,sigma_v)*Theta[:,0])
                
            count +=1

    return Q

def plotQ(Theta,phi,k):
    Q = calcQ(Theta,phi)
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
    
def eGreedy(epsilon):
    n = np.random.randint(100)
    ep = epsilon*100
    if (ep>n):
        e = 1
    else:
        e = 0
    return e


def calcPhi(s,phi,c,sigma_p,sigma_v):
    phiNew = np.zeros(len(phi))
    d = np.zeros(len(phi)*len(s)).reshape(len(phi),len(s))
    for i in range(len(phi)):
        for j in range(len(s)):
            if (j==0):
                d[i,j] = float((s[j] - c[i,j]))**2/float(sigma_p)
            if (j==1):
                d[i,j] = float((s[j] - c[i,j]))**2/float(sigma_v)
        
        phiNew[i] = np.exp(-(d[i,0]+d[i,1])/2)
        
    return phiNew

def findMaxIndex(Q):
    m = []
    i = []
    for i in range(len(Q[:,0])):
        l = list(Q[i,:])
        m.append(max(l))
        i.append(l.index(max(l)))
    
    return i[m.index(max(m))],max(m)
            
                
    

def linearQ(Theta,c,phi,sigma_p,sigma_v,episode,steps,epsilon,gamma,alpha,lamb):
    steps_til_end=[]
    Q = np.zeros(len(Theta[0,:]))
    Qnew = np.zeros(len(Theta[0,:]))
    for ep in range(episode):
        print('#episode:',ep)
        print('___________________________________________________________________')
        e = np.zeros(len(Theta[:,0])*len(Theta[0,:])).reshape(len(Theta[:,0]),len(Theta[0,:]))
        s = env._reset()
        
        for st in range(steps):
            
            
            eps = eGreedy(epsilon)
            phi = calcPhi(s,phi,c,sigma_p,sigma_v)
            for i in range(len(Theta[0,:])):
                Q[i]=sum(phi*Theta[:,i])
#             print('Q for all actions:',Q)
            l = list(Q)
            astar = l.index(max(l))
            a = eps*np.random.randint(env.action_space.n) + (1-eps)*astar
            if (st%50 == 0):
                print('state:' ,s,'action:',a)
            if (not(a ==astar)):
                e[:,a] = np.zeros(len(Theta[:,0]))
            snew,rnew,done,_ = env._step(a)
            if (st%50 == 0):
                print('new state:',snew,'reward:',rnew)
#             print('______________________________________________')
#             env._render()
            
            e[:,a] = e[:,a] + calcPhi(s,phi,c,sigma_p,sigma_v)
            
            
            if (s[0]>=0.5):
                for i in range(len(Theta[0,:])):
                    for j in range(len(Theta[:,0])):
                        Theta[j,i] = Theta[j,i] + alpha*e[j,i]*(rnew-Q[i])
#                 if (ep%10 == 0 and ep > 0):
#                     plotQ(Theta,phi,ep)
                print('done in:',st,'steps')
                steps_til_end.append(st)
                break
            
            
            phiNew = calcPhi(snew,phi,c,sigma_p,sigma_v)
            for i in range(len(Theta[0,:])):
                Qnew[i]=sum(phiNew*Theta[:,i])
            l = list(Qnew)
            Qstar = max(l)
            
            for i in range(len(Theta[0,:])):
                for j in range(len(Theta[:,0])):
                    Theta[j,i] = Theta[j,i] + alpha*e[j,i]*(rnew+gamma*Qstar-Q[i])
            
            e = gamma*lamb*e
            s = snew
            
    return steps_til_end
        
        

env=gym.make('MountainCar-v0')

n_p = 4
n_v = 8
episode = 200
steps = int(1e5)
epsilon = 0.0
gamma = 0.99
alpha = 0.001
lamb = 0.9
sigma_p = 0.04
sigma_v = 0.0004
phi = np.zeros(n_p*n_v)
Theta = np.zeros(n_p*n_v*env.action_space.n).reshape(n_p*n_v,env.action_space.n)
c_p = np.linspace(-1.2, 0.5, n_p)
c_v = np.linspace(-0.07,0.07,n_v)
c = np.zeros(n_p*n_v*2).reshape(n_p*n_v,2)
count = 0
for i in range(len(c_v)):
    for j in range(len(c_p)):
        c[count,:] = np.array([c_p[j],c_v[i]])
        count +=1


step_til_end = linearQ(Theta, c, phi, sigma_p, sigma_v, episode, steps, epsilon, gamma, alpha, lamb)
np.save(r'/home/jack/Documents/LiClipse Workspace/RL/mountain_car_data/steps_til_end_rbf.npy',step_til_end) 

