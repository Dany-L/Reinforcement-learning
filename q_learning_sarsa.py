import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from cProfile import label

class windyGridworld():

    def __init__(self,start,end):
#         action = [left up right down]
#         action = [0 1 2 3]
        self.states = np.arange(70).reshape(7,10)
        self.action_space = [0,1,2,3]
        self.size = 70
        self.done = False
        self.state = start
        self.endstate = end
        self.Q = np.zeros(len(self.action_space)*self.size).reshape(self.size,len(self.action_space))
#         self.Q = np.zeros(len(self.action_space)*self.size).reshape(self.size,len(self.action_space))
#         self.Q_q = np.zeros(70).reshape(7,10)
#         self.Q = np.array([[ -2.55000000e+01,  -3.67000000e+01,  -6.73000000e+01,  -2.16280000e+03, -4.20800000e+03,  -5.67300000e+03,  -6.38580000e+03,  -5.29090000e+03 ,-2.66340000e+03,  -8.64000000e+01],[ -2.55000000e+01,  -2.55000000e+01,  -3.66000000e+01,  -2.76000000e+01, -2.19300000e+02,  -1.75900000e+02,  -1.04000000e+01,  -7.47000000e+01 ,-7.84000000e+01,  -8.60000000e+01],[ -2.54000000e+01,  -2.54000000e+01,  -2.55000000e+01,  -2.58000000e+01, -1.13000000e+01,  -9.80000000e+00,  -8.40000000e+00,  -1.86000000e+01, -3.92000000e+01 , -7.82000000e+01],[ -2.52000000e+01,  -2.54000000e+01,  -2.52000000e+01 , -2.81000000e+01, -1.50800000e+02,  -9.67000000e+01,  -2.27000000e+01,   0.00000000e+00 ,-1.99000000e+01  ,-3.92000000e+01],[ -2.52000000e+01,  -2.51000000e+01,  -2.50000000e+01,  -2.40000000e+01, -1.93000000e+01,  -2.08000000e+01,   0.00000000e+00,  -4.60000000e+00, -1.02000000e+01,  -1.99000000e+01],[ -2.49000000e+01,  -2.46000000e+01,  -2.42000000e+01,  -2.39000000e+01, -2.20000000e+01,   0.00000000e+00,   0.00000000e+00,  -4.50000000e+00, -4.90000000e+00,  -1.01000000e+01],[ -2.44000000e+01,  -2.44000000e+01 , -2.41000000e+01 , -2.28000000e+01, 0.00000000e+00  , 0.00000000e+00   ,0.00000000e+00  , 0.00000000e+00 ,-4.50000000e+00  ,-4.90000000e+00]])

        
    def _step(self, action,reward):
        reward -= 1
#         left
        pos = np.where(self.states == self.state)
        if (action==0):
            if (pos[1] == 0):
                self.state = self.state
#                 reward += 1
            else:
                w = self.wind(pos)
                newPos = list(pos)
                newPos[0] -= w    
                if (newPos[0] < 0):
                    newPos[0] = 0  
                newPos = tuple(newPos)
                self.state = int(self.states[newPos])
                self.state = self.state - 1
                
#         right
        if (action==2):
            if (pos[1] == 9):
                self.state = self.state
#                 reward += 1
            else:
                w = self.wind(pos)
                newPos = list(pos)
                newPos[0] -= w
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
                w = self.wind(pos)
                newPos = list(pos)
                newPos[0] -= w
                newPos[0] -=1
                if (newPos[0] < 0):
                    newPos[0] = 0  
                newPos = tuple(newPos)
                self.state = int(self.states[newPos])
#         down
        if (action==3):
            if (pos[0] == 6):
                self.state = self.state
#                 reward += 1
            else:
                w = self.wind(pos)
                newPos = list(pos)
                newPos[0] -= w
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
        
    def wind(self,pos):
        w = 0
        if (int(pos[1])==3 or int(pos[1])==4 or int(pos[1])==5 or int(pos[1])==8):
            w=1
        elif (int(pos[1])==6 or int(pos[1])==7):
            w = 2
        return w
            
    def eGreedy(self,epsilon):
        m = 100
        n = epsilon * 100
        p = np.random.randint(m)
        if (p>(m-n-1)):
            epsilon = 1
        else:
            epsilon = 0
        return epsilon
        
start = 30
end = 37
c = windyGridworld(start,end)
c._render()
episode = 800000
step = 100
alpha = 0.1
gamma = 1
epsilon = 0.1
num = 0
num_ges = []
num2 = []
count = []

# algo = 0 -> SARSA algo = 1 -> Q-learning
def rlwindy(episode,step,alpha,gamma,epsilon,algo):
    if (algo==0):
        for j in range(step):
            num = 0
            row = []
            reward = 0
             
            #init a starting state s
            c._reset()
            #select an action a from s using a policy pi derived from Q e-greedy
            e = c.eGreedy(epsilon)
            l = list(c.Q[c.state,:])
            a = e*np.random.randint(len(c.action_space)) + (1-e)*l.index(max(l))
             
            row.append(c.state)
            for i in range(episode):
        #         print('old_state: ',c.state,' action: ', a, ' epsilon:', e)
         
                s = c.state
                #execute a observe r,s'
                snew,rnew,done = c._step(a, reward)
                row.append(snew)
                 
                #Derive pi from Q, then selection action a' from s'
                e = c.eGreedy(epsilon)
                l = list(c.Q[c.state,:])
                anew = e*np.random.randint(len(c.action_space)) + (1-e)*l.index(max(l))
                 
                #update Q_q
                c.Q[s,a] = c.Q[s,a] + alpha * (rnew + gamma*c.Q[snew,anew] - c.Q[s,a])
                 
                #s = s' and a = a'
                a = anew
                 
                num += 1
        #         print(c.Q)
                if done:
                    if (j>0):
                        num_ges.append(num_ges[j-1]+num)
                    else:
                        num_ges.append(num)
                    count.append(j)
                    num2.append(num)
                    print('terminal state: ', c.state,'steps: ',num)
                    if (num < 18):
                        print(row)
                    break
         
             
            num = 0
        return c.Q,row,num2,count
    elif(algo == 1):
        for j in range(step):
            num = 0
            row = []
            reward = 0
            #init a starting state s
            c._reset()
           
               
            row.append(c.state)
            for i in range(episode):
                   
        #         select an action a from s using a plicy pi derived from Q (epsilon greedy)
                e = c.eGreedy(epsilon)
                l = list(c.Q[c.state,:])
                a = e*np.random.randint(len(c.action_space)) + (1-e)*l.index(max(l))
               
                s = c.state
        #         execute a, observe r, s'
                snew,r,done = c._step(a, reward)
                   
                row.append(snew)
                   
                #update Q
                c.Q[s,a] = c.Q[s,a] + alpha*(r + gamma*max(c.Q[snew,:]) - c.Q[s,a])
                   
                #s = s' and a = a'
                           
                num += 1
        #         print(c.Q)
                if done:
                    if (j>0):
                        num_ges.append(num_ges[j-1]+num)
                    else:
                        num_ges.append(num)
                    count.append(j)
                    num2.append(num)
                    print('terminal state: ', c.state,'steps: ',num)
                    if (num < 18):
                        print(row)
                    break
           
               
            num = 0
        return c.Q,row,num2,count

Q,row,num2,count = rlwindy(episode, step, alpha, gamma, epsilon,0)
# 1b)
#-------------------------------------------------------------------------------
x=[]
y=[]
dx =[]
dy = []
plt.subplot(211)
for k in range(len(row)):
    pos = np.where(c.states == row[k])
    y.append(int(-pos[0]))
    x.append(int(pos[1]))
    if (k+1 < len(row)):
        posnext = np.where(c.states == row[k+1])
        dy=(int(-posnext[0])-y[k]) - (int(-posnext[0])-y[k])*0.6
        dx=(int(posnext[1])-x[k]) - (int(posnext[1])-x[k])*0.6
        plt.arrow(int(x[k]), int(y[k]), dx, dy,head_width=0.2, head_length=0.5, fc='r', ec='r')
  
plt.plot(x,y,'ro',label = "SARSA")
plt.plot(0,-3,'cx',label = 'start')
plt.plot(7,-3,'gx',label = 'goal')   
plt.grid()
plt.legend()
plt.plot(x,y,'r')
#-------------------------------------------------------------------------------
# 1a)
# plt.plot(count,num2,label="SARSA")
# plt.plot(num_ges,count,label='SARSA')


c=windyGridworld(start,end)
c._render()
# episode = 800000
# step = 100
# alpha = 0.1
# gamma = 1
# epsilon = 0.1
# num = 0
# num_ges = []
# num2 = []
# count = []

Q_q,row_q,num2_q,count_q = rlwindy(episode, step, alpha, gamma, epsilon,1)

# 1b)
#-------------------------------------------------------------------------------
x=[]
y=[]
dx =[]
dy = []
plt.subplot(212)
for k in range(len(row_q)):
    pos = np.where(c.states == row_q[k])
    y.append(int(-pos[0]))
    x.append(int(pos[1]))
    if (k+1 < len(row_q)):
        posnext = np.where(c.states == row_q[k+1])
        dy=(int(-posnext[0])-y[k]) - (int(-posnext[0])-y[k])*0.6
        dx=(int(posnext[1])-x[k]) - (int(posnext[1])-x[k])*0.6
        plt.arrow(int(x[k]), int(y[k]), dx, dy,head_width=0.2, head_length=0.5, fc='b', ec='b')
plt.plot(x,y,'bo',label="Q-learning")
plt.plot(0,-3,'cx',label = 'start')
plt.plot(7,-3,'gx',label = 'goal')   
plt.grid()
# ------------------------------------------------------------------------------
# 1a)
# plt.plot(count_q,num2_q,label="Q-learning")
# plt.xlabel('#episodes')
# plt.ylabel('#steps')
# plt.plot(num_ges,count,label="Q-learning")
 
plt.legend()
plt.show()
       