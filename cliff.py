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
        self.reward = 0
        self.states = np.arange(self.x*self.y).reshape(self.y,self.x)
        self.action_space = [0,1,2,3]
        self.size = self.x*self.y
        self.done = False
        self.state = start
        self.endstate = end
        self.Q = np.zeros(len(self.action_space)*self.size).reshape(self.size,len(self.action_space))
        
    def _step(self, action,reward):
        self.reward -= 1
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
        
        if (int(pos[0])==self.y-1):
            if (pos[1]>0 and pos[1]<self.x-1):
                self.reward = -100
                self.state = start
            
        if (self.state == self.endstate):
            self.reward = 0
            self.done = True
        
#         print('new state: ',self.state, ', reward: ', reward)
        return self.state, self.reward, self.done
    
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
    
start = 36
end = 47
episode = 10
step = 100000
alpha = 0.1
gamma = 1
epsilon = 0.1
c = cliffWalking(start,end)
[snew,rnew,done]=c._step(3,0)
print('new state:',snew,'reward:',rnew)
# print(c.state)

def rlcliff(c,episode,step,alpha,gamma,epsilon,algo):

    num2 = []
    count = []
    num_ges = []
    rnew = 0
    rew =[]

    if (algo==0):
        for j in range(episode):
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
            for i in range(step):
        #         print('old_state: ',c.state,' action: ', a, ' epsilon:', e)
                r = rnew
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
                    rew.append(r)
                    count.append(j)
                    num2.append(num)
                    print('terminal state: ', c.state,'steps: ',num)
                    break
         
             
            num = 0
        return c.Q,row,rew,count
    elif(algo == 1):
        for j in range(episode):
            num = 0
            row = []
            reward = 0
            #init a starting state s
            c._reset()
           
            row.append(c.state)
            for i in range(step):
                r = rnew
        #         select an action a from s using a plicy pi derived from Q (epsilon greedy)
                e = c.eGreedy(epsilon)
                l = list(c.Q[c.state,:])
                a = e*np.random.randint(len(c.action_space)) + (1-e)*l.index(max(l))
               
                s = c.state
        #         execute a, observe r, s'
                snew,rnew,done = c._step(a, reward)
                   
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
                    rew.append(r)
                    num2.append(num)
                    print('terminal state: ', c.state,'steps: ',num)
                    if (num < 18):
                        print(row)
                    break
           
               
            num = 0
        return c.Q,row,rew,count


def plotPath(row,color,algo,endstate):
    x=[]
    y=[]
    dx =[]
    dy = []
    for k in range(len(row)):
        pos = np.where(c.states == row[k])
        y.append(int(-pos[0]))
        x.append(int(pos[1]))
        if (k+1 < len(row)):
            posnext = np.where(c.states == row[k+1])
            dy=(int(-posnext[0])-y[k]) - (int(-posnext[0])-y[k])*0.6
            dx=(int(posnext[1])-x[k]) - (int(posnext[1])-x[k])*0.6
            plt.arrow(int(x[k]), int(y[k]), dx, dy,head_width=0.2, head_length=0.5, fc=color, ec=color)
       
    plt.plot(x,y,'ro',label = algo)
    plt.plot(0,-3,'cx',label = 'start')
    plt.plot(11,-3,'kx',label = 'original goal')
    plt.plot(endstate[1],-endstate[0],'gx',label = 'goal')   
    plt.grid()
    plt.legend()
    plt.plot(x,y,color)


Q,row,rew,count = rlcliff(c, episode, step, alpha, gamma, epsilon, 0)
Q_q,row_q,rew_q,count_q = rlcliff(c, episode, step, alpha, gamma, epsilon, 1)
print(row)
plt.plot(count,rew,label='SARSA')
plt.plot(count_q,rew_q,label='Q-learning')
plt.legend
# plotPath(row, 'r', 'SARSA',[11,3])
plt.legend()
plt.grid()
plt.show()


# For all comparisons, plot the learning graph having episode as x-axis, reward per episode as y-axis, smoothed by
# averaging the reward sums from 10 successive episodes.



# Q,row,num2,count,num_ges = rlwindy(c,episode, step, alpha, gamma, epsilon,0)
    