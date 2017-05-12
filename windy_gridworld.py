import numpy as np

class windyGridworld():

    def __init__(self,start):
#         action = [left up right down]
#         action = [0 1 2 3]
        self.states = np.arange(70).reshape(7,10)
        self.action_space = [0,1,2,3]
        self.size = 70
        self.done = False
        self.state = start
        self.Q = np.zeros(70).reshape(7,10)
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
          
        if (self.state == 37):
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
        
    def renderFcn (self):
        pos = np.where(self.states == self.state)
        pos = list(pos)
#         left
        pos[1] -= 1
        if (pos[1]<0):
            pos[1] = 0
        l = self.Q[tuple(pos)]
        pos = np.where(self.states == self.state)
        pos = list(pos)
#         right
        pos[1] += 1
        if (pos[1]>9):
            pos[1] = 9
        r = self.Q[tuple(pos)]
        pos = np.where(self.states == self.state)
        pos = list(pos)
#         down
        pos[0] += 1
        if (pos[0]>6):
            pos[0] = 6
        d = self.Q[tuple(pos)]
        pos = np.where(self.states == self.state)
        pos = list(pos)
#         up
        pos[0] -= 1
        if (pos[0]<0):
            pos[0] = 0
        u = self.Q[tuple(pos)]
        pos = np.where(self.states == self.state)
        pos = list(pos)
              
        direction = [l,u,r,d]
        
        return direction.index(max(direction))
            
    def eGreedy(self):
        n = np.random.randint(10)
        if (n>8):
            epsilon = 1
        else:
            epsilon = 0
        return epsilon
        
start = 30
c = windyGridworld(start)
c._render()
reward = 0
episode = 10000
step = 100
alpha = 0.1
gamma = 1
num = 0
e = c.eGreedy()

for j in range(step):
    
    row = []
    c._reset()
    a = e*np.random.randint(len(c.action_space)) + (1-e)*c.renderFcn()
    row.append(c.state)
    for i in range(episode):
        print('old_state: ',c.state,' action: ', a, ' epsilon:', e)
        snew,rnew,done = c._step(a, reward)
        row.append(snew)
        
        e = c.eGreedy()
        anew = e*np.random.randint(len(c.action_space)) + (1-e)*c.renderFcn()
        pos = np.where(c.states == c.state)
        posnew = np.where(c.states == snew)
        
        c.Q[pos] = c.Q[pos] + alpha * (rnew + gamma*c.Q[posnew] - c.Q[pos])
        
        a = anew
        num += 1
#         print(c.Q)
        if done:
            print('terminal state: ', c.state,'steps: ',num)
            break
    
#     c.done = False
    num = 0
    
#     print('----------------------')
#     print(c.Q)

print(row)
#     
# print('a',a)
# iter = 10000
# snew,rnew,done = c._step(2, reward)



# for i in range(iter):
#     a = np.random.randint(len(c.action_space))
#     print('old_state',c.state,', action: ',a)
#     snew,rnew,done = c._step(a, reward)
# #     print('old sate: ', c.state ,', new state: ',snew,' reward: ',rnew, ' action: ', a)
#     if done:
#         print('goal sate: ',snew,' total_reward: ',rnew, ' action: ', a)
#         break
#     reward = rnew
    



        