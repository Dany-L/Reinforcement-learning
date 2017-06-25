import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class Four_room_domain():
    
    def __init__(self,start,goal):
        self.state = start
        self.room = 0
        self.states = np.arange(13*13).reshape(13,13)
        
        self.hallways = [45,80,136,100]
        
        self.h = {}
        self.h['hallway_room1'] = [45,80]
        self.h['hallway_room2'] = [80,136]
        self.h['hallway_room3'] = [45,100]
        self.h['hallway_room4'] = [100,136]        
        
        self.current_room = 0
        self.size_room = []
        
        self.size1 = [5,5]
        self.size2 = [5,5]
        self.size3 = [5,6]
        self.size4 = [5,4]
        
        self.w = {}
        #room 1 and 2
        self.room1 = np.zeros(self.size1[0]*self.size1[1]).reshape(self.size1[1],self.size1[0])
        self.room2 = np.zeros(self.size2[0]*self.size2[1]).reshape(self.size2[1],self.size2[0])
        startr1 = 14
        startr2 = 92
        count = 0
        for i in range(self.size1[1]):
            self.room1[i,:] = np.arange(startr1+count,startr1+count+self.size1[0])  
            self.room2[i,:] = np.arange(startr2+count,startr2+count+self.size2[0])
            count += 13
        # room 3
        self.room3 = np.zeros(self.size3[0]*self.size3[1]).reshape(self.size3[1],self.size3[0])
        count = 0
        startr3 = 20
        for i in range(self.size3[1]):
            self.room3[i,:] = np.arange(startr3+count,startr3+count+self.size3[0])  
            count += 13
        #room 4
        self.room4 = np.zeros(self.size4[0]*self.size4[1]).reshape(self.size4[1],self.size4[0])
        count = 0
        startr4 = 111
        for i in range(self.size4[1]):
            self.room4[i,:] = np.arange(startr4+count,startr4+count+self.size4[0])  
            count += 13
            
        self.w['room1'] = self.room1
        self.w['room2'] = self.room2
        self.w['room3'] = self.room3
        self.w['room4'] = self.room4
        
        self.V = np.zeros(13*13).reshape(13,13)
            
        self.action_space = [0, 1, 2, 3] #0= up, 1 = right, 2=down, 3 = left
        
        self.reward = 0
        
        self.goal = goal
        
        self.done = False
        
    def move_up(self):
        if not(int(np.where(self.state == self.current_room)[0]) == 0) or self.state - 13 == self.hallways[1] or self.state -13 == self.hallways[3]:
            self.state -= 13    
    def move_right(self):
        if not(int(np.where(self.state == self.current_room)[1]) == self.size_room[1]-1)or self.state + 1 == self.hallways[0] or self.state + 1 == self.hallways[2]:
            self.state += 1
    def move_down(self):
        if not(int(np.where(self.state == self.current_room)[0]) == self.size_room[0]-1) or self.state + 13 == self.hallways[1] or self.state +13 == self.hallways[3]:
            self.state += 13
    def move_left(self):
        if not(int(np.where(self.state == self.current_room)[1]) == 0) or self.state - 1 == self.hallways[0] or self.state - 1 == self.hallways[2]:
            self.state -= 1
    
    
    def move(self,a,sucess):
        #rewards are zero on each state trainsiton except for the terminal state r = 1
        
        # find out in which room the current state is
        for i in range(4):
            if not(not(len(np.where(self.state == self.w['room{0}'.format(i+1)])[0])) and not(len(np.where(self.state == self.w['room{0}'.format(i+1)])[1]))):
                self.room = i+1
        self.current_room = self.w['room{}'.format(self.room)]
        self.size_room = self.current_room.shape
        
        
        # up
        if (a == 0):
            if sucess:
                self.move_up()
            else:
                a_rand = int(np.random.choice([1,2,3],1))
                if (a_rand == 1):
                    self.move_right()
                if (a_rand == 2):
                    self.move_down()
                if (a_rand == 3):
                    self.move_left()
         
        # right
        if (a == 1):
            if sucess:
                self.move_right()
            else:
                a_rand = int(np.random.choice([0,2,3],1))
                if (a_rand == 0):
                    self.move_up()
                if (a_rand == 2):
                    self.move_down()
                if (a_rand == 3):
                    self.move_left()
         
        #down
        if (a == 2):
            if sucess:
                self.move_down()
            else:
                a_rand = int(np.random.choice([0,1,3],1))
                if (a_rand == 0):
                    self.move_up()
                if (a_rand == 1):
                    self.move_right()
                if (a_rand == 3):
                    self.move_left()
        
        #left
        if (a == 3):
            if sucess:
                self.move_right()
            else:
                a_rand = int(np.random.choice([0,1,2],1))
                if (a_rand == 0):
                    self.move_up()
                if (a_rand == 1):
                    self.move_right()
                if (a_rand == 2):
                    self.move_down()
                
#         print(self.state)
        
        if (self.state == self.goal):
            self.reward = 1
            self.done = True
    
    def reset(self,start):
        self.state = start
        self.reward = 0

def standard_Value_iteration_Algorithm(r,V,gamma,room):
    states = r.states
    R = np.zeros(r.states.size).reshape(r.states.shape[0],r.states.shape[1])
    for i in range(4):
        pos_h = np.where(r.hallways[i]==r.states)
        R[pos_h] = 1
        
    optimal_V = np.zeros(V.size).reshape(V.shape[0],V.shape[1])
    old_V = np.zeros(V.size).reshape(V.shape[0],V.shape[1])
    
    actual_room = r.w['room{}'.format(room)]
    
    for k in range(100):
        
        for i in range(actual_room.shape[0]):
            for j in range(actual_room.shape[1]):
                state = actual_room[i,j]
                r_u = R[np.where(state-13==states)]
                r_r = R[np.where(state+1==states)]
                r_d = R[np.where(state+13==states)]
                r_l = R[np.where(state-1==states)]
                
                if i==0:
                    action1 = 0
                    if j == 0:
                        action2 = old_V[i,j+1]
                        action3 = old_V[i+1,j]
                        action4 = 0
                    elif j == actual_room.shape[1]-1:
                        action2 = 0
                        action3 = old_V[i+1,j]
                        action4 = old_V[i,j-1]
                    else:
                        action2 = old_V[i,j+1]
                        action3 = old_V[i+1,j]
                        action4 = old_V[i,j-1]
                elif i == actual_room.shape[0]-1:
                    action3 = 0
                    if j == 0:
                        action1 = old_V[i-1,j]
                        action2 = old_V[i,j+1]
                        action4 = 0
                    elif j == actual_room.shape[1]-1:
                        action1 = old_V[i-1,j]
                        action2 = 0
                        action4 = old_V[i,j-1]
                    else:
                        action1 = old_V[i-1,j]
                        action2 = old_V[i,j+1]
                        action4 = old_V[i,j-1]
                elif j == 0:
                    action1 = old_V[i-1,j]
                    action2 = old_V[i,j+1]
                    action3 = old_V[i+1,j]
                    action4 = 0
                elif j == actual_room.shape[1]-1:
                    action1 = old_V[i-1,j]
                    action2 = 0
                    action3 = old_V[i+1,j]
                    action4 = old_V[i,j-1]
                else:
                    action1 = old_V[i-1,j]
                    action2 = old_V[i,j+1]
                    action3 = old_V[i+1,j]
                    action4 = old_V[i,j-1]
                
                action_up = float(r_u + gamma*(2/float(3)*action1+1/float(9)*action2+1/float(9)*action3+1/float(9)*action4))
                action_right = float(r_r + gamma*(1/float(9)*action1+2/float(3)*action2+1/float(9)*action3+1/float(9)*action4))
                action_down = float(r_d + gamma*(1/float(9)*action1+1/float(9)*action2+2/float(3)*action3+1/float(9)*action4))
                action_left = float(r_l + gamma*(1/float(9)*action1+1/float(9)*action2+1/float(9)*action3+2/float(3)*action4))
                
                optimal_V[i,j] = np.max([action_up,action_right,action_down,action_left])
        
        print(optimal_V)
                
        old_V = optimal_V
                
    return optimal_V            
            
        
#         for i in range(1000):
#             r.reset(int(np.random.choice(room,1)))
#             
#             pos = np.where(r.state == r.states)
#             if pos[0] == 0:
#                 if pos[1] == 0:
#                     neighbours = [-1,r.state+1,r.state+13,-1]
#                 elif pos[1] == V.shape[1]-1:
#                     neigbours = [-1,-1,r.state+13,r.state-1]
#                 else:
#                     neighbours = [-1,r.state+1,r.state-1,r.state+13]
#                     
#             
#             elif pos[0] == V.shape[0]-1:
#                 if pos[1] == 0:
#                     neighbours = [r.state-13,r.state+1,-1,-1]
#                 elif pos[1] == V.shape[1]-1:
#                     neighbours = [r.state-13,-1,-1,r.state-1]
#                 else:
#                     neighbours = [r.state-13,r.state+1,-1,r.state+13]
#             
#             elif pos[1] == 0:
#                 neigbours = [r.state-13,r.state+1,r.state+13,-1]
#             elif pos[1] == V.shape[1]:
#                 neighbours = [r.state-13,-1,r.state+13,r.state-1]
#             
#             else:
#                 neighbours = [r.state-13,r.state+1,r.state+13,r.state-1]
#             
#             v_temp = []
#             for i in range(len(neigbours)):
#                 v_pos = np.where(neighbours[i]==room)
#                 v_temp.append(V[v_pos])
#             
#             a = np.argmax(v_temp)
#             
#             p = np.random.randint(100)
#             if p > 66:
#                 sucess = True
#             else:
#                 sucess = False
#             
#             r.move(a,sucess)
#             
#             optimal_V[pos] = r.reward + gamma*float(2/3)*old_V[pos]
#             
#             if r.done:
#                 break
#         
#         old_V = optimal_V
        
        
    
    return optimal_V

hallways = [45,80,136,100]
r = Four_room_domain(14,100)

#room 1
r1 = Four_room_domain(14,hallways[0])
V_r1_1 = np.zeros(r1.room1.shape[0]*r1.room1.shape[1]).reshape(r1.room1.shape[0],r1.room1.shape[1])
r2 = Four_room_domain(14,hallways[1])
V_r1_2 = np.zeros(r1.room1.shape[0]*r1.room1.shape[1]).reshape(r1.room1.shape[0],r1.room1.shape[1])
#room 2
r3 = Four_room_domain(20,hallways[1])
V_r2_1 =np.zeros(r2.room2.shape[0]*r2.room2.shape[1]).reshape(r2.room2.shape[0],r2.room2.shape[1])
r4 = Four_room_domain(20,hallways[2]) 
V_r2_2 = np.zeros(r2.room2.shape[0]*r2.room2.shape[1]).reshape(r2.room2.shape[0],r2.room2.shape[1])
#room 3
r5 = Four_room_domain(92,hallways[0])
V_r3_1 = np.zeros(r3.room3.shape[0]*r3.room3.shape[1]).reshape(r3.room3.shape[0],r3.room3.shape[1])
r6 = Four_room_domain(92,hallways[3])
V_r3_2 = np.zeros(r3.room3.shape[0]*r3.room3.shape[1]).reshape(r3.room3.shape[0],r3.room3.shape[1])
#room 4
r7 = Four_room_domain(111,hallways[2])
V_r4_1 = np.zeros(r4.room4.shape[0]*r4.room4.shape[1]).reshape(r4.room4.shape[0],r4.room4.shape[1])
r8 = Four_room_domain(111,hallways[3])
V_r4_2 = np.zeros(r4.room4.shape[0]*r4.room4.shape[1]).reshape(r4.room4.shape[0],r4.room4.shape[1])

opt_V_1_1 = standard_Value_iteration_Algorithm(r1, V_r1_1, 0.9, 1)
opt_V_1_2 = standard_Value_iteration_Algorithm(r2, V_r1_2, 0.9, 1)
opt_V_2_1 = standard_Value_iteration_Algorithm(r3, V_r2_1, 0.9, 2)
opt_V_2_2 = standard_Value_iteration_Algorithm(r4, V_r2_2, 0.9, 2)
opt_V_3_1 = standard_Value_iteration_Algorithm(r5, V_r3_1, 0.9, 3)
opt_V_3_2 = standard_Value_iteration_Algorithm(r6, V_r3_2, 0.9, 3)
opt_V_4_1 = standard_Value_iteration_Algorithm(r7, V_r4_1, 0.9, 4)
opt_V_4_2 = standard_Value_iteration_Algorithm(r8, V_r4_2, 0.9, 4)

print('V_1_1',np.round(opt_V_1_1,2))
print('V_1_2',np.round(opt_V_1_2,2))
print('V_2_1',np.round(opt_V_2_1,2))
print('V_2_2',np.round(opt_V_2_2,2))
print('V_3_1',np.round(opt_V_3_1,2))
print('V_3_2',np.round(opt_V_3_2,2))
print('V_4_1',np.round(opt_V_4_1,2))
print('V_4_2',np.round(opt_V_4_2,2))



# r.move(3,False)
