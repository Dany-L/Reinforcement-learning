import gym
import numpy as np
import gym_random_walk

env = gym.make('random_walk-v0')

V = [0, 0.5, 0.5,0.5,0.5,0.5,0]
V_MC = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]
alpha = 0.1
gamma = 1

num2 = 100

for episodes in range(num2):

    env.reset()
    env.render()
    s=3
    total_reward = 0
    done = False
    states =[]
    
    num = 100
    
    for i in range(num):
        a = np.random.randint(env.action_space.n)
        print("action: " ,a)
        s1,reward,done, _ = env.step(a)
        env.render()

#        TD(0) 
        V[s] = (1-alpha)*V[s] + alpha*(reward + gamma*V[s1])
        
#         reward and states for MC
        states.append(s1)
        total_reward += reward
        
#         update state
        s = s1
        
        if done:
            for j in range(len(states)):
                counter = states[j]
#                 calculate MC
                V_MC[counter] = V_MC[counter] +alpha*(total_reward-V_MC[counter])
            print("endstate reached")
            break
        
    print("TD",V,"MC",V_MC)
