import logging
import os, sys

import gym
from gym.wrappers import Monitor
import gym_ple
import pygame

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

  
env = gym.make('FlappyBird-v0')
agent = RandomAgent(env.action_space)  
env.reset()
pygame.display.update()
episode_count = 100
reward = 0
done = False
# for i in range(1):
ob = env.reset()

while True:
    ob,rew,done,_ = env.step(0)
    pygame.display.update()
    
    if done:
        break
    

#     while True:
#         action = agent.act(ob, reward, done)
#         ob, reward, done, _ = env.step(action)
#         pygame.display.update()
# 
#         if done:
#             break

#     env.close()

        

