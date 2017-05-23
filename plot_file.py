import numpy as np
import matplotlib.pyplot as plt

# deltaT = np.load(r'/home/jack/Documents/LiClipse Workspace/RL/deltaT.npy')
# n = [20,80,120]

step_til_end_rbf = np.load(r'/home/jack/Documents/LiClipse Workspace/RL/mountain_car_data/steps_til_end_rbf.npy')
# step_til_end_1 = np.load(r'/home/jack/Documents/LiClipse Workspace/RL/step_til_end_1.npy')
# step_til_end_2 = np.load(r'/home/jack/Documents/LiClipse Workspace/RL/step_til_end_2.npy')

episode = 200
# plt.subplot(211)
plt.plot(range(episode),step_til_end_rbf)
# plt.plot(range(episode),step_til_end_1,label = str(n[1]))
# plt.plot(range(episode),step_til_end_2,label = str(n[2]))
plt.ylabel('#steps unit goal is reached')
plt.xlabel('#episodes')
# plt.legend()
plt.grid()

# plt.subplot(212)
# plt.plot(n,deltaT,'xr')
# plt.xlabel('intervals')
# plt.ylabel('time in seconds')
# 
# plt.grid()
plt.show()
# np.load(outfile)
# 
# plt.plot(sum_episode,range(episode))
# plt.xlabel('averaged number of steps per episode')
# plt.ylabel('number of episodes')
# plt.grid()
# 
# plt.show()