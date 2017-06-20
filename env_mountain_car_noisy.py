import gym.envs.classic_control
from gym import spaces
import numpy as np

class GaussianNoiseCartPoleEnv(gym.envs.classic_control.CartPoleEnv):
    def __init__(self):
        super(GaussianNoiseCartPoleEnv, self).__init__()
        self.action_space = spaces.Box(-10.0, 10.0, shape = (1,))

    def _step(self, action):
        x, x_dot, theta, theta_dot = self.state

        GRAVITY = self.gravity
        MASSCART = self.masscart
        m1 = self.masspole
        m2 = m1 + MASSCART
        LENGTH = self.length
        MASSPOLE = self.masspole
        POLE = MASSPOLE * LENGTH
        FORCE_MAG = self.force_mag
        TAU = self.tau

        eps_1 = np.random.normal(0.0, 0.01)
        eps_2 = np.random.normal(0.0, 0.0001)

        temp = (action + POLE*theta_dot**2*np.sin(theta))/m2
        thetaacc = (GRAVITY*np.sin(theta) - np.cos(theta)*temp)/(LENGTH*(4.0/3.0 - m1*np.cos(theta)**2/m2))
        xacc = temp - POLE*thetaacc*np.cos(theta)/m2
        x_next = x + TAU*x_dot + eps_1
        x_dot_next = x_dot + TAU*xacc+eps_1
        theta_next = theta + TAU*theta_dot + eps_2
        theta_dot_next = theta_dot + TAU*thetaacc + eps_2

        self.state = (x_next, x_dot_next, theta_next, theta_dot_next)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        else:
            reward = -1.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.steps_beyond_done = None
        return np.array(self.state)