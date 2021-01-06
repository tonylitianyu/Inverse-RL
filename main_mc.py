import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
from IRL2 import IRL_LP




env = gym.make('MountainCar-v0')


x_discrete =100
x_min = env.env.min_position
x_max = env.env.max_position
x_stepsize = (x_max - x_min)/x_discrete
print(x_stepsize)


v_discrete = 5
v_min = -env.env.max_speed
v_max = env.env.max_speed
v_stepsize = (v_max - v_min)/v_discrete
print(v_stepsize)




