import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np

from IRL_MaxEnt import IRL_MaxEnt, RewardNet



env = gym.make('gridworld-v0')
env.visual = True

tran = env.transition_prob()
reward_table = env.reward_state.flatten()
#print(reward_table)

irl_agent = IRL_MaxEnt(0,1,env.grid_size**2,4,tran,0.9)

v = irl_agent.approx_value_iteration(reward_table)
print(v.reshape((env.grid_size,env.grid_size)))
