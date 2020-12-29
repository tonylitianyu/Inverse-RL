import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
#from IRL import IRL_LP
from IRL2 import IRL_LP

def plot_reward_surface(reward, grid_size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, grid_size, 1)
    X, Y = np.meshgrid(x, y)
    zs = reward
    Z = zs.reshape(X.shape)
    ax.view_init(45, 135)
    ax.plot_surface(X, Y, Z,alpha=0.5,cmap='jet', rstride=1, cstride=1, edgecolors='k', lw=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Reward Values')

    plt.show()




env = gym.make('gridworld-v0')
env.visual = True


#optimal policy calculated by value iteration
print("optimal policy calculated by value iteration")
v = env.value_iteration(0.9)
print(v)
pi, pi_visual = env.generate_policy(v,0.9)
print(pi)
print(pi_visual)


#recover reward from IRL
tran = env.transition_prob()
myIRL = IRL_LP(env.grid_size,4,tran,pi.flatten(), 0.1,3,1.0)
myIRL_rewards = myIRL.solve()



#test recovered reward
print("test recovered reward")
env.reward_state = myIRL_rewards
v = env.value_iteration(0.9)
print(v)
pi, pi_visual = env.generate_policy(v,0.9)
print(pi)
print(pi_visual)


#render
env.render()
plot_reward_surface(myIRL_rewards, env.grid_size)








