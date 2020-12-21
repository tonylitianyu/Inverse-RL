import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
from IRL import IRL_LP

def get_reward_arr(env):
    r = np.zeros((4,4))
    for i in range(len(r)):
        for j in range(len(r[0])):
            r_val = env.get_reward([i,j])
            r[i][j] = r_val

    return r

def draw_3d_hist(data):
    data_arr = np.array(data)
    fig = plt.figure(1)
    ax = fig.add_subplot(111,projection='3d')
    x_data,y_data = np.meshgrid(np.arange(data_arr.shape[1]), np.arange(data_arr.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_arr.flatten()
    ax.bar3d(x_data,y_data,np.zeros(len(z_data)),1,1,z_data)
    plt.ylim(max(y_data)+1, 0)
    plt.show()


env = gym.make('gridworld-v0')
env.visual = True
env.render()
print(env.policy)

v = env.value_iteration(0.9)
pi = env.generate_policy(v,0.9)
print(pi)

# print(v)
#r_data = get_reward_arr(env)
#draw_3d_hist(r_data)

irl = IRL_LP(env,0.9)
irl_result = irl.solve()
print(irl_result)
draw_3d_hist(irl_result)


for i in range(1):
    
    action = env.policy[env.agent_pos[0]][env.agent_pos[1]]#env.get_keyboard_action(input())#env.action_space.sample()
    new_state, reward, done = env.step(action)

    env.render()

    if done:
        print(reward)
        break




