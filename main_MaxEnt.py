import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
import random
from IRL_MaxEnt import IRL_MaxEnt, RewardNet



env = gym.make('gridworld-v0')
env.visual = True

#Calculating Input
tran = env.transition_prob()
reward_table = env.reward_state.flatten()
print(reward_table.reshape((5,5)))


##################################
#####Algorithm 1 starts here######
##################################

irl_agent = IRL_MaxEnt(1,env.grid_size**2,4,tran,0.9) #Initialization
expert_policy_action, expert_policy_state = irl_agent.approx_value_iteration(reward_table)

def generate_expert_demo(n_state, goal_idx, policy, n_traj, max_step):
    all_demo_trajs = []
    for i in range(0, n_traj):
        idx = random.randint(0,24)
        traj = [idx]

        while idx != goal_idx:
            idx = policy[idx]
            traj.append(idx)

        curr_len = len(traj)
        for k in range(curr_len, max_step):
            traj.append(goal_idx)

        all_demo_trajs.append(traj)
    return all_demo_trajs

#generate expert demonstration trajectory with same length
n_traj = 15
max_traj_step = 8
expert_demo = generate_expert_demo(env.grid_size**2, 9, expert_policy_state, n_traj, max_traj_step)




####Training

for i in range(0,10):
    curr_reward_table = irl_agent.initialize_training_episode()
    #print(curr_reward_table.reshape((5,5)))

    policy_action, policy_state = irl_agent.approx_value_iteration(curr_reward_table)
    #print(policy_action.reshape((env.grid_size,env.grid_size)))

    
    #print(expert_demo)
    svf = irl_agent.policy_propagation(policy_action, expert_demo, max_traj_step,9)


    #Determine Maximum Entropy Loss and Gradients
    expert_freq = irl_agent.expert_state_freq(expert_demo)

    #print(svf)

    irl_agent.train_network(expert_freq, svf, curr_reward_table)








