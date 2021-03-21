import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import random
import time
from IRL_Gripper import IRL_MaxEnt, RewardNet



grid_side_length = 5
goal_idx = (1,0,0)#gxy g{open, close}={0,1}
n_traj = 3
n_episode = 100

env = gym.make('gripperworld-v0', grid_size = grid_side_length, goal_idx = goal_idx)
env.visual = True

#Calculating Input
tran = env.transition_prob()
#print(tran)
reward_table = env.reward_state.flatten()



##################################
#####Algorithm 1 starts here######
##################################

irl_agent = IRL_MaxEnt(3,(env.grid_size**2)*2,6,tran,0.9) #Initialization
expert_policy_action, expert_policy_state = irl_agent.approx_value_iteration(reward_table)
#expert_policy_action=expert_policy_action.reshape((2,env.grid_size,env.grid_size))
# print(len(expert_policy_action))
# print(env.visualize_policy(expert_policy_action,3,3))


def generate_expert_demo(n_state, goal_idx, policy, n_traj, max_step):
    all_demo_trajs = []
    for i in range(0, n_traj):
        idx = random.randint(0,((env.grid_size**2)*2)-1)
        traj = [idx]

        while idx != goal_idx:
            idx = policy[idx]
            traj.append(idx)

        curr_len = len(traj)
        for k in range(curr_len, max_step):
            traj.append(goal_idx)

        all_demo_trajs.append(traj)
    return all_demo_trajs

# #generate expert demonstration trajectory with same length
max_traj_step = env.grid_size*2
expert_demo = generate_expert_demo((env.grid_size**2)*2, env.XYGToFlat(goal_idx[0], goal_idx[2], goal_idx[1], (env.grid_size**2)*2), expert_policy_state, n_traj, max_traj_step)
#expert_demo = [[17, 13, 9, 0, 0, 0]]
print(expert_demo)




# ####Training

start = time.time()

for i in range(0,n_episode):
    curr_reward_table = irl_agent.initialize_training_episode()
    irl_agent.print_episode_info(i)


    policy_action, policy_state = irl_agent.approx_value_iteration(curr_reward_table)
    #print(policy_action.reshape((env.grid_size,env.grid_size)))

    
    #print(expert_demo)
    svf = irl_agent.policy_propagation(policy_action, expert_demo, max_traj_step)


    #Determine Maximum Entropy Loss and Gradients
    expert_freq = irl_agent.expert_state_freq(expert_demo)
    #print(expert_freq)
    #print(svf)

    irl_agent.train_network(expert_freq, svf)

irl_agent.print_final_reward_table(env.grid_size, env.grid_size)



final_reward_table = irl_agent.initialize_training_episode()
policy_action, policy_state = irl_agent.approx_value_iteration(final_reward_table)
print(env.visualize_policy(policy_action.reshape(2,env.grid_size, env.grid_size), env.grid_size, env.grid_size))



end = time.time()
print("Total training time: "+ str(end - start))




