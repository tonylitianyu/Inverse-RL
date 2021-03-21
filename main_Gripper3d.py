import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import random
import time
from IRL_Gripper3d import IRL_MaxEnt, RewardNet



grid_side_length = 3
goal_idx = (1,1,0,0)#gxyz g{open, close}={0,1}
n_traj = 1
n_episode = 40

env = gym.make('gripperworld3d-v0', grid_size = grid_side_length, goal_idx = goal_idx)
env.visual = True

#Calculating Input
tran = env.transition_prob()
#print(tran)
reward_table = env.reward_state.flatten()
print(reward_table)
#print(env.flatToXYZG(40,54))



##################################
#####Algorithm 1 starts here######
##################################

irl_agent = IRL_MaxEnt(4,(env.grid_size**3)*2,8,tran,0.9) #Initialization
expert_policy_action, expert_policy_state = irl_agent.approx_value_iteration(reward_table)
expert_policy_action=expert_policy_action.reshape((2,env.grid_size,env.grid_size,env.grid_size))
# print(expert_policy_action.shape)
# # # print(len(expert_policy_action))
print(env.visualize_policy(expert_policy_action,3,3,3))


def generate_expert_demo(n_state, goal_idx, policy, n_traj, max_step):
    all_demo_trajs = []
    for i in range(0, n_traj):
        idx = random.randint(0,((env.grid_size**3)*2)-1)
        traj = [idx]


        while idx != goal_idx:
            
            idx = policy[idx]
            traj.append(idx)
        
        curr_len = len(traj)
        for k in range(curr_len, max_step):
            traj.append(goal_idx)

        all_demo_trajs.append(traj)
    return all_demo_trajs

# # #generate expert demonstration trajectory with same length
max_traj_step = 10#env.grid_size*6
expert_demo = generate_expert_demo((env.grid_size**3)*2, env.XYZGToFlat(goal_idx[0], goal_idx[2], goal_idx[1], goal_idx[3], (env.grid_size**3)*2), expert_policy_state, n_traj, max_traj_step)


#expert_demo = [[20, 19, 18, 9, 0, 27, 36, 45, 46, 47]]
print(expert_demo)





# # # ####Training

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

irl_agent.print_final_reward_table(env.grid_size, env.grid_size, env.grid_size)


#after getting the traied reward table, using simple value iteration to get policy
final_reward_table = irl_agent.initialize_training_episode()
policy_action, policy_state = irl_agent.approx_value_iteration(final_reward_table)

policy_action_unflat = policy_action.reshape(2,env.grid_size, env.grid_size, env.grid_size)

policy_action_visual = env.visualize_policy(policy_action_unflat, env.grid_size, env.grid_size, env.grid_size)
print(policy_action_visual)
np.save('trained_model/optimal_action.npy', policy_action_visual)



end = time.time()
print("Total training time: "+ str(end - start))




