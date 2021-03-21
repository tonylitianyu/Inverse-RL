import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
from IRL_MaxEnt import IRL_MaxEnt, RewardNet





# grid_side_length = 3
# goal_idx = 1
# n_traj = 1
# n_episode = 10

env = gym.make('CartPole-v1')
print(env.action_space.n)
lr = 0.1
gamma = 0.95
eps = 40000
total = 0
total_reward = 0
prior_reward = 0
Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
epsilon = 1
epsilon_decay_value = 0.99995

q_table = np.random.uniform(low = 0, high = 1, size = (Observation + [env.action_space.n]))
print(q_table.shape)

def get_discrete_state(state):
    discrete_state = state/np_array_win_size+ np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))



for episode in range(eps + 1):
    t0 = time.time()
    discrete_state = get_discrete_state(env.reset())
    done = False
    episode_reward = 0

    
    while done == False:
        if np.random.random() > epsilon:
            action = np.argmax([q_table[discrete_state]])
        else:
            action = np.random.randint(0, env.action_space.n)


        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        
        # if episode % 5 == 0 and episode != 0:
        #     print(episode)
        #     env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1-lr)*current_q+lr*(reward + gamma * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    if epsilon > 0.05:
        if episode_reward > prior_reward and eps > 10000:
            epsilon = math.pow(epsilon_decay_value, eps - 10000)


    t1 = time.time()
    eps_total = t1-t0
    total = total + eps_total

    total_reward += episode_reward
    prior_reward = episode_reward


    if episode % 1000 == 0: 
        mean = total / 1000
        print("Time Average: " + str(mean))
        total = 0

        mean_reward = total_reward / 1000
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0



discrete_state = get_discrete_state(env.reset())
done = False
while not done:
    action = np.argmax([q_table[discrete_state]])
    new_state, reward, done, _ = env.step(action)
    new_discrete_state = get_discrete_state(new_state)
    discrete_state = new_discrete_state
    
    env.render()
    



env.close()

    





#Calculating Input
# tran = env.transition_prob()
# reward_table = env.reward_state.flatten()
# print(reward_table.reshape((env.grid_size,env.grid_size)))


##################################
#####Algorithm 1 starts here######
##################################

# irl_agent = IRL_MaxEnt(env.grid_size**2,env.grid_size**2,4,tran,0.9) #Initialization
# expert_policy_action, expert_policy_state = irl_agent.approx_value_iteration(reward_table)

# def generate_expert_demo(n_state, goal_idx, policy, n_traj, max_step):
#     all_demo_trajs = []
#     for i in range(0, n_traj):
#         idx = random.randint(0,(env.grid_size**2)-1)
#         traj = [idx]

#         while idx != goal_idx:
#             idx = policy[idx]
#             traj.append(idx)

#         curr_len = len(traj)
#         for k in range(curr_len, max_step):
#             traj.append(goal_idx)

#         all_demo_trajs.append(traj)
#     return all_demo_trajs

# #generate expert demonstration trajectory with same length
# max_traj_step = env.grid_size*2
# expert_demo = generate_expert_demo(env.grid_size**2, goal_idx, expert_policy_state, n_traj, max_traj_step)
# print(expert_demo)




# ####Training

# start = time.time()

# for i in range(0,n_episode):
#     curr_reward_table = irl_agent.initialize_training_episode()
#     irl_agent.print_episode_info(i)


#     policy_action, policy_state = irl_agent.approx_value_iteration(curr_reward_table)
#     #print(policy_action.reshape((env.grid_size,env.grid_size)))

    
#     #print(expert_demo)
#     svf = irl_agent.policy_propagation(policy_action, expert_demo, max_traj_step)


#     #Determine Maximum Entropy Loss and Gradients
#     expert_freq = irl_agent.expert_state_freq(expert_demo)
#     print(expert_freq)
#     #print(svf)

#     irl_agent.train_network(expert_freq, svf)

# irl_agent.print_final_reward_table(env.grid_size, env.grid_size)



# final_reward_table = irl_agent.initialize_training_episode()
# policy_action, policy_state = irl_agent.approx_value_iteration(final_reward_table)
# print(env.visualize_policy(policy_action.reshape(env.grid_size, env.grid_size), env.grid_size, env.grid_size))



# end = time.time()
# print("Total training time: "+ str(end - start))




