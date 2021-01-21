import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Running on the CPU")




class RewardNet(nn.Module):
    def __init__(self, n_input,n_hidden):
        super(RewardNet, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden


        self.fc1 = nn.Linear(self.n_input, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.n_hidden, 1)


    def forward(self, input):
        h1 = functional.relu(self.fc1(input))
        h2 = functional.relu(self.fc2(h1))
        reward = functional.relu(self.fc3(h2))
        return reward


    

class IRL_MaxEnt:
    def __init__(self, expert_state_freq, n_features, n_states, n_actions, transition, gamma):
        self.expert_state_freq = expert_state_freq
        self.n_features = n_features
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition = np.transpose(transition,(0,2,1)) #(state, action, state)
        
        self.gamma = gamma


        self.r_model = RewardNet(n_features, 64).to(device)
        self.optimizer = optim.Adam(self.r_model.parameters(), lr=0.01)


    
    def approx_value_iteration(self, curr_reward_table):
        '''Algorithm 2 Approximate Value Iteration in the paper
            Args:
                curr_reward_table (list) - current reward table from the neural net (will be delete)
            Returns:
                best_policy (list) - best "next state" index for each state
        '''
        value_table = self.find_value_table(curr_reward_table)
        best_policy = self.find_policy(value_table, curr_reward_table)

        return best_policy


    def find_value_table(self, curr_reward_table):
        V = np.zeros(self.n_states)
        V_res = np.zeros(self.n_states)
        eps = 0.001

        # def softmax(next_value_array):
        #     if abs(sum(next_value_array)) < 1e-3:
        #         return 0.0

        #     e = np.exp(next_value_array-np.max(next_value_array))
        #     return max(e/sum(e))


        while True:
            delta = 0
            V = V_res.copy()
            for i in range(0,self.n_states):
                curr_reward = curr_reward_table[i]#self.r_model(i)
                
                next_state_vals = []
                for k in range(0,self.n_actions):
                    next_s = self.transition[i,k,:]
                    for j in range(0,len(next_s)):
                        if next_s[j] > 0.0:
                            value = curr_reward+self.gamma*V[j]
                            next_state_vals.append(value)

                V_res[i] = max(next_state_vals)
                delta = max(delta, np.abs(V[i]-V_res[i]))

            if delta < eps:
                break

        return V_res


    def find_policy(self, value_table, curr_reward_table):
        opti_policy = np.zeros(self.n_states)
        for i in range(0,self.n_states):
            curr_reward = curr_reward_table[i]
            next_best_dic = {}
            for k in range(0,self.n_actions):
                next_s = self.transition[i,k,:]
                for j in range(0, len(next_s)):
                    if next_s[j] > 0.0:
                        value = curr_reward+self.gamma*value_table[j]
                        next_best_dic[j] = value

            next_best_state = max(next_best_dic, key=next_best_dic.get)
            opti_policy[i] = next_best_state

        return opti_policy


    def policy_propagation(self, best_policy):
        



            




                












        
