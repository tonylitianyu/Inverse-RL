import numpy as np
import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional
from torch.autograd import Variable




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
        reward = self.fc3(h2)
        return reward

    def get_gradient(self, expert_freq, curr_policy_freq):
        return (expert_freq - curr_policy_freq).view(-1,1)



    

class IRL_MaxEnt:
    def __init__(self, n_features, n_states, n_actions, transition, gamma):

        self.n_features = n_features
        self.n_states = n_states
        self.n_side_length = int((self.n_states/2)**(1/3))
        print(self.n_side_length)
        self.n_actions = n_actions
        self.transition = np.transpose(transition,(0,2,1)) #(state, action, state)
        
        self.gamma = gamma


        #self.feat_map = np.eye(self.n_states)



        self.feat_map = np.zeros((self.n_states, 4))
        j = 0
        k = 0
        m = 0
        for i in range(0, self.n_states):
            self.feat_map[i][0] = j
            self.feat_map[i][1] = k
            self.feat_map[i][2] = m

            if i >= self.n_states/2:
                self.feat_map[i][3] = 1
            else:
                self.feat_map[i][3] = 0

            j += 1
            j %= self.n_side_length
            if j == 0:
                k += 1
                if k == self.n_side_length:
                    m += 1
                    m %= self.n_side_length
                k %= self.n_side_length
                



        self.feat_map = torch.from_numpy(self.feat_map).type(torch.FloatTensor).to(device)

        #self.r_model = RewardNet(self.n_side_length*2, 64).to(device)
        self.r_model = RewardNet(n_features, 128).to(device)
        self.optimizer = optim.Adagrad(self.r_model.parameters(), lr=0.01, weight_decay=1e-4)


    def initialize_training_episode(self):

        self.optimizer.zero_grad()


        self.curr_reward_table = self.r_model(self.feat_map)
        

        return self.curr_reward_table
    
    def approx_value_iteration(self, curr_reward_table):
        '''Algorithm 2 Approximate Value Iteration in the paper
            Args:
                curr_reward_table (list) - current reward table from the neural net (will be delete)
            Returns:
                best_policy (list) - best "next state" index for each state
        '''
        value_table = self.find_value_table(curr_reward_table)
        return self.find_policy(value_table, curr_reward_table)



    def find_value_table(self, curr_reward_table):
        V = np.zeros(self.n_states)
        V_res = np.zeros(self.n_states)
        eps = 0.001

        # def softmax(next_value_array):
        #     if abs(sum(next_value_array)) < 1e-3:
        #         return 0.0

        #     e = np.exp(next_value_array-np.max(next_value_array))
        ##     return max(e/sum(e))


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
        opti_policy_state = np.zeros(self.n_states, dtype=int)
        opti_policy_action = np.zeros(self.n_states, dtype=int)
        for i in range(0,self.n_states):
            curr_reward = curr_reward_table[i]
            next_best_dic = {}
            for k in range(0,self.n_actions):
                next_s = self.transition[i,k,:]
                for j in range(0, len(next_s)):
                    if next_s[j] > 0.0:
                        value = curr_reward+self.gamma*value_table[j]
                        next_best_dic[(k,j)] = value

            next_best = max(next_best_dic, key=next_best_dic.get) #(action, next_state)
            opti_policy_action[i] = next_best[0]
            opti_policy_state[i] = next_best[1]
            

        return opti_policy_action, opti_policy_state


    def policy_propagation(self, policy_action, expert_traj, max_step):

        E = np.zeros((self.n_states, max_step))
        for i in expert_traj:
            E[i[0]][0] = 1

        E[:,0] /= len(expert_traj)

        for t in range(0, max_step-1):
            for s in range(self.n_states):

                for next_s in range(self.n_states):
                    E[next_s][t+1] += E[s][t]*self.transition[s, policy_action[s], next_s]

        state_visit_feq = np.sum(E,1)

        return state_visit_feq



    
    def expert_state_freq(self,expert_traj):
        uD = np.zeros(self.n_states)

        for traj in expert_traj:
            for t in traj:
                uD[t] += 1

        uD /= len(expert_traj)


        return uD

    def train_network(self, expert_freq, curr_policy_freq):


        expert_freq = torch.from_numpy(expert_freq)
        curr_policy_freq = torch.from_numpy(curr_policy_freq)

        r_gradient = self.r_model.get_gradient(expert_freq, curr_policy_freq).to(device)
        # self.curr_reward_table.retain_grad()
        #print(-r_gradient)
        # self.curr_reward_table.grad = -r_gradient.to(dtype=torch.float32).to(device)
        # self.curr_reward_table.backward(torch.ones(self.n_states,1).to(device))
        # self.optimizer.step()
        
        ##self.curr_reward_table.backward(gradient=r_gradient.to(dtype=torch.float32).to(device))

        torch.nn.utils.clip_grad_norm_(self.r_model.parameters(), 100)
        

        self.curr_reward_table.backward(-r_gradient.to(dtype=torch.float32).to(device))
        self.optimizer.step()
        


    def print_episode_info(self, episode_number):

        print("Episode " + str(episode_number) + " max index: " + str(torch.argmax(self.curr_reward_table)))

    def print_final_reward_table(self, grid_width, grid_length, grid_height):
        print(self.curr_reward_table.reshape((2,grid_width,grid_length, grid_height))/torch.max(self.curr_reward_table))









        



            




                












        
