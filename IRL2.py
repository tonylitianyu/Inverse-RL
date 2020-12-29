import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from cvxopt import matrix, solvers

A_test = []
B_test = []
C_test = []

class IRL_LP:

    def __init__(self, state_size,action_size,transition, policy, gamma,l1=5.0, max_reward=1.0):

        self.N_STATES = state_size
        self.N_ACTIONS = action_size

        self.p = transition

        self.policy = policy

        self.gamma = gamma
        self.l1 = l1
        self.max_reward = max_reward





    def build_c_mat(self):
        c = np.zeros(3*self.N_STATES)

        for i in range(self.N_STATES, 2*self.N_STATES):
            c[i] = 1

        for j in range(2*self.N_STATES, 3*self.N_STATES):
            c[j] = -self.l1

        c = -c
        C_test = c

        return matrix(c),c


    def build_b_mat(self):

        b1 = np.zeros((self.N_STATES*(self.N_ACTIONS-1),1))
        b2 = np.zeros((self.N_STATES*(self.N_ACTIONS-1),1))

        b3 = np.zeros((self.N_STATES,1))
        b4 = np.zeros((self.N_STATES,1))

        b5 = self.max_reward*np.ones((self.N_STATES,1))
        b6 = np.zeros((self.N_STATES,1))#self.max_reward*np.ones((self.N_STATES,1))

        b = np.vstack((b1,b2,b5,b6,b3,b4))
        # N_sub_optimal_action = self.N_ACTIONS-1
        # b = np.zeros(2*N_sub_optimal_action*self.N_STATES+4*self.N_STATES)

        # start_idx = 2*N_sub_optimal_action*self.N_STATES
        # for i in range(self.N_STATES):
        #     b[start_idx+i] = 1

        B_test = b
        return matrix(b), b

    def build_A_mat(self):

        T_array = []
        I_array = []
        transition_p = self.p
        for i in range(self.N_STATES):
            for j in range(self.N_ACTIONS):
                if j != self.policy[i]:

                    pa1pa = transition_p[i,:,self.policy[i]] - transition_p[i,:,j]
                    
                    igammapa1 = np.linalg.inv(np.identity(self.N_STATES)- self.gamma*transition_p[:,:,self.policy[i]])
                    T = -np.dot(pa1pa, igammapa1)
                    T_array.append(T)


                    I = np.eye(1, self.N_STATES, i)#
                    I_array.append(I)



        T_s = np.vstack(T_array)

        I_s = np.eye(self.N_STATES)

        A_l = np.vstack([T_s, T_s, I_s, -I_s, I_s, -I_s])

        I_suboptimal = np.vstack(I_array)



        A_m_zero_top = np.zeros((self.N_STATES*(self.N_ACTIONS-1), self.N_STATES))
        A_m_zero_mid_btm = np.zeros((self.N_STATES, self.N_STATES))

        A_m = np.vstack([A_m_zero_top, I_suboptimal, A_m_zero_mid_btm, A_m_zero_mid_btm, A_m_zero_mid_btm, A_m_zero_mid_btm])

        A_r = np.vstack([A_m_zero_top, A_m_zero_top, A_m_zero_mid_btm, A_m_zero_mid_btm, -I_s, -I_s])

        A = np.hstack([A_l, A_m, A_r])


        A_test = A

        return matrix(A), A



    def solve(self):

        A_mat,A = self.build_A_mat()
        b_mat,b = self.build_b_mat()
        c_mat,c = self.build_c_mat()
        print(c.shape)

        soln = solvers.lp(c_mat, A_mat, b_mat)
        

        rewards = np.array(soln["x"][:self.N_STATES])

        # rewards_norm = np.linalg.norm(rewards)
        # rewards = np.array((rewards/rewards_norm))

        with open('policy.npy', 'wb') as f:
            np.save(f, self.policy)

        return rewards.reshape((5,5))/max(rewards)

