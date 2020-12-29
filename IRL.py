import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from cvxopt import matrix, solvers


#https://pic1.zhimg.com/80/v2-f942c58661df973656674403082afda4_720w.jpg

class IRL_LP:

    def __init__(self):
        # self.env = env
        # self.N_STATES = env.grid_size*env.grid_size
        # self.N_ACTIONS = len(env.action)

        # self.p = np.ones((self.N_STATES, self.N_ACTIONS,self.N_STATES))#!!!!!!!!!!!!!!!!
        # self.build_transition_dynamics()

        # self.policy = env.policy.flatten()
        # self.gamma = gamma
        # self.l1 = l1
        # self.max_reward = max_reward


        self.A_test = []
        self.B_test = []
        self.C_test = []

    def build_transition_dynamics(self):

        for i in range(self.N_STATES):
            for a in range(self.N_ACTIONS):
                for j in range(self.N_STATES):

                    iy, ix = self.flatToXY(i)
                    jy, jx = self.flatToXY(j)
                    self.p[i][a][j] = self.env.transition_prob(iy,ix,a,jy,jx)


        print(self.p)

    def flatToXY(self, i):
        
        y = int(i/self.env.grid_size)
        x = int(i%self.env.grid_size)

        return y, x



    def build_c_mat(self):
        c = np.zeros(3*self.N_STATES)

        for i in range(self.N_STATES, 2*self.N_STATES):
            c[i] = 1

        for j in range(2*self.N_STATES, 3*self.N_STATES):
            c[j] = -self.l1

        c = -c
        self.C_test = c
        return matrix(c),c


    def build_b_mat(self):

        b1 = np.zeros((self.N_STATES*(self.N_ACTIONS-1),1))
        b2 = np.zeros((self.N_STATES*(self.N_ACTIONS-1),1))

        b3 = np.zeros((self.N_STATES,1))
        b4 = np.zeros((self.N_STATES,1))

        b5 = self.max_reward*np.ones((self.N_STATES,1))
        b6 = self.max_reward*np.ones((self.N_STATES,1))

        b = np.vstack((b1,b2,b3,b4,b5,b6))

        self.B_test = b
        return matrix(b), b

    def build_A_mat(self):

        T_array = []
        I_array = []
        transition_p = np.transpose(self.p, (1, 0, 2))
        for i in range(self.N_STATES):
            for j in range(0,self.N_ACTIONS):
                if j != self.policy[i]:

                    pa1pa = transition_p[self.policy[i],i] - transition_p[j,i]
                    igammapa1 = np.linalg.inv(np.eye(self.N_STATES)- self.gamma*transition_p[self.policy[i]])
                    T = -np.dot(pa1pa, igammapa1)
                    T_array.append(T)


                    I = np.eye(1, self.N_STATES, i)#
                    I_array.append(I)



        T_s = np.vstack(T_array)

        I_s = np.eye(self.N_STATES)

        A_l = np.vstack([T_s, T_s, -I_s, I_s, -I_s, I_s])

        I_suboptimal = np.vstack(I_array)



        A_m_zero_top = np.zeros((self.N_STATES*(self.N_ACTIONS-1), self.N_STATES))
        A_m_zero_mid_btm = np.zeros((self.N_STATES, self.N_STATES))

        A_m = np.vstack([I_suboptimal, A_m_zero_top, A_m_zero_mid_btm, A_m_zero_mid_btm, A_m_zero_mid_btm, A_m_zero_mid_btm])

        A_r = np.vstack([A_m_zero_top, A_m_zero_top, -I_s, -I_s, A_m_zero_mid_btm, A_m_zero_mid_btm])

        A = np.hstack([A_l, A_m, A_r])


        self.A_test = A

        return matrix(A), A



    def solve(self):

        A_mat,A = self.build_A_mat()
        b_mat,b = self.build_b_mat()
        c_mat,c = self.build_c_mat()
        print(c)

        soln = solvers.lp(c_mat, A_mat, b_mat)
        

        rewards = np.array(soln["x"][:self.N_STATES])

        # rewards_norm = np.linalg.norm(rewards)
        # rewards = np.array((rewards/rewards_norm))


        return (rewards/max(rewards)).reshape((5,5))



