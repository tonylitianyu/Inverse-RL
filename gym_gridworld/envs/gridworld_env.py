import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
import matplotlib.pyplot as plt


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3





class GridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5, goal_idx=0):
        self.action = [UP, DOWN, LEFT, RIGHT]
        self.action_space = spaces.Discrete(4)
        self.action_grid_change = {UP:[-1,0],DOWN:[1,0], LEFT:[0,-1],RIGHT:[0,1]}

        self.grid_size = grid_size
        self.grid_width = self.grid_size     #verticle 
        self.grid_length = self.grid_size    #horizontal
        self.state_space = spaces.Box(low=0, high=1, shape=[grid_size,grid_size,1])
        self.goal_idx = goal_idx

        self.reset()
        

    def step(self, action):
        next_pos = [self.curr_y+self.action_grid_change[action][0],self.curr_x+self.action_grid_change[action][1]]
        done = False
        if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.grid_width-1 or next_pos[1] > self.grid_length-1:
            return self.state, self.get_reward([self.curr_y, self.curr_x]), done

        self.curr_y = next_pos[0]
        self.curr_x = next_pos[1]
        self.state_update()

        if self.curr_y == self.goal_y and self.curr_x == self.goal_x:
            done = True

        return self.state, self.get_reward([self.curr_y, self.curr_x]), done

    
    def state_update(self):
        self.state = np.zeros((self.grid_width, self.grid_length))
        self.state[self.curr_y][self.curr_x] = 1.0

        self.visual_space = np.zeros((self.grid_width, self.grid_length))
        self.visual_space[self.curr_y][self.curr_x] = 1
        self.visual_space[self.goal_y][self.goal_x] = 2

    def idx_to_xy(self, idx):
        x = idx % self.grid_size
        y = int(idx/self.grid_size)

        return x, y


    def reset(self):
        
        self.goal_x, self.goal_y = self.idx_to_xy(self.goal_idx)

        self.reward_state = np.zeros((self.grid_width, self.grid_length))
        self.reward_state[self.goal_y][self.goal_x] = 1.0

        self.curr_y = self.grid_width-1
        self.curr_x = 0

        self.state_update()


    def render(self, mode='human'):
        fig = plt.figure(0)
        plt.clf()
        plt.imshow(self.visual_space)
        fig.canvas.draw()
        plt.pause(0.00001)
            

    def get_reward(self,pos):
        return self.reward_state[pos[0]][pos[1]]



    def close(self):
        pass


    def transition_prob(self):
        N_total_states = self.grid_width*self.grid_length
        p = np.zeros((N_total_states, N_total_states, len(self.action)))

        def flatToXY(i):
            
            y = int(i/self.grid_width)
            x = int(i%self.grid_length)

            return y, x

        def XYToFlat(y, x):
            i = y*self.grid_length+x

            return i

        for s in range(N_total_states):
            for a in range(len(self.action)):
                iy, ix = flatToXY(s)

                next_pos = [iy+self.action_grid_change[a][0],ix+self.action_grid_change[a][1]]
                if next_pos[0] >= 0 and next_pos[1] >= 0 and next_pos[0] <= self.grid_width-1 and next_pos[1] <= self.grid_length-1:
                    
                    i = XYToFlat(next_pos[0], next_pos[1])
                    p[s][i][a] = 1.0
                else:
                    p[s][s][a] = 1.0

        return p

                    


        


    def get_keyboard_action(self, action):
        if action == 'w':
            return UP
        
        if action == 's':
            return DOWN

        if action == 'a':
            return LEFT

        if action == 'd':
            return RIGHT
        



    def value_iteration(self, gamma):
        V = np.zeros((self.grid_width,self.grid_length))
        for q in range(0,100):
            for i in range(0,len(self.state)):
                for j in range(0,len(self.state[0])):
                    if i == self.goal_y and j == self.goal_x:
                        V[i][j] = 1.0
                        continue


                    neighbor_vals = {}
                    for k in self.action:
                        neighbor_y = i+self.action_grid_change[k][0]
                        neighbor_x = j+self.action_grid_change[k][1]
                        reward = self.get_reward([i,j])
                        
                        if neighbor_y < 0 or neighbor_y >= len(V):
                            continue
                        if neighbor_x < 0 or neighbor_x >= len(V[0]):
                            continue
                        
                        neighbor_val = V[neighbor_y][neighbor_x]
                        value = reward+gamma*neighbor_val
                        neighbor_vals[k] = value
                    #print(neighbor_vals,i,j)

                    max_value_action = max(neighbor_vals, key=neighbor_vals.get)
                    #print(max_value_action,i,j)
                    max_y = i + self.action_grid_change[max_value_action][0]
                    max_x = j + self.action_grid_change[max_value_action][1]
                    curr_reward = self.get_reward([i,j])
                    V[i][j] = curr_reward+gamma*V[max_y][max_x]
        
        return V

    def generate_policy(self, V, gamma):
        opti_policy = np.zeros((self.grid_width,self.grid_length))
        for i in range(0,len(self.state)):
            for j in range(0,len(self.state[0])):
                if i == self.goal_y and j == self.goal_x:
                    opti_policy[i][j] = RIGHT
                    continue

                neighbor_vals = {}
                for k in self.action:
                    neighbor_y = i+self.action_grid_change[k][0]
                    neighbor_x = j+self.action_grid_change[k][1]
                    reward = self.get_reward([i,j])
                    
                    if neighbor_y < 0 or neighbor_y >= len(V):
                        continue
                    if neighbor_x < 0 or neighbor_x >= len(V[0]):
                        continue
                    
                    neighbor_val = V[neighbor_y][neighbor_x]
                    value = reward+gamma*neighbor_val
                    neighbor_vals[k] = value

                max_value_action = max(neighbor_vals, key=neighbor_vals.get)
                opti_policy[i][j] = max_value_action



        return np.array(opti_policy, dtype=int), self.visualize_policy(opti_policy, self.grid_width, self.grid_length)

    def visualize_policy(self, policy, grid_width, grid_length):
        visual_policy = np.empty((grid_width,grid_length)).astype('U')
        for i in range(0,len(policy)):
            for j in range(0,len(policy[0])):
                if policy[i][j] == UP:
                    visual_policy[i][j] = '\u2191'
                elif policy[i][j] == DOWN:
                    visual_policy[i][j] = '\u2193'
                elif policy[i][j] == LEFT:
                    visual_policy[i][j] = '\u2190'
                else:
                    visual_policy[i][j] = '\u2192'
        return np.array(visual_policy)


                

                

