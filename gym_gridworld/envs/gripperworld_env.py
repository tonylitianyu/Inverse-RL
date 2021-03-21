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
GRIPPER_OPEN = 4
GRIPPER_CLOSE = 5





class GripperEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5, goal_idx=(0,0,0)):
        self.action = [UP, DOWN, LEFT, RIGHT, GRIPPER_OPEN, GRIPPER_CLOSE]
        self.action_space = spaces.Discrete(6)
        self.action_grid_change = {GRIPPER_OPEN:[-1,0,0], GRIPPER_CLOSE:[1,0,0], UP:[0,-1,0],DOWN:[0,1,0], LEFT:[0,0,-1],RIGHT:[0,0,1]}

        self.grid_size = grid_size
        self.grid_width = self.grid_size     #verticle 
        self.grid_length = self.grid_size    #horizontal
        self.state_space = spaces.Box(low=0, high=1, shape=[2,grid_size,grid_size])
        self.goal_idx = goal_idx

        self.reset()
        

    def step(self, action):
        next_pos = [self.curr_gripper+self.action_grid_change[action][0], self.curr_y+self.action_grid_change[action][1],self.curr_x+self.action_grid_change[action][2]]
        done = False
        if next_pos[1] < 0 or next_pos[2] < 0 or next_pos[1] > self.grid_width-1 or next_pos[2] > self.grid_length-1 or next_pos[0] > 1 or next_pos[0] < 0:
            return self.state, self.get_reward([self.curr_y, self.curr_x]), done

        self.curr_y = next_pos[1]
        self.curr_x = next_pos[2]
        self.curr_gripper = next_pos[0]
        self.state_update()

        if self.curr_y == self.goal_y and self.curr_x == self.goal_x and self.curr_gripper == self.goal_gripper:
            done = True

        return self.state, self.get_reward([self.curr_y, self.curr_x,self.curr_gripper]), done

    
    def state_update(self):
        self.state = np.zeros((2,self.grid_width, self.grid_length))
        self.state[self.curr_gripper][self.curr_y][self.curr_x] = 1.0

        self.visual_space = np.zeros((2,self.grid_width, self.grid_length))
        self.visual_space[self.curr_gripper][self.curr_y][self.curr_x] = 1
        self.visual_space[self.goal_gripper][self.goal_y][self.goal_x] = 2

    # def idx_to_xy(self, idx):
    #     x = idx % self.grid_size
    #     y = int(idx/self.grid_size)

    #     return x, y


    def reset(self):
        
        self.goal_gripper, self.goal_x, self.goal_y = self.goal_idx

        self.reward_state = np.zeros((2,self.grid_width, self.grid_length))
        self.reward_state[self.goal_gripper][self.goal_y][self.goal_x] = 1.0

        self.curr_y = self.grid_width-1
        self.curr_x = 0
        self.curr_gripper = 0

        self.state_update()


    def render(self, mode='human'):
        fig = plt.figure(0)
        plt.clf()
        plt.imshow(self.visual_space)
        fig.canvas.draw()
        plt.pause(0.00001)
            

    def get_reward(self,pos):
        return self.reward_state[pos[0]][pos[1]][pos[2]]



    def close(self):
        pass

    def flatToXYG(self,i,n):

        g = 0
        if i < int(n/2) - 1:
            g = 0 #open
        else:
            g = 1 #close

        
        
        y = int((i%(self.grid_width*self.grid_length))/self.grid_width)
        x = int((i%(self.grid_width*self.grid_length))%self.grid_length)

        return g, y, x

    def XYGToFlat(self,g, y, x, n):
        i = y*self.grid_length+x+(g*int(n/2))

        return i

    def transition_prob(self):
        N_total_states = self.grid_width*self.grid_length*2
        p = np.zeros((N_total_states, N_total_states, len(self.action)))


        for s in range(N_total_states):
            for a in range(len(self.action)):
                ig, iy, ix = self.flatToXYG(s, N_total_states)
                

                next_pos = [ig+self.action_grid_change[a][0], iy+self.action_grid_change[a][1],ix+self.action_grid_change[a][2]]
                if next_pos[1] >= 0 and next_pos[2] >= 0 and next_pos[0] >= 0 and next_pos[1] <= self.grid_width-1 and next_pos[2] <= self.grid_length-1 and next_pos[0] <= 1:
                    
                    i = self.XYGToFlat(next_pos[0], next_pos[1], next_pos[2], N_total_states)
                    # print("Current state is " + str(s) + " take action " + str(a) + " reach " + str(i))
                    # print("XYGToFlat "+ str(next_pos[0]) + " " + str(next_pos[1]) +" "+ str(next_pos[2]) + " is "+str(i))
                    # print("======================")
                    p[s][i][a] = 1.0
                else:
                    p[s][s][a] = 1.0

        return p



    def visualize_policy(self, policy, grid_width, grid_length):
        visual_policy = np.empty((2,grid_width,grid_length)).astype('U')
        for i in range(0,len(policy[0])):
            for j in range(0,len(policy[1])):
                for k in range(0,2):
                    if policy[k][i][j] == UP:
                        visual_policy[k][i][j] = '\u2191'
                    elif policy[k][i][j] == DOWN:
                        visual_policy[k][i][j] = '\u2193'
                    elif policy[k][i][j] == LEFT:
                        visual_policy[k][i][j] = '\u2190'
                    elif policy[k][i][j] == RIGHT:
                        visual_policy[k][i][j] = '\u2192'
                    elif policy[k][i][j] == GRIPPER_CLOSE:
                        visual_policy[k][i][j] = '\u2206'
                    else:
                        visual_policy[k][i][j] = '\u2207'
        return np.array(visual_policy)


                

                

