import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
import matplotlib.pyplot as plt


FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
UP = 4
DOWN = 5
GRIPPER_OPEN = 6
GRIPPER_CLOSE = 7





class Gripper3dEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5, goal_idx=(0,0,0,0)):
        self.action = [FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN, GRIPPER_OPEN, GRIPPER_CLOSE]
        self.action_space = spaces.Discrete(6)
        self.action_grid_change = {GRIPPER_OPEN:[-1,0,0,0], GRIPPER_CLOSE:[1,0,0,0], FORWARD:[0,-1,0,0],BACKWARD:[0,1,0,0], LEFT:[0,0,-1,0],RIGHT:[0,0,1,0], UP:[0,0,0,1], DOWN:[0,0,0,-1]}

        self.grid_size = grid_size
        self.grid_width = self.grid_size     #verticle 
        self.grid_length = self.grid_size    #horizontal
        self.grid_height = self.grid_size    #z
        self.state_space = spaces.Box(low=0, high=1, shape=[2,grid_size,grid_size,grid_size])
        self.goal_idx = goal_idx

        self.reset()
        

    def step(self, action):
        next_pos = [self.curr_gripper+self.action_grid_change[action][0], self.curr_y+self.action_grid_change[action][1],self.curr_x+self.action_grid_change[action][2],self.curr_z+self.action_grid_change[action][3]]
        done = False
        if next_pos[1] < 0 or next_pos[2] < 0 or next_pos[3] < 0 or next_pos[1] > self.grid_width-1 or next_pos[2] > self.grid_length-1 or next_pos[3] > self.grid_height-1 or next_pos[0] > 1 or next_pos[0] < 0:
            return self.state, self.get_reward([self.curr_gripper, self.curr_y, self.curr_x, self.curr_z]), done

        self.curr_y = next_pos[1]
        self.curr_x = next_pos[2]
        self.curr_z = next_pos[3]
        self.curr_gripper = next_pos[0]
        self.state_update()

        if self.curr_y == self.goal_y and self.curr_x == self.goal_x and self.curr_z == self.goal_z and self.curr_gripper == self.goal_gripper:
            done = True

        return self.state, self.get_reward([self.curr_gripper, self.curr_y, self.curr_x, self.curr_z]), done

    
    def state_update(self):
        self.state = np.zeros((2,self.grid_width, self.grid_length, self.grid_height))
        self.state[self.curr_gripper][self.curr_y][self.curr_x][self.curr_z] = 1.0

        self.visual_space = np.zeros((2,self.grid_width, self.grid_length, self.grid_height))
        self.visual_space[self.curr_gripper][self.curr_y][self.curr_x][self.curr_z] = 1
        self.visual_space[self.goal_gripper][self.goal_y][self.goal_x][self.goal_z] = 2

    # def idx_to_xy(self, idx):
    #     x = idx % self.grid_size
    #     y = int(idx/self.grid_size)

    #     return x, y


    def reset(self):
        
        self.goal_gripper, self.goal_x, self.goal_y, self.goal_z = self.goal_idx

        self.reward_state = np.zeros((2,self.grid_width, self.grid_length, self.grid_height))
        self.reward_state[self.goal_gripper][self.goal_y][self.goal_x][self.goal_z] = 1.0

        self.curr_z = 0
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
        return self.reward_state[pos[0]][pos[1]][pos[2]][pos[3]]



    def close(self):
        pass

    def flatToXYZG(self,i,n):

        g = 0
        if i < int(n/2) - 1:
            g = 0 #open
        else:
            g = 1 #close

        
        z = int((i%(self.grid_width*self.grid_length*self.grid_height))/(self.grid_width*self.grid_length))
        y = int(((i - z*(self.grid_width*self.grid_length))%(self.grid_width*self.grid_length))/(self.grid_width))
        x = int(((i - z*(self.grid_width*self.grid_length))%(self.grid_width*self.grid_length))%(self.grid_length))
        
        return g, y, x, z

    def XYZGToFlat(self,g, y, x, z, n):
        i = y*self.grid_length+x+(g*int(n/2))+z*(self.grid_width*self.grid_length)

        return i

    def transition_prob(self):
        N_total_states = self.grid_height*self.grid_width*self.grid_length*2
        p = np.zeros((N_total_states, N_total_states, len(self.action)))

        for s in range(N_total_states):
            for a in range(len(self.action)):
                ig, iy, ix, iz = self.flatToXYZG(s, N_total_states)
                

                next_pos = [ig+self.action_grid_change[a][0], iy+self.action_grid_change[a][1],ix+self.action_grid_change[a][2], iz+self.action_grid_change[a][3]]
                if next_pos[1] >= 0 and next_pos[2] >= 0 and next_pos[3] >= 0 and next_pos[0] >= 0 and next_pos[1] <= self.grid_width-1 and next_pos[2] <= self.grid_length-1 and next_pos[3] <= self.grid_height-1 and next_pos[0] <= 1:
                    
                    i = self.XYZGToFlat(next_pos[0], next_pos[1], next_pos[2], next_pos[3],N_total_states)
                    # print("Current state is " + str(s) + " take action " + str(a) + " reach " + str(i))
                    # print("XYGToFlat "+ str(next_pos[0]) + " " + str(next_pos[1]) +" "+ str(next_pos[2]) + " is "+str(i))
                    # print("======================")
                    p[s][i][a] = 1.0
                else:
                    p[s][s][a] = 1.0

        return p



    def visualize_policy(self, policy, grid_width, grid_length, grid_height):
        visual_policy = np.empty((2,grid_width,grid_length, grid_height)).astype('U')

        for i in range(0,grid_width):
            for j in range(0,grid_length):
                for k in range(0,2):
                    for p in range(0,grid_height):
                        if policy[k][i][j][p] == FORWARD:
                            visual_policy[k][i][j][p] = 'FORWARD'
                        elif policy[k][i][j][p] == BACKWARD:
                            visual_policy[k][i][j][p] = 'BACKWARD'
                        elif policy[k][i][j][p] == LEFT:
                            visual_policy[k][i][j][p] = 'LEFT'
                        elif policy[k][i][j][p] == RIGHT:
                            visual_policy[k][i][j][p] = 'RIGHT'
                        elif policy[k][i][j][p] == GRIPPER_CLOSE:
                            visual_policy[k][i][j][p] = 'CLOSE'
                        elif policy[k][i][j][p] == GRIPPER_OPEN:
                            visual_policy[k][i][j][p] = 'OPEN'
                        elif policy[k][i][j][p] == UP:
                            visual_policy[k][i][j][p] = 'UP'
                        elif policy[k][i][j][p] == DOWN:
                            visual_policy[k][i][j][p] = 'DOWN'
                        else:
                            visual_policy[k][i][j][p] = 'STAY'
        return np.array(visual_policy)