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

EMPTY = [0.0,0.0,0.0]
AGENT = [0.0,0.0,1.0]
GOAL = [0.0,1.0,0.0]
START = [1.0,0.0,0.0]

GRID_SIZE = 4

class GridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action = [UP, DOWN, LEFT, RIGHT]
        self.action_space = spaces.Discrete(4)
        self.action_grid_change = {UP:[-1,0],DOWN:[1,0], LEFT:[0,-1],RIGHT:[0,1]}

        self.state_space = spaces.Box(low=0, high=3, shape=[GRID_SIZE,GRID_SIZE,3])

        #for reset
        self.start_pos_temp = [0,0]
        self.goal_pos_temp = [3,3]

        self.start_pos = copy.deepcopy(self.start_pos_temp)
        self.goal_pos = copy.deepcopy(self.goal_pos_temp)
        self.agent_pos = copy.deepcopy(self.start_pos_temp)

        self.curr_state = np.zeros((GRID_SIZE,GRID_SIZE,3))
        self.set_state_pos(self.goal_pos, GOAL)
        self.set_state_pos(self.start_pos, START)
        self.set_state_pos(self.agent_pos, AGENT)

        self.visual = False


        #self.policy = np.array([[RIGHT,RIGHT,RIGHT,DOWN],[RIGHT,RIGHT,RIGHT,DOWN],[RIGHT,RIGHT,RIGHT,DOWN],[RIGHT,RIGHT,RIGHT,LEFT]])
        self.policy = np.array([[RIGHT,RIGHT,RIGHT,DOWN],[DOWN,RIGHT,DOWN,DOWN],[DOWN,RIGHT,DOWN,DOWN],[RIGHT,RIGHT,RIGHT,DOWN]])


        self.grid_size = GRID_SIZE



    def step(self, action):
        next_agent_pos = [self.agent_pos[0]+self.action_grid_change[action][0],
                            self.agent_pos[1]+self.action_grid_change[action][1]]
        
        reward_val = -1.0
        done = False

        if next_agent_pos[0] < 0 or next_agent_pos[0] >= len(self.curr_state):
            return self.curr_state, reward_val, done
        
        if next_agent_pos[1] < 0 or next_agent_pos[1] >= len(self.curr_state):
            return self.curr_state, reward_val, done
        

        self.set_state_pos(self.agent_pos, EMPTY)
        self.set_state_pos(self.start_pos, START)
        self.set_state_pos(self.goal_pos, GOAL)
        
        self.set_state_pos(next_agent_pos, AGENT)
        self.agent_pos = copy.deepcopy(next_agent_pos)
        

        if self.agent_pos == self.goal_pos:
            reward_val = 0.0
            done = True

        return self.state_space, reward_val, done

        

    def reset(self):
        self.curr_state = np.zeros((GRID_SIZE,GRID_SIZE,3))
        self.set_state_pos(self.start_pos_temp, START)
        self.set_state_pos(self.goal_pos_temp, GOAL)
        self.set_state_pos(self.agent_pos, AGENT)
        self.start_pos = copy.deepcopy(self.start_pos_temp)
        self.goal_pos = copy.deepcopy(self.goal_pos_temp)
        self.agent_pos = copy.deepcopy(self.start_pos_temp)

    def render(self, mode='human'):
        if self.visual == False:
            return
        else:
            fig = plt.figure(0)
            plt.clf()
            plt.imshow(self.curr_state)
            fig.canvas.draw()
            plt.pause(0.00001)
            

    def get_reward(self,pos):
        #next_pos = [pos[0]+self.action_grid_change[action][0],pos[1]+self.action_grid_change[action][1]]
        if pos == self.goal_pos:
            return 1.0
        
        return 0.0



    def close(self):
        pass

    def transition_prob(self,iy,ix,action, jy,jx):
        next_agent_pos = [iy+self.action_grid_change[action][0],
                            ix+self.action_grid_change[action][1]]

        if next_agent_pos == [jy,jx]:
            return 1.0
        else:
            return 0.0
        




    def set_state_pos(self, update_pos, target_name):
        self.curr_state[update_pos[0]][update_pos[1]] = target_name


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
        V = np.zeros((GRID_SIZE,GRID_SIZE))
        for q in range(0,20):
            for i in range(0,len(self.curr_state)):
                for j in range(0,len(self.curr_state[0])):
                    if i == self.goal_pos[0] and j == self.goal_pos[1]:
                        V[i][j] = 0.0
                        continue


                    neighbor_vals = {}
                    for k in self.action:
                        neighbor_y = i+self.action_grid_change[k][0]
                        neighbor_x = j+self.action_grid_change[k][1]
                        reward = self.get_reward([neighbor_y,neighbor_x])
                        
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
                    max_reward = self.get_reward([max_y,max_x])
                    V[i][j] = max_reward+gamma*V[max_y][max_x]
        
        return V

    def generate_policy(self, V, gamma):
        opti_policy = np.zeros((GRID_SIZE,GRID_SIZE))
        for i in range(0,len(self.curr_state)):
            for j in range(0,len(self.curr_state[0])):
                if i == self.goal_pos[0] and j == self.goal_pos[1]:
                    opti_policy[i][j] = RIGHT
                    continue

                neighbor_vals = {}
                for k in self.action:
                    neighbor_y = i+self.action_grid_change[k][0]
                    neighbor_x = j+self.action_grid_change[k][1]
                    reward = self.get_reward([neighbor_y,neighbor_x])
                    
                    if neighbor_y < 0 or neighbor_y >= len(V):
                        continue
                    if neighbor_x < 0 or neighbor_x >= len(V[0]):
                        continue
                    
                    neighbor_val = V[neighbor_y][neighbor_x]
                    value = reward+gamma*neighbor_val
                    neighbor_vals[k] = value

                max_value_action = max(neighbor_vals, key=neighbor_vals.get)
                opti_policy[i][j] = max_value_action
        self.policy = np.array(opti_policy, dtype=int)
        return opti_policy

                

                

