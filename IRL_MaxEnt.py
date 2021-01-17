import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional

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
        self.expert_state_freq = self.expert_state_freq
        self.n_features = n_features
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition = transition
        self.gamma = gamma

        
