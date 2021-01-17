import gym
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from IRL_MaxEnt import IRL_MaxEnt, RewardNet




if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Running on the CPU")


n_features = 1



r_model = RewardNet(n_features, 64).to(device)
optimizer = optim.Adam(r_model.parameters(), lr=0.01)



