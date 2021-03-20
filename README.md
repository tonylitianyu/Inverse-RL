# Inverse Reinforcement Learning

This repo contains implementation of the following papers on inverse reinforcement learning. The purpose is to achieve robot learning from demonstration via inverse reinforcement learning.


## Linear Programming
Inverse reinforcement learning using linear programming based on the paper:  
*Ng, Andrew Y., and Stuart J. Russell. "Algorithms for inverse reinforcement learning." Icml. Vol. 1. 2000.*

```
python3 main.py
```

## Maximum Entropy Deep Inverse Reinforcement Learning
Inverse reinforcement learning using maximum entropy deep learning based on the paper:
*Wulfmeier, Markus, Peter Ondruska, and Ingmar Posner. "Maximum entropy deep inverse reinforcement learning." arXiv preprint arXiv:1507.04888 (2015).*

For 2D Environment with discretized (x,y) states
```
python3 main_MaxEnt.py
```

For 2D Environment with discretized (x,y,gripper) states
```
python3 main_Gripper.py
```

For 3D Environment with discretized (x,y,z,gripper) states
```
python3 main_Gripper3d.py
```

For simulation of the policy in pybullet. The models are saved in ```trained_model``` folder
```
python3 simulation.py
```

## Dependencies
```
gym
pytorch
```



