from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gym_gridworld.envs:GridEnv',
)

register(
    id='gripperworld-v0',
    entry_point='gym_gridworld.envs:GripperEnv',
)

register(
    id='gripperworld3d-v0',
    entry_point='gym_gridworld.envs:Gripper3dEnv',
)