import isaacgym
import isaacgymenvs
import torch


envs = isaacgymenvs.make(
    seed=0, 
    task="Ant", 
    num_envs=2, 
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0
)
