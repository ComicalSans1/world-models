import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

env = gym.make("CarRacing-v3")

def generate_episode(n_ep, env):
    for _ in range(n_ep):
        observation, info = env.reset()
        observations = []
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            observations.append(observation)
            if done:
                break
    return observations

observations = generate_episode(1, env)
observations = torch.tensor(np.array(observations))

