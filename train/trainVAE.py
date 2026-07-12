from models.VAE import VAE

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from torch.utils.data import DataLoader

def loss_fn(data, reconst, mu, logstd):
    bce = torch.nn.functional.mse_loss(reconst, data, reduction='sum')
    KL_Divergence = -0.5 * torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(2 * logstd))

    return bce + KL_Divergence

def train_VAE(n_ep, loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_ep = 20

    for epoch in range(n_ep):
        total_loss = 0.0
        for batch_idx, data in enumerate(loader):
            optimizer.zero_grad()
            
            mu, logstd, decoded = model(data.float() / 256)
            loss = loss_fn(data.float() / 256, decoded, mu, logstd)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader.dataset)
        print(
            "Epoch {}/{}: loss={:.4f}".format(epoch + 1, n_ep, epoch_loss)
        )

if __name__ == "__main__":
    observations = generate_episode(1, env)
    observations = torch.tensor(np.array(observations))
    dataset = observations
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_ep = 20