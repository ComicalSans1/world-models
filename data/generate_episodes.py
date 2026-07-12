import gymnasium as gym
import torch
from pathlib import Path

env = gym.make("CarRacing-v3")

def generate_episode_VAE(n_ep, env):
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

def generate_episode_RNN(n_ep, env):
    
    observations = []
    actions = []
    rewards = []
    dones = []
    next_states = []

    observation, info = env.reset()

    for _ in range(n_ep + 1):
        temp_obs = []
        temp_acts = []
        temp_rews = []
        temp_dones = []
        temp_next = []
        
        terminated_early = False

        for _ in range(100):
            temp_obs.append(observation)
            
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            temp_acts.append(action)
            temp_rews.append(reward)
            temp_dones.append(done)
            temp_next.append(observation)
            
            if done:
                observation, info = env.reset()
                terminated_early = True
                break
        
        if not terminated_early:
            observations.append(temp_obs)
            actions.append(temp_acts)
            rewards.append(temp_rews)
            dones.append(temp_dones)
            next_states.append(temp_next)
                
    return observations, actions, rewards, dones, next_states

if __name__ == "__main__":
    observations, actions, rewards, dones, next_states = generate_episode_RNN(20, env)

    tensors_dict = {
        "observations" : observations,
        "actions" : actions,
        "rewards" : rewards,
        "dones" : dones,
        "next_states" : next_states,
    }

    script_dir = Path(__file__).resolve().parent
    current_dir = script_dir / 'observations.pt'
    torch.save(tensors_dict, current_dir)