import numpy as np
# import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.n1 = 128
        self.n2 = 64
        self.fc1 = nn.Linear(state_dim, self.n1)
        self.fc2 = nn.Linear(self.n1, self.n2)
        self.fc3 = nn.Linear(self.n2, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.n1 = 128
        self.n2 = 64
        self.fc1 = nn.Linear(state_dim + action_dim, self.n1)
        self.fc2 = nn.Linear(self.n1, self.n2)
        self.fc3 = nn.Linear(self.n2, 1)
    
    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state,action], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        # print(f"add : {experience[0]}")
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)
    
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_low, action_high):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.gamma = 0.99 # discount factor
        self.tau = 0.005 # soft target update (polyak averaging)

        self.update_target_networks()

    def update_target_networks(self):
        for target_param, para in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)
            # target_param.data.copy_(para.data)
        for target_param, para in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)
            # target_param.data.copy_(para.data)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        print(f"raw ac : {action}")
        return np.clip(action + noise * np.random.randn(len(action)), -1, 1)
    
    def train(self, batch_size=4):
        if self.replay_buffer.size() < batch_size:
            return
        experiences = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        print(f"state : {states}")
        print(f"ac : {actions}")
        print(f"rw : {rewards}")
        print(f"dn : {dones}")
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.update_target_networks()

if __name__ == "__main__":

    controller_name = "OSC_POSE"
    controller_config = load_controller_config(default_controller=controller_name)
    # create environment with selected grippers
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="UR5e",
            controller_configs=controller_config,
            reward_shaping=True,
            reward_scale=1.0,
            has_renderer=True,  # make sure we can render to the screen
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            use_camera_obs=False,  # do not use pixel observations
            # control_freq=50,  # control should happen fast enough so that simulation looks smoother
            camera_names="frontview",
        ),
        keys=["robot0_eef_pos","cube_pos"]
    )
    obs = env.reset()
    # print(f"a space : {env.action_space}")
    # print(f"a dim : {env.action_space.shape[0]}")
    # print(f"a low : {env.action_space.low[0]}")
    # print(f"a high : {env.action_space.high[0]}")
    # print(f"obs space : {env.observation_space.shape[0]}")
    # print(f"obs space : {dir(env)}")
    # print(f"obs space : {obs}")

    agent = DDPGAgent(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.shape[0],
                      action_low=env.action_space.low[0],
                      action_high=env.action_space.high[0])
    
    num_episodes = 5

    for episode in range(num_episodes):
        obs = env.reset()
        state = obs[0]
        total_reward = 0
        while True:
            # print(f"st3 : {state.shape}")
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # print(f"ns : {next_state.shape}")
            agent.replay_buffer.add((state, action, reward, next_state, float(done)))

            agent.train()
            state = next_state            
            total_reward += reward

            if done:
                print(f"Episode {episode} finished with total reward : {total_reward}")
                break
            env.render()
    env.close()