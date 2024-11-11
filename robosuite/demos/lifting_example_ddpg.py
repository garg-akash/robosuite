import os, sys, random
from itertools import count
import numpy as np
# import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env_name = "Lift"
script_name = os.path.basename(__file__)
directory = './exp' + script_name + env_name +'/'
update_iteration = 200
log_interval = 50
mode = "test"
train_episodes = 1000000
test_iterations = 10
render_training = False
render_interval = 10

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high):
        super(Actor, self).__init__()
        self.n1 = 400
        self.n2 = 300
        self.fc1 = nn.Linear(state_dim, self.n1)
        self.fc2 = nn.Linear(self.n1, self.n2)
        self.fc3 = nn.Linear(self.n2, action_dim)
        self.action_high = action_high
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.action_high * torch.tanh(self.fc3(x))
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.n1 = 400
        self.n2 = 300
        self.fc1 = nn.Linear(state_dim + action_dim, self.n1)
        self.fc2 = nn.Linear(self.n1, self.n2)
        self.fc3 = nn.Linear(self.n2, 1)
    
    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state,action], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size):
        # self.buffer = deque(maxlen=max_size)
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, experience):
        # self.buffer.append(experience)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = experience
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(experience)

    def sample(self, batch_size):
        # return random.sample(self.buffer, batch_size)
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    # def size(self):
    #     return len(self.buffer)
    
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_low, action_high):
        self.actor = Actor(state_dim, action_dim, action_high).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, action_high).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(max_size=1000000)
        self.gamma = 0.99 # discount factor
        self.tau = 0.005 # soft target update (polyak averaging)

        self.action_dim = action_dim
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.update_target_networks()

    def update_target_networks(self):
        for target_param, para in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)
            # target_param.data.copy_(para.data)
        for target_param, para in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)
            # target_param.data.copy_(para.data)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # print(f"raw ac : {action}")
        return np.clip(action + noise * np.random.normal(0, noise, size=self.action_dim), -1, 1)
    
    def train(self, batch_size=100):
        for it in range(update_iteration):
            states, next_states, actions, rewards, dones = self.replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)
            # print(f"state : {states}")
            # print(f"ac : {actions}")
            # print(f"rw : {rewards}")
            # print(f"dn : {dones}")
            with torch.no_grad():
                target_actions = self.target_actor(next_states)
                target_q = self.target_critic(next_states, target_actions)
                target_q = rewards + ((1 - dones) * self.gamma * target_q)

            current_q = self.critic(states, actions)
            # critic_loss = nn.MSELoss()(current_q, target_q)
            critic_loss = nn.functional.mse_loss(current_q, target_q)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.update_target_networks()
            
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

if __name__ == "__main__":

    controller_name = "OSC_POSE"
    controller_config = load_controller_config(default_controller=controller_name)
    # create environment with selected grippers
    env = GymWrapper(
        suite.make(
            env_name,
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
    
    agent = DDPGAgent(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.shape[0],
                      action_low=env.action_space.low[0],
                      action_high=env.action_space.high[0])
    
    if mode == "train":
        agent.load()
        total_step = 0
        for episode in range(train_episodes):
            obs = env.reset()
            state = obs[0]
            total_reward = 0
            step = 0
            for t in count():
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if render_training and episode % render_interval == 0:
                    env.render()
                agent.replay_buffer.add((state, next_state, action, reward, np.float(done)))

                state = next_state            
                if done:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, episode, total_reward))
            agent.train()
            if episode % log_interval == 0:
                agent.save()
    
    elif mode == "test":
        agent.load()
        ep_r = 0
        for i in range(test_iterations):
            obs = env.reset()
            state = obs[0]
            for t in count():
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_r += reward
                env.render()
                if done:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state
    env.close()