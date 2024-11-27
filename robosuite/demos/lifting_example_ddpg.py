import os, sys, random
from pathlib import Path
from itertools import count
import numpy as np
# import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from utils.common_utils import load_config, set_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "Lift"
robot_name = "UR5e"
controller_name = "OSC_POSE"
script_name = os.path.basename(__file__)
directory = './exp' + script_name + env_name +'/'
# update_iteration = 200
mode = "test"
to_load = mode == "test"
epoch_to_load = 40
num_epochs = 180
num_episodes = 75
episode_horizon = 150
n_fc1 = 500
n_fc2 = 500
test_epochs = 10
render_training = False
render_interval = 10
tau = 0.005
gamma = 0.99
lr_actor = 0.001
lr_critic = 0.001
eps = 1.5 # for noise
eps_decay = 0.0000015 # eps / episode_horizon
batch_size = 64
buffer_max_size = 50000

# current_path = Path(os.getcwd()).resolve()
current_path = os.path.dirname(os.path.abspath(__file__))
logging_path = current_path + '/utils/default_logger.conf'
saving_path = current_path + '/weights'
os.makedirs(saving_path, exist_ok=True)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high, n1, n2, init_w=3e-3):
        super(Actor, self).__init__()
        # self.n1 = 300
        # self.n2 = 300
        self.fc1 = nn.Linear(state_dim, n1)
        self.bn1 = nn.LayerNorm(n1)
        self.fc2 = nn.Linear(n1, n2)
        self.bn2 = nn.LayerNorm(n2)
        self.fc3 = nn.Linear(n2, action_dim)
        self.init_weights(init_w)
        self.action_high = action_high
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.action_high * torch.tanh(self.fc3(x))
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n1, n2, init_w=3e-3):
        super(Critic, self).__init__()
        # self.n1 = 400
        # self.n2 = 300
        self.fc1 = nn.Linear(state_dim + action_dim, n1)
        self.bn1 = nn.LayerNorm(n1)
        self.fc2 = nn.Linear(n1, n2)
        self.bn2 = nn.LayerNorm(n2)
        self.fc3 = nn.Linear(n2, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        x = torch.relu(self.bn1(self.fc1(torch.cat([state,action], 1))))
        x = torch.relu(self.bn2(self.fc2(x)))
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
    def __init__(self, state_dim, action_dim, action_low, action_high, n1, n2, 
                 lr_actor, lr_critic, to_load, epoch_to_load, buffer_max_size, gamma, tau, eps, eps_decay, batch_size):
        self.actor = Actor(state_dim, action_dim, action_high, n1, n2).to(device)
        self.critic = Critic(state_dim, action_dim, n1, n2).to(device)
        self.target_actor = Actor(state_dim, action_dim, action_high, n1, n2).to(device)
        self.target_critic = Critic(state_dim, action_dim, n1, n2).to(device)

        # if to_load:
        #     self.load(epoch_to_load)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        if to_load == False:
            self.update_target_networks(is_hard=True)

        self.replay_buffer = ReplayBuffer(max_size=buffer_max_size)
        self.gamma = gamma # discount factor
        self.tau = tau # soft target update (polyak averaging)
        self.eps = eps
        self.eps_decay = eps_decay
        self.batch_size = batch_size

        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.update_target_networks()

    def update_target_networks(self, is_hard=True):
        if is_hard:
            for target_param, para in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(para.data)
            for target_param, para in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(para.data)
        else:
            for target_param, para in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)
                # target_param.data.copy_(para.data)
            for target_param, para in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)
                # target_param.data.copy_(para.data)

    def select_action(self, state, epoch=1, total_epochs=1, noise=0.15, min_noise = 0.01, is_training=True):
        noise_scale = max(noise * (1 - epoch / total_epochs), min_noise)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # print(f"raw ac : {action}")
        action = np.clip(action + is_training * max(self.eps,0) * np.random.normal(0, noise_scale, size=self.action_dim), self.action_low, self.action_high)
        self.eps = max(self.eps - self.eps_decay, 0)
        return action
    
    def train(self):
        # for it in range(update_iteration): # TO CHECK
        states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = abs(1 - torch.FloatTensor(dones)).to(device)
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions)
            target_q = rewards + (dones * self.gamma * target_q)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        # critic_loss = nn.functional.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.update_target_networks(is_hard=False)
        
        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1

    def save(self, dir, epoch):
        torch.save(self.actor.state_dict(), dir + "/actor_{}.pt".format(epoch+1))
        torch.save(self.critic.state_dict(), dir + "/critic_{}.pt".format(epoch+1))
        torch.save(self.target_actor.state_dict(), dir + "/target_actor_{}.pt".format(epoch+1))
        torch.save(self.target_critic.state_dict(), dir + "/target_critic_{}.pt".format(epoch+1))

    def load(self, dir, epoch):
        self.actor.load_state_dict(torch.load(dir + "/actor_{}.pt".format(epoch)))
        self.critic.load_state_dict(torch.load(dir + "/critic_{}.pt".format(epoch)))
        self.target_actor.load_state_dict(torch.load(dir + "/target_actor_{}.pt".format(epoch)))
        self.target_critic.load_state_dict(torch.load(dir + "/target_critic_{}.pt".format(epoch)))
        print("====================================")
        print(f"model {epoch} has been loaded...")
        print("====================================")

if __name__ == "__main__":

    controller_config = load_controller_config(default_controller=controller_name)
    # create environment with selected grippers
    env = GymWrapper(
        suite.make(
            env_name,
            robots=robot_name,
            controller_configs=controller_config,
            reward_shaping=True,
            reward_scale=1.0,
            has_renderer=True,  # make sure we can render to the screen
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            use_camera_obs=False,  # do not use pixel observations
            # control_freq=50,  # control should happen fast enough so that simulation looks smoother
            camera_names="frontview",
        ),
        keys=["robot0_proprio-state","object-state"]
    )
    agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0],
                      env.action_space.low[0], env.action_space.high[0], n_fc1, n_fc2,
                      lr_actor, lr_critic, to_load, epoch_to_load, buffer_max_size, gamma, tau, eps, eps_decay, batch_size)
    print("sap",env.action_space)
    a = torch.prod(torch.Tensor(env.action_space.shape))
    print("a",a)
    if mode == "train":
        # Set logging
        log = set_logging(logging_path)
        log.info("Environment: {} \n Robot: {}\n Controller {}\n".format(env_name, robot_name, controller_name))
        log.info("Training started...")
        # agent.load()
        total_step = 0
        for epoch in range(num_epochs):
            epoch_reward = 0
            for episode in range(num_episodes):
                obs = env.reset()
                state = obs[0]
                episode_reward = 0
                done = False
                step = 0
                while (done == False) & (step <= episode_horizon):
                    action = agent.select_action(state, epoch=epoch, total_epochs=num_epochs)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    if render_training and episode % render_interval == 0:
                        env.render()
                    agent.replay_buffer.add((state, next_state, action, reward, np.float(done)))

                    state = next_state    
                    agent.train()

                    step += 1
                    episode_reward += reward
                epoch_reward += episode_reward
                total_step += step+1
                # log.info("Episode:{} Steps:\t{}  Total Reward:\t{:0.2f}".format(episode, step, episode_reward))
            log.info("Epoch:{} Epoch Reward:\t{:0.2f}".format(epoch, epoch_reward/num_episodes))
            agent.save(dir=saving_path, epoch=epoch)
    
    elif mode == "test":
        agent.load(dir=saving_path, epoch=epoch_to_load)
        for i in range(test_epochs):
            obs = env.reset()
            state = obs[0]
            done = False
            step = 0
            ep_r = 0
            while (done == False) & (step <= episode_horizon):
                action = agent.select_action(state, is_training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1
                ep_r += reward
                env.render()
            print("Game:{} Reward:\t{:0.2f}".format(i, ep_r))
    env.close()