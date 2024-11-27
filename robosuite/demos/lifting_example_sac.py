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
mode = "train"
to_load = mode == "test"
epoch_to_load = 129
num_epochs = 150
num_episodes = 75
episode_horizon = 150
n_fc1 = 256
n_fc2 = 256
test_epochs = 10
render_training = False
render_interval = 10
tau = 0.005
gamma = 0.99
alpha = 0.2
lr_actor = 0.003
lr_critic = 0.003
lr_alpha = 0.003
automatic_entropy_tuning = False
batch_size = 128
buffer_max_size = 500000
target_update_interval = 1

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
    def __init__(self, state_dim, action_dim, n1, n2, action_low, action_high, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, n1)
        self.bn1 = nn.LayerNorm(n1)
        self.fc2 = nn.Linear(n1, n2)
        self.bn2 = nn.LayerNorm(n2)
        self.mean_layer = nn.Linear(n2, action_dim)
        self.log_std_layer = nn.Linear(n2, action_dim)
        self.init_weights(init_w)
        self.min_log_std = -20
        self.max_log_std = 2
        self.epsilon = 1e-6

        self.action_scale = torch.FloatTensor([float(action_high - action_low) / 2.])
        self.action_bias = torch.FloatTensor([float(action_high + action_low) / 2.])
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.mean_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mean, log_std
    
    def sample_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        if deterministic:
            action = mean
            return action
        normal = Normal(mean, std)
        x_t = normal.rsample() # sampling with reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n1, n2, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1_q1 = nn.Linear(state_dim + action_dim, n1)
        self.bn1_q1 = nn.LayerNorm(n1)
        self.fc2_q1 = nn.Linear(n1, n2)
        self.bn2_q1 = nn.LayerNorm(n2)
        self.fc3_q1 = nn.Linear(n2, 1)

        # self.fc1_q2 = nn.Linear(state_dim + action_dim, n1)
        # self.bn1_q2 = nn.LayerNorm(n1)
        # self.fc2_q2 = nn.Linear(n1, n2)
        # self.bn2_q2 = nn.LayerNorm(n2)
        # self.fc3_q2 = nn.Linear(n2, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1_q1.weight.data = fanin_init(self.fc1_q1.weight.data.size())
        self.fc2_q1.weight.data = fanin_init(self.fc2_q1.weight.data.size())
        self.fc3_q1.weight.data.uniform_(-init_w, init_w)

        # self.fc1_q2.weight.data = fanin_init(self.fc1_q2.weight.data.size())
        # self.fc2_q2.weight.data = fanin_init(self.fc2_q2.weight.data.size())
        # self.fc3_q2.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        x1 = torch.relu(self.bn1_q1(self.fc1_q1(torch.cat([state,action], 1))))
        x1 = torch.relu(self.bn2_q1(self.fc2_q1(x1)))
        q1 = self.fc3_q1(x1)

        # x2 = torch.relu(self.bn1_q2(self.fc1_q2(torch.cat([state,action], 1))))
        # x2 = torch.relu(self.bn2_q2(self.fc2_q2(x2)))
        # q2 = self.fc3_q2(x2)

        # return q1, q2
        return q1

class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, experience):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = experience
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(experience)

    def sample(self, batch_size):
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
    
class SACAgent:
    def __init__(self, state_dim, action_dim, action_low, action_high, n1, n2, 
                 lr_actor, lr_critic, lr_alpha, to_load, epoch_to_load, buffer_max_size, gamma, tau, batch_size,
                 alpha, automatic_entropy_tuning, target_update_interval):
        # action_dim = action_space_shape[0]
        self.actor = Actor(state_dim, action_dim, n1, n2, action_low, action_high).to(device)
        self.critic_1 = Critic(state_dim, action_dim, n1, n2).to(device)
        self.critic_2 = Critic(state_dim, action_dim, n1, n2).to(device)
        # self.target_actor = Actor(state_dim, action_dim, action_high, n1, n2).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim, n1, n2).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim, n1, n2).to(device)

        # if to_load:
        #     self.load(epoch_to_load)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim_1 = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_optim_2 = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        if to_load == False:
            self.update_target_networks(is_hard=True)

        self.replay_buffer = ReplayBuffer(max_size=buffer_max_size)
        self.gamma = gamma # discount factor
        self.tau = tau # soft target update (polyak averaging)
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_update_interval = target_update_interval
        if self.automatic_entropy_tuning is True:
            # self.target_entropy = -torch.prod(torch.Tensor(action_space_shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)


        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.update_target_networks()

    def update_target_networks(self, is_hard=True):
        if is_hard:
            # for target_param, para in zip(self.target_actor.parameters(), self.actor.parameters()):
            #     target_param.data.copy_(para.data)
            for target_param, para in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(para.data)
            for target_param, para in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(para.data)
        else:
            # for target_param, para in zip(self.target_actor.parameters(), self.actor.parameters()):
            #     target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)
            for target_param, para in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)
            for target_param, para in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + para.data * self.tau)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _ = self.actor.sample_action(state)
        return action.cpu().data.numpy().flatten()
    
    def train(self, iter):
        if len(self.replay_buffer.storage) < self.batch_size:
            return
        
        states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = abs(1 - torch.FloatTensor(dones)).to(device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_action(next_states)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs # TODO
            target_q = rewards + (dones * self.gamma * target_q)

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss1 = nn.MSELoss()(current_q1, target_q)
        critic_loss2 = nn.MSELoss()(current_q2, target_q)

        self.critic_optim_1.zero_grad()
        critic_loss1.backward()
        self.critic_optim_1.step()

        self.critic_optim_2.zero_grad()
        critic_loss2.backward()
        self.critic_optim_2.step()

        pi, log_pi = self.actor.sample_action(states)
        q1 = self.critic_1(states, pi)
        q2 = self.critic_2(states, pi)
        q = torch.min(q1, q2)
        actor_loss = -(q - self.alpha * log_pi).mean() # TODO

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if iter % self.target_update_interval == 0: # delayed policy update
            self.update_target_networks(is_hard=False)
        
        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1

    def save(self, dir, epoch):
        torch.save(self.actor.state_dict(), dir + "/actor_{}.pt".format(epoch+1))
        torch.save(self.critic_1.state_dict(), dir + "/critic_1_{}.pt".format(epoch+1))
        torch.save(self.critic_2.state_dict(), dir + "/critic_2_{}.pt".format(epoch+1))
        torch.save(self.target_critic_1.state_dict(), dir + "/target_critic_1_{}.pt".format(epoch+1))
        torch.save(self.target_critic_2.state_dict(), dir + "/target_critic_2_{}.pt".format(epoch+1))

    def load(self, dir, epoch):
        self.actor.load_state_dict(torch.load(dir + "/actor_{}.pt".format(epoch)))
        self.critic_1.load_state_dict(torch.load(dir + "/critic_1_{}.pt".format(epoch)))
        self.critic_2.load_state_dict(torch.load(dir + "/critic_2_{}.pt".format(epoch)))
        self.target_critic_1.load_state_dict(torch.load(dir + "/target_critic_1_{}.pt".format(epoch)))
        self.target_critic_2.load_state_dict(torch.load(dir + "/target_critic_2_{}.pt".format(epoch)))
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
    print("1:",env.observation_space.shape[0])
    print("1:",env.action_space.shape)
    print("1:",env.action_space.low[0])
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0],
                      env.action_space.low[0], env.action_space.high[0], n_fc1, n_fc2,
                      lr_actor, lr_critic, lr_alpha, to_load, epoch_to_load, buffer_max_size, gamma, tau, batch_size,
                      alpha, automatic_entropy_tuning, target_update_interval)
    
    if mode == "train":
        # Set logging
        log = set_logging(logging_path)
        log.info("Environment: {} \n Robot: {}\n Controller {}\n SAC\n".format(env_name, robot_name, controller_name))
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
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    if render_training and episode % render_interval == 0:
                        env.render()
                    agent.replay_buffer.add((state, next_state, action, reward, np.float(done)))

                    state = next_state    
                    agent.train(step)

                    step += 1
                    episode_reward += reward
                epoch_reward += episode_reward
                total_step += step+1
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