import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

__REDUCE__ = lambda b : 'mean' if b else 'none'

def l1(pred, target, reduce=False):
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))

def mse(pred, target, reduce=False):
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))
     
def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)

def ema(m, m_target, tau):
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau) # Linear interpolation in place
        
def emaInd(p_params, p_target_params, tau):
    with torch.no_grad():
        for p, p_target in zip(p_params, p_target_params):
            p_target.data.lerp_(p.data, tau) # Linear interpolation in place

def set_requires_gradient(net, value):
    for p in net.parameters():
        p.requires_grad_(value)

def enc(cfg):
    # if cfg.modality == "Pixels" TODO
    layers = []
    layers.append(nn.Linear(cfg.obs_shape[0], cfg.enc_dim))
    layers.append(nn.ELU())
    layers.append(nn.Linear(cfg.enc_dim,cfg.latent_dim))
    return nn.Sequential(*layers)

def q(cfg, act_fn=nn.ELU()):
    layers = []
    layers.append(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim))
    layers.append(nn.LayerNorm(cfg.mlp_dim))
    layers.append(nn.Tanh())
    layers.append(nn.Linear(cfg.mlp_dim, cfg.mlp_dim))
    layers.append(nn.ELU())
    layers.append(nn.Linear(cfg.mlp_dim, 1))
    return nn.Sequential(*layers)

def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    if isinstance(mlp_dim,int):
        mlp_dim = [mlp_dim, mlp_dim]
    layers = []
    layers.append(nn.Linear(in_dim, mlp_dim[0]))
    layers.append(act_fn)
    layers.append(nn.Linear(mlp_dim[0], mlp_dim[1]))
    layers.append(act_fn)
    layers.append(nn.Linear(mlp_dim[1], out_dim))
    return nn.Sequential(*layers)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps =torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class Episode(object):
    def __init__(self, cfg, init_obs):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        dtype = torch.float32 if cfg.modality == 'State' else torch.uint8
        self.obs = torch.empty((cfg.episode_length+1, *init_obs.shape), dtype=dtype, device=self.device)
        self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
        self.action = torch.empty((cfg.episode_length, cfg.action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((cfg.episode_length, ), dtype=torch.float32, device=self.device)
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0
    
    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done):
        self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.device)
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1

class ReplayBuffer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        dtype = torch.float32 if cfg.modality == 'State' else torch.uint8
        self.capacity = (min(cfg.train_steps, cfg.max_buffer_size) // cfg.episode_length) * cfg.episode_length
        obs_shape = cfg.obs_shape if cfg.modality == 'State' else (3, *cfg.obs_shape[-2:])
        self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
        self._last_obs = torch.empty((self.capacity//cfg.episode_length, *cfg.obs_shape), dtype=dtype, device=self.device)
        self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
        self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
        self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self.eps = 1e-6
        self._full = False
        self.idx = 0
        print("Buffer capacity : ", self.capacity)

    def __add__(self, episode: Episode):
        self.add(episode)
        return self
    
    def add(self, episode):
        # idx_next = (self.idx + self.cfg.episode_length) % self.capacity
        # self._full = self._full or self.idx > idx_next
        # self.idx = idx_next if self.idx > idx_next else self.idx
        # print("idx : ", [self.idx,idx_next,self._full])

        self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1] if self.cfg.modality == 'State' else episode.obs[:-1, -3:] # TODO
        self._last_obs[self.idx//self.cfg.episode_length] = episode.obs[-1]
        self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
        self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
        if self._full:
            max_priority = self._priorities.max().to(self.device).item()
        else:
            max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
        mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
        new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device) # initially set priorities for all transitions in the current episode to max_priority
        new_priorities[mask] = 0 # priorities corresponding to the last horizon steps are set to 0.
        self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
        # self.idx = idx_next

        idx_next = (self.idx + self.cfg.episode_length) % self.capacity
        self._full = self._full or self.idx > idx_next
        # print("idx : ", [self.idx,self._full])
        self.idx = idx_next

        # self.idx = (self.idx + self.cfg.episode_length) % self.capacity
        # self._full = self._full or self.idx == 0

        ## Not good as mask computation in sample is based on the fact that idx / episode_length signifies last observation
        ## If you break storage between remaining part of end and starting part, the above condition is no longer true
        # start_idx = self.idx
        # end_idx = (self.idx + self.cfg.episode_length) % self.capacity
        # print("storing : ", [start_idx,end_idx])
        # if start_idx < end_idx:
        #     self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1] # TODO
        #     self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
        #     self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
        # else:
        #     self._full = True
        #     part_1_len = self.capacity - start_idx
        #     part_2_len = self.cfg.episode_length - part_1_len

        #     self._obs[start_idx:start_idx+part_1_len] = episode.obs[:part_1_len]
        #     self._obs[0:part_2_len] = episode.obs[part_1_len:-1]

        #     self._action[start_idx:start_idx+part_1_len] = episode.action[:part_1_len]
        #     self._action[0:part_2_len] = episode.action[part_1_len:]
            
        #     self._reward[start_idx:start_idx+part_1_len] = episode.reward[:part_1_len]
        #     self._reward[0:part_2_len] = episode.reward[part_1_len:]

        # self._last_obs[self.idx//self.cfg.episode_length] = episode.obs[-1] # stores last observ from each episode

        # if self._full:
        #     max_priority = self._priorities.max().to(self.device).item()
        # else:  
        #     max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
        
        # mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
        # new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
        # new_priorities[mask] = 0
        # if start_idx < end_idx:
        #     self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
        # else:
        #     part_1_len = self.capacity - start_idx
        #     part_2_len = self.cfg.episode_length - part_1_len

        #     self._priorities[start_idx:start_idx+part_1_len] = new_priorities[:part_1_len]
        #     self._priorities[0:part_2_len] = new_priorities[part_1_len:]
        # self.idx = end_idx
        ##
    
    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self.eps

    def _get_obs(self, arr, idxs):
        if self.cfg.modality == "State":
            return arr[idxs]
        # TODO

    def sample(self):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max() # (batch_size)
        obs = self._get_obs(self._obs, idxs) # (batch_size,obs_dim)
        next_obs_shape = self._last_obs.shape[1:] # TODO
        next_obs = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
        reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
        action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
        for t in range(self.cfg.horizon+1):
            _idxs = idxs + t
            next_obs[t,:] = self._get_obs(self._obs, _idxs+1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]

        mask = (_idxs + 1) % self.cfg.episode_length == 0 # remember idxs are based index
        next_obs[-1,mask] = self._last_obs[_idxs[mask]//self.cfg.episode_length].cuda().float()
        if not action.is_cuda:
            action, reward, idxs, weights = action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()
        
        return obs, next_obs, action, reward.unsqueeze(2), idxs, weights
    
def linear_schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final