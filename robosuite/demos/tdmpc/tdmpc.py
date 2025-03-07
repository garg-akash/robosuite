import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

import tdmpc.helper as h

class TOLD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
        self.apply(h.orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, value=True):
        h.set_requires_gradient(self._Q1, value)
        h.set_requires_gradient(self._Q2, value)

    def h(self, obs):
        return self._encoder(obs)
    
    def next(self, z, action):
        x = torch.cat([z,action], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, std=0):
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu,std).sample(clip=0.3)
        return mu 

    def Q(self, z, a):
        x = torch.cat([z,a], dim=-1)
        return self._Q1(x), self._Q2(x)

class TDMPC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = TOLD(cfg).cuda()
        self.model_target = deepcopy(self.model)
        # self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.q_params = list(self.model._Q1.parameters()) + list(self.model._Q2.parameters())
        self.q_target_params = list(self.model_target._Q1.parameters()) + list(self.model_target._Q2.parameters())
        self.other_params = []
        for name, param in self.model.named_parameters():
            if '_Q1' not in name and '_Q2' not in name:
                self.other_params.append(param)
        self.other_target_params = []
        for name, param in self.model_target.named_parameters():
            if '_Q1' not in name and '_Q2' not in name:
                self.other_target_params.append(param)
        optim_params = [
            {'params' : self.q_params, 'lr' : self.cfg.q_lr},
            {'params' : self.other_params, 'lr' : self.cfg.lr}
        ]
        self.optim = torch.optim.Adam(optim_params)
        # self.aug = h.RandomShiftsAug(cfg) TODO
        self.model.eval()
        self.model_target.eval()
    
    def state_dict(self):
        return {'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict()}

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])
        self.model_target.load_state_dict(d['model_target'])
        print("model loaded")

    @torch.no_grad()
    def estimate_value(self, z, a, horizon):
        value, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, a[t])
            value += discount * reward
            discount *= self.cfg.discount
        value += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return value
    
    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        if step is not None and step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # print("obs : ", obs.shape)
        # if step is not None:
        #     horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        # else:
        #     horizon = self.cfg.horizon
        horizon = self.cfg.horizon
        # print(f"step {step} horizon {horizon}")
        
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0 :
            pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            z = self.model.h(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z, _ = self.model.next(z,pi_actions[t])

        z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]
            
        # CEM Iteration
        for i in range(self.cfg.iterations):
            action = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device), -1, 1)
            if num_pi_trajs > 0:
                action = torch.cat([action,pi_actions], dim=1)
            value = self.estimate_value(z, action, horizon).nan_to_num_(0)
            # print("value : ", value.shape) # (num_samples,1)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_values, elite_actions = value[elite_idxs], action[:, elite_idxs]
            max_value = elite_values.max(0)[0] # torch.max(elite_values,dim=1)
            score = torch.exp(self.cfg.temperature*(elite_values - max_value))
            score /= score.sum(0)

            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) **2, dim=1) / (score.sum(0) + 1e-9))
            _std =  _std.clamp(self.std, 2)

            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std 
        
        score = score.squeeze(1).cpu().numpy()
        # print("action shape : ", action.shape)
        # print("score shape : ", score.shape)
        # print("elite actions shape : ", elite_actions.shape)
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        # print("actions shape : ", actions.shape)
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        # print("a shape : ", a.shape)
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=self.device)
        return a

    def update_pi(self, zs):
        self.pi_optim.zero_grad(set_to_none=False)
        self.model.track_q_grad(False)
        loss = 0
        for t,z in enumerate(zs):
            ac = self.model.pi(z, self.cfg.min_std)
            Q = torch.min(*self.model.Q(z,ac))
            loss += -Q.mean() * (self.cfg.rho ** t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return loss.item()

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        next_z = self.model.h(next_obs) # IMP : this is not next z from dynamics network
        td_target = reward + self.cfg.discount * \
                    torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
        return td_target
    
    def update(self, replay_buffer, step):
        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        z = self.model.h(obs) # [batch_size,latent_dim] # TODO
        zs = [z.detach()] # stop tracking gradient of the encoder network
        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        c_loss = []
        c_loss_catch = True
        for t in range(self.cfg.horizon):
            z_next,reward_pred = self.model.next(z, action[t])
            
            Q1_pred, Q2_pred = self.model.Q(z, action[t])
            with torch.no_grad():
                next_obs = next_obses[t] # self.aug(next_obses[t]) # TODO
                z_next_target = self.model_target.h(next_obs)
                td_target = self._td_target(next_obs, reward[t])
            zs.append(z_next.detach())

            rho = self.cfg.rho ** t
            reward_loss += rho * h.mse(reward_pred, reward[t])
            consistency_loss += rho * torch.mean(h.mse(z_next, z_next_target), dim=1, keepdim=True)
            value_loss += rho * (h.mse(Q1_pred, td_target) + h.mse(Q2_pred, td_target))
            priority_loss += rho * (h.l1(Q1_pred, td_target) + h.l1(Q2_pred, td_target))
            
            z = z_next  

            curr_loss = float(consistency_loss.mean().item())
            if t > 0 and c_loss_catch:
                diff = curr_loss - prev_consistency_loss
                if abs(diff) > 50:
                    c_loss_catch = False
                    c_loss.append([step, t, curr_loss, prev_consistency_loss, diff]) 
                    print(f"c loss : ", c_loss)
            prev_consistency_loss = curr_loss
        
        total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                    self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                    self.cfg.value_coef * value_loss.clamp(max=1e4)
        weighted_loss = (total_loss.squeeze(1) * weights).mean()
        weighted_loss.register_hook(lambda grad : grad * (1/self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())
        
        pi_loss = self.update_pi(zs)
        if step % self.cfg.update_freq == 0:
            h.emaInd(self.q_params, self.q_target_params, self.cfg.q_tau)
            h.emaInd(self.other_params, self.other_target_params, self.cfg.tau)
            # h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm),
                'c_loss': c_loss}
