import os
import numpy as np
import torch
import time
import random
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
import re

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from tdmpc.tdmpc import TDMPC
from tdmpc.helper import Episode, ReplayBuffer
import tdmpc.logger as logger

__CONFIG__, __LOGS__ = 'tdmpc/default.yaml', 'tdmpc/logs'

def parse_cfg(cfg_path):
    base = OmegaConf.load(cfg_path)

    # Algebraic expressions
    for k,v in base.items():
        if isinstance(v, str):
            match = re.match(r'(\d+)([+\-*/])(\d+)', v)
            if match:
                base[k] = eval(match.group(1) + match.group(2) + match.group(3))
                if isinstance(base[k], float) and base[k].is_integer():
                    base[k] = int(base[k])
                         
    base.task_title = base.env_name + ' ' + base.robot_name
    base.task = base.env_name + '-' + base.robot_name
    base.exp_name = str(base.get('exp_name', 'default'))
    return base

def evaluate(env, agent, num_episodes, step, env_step, video, ep_length):
    episode_rewards = []
    # print("evaluate called")
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset()[0], False, 0, 0
        if video: video.init(env, enabled=(i==0))
        while not done and t < ep_length:
            action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
            # obs, reward, done, _ = env.step(action.cpu().numpy())
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            # frame = obs["frontview" + "_image"]
            done = terminated or truncated
            ep_reward += reward
            if video: video.record(env)
            # else:
            #     print("video recorder is none")
            t += 1
        # print("eval done after : ", t)
        episode_rewards.append(ep_reward)
        if video: video.save(env_step)
    return np.nanmean(episode_rewards)

def train(cfg):
    print(cfg.env_name)
    controller_config = load_controller_config(default_controller=cfg.controller_name)
    if cfg.modality == 'State':
        keys = ["robot0_proprio-state","object-state"]

    env = GymWrapper(
        suite.make(
            cfg.env_name,
            robots=cfg.robot_name,
            controller_configs=controller_config,
            reward_shaping=True,
            reward_scale=1.0,
            has_renderer=True,  # make sure we can render to the screen
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            use_camera_obs=cfg.use_camera_obs,  # do not use pixel observations
            # control_freq=50,  # control should happen fast enough so that simulation looks smoother
            camera_names="frontview",
        ),
        keys=keys
    )
    # print(env._get_observations())
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_dim = env.action_space.shape[0]
    
    # set_seed(cfg.seed)
    timestamp = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    work_dir = Path(__file__).parent.resolve() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed) / timestamp
    agent, buffer = TDMPC(cfg), ReplayBuffer(cfg)
    
    obs_dims = env.observation_space.shape[0]
    print("Obs : ", cfg.obs_shape)

    # Run training
    L = logger.Logger(work_dir, cfg)

    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
        # collect trajectory
        # obs = env.reset()[0] # robosiute version difference
        obs = env.reset()[0]
        # print("0 : ", obs.shape)
        episode = Episode(cfg, obs)
        ep_len = 0
        # TODO : use episode_length as horizon in the env
        while not episode.done and ep_len < cfg.episode_length:
            action = agent.plan(obs, step=step, t0=episode.first)
            # obs, reward, done, _ = env.step(action.cpu().numpy())
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            # frame = obs["frontview" + "_image"]
            done = terminated or truncated
            # print("term : ", terminated)
            # print("trun : ", truncated)
            # print("1 : ", obs)
            # print(type(obs[0]))
            # print("2 :", (np.array(obs[0])).shape)
            episode += (obs, action, reward, done)
            ep_len += 1
        assert len(episode) == cfg.episode_length # ?
        buffer += episode

        # update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step+i))

        # Log training episode
        episode_idx += 1
        env_step = int(step*cfg.action_repeat) # wrapper needed so keep action_repeat = 1 in cfg
        common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

		# Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video, cfg.episode_length)
            L.log(common_metrics, category='eval')
            # Save model
            L.saveModel(env_step, agent)

    L.finish(agent)
    print('Training completed successfully')

def test(cfg):
    cfg.use_wandb = False
    controller_config = load_controller_config(default_controller=cfg.controller_name)
    if cfg.modality == 'State':
        keys = ["robot0_proprio-state","object-state"]

    env = GymWrapper(
        suite.make(
            cfg.env_name,
            robots=cfg.robot_name,
            controller_configs=controller_config,
            reward_shaping=True,
            reward_scale=1.0,
            has_renderer=True,  # make sure we can render to the screen
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            use_camera_obs=cfg.use_camera_obs,  # do not use pixel observations
            # control_freq=50,  # control should happen fast enough so that simulation looks smoother
            camera_names="frontview",
        ),
        keys=keys
    )
    # print(env._get_observations())
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_dim = env.action_space.shape[0]
    
    # set_seed(cfg.seed)
    timestamp = '21-02-2025_12-42-57'#input("Enter the timestamp to load : ")
    work_dir = Path(__file__).parent.resolve() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed) / timestamp
    agent = TDMPC(cfg)
    
    obs_dims = env.observation_space.shape[0]
    print("Obs : ", cfg.obs_shape)

    # Run training
    step_to_load = input("Enter the step to load")
    L = logger.Logger(work_dir, cfg)
    L.loadModel(step_to_load, agent)
    # episode_idx, start_time = 0, time.time()
    # for idx, step in enumerate(range(0, cfg.test_steps+cfg.episode_length, cfg.episode_length)):
    #     # collect trajectory
    #     # obs = env.reset()[0] # robosiute version difference
    #     obs = env.reset()[0]
    #     # print("0 : ", obs.shape)
    #     episode = Episode(cfg, obs)
    #     ep_len = 0
    #     while not episode.done and ep_len < cfg.episode_length:
    #         action = agent.plan(obs, step=step, t0=episode.first)
    #         # obs, reward, done, _ = env.step(action.cpu().numpy())
    #         obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
    #         # frame = obs["frontview" + "_image"]
    #         done = terminated or truncated
    #         # print("term : ", terminated)
    #         # print("trun : ", truncated)
    #         # print("1 : ", obs)
    #         # print(type(obs[0]))
    #         # print("2 :", (np.array(obs[0])).shape)
    #         episode += (obs, action, reward, done)
    #         ep_len += 1
    #     print(f"Game : {idx}, Reward : {episode.cumulative_reward}")

    for i in range(cfg.test_episodes):
        obs, done, ep_reward, t = env.reset()[0], False, 0, 0
        while not done and t < cfg.episode_length:
            action = agent.plan(obs, eval_mode=True, t0=t==0)
            env.render()
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            ep_reward += reward
            t += 1
        print(f"Game : {i}, Reward : {ep_reward}")


if __name__ == '__main__':  
    cfg = parse_cfg(Path(__file__).parent.resolve() / __CONFIG__)
    if cfg.mode == 'training':
        train(cfg)
    else:
        test(cfg)