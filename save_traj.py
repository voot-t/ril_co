import argparse
import os
# workaround to unpickle olf model files
import sys
from os import path, listdir

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.arguments import get_args
import time 
import random 
import h5py 
import pathlib 
from itertools import count

import lmdb 
import seaborn as sns
import re
import pyarrow as pa
import pickle 
import pathlib

sys.path.append('a2c_ppo_acktr')

## save data with policy snapshots
def save_traj_perf():

    args = get_args()

    method_type = "RL"
    method_name = args.algo.upper()
    hypers = "ec%0.5f" % args.entropy_coef
    exp_name = "%s-%s_s%d" % (method_name, hypers, 1)
    model_name = "%s-%s" % (args.env_name, exp_name)
    save_path = os.path.join(args.save_dir, args.algo.upper(), args.env_name)
    traj_deterministic = args.traj_deterministic

    ## Approximated relative performance w.r.t. the expert performance: 
    ## 1.0 is to save expert data, 0.4 is to save non-expert data with performance ~40% of expert, 0.0 is a random initial policy.
    ## Choose one from 1.0, 0.4, 0.3, 0.2, 0.1, 0.0. (Looping through the list makes rng inconsistent).
    perf_list = [0.0]

    demo_file_size = 10000
    clipob = 10    #default in baselines code 
    epsilon = 1e-8
    max_step = 1000 

    for perf in perf_list:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)

        env = make_vec_envs(
            args.env_name,
            args.seed + 1000,
            1,
            None,
            None,
            device='cpu',
            allow_early_resets=False)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print("State dim: %d, action dim: %d" % (state_dim, action_dim))

        if args.env_name == "HalfCheetahBulletEnv-v0":  
            if perf == 1.0: 
                load_step = 10000000 # 2500 
            if perf == 0.4:
                load_step =   576640 # 1300  
            if perf == 0.3:
                load_step =   384640 # 1000 
            if perf == 0.2: 
                load_step =   256640 # 700 
            if perf == 0.1:
                load_step =   128640 # -1100 
            if perf == 0.0:
                load_step =   640 # -1000 

        if args.env_name == "AntBulletEnv-v0":   
            if perf == 1.0: 
                load_step = 10000000 # 3500  
            if perf == 0.4:
                load_step =   704640 # 1400 
            if perf == 0.3:
                load_step =   576640 # 1000 
            if perf == 0.2:
                load_step =   384640 # 700 
            if perf == 0.1:
                load_step =   128640  # 400 
            if perf == 0.0:
                load_step =   640 # 30 

        if args.env_name == "HopperBulletEnv-v0":   
            if perf == 1.0: 
                load_step = 10000000 # 2300 
            if perf == 0.4:
                load_step =   4416640 # 1100
            if perf == 0.3:
                load_step =   384640 # 1000
            if perf == 0.2: 
                load_step =   256640 # 900
            if perf == 0.1:
                load_step =   128640 # 600 
            if perf == 0.0:
                load_step =   640 # 40 

        if args.env_name == "Walker2DBulletEnv-v0":   
            if perf == 1.0: 
                load_step = 10000000 # 2700
            if perf == 0.4:
                load_step =  1024640 # 800     
            if perf == 0.3:
                load_step =   576640 # 600     
            if perf == 0.2:
                load_step =   384640 # 700  
            if perf == 0.1: 
                load_step =   256640 # 100 
            if perf == 0.0:
                load_step =   640 # 16  

        # We need to use the same statistics for normalization as used in training
        actor_critic = None 
        ob_rms = None 
        actor_critic, ob_rms = \
                    torch.load( os.path.join(save_path, model_name + ("T%d.pt" % load_step)))

        vec_norm = get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = None # ob_rms will be used manually in order to save unnormalize demonstrations.

        obs_list = []     
        act_lst = []     
        nobs_list = []
        mask_list = []
        reward_list = []

        total_step = 0
        avg_reward_episode = 0
        print(model_name + ("T%d.pt" % load_step))
        for i_episode in count():
            obs = env.reset()
            sum_rewards = 0
            recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)

            t = 0
            while True:
                
                ## normalize obs used by trained models
                obs_normalize = np.clip((obs.numpy() - ob_rms.mean) /
                                np.sqrt(ob_rms.var + epsilon),
                                -clipob, clipob)
                obs_normalize = torch.from_numpy(obs_normalize).float()

                with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                        obs_normalize, recurrent_hidden_states, masks, deterministic=traj_deterministic )

                # Obser reward and next obs (unnormalized) 
                next_obs, reward, done, infos = env.step(action)
                sum_rewards += reward 
                masks.fill_(0.0 if done else 1.0)

                obs_list.append(obs.numpy())
                act_lst.append(action.numpy())
                nobs_list.append(next_obs.numpy())
                mask_list.append(int(not done))
                reward_list.append(reward.numpy())

                obs = next_obs 
                total_step += 1
                t += 1

                if done:
                    print("Episode %2d: Sum rewards %0.2f, Steps %d" % (i_episode, sum_rewards, t))
                    break 

            avg_reward_episode += sum_rewards

            # if i_episode % 10 == 0:
            #     print('Episode %2d reward: %.2f' % (i_episode, sum_rewards))

            if total_step >= demo_file_size:
                break


        """ save data """
        obs_array = np.vstack(obs_list)
        act_array = np.vstack(act_lst)
        nobs_array = np.vstack(nobs_list)
        mask_array = np.vstack(mask_list)
        reward_array = np.vstack(reward_list)

        print("Total steps %d, total episode %d, AVG reward: %f" % ( total_step, i_episode + 1, avg_reward_episode/(i_episode+1)))

        traj_path = "./imitation_data/%s/" % (args.env_name)
        pathlib.Path(traj_path).mkdir(parents=True, exist_ok=True) 
        traj_filename = traj_path + ("/%s_TRAJ-N%d_P%0.1f" % (args.env_name, demo_file_size, perf))
        if traj_deterministic:
            traj_filename += "_det"
        else:
            traj_filename += "_sto"

        hf = h5py.File(traj_filename + ".h5", 'w')
        hf.create_dataset('model_file', data=model_name + ("T%d.pt" % load_step))  
        hf.create_dataset('obs_array', data=obs_array)
        hf.create_dataset('act_array', data=act_array)
        hf.create_dataset('nobs_array', data=nobs_array)
        hf.create_dataset('mask_array', data=mask_array)
        hf.create_dataset('reward_array', data=reward_array)
        hf.create_dataset('obs_rms_mean', data=ob_rms.mean)
        hf.create_dataset('obs_rms_var', data=ob_rms.var)

        print("TRAJs are saved as %s" % traj_filename)

## save data with guassian noise
def save_traj_noise():

    args = get_args()

    method_type = "RL"
    method_name = args.algo.upper()
    hypers = "ec%0.5f" % args.entropy_coef
    exp_name = "%s-%s_s%d" % (method_name, hypers, 1)
    model_name = "%s-%s" % (args.env_name, exp_name)
    save_path = os.path.join(args.save_dir, args.algo.upper(), args.env_name)

    demo_file_size = 10000
    clipob = 10    #default in baselines code 
    epsilon = 1e-8
    max_step = 1000 
    traj_deterministic = args.traj_deterministic

    # 1.0, 0.4, 0.3, 0.2, 0.1, 0.0  # choose one from this. Looping through the list makes rng not consistent.
    perf_list = [1.0 ]

    for perf in perf_list:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)

        env = make_vec_envs(
            args.env_name,
            args.seed + 1000,
            1,
            None,
            None,
            device='cpu',
            allow_early_resets=False)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print("State dim: %d, action dim: %d" % (state_dim, action_dim))

        a_high = np.asscalar(env.action_space.high[0])
        a_low = np.asscalar(env.action_space.low[0])

        if args.env_name == "AntBulletEnv-v0":   
            load_step = 10000000
            if perf == 1.0: 
                noise_level = 0  # 3500
            if perf == 0.4:
                noise_level = 1.0 # 1500
            if perf == 0.3:
                noise_level = 1.2 # 1000
            if perf == 0.2: 
                noise_level = 1.3 # 800
            if perf == 0.1:
                noise_level = 1.4 # 500
            if perf == 0.0:
                noise_level = 1.5 # 400
        else:
            raise NotImplementedError 

        # We need to use the same statistics for normalization as used in training
        actor_critic, ob_rms = \
                    torch.load( os.path.join(save_path, model_name + ("T%d.pt" % load_step)))

        vec_norm = get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = None # ob_rms will be used manually in order to save unnormalize demonstrations.

        obs_list = []     
        act_lst = []     
        nobs_list = []
        mask_list = []
        reward_list = []

        total_step = 0
        avg_reward_episode = 0
        print(model_name + ("T%d.pt" % load_step))
        for i_episode in count():
            obs = env.reset()
            sum_rewards = 0
            recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)

            t = 0
            while True:
                
                ## normalize obs used by trained models
                obs_normalize = np.clip((obs.numpy() - ob_rms.mean) /
                                np.sqrt(ob_rms.var + epsilon),
                                -clipob, clipob)
                obs_normalize = torch.from_numpy(obs_normalize).float()

                with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                        obs_normalize, recurrent_hidden_states, masks, deterministic=traj_deterministic )

                ## add noise 
                if noise_level > 0:
                    # action = action + torch.FloatTensor(np.random.normal(0, noise_level, (1, action_dim)))
                    action = action + torch.normal(mean=0, std=noise_level, size=action.size())
                        
                # Observe reward and next obs (unnormalized) 
                next_obs, reward, done, infos = env.step(action)

                sum_rewards += reward 
                masks.fill_(0.0 if done else 1.0)

                obs_list.append(obs.numpy())
                act_lst.append(action.numpy())
                nobs_list.append(next_obs.numpy())
                mask_list.append(int(not done))
                reward_list.append(reward.numpy())

                obs = next_obs 
                total_step += 1
                t += 1

                if done:
                    print("Episode %2d: Sum rewards %0.2f, Steps %d" % (i_episode, sum_rewards, t))
                    break 

            avg_reward_episode += sum_rewards

            # if i_episode % 10 == 0:
            #     print('Episode %2d reward: %.2f' % (i_episode, sum_rewards))

            if total_step >= demo_file_size:
                break


        """ save data """
        obs_array = np.vstack(obs_list)
        act_array = np.vstack(act_lst)
        nobs_array = np.vstack(nobs_list)
        mask_array = np.vstack(mask_list)
        reward_array = np.vstack(reward_list)

        print("Total steps %d, total episode %d, AVG reward: %f" % ( total_step, i_episode + 1, avg_reward_episode/(i_episode+1)))

        traj_path = "./imitation_data/%s/" % (args.env_name)
        pathlib.Path(traj_path).mkdir(parents=True, exist_ok=True) 
        traj_filename = traj_path + ("/%s_TRAJ-N%d_A%0.1f" % (args.env_name, demo_file_size, perf))
        if traj_deterministic:
            traj_filename += "_det"
        else:
            traj_filename += "_sto"

        hf = h5py.File(traj_filename + ".h5", 'w')
        hf.create_dataset('model_file', data=model_name + ("T%d.pt" % load_step))  
        hf.create_dataset('obs_array', data=obs_array)
        hf.create_dataset('act_array', data=act_array)
        hf.create_dataset('nobs_array', data=nobs_array)
        hf.create_dataset('mask_array', data=mask_array)
        hf.create_dataset('reward_array', data=reward_array)
        hf.create_dataset('obs_rms_mean', data=ob_rms.mean)
        hf.create_dataset('obs_rms_var', data=ob_rms.var)

        print("TRAJs are saved as %s" % traj_filename)

if __name__ == "__main__":
    # save_traj_perf()
    save_traj_noise()