import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

import gym 
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from a2c_ppo_acktr.arguments import get_args
import time 
import random 

from matplotlib import pyplot as plt

sys.path.append('a2c_ppo_acktr')

args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)
random.seed(args.seed)

load_expert = True 
perf = 0.1  # Choose expert and non-expert data to visualize. Can be one of [1.0, 0.4, 0.3, 0.2, 0.1, 0.0]. See save_traj.py for details. 
if load_expert:
    if perf == 1.0:
        eval_log_dir = os.path.expanduser("./videos/%s_expert" % args.env_name)
    else:
        eval_log_dir = os.path.expanduser("./videos/%s_expertP%0.1f" % (args.env_name , perf))
else:
    eval_log_dir = os.path.expanduser("./videos/%s_rilco" % args.env_name)

utils.cleanup_log_dir(eval_log_dir) 

## cannot monitor with vector env. 
# env = make_vec_envs(
#     args.env_name,
#     args.seed + 1000,
#     num_processes = 1,
#     gamma = None,
#     log_dir = None,   # need this so that infos include atari scores
#     device = 'cpu',
#     allow_early_resets = True)
# render_func = get_render_func(env)

env = gym.make(args.env_name) 
env = gym.wrappers.Monitor(env, eval_log_dir, video_callable=lambda episode_id: True,force=True)
render_func = env.render 

# Get a render function
if render_func is not None:
    render_func('human')

obs = env.reset()

if load_expert:
    ## expert/non-expert policy
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

    load_dir = "./trained_models/ACKTR/%s" % args.env_name
    model_file =  "%s-ACKTR-ec0.01000_s1T%d.pt" % (args.env_name, load_step)

else:   
    ## RIL-Co policy at 20000000 time steps. 
    load_step = 20000000
    load_dir = "./trained_models/ACKTR_RIL_CO/%s" % args.env_name
    model_file = "%s-np0.4_det-ACKTR_RIL_CO-ec0.01000_gp10.000_apl_sat1_s1T%d.pt" % (args.env_name, load_step )

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(load_dir, model_file))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

# if args.env_name.find('Bullet') > -1:
#     import pybullet as p
#     torsoId = -1
#     for i in range(p.getNumBodies()):
#         if p.getBodyInfo(i)[0].decode() == "torso":
#             torsoId = i

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(torch.device("cuda:0"))
masks = torch.zeros(1, 1).to(torch.device("cuda:0"))

t = 0
eval_rewards = 0

clipob = 10    #default in baselines code 
epsilon = 1e-8
ob_rms_sd = np.sqrt(ob_rms.var + epsilon)
ob_rms_mean = ob_rms.mean
while True:

    with torch.no_grad():

        obs_normalize = np.clip((obs - ob_rms_mean) / ob_rms_sd, -clipob, clipob)

        if load_expert: # expert policy network is saved as cpu object...
            value, action, _, recurrent_hidden_states = actor_critic.act(
                torch.FloatTensor(obs_normalize).to(torch.device("cpu")).unsqueeze(0), recurrent_hidden_states, masks, deterministic=True )
        else:   # learned policy network is saved as cuda object...
            value, action, _, recurrent_hidden_states = actor_critic.act(
                torch.FloatTensor(obs_normalize).to(torch.device("cuda:0")).unsqueeze(0), recurrent_hidden_states, masks, deterministic=True )

        obs, reward, done, infos = env.step(action.to(torch.device("cpu")).squeeze().numpy())
        eval_rewards += reward

        masks.fill_(0.0 if done else 1.0)

        t += 1

        # if args.env_name.find('Bullet') > -1:
        #     if torsoId > -1:
        #         distance = 3 # 5
        #         yaw = 0 # 0
        #         humanPos, humanOrn = p.getBasePositionAndOrientation(1)
        #         # print(humanPos) 
        #         # print(humanOrn)
        #         # p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
        #         p.resetDebugVisualizerCamera(distance, yaw, -40, humanPos)

        if t == 1000:
            print("Sum rewards: %0.1f" % eval_rewards) 
            break 

        if render_func is not None:
            render_func('human')
            # time.sleep(0.01)
