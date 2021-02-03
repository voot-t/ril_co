import copy
import glob
import os
import time
from collections import deque

import random 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

import pathlib
import os
import sys
from colorama import init
from termcolor import cprint, colored
init(autoreset=True)
p_color = "yellow"
def t_format(text, text_length=0):
    if text_length==0:
        return "%-10s" % text 
    if text_length==0.5:
        return "%-15s" % text 
    elif text_length==1:
        return "%-20s" % text
    elif text_length==2:
        return "%-25s" % text
    elif text_length==3:
        return "%-30s" % text
    elif text_length==4:
        return "%-40s" % text
    else:
        return "%-25s" % text

device_cpu = torch.device("cpu")

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    print("Env obs space shape")
    print(envs.observation_space.shape)
    clip_action = False 
    a_high = None 
    a_low = None 

    # if args.algo == 'a2c':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         alpha=args.alpha,
    #         max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'ppo':
    #     agent = algo.PPO(
    #         actor_critic,
    #         args.clip_param,
    #         args.ppo_epoch,
    #         args.num_mini_batch,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'acktr':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    method_type = "RL"
    method_name = args.algo.upper()
    hypers = "ec%0.5f" % args.entropy_coef
    exp_name = "%s-%s_s%d" % (method_name, hypers, args.seed)

    if args.gail:
        if args.is_atari:   # Do not have GAIL implementation for Atari games
            raise NotImplementedError   

        from a2c_ppo_acktr.algo import ail, ril, vild, bc
        if args.il_algo.upper() == "AIL":
            discr = ail.AIL(
                envs.observation_space, envs.action_space, device, args)
        elif args.il_algo.upper() == "AIRL":
            discr = ail.AIRL(
                envs.observation_space, envs.action_space, device, args)
        elif args.il_algo.upper() == "FAIRL":
            discr = ail.FAIRL(
                envs.observation_space, envs.action_space, device, args)
        elif args.il_algo.upper() == "VILD":
            discr = vild.VILD(
                envs.observation_space, envs.action_space, device, args)
        elif args.il_algo.upper() == "RIL_CO":
            discr = ril.RIL_CO(
                envs.observation_space, envs.action_space, device, args)
        elif args.il_algo.upper() == "RIL":
            discr = ril.RIL(
                envs.observation_space, envs.action_space, device, args)
        elif args.il_algo.upper() == "BC":
            discr = bc.BC(actor_critic, 
                envs.observation_space, envs.action_space, device, args)

        # set result file name. 
        method_type = "IL"
        method_name = args.algo.upper() + "_" + args.il_algo.upper() 

        hypers += "_gp%0.3f" % args.gp_lambda    
        
        if args.noise_type == "policy":     # noisy policy (sub-optimal policies)
            traj_name = "np%0.1f" % args.noise_prior 
        elif args.noise_type == "action":   # noisy action
            traj_name = "na%0.1f" % args.noise_prior 
        if args.traj_deterministic:
            traj_name += "_det"
        else:
            traj_name += "_sto" 

        if "AIL" in args.il_algo.upper():
            hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate)
            
        if "VILD" in args.il_algo.upper():
            hypers += "_%s" % (args.ail_loss_type)
            if args.ail_saturate != 1:
                hypers += "_sat%d" % (args.ail_saturate)
                            
        if "RIL" in args.il_algo.upper() :
            hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate) 

        if args.reward_std: hypers += "_rs"

        if "BC" in args.il_algo.upper() :
            method_name = args.il_algo.upper()
            hypers = "bc"

        exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, args.seed)

    result_path = "./results_%s/%s/%s/%s-%s" % (method_type, method_name, args.env_name, args.env_name, exp_name)
    pathlib.Path("./results_%s/%s/%s" % (method_type, method_name, args.env_name)).mkdir(parents=True, exist_ok=True) 
    print("Running %s" % (colored(method_name, p_color)))
    print("%s result will be saved at %s" % (colored(method_name, p_color), colored(result_path, p_color)))

    model_name = "%s-%s" % (args.env_name, exp_name)
    save_path = os.path.join(args.save_dir, method_name, args.env_name)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 

    envs = None 
    
    if args.eval_interval is not None:     
        eval_envs = make_vec_envs(args.env_name, args.seed, 1,
                              None, eval_log_dir, device, True)

    # rollouts = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs.observation_space.shape, envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size)

    # obs = envs.reset()
    # rollouts.obs[0].copy_(obs)
    # rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    tt_g, tt_d = 0, 0
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    print("Update iterations: %d" % num_updates)  #  ~~ 15000
    for j in range(num_updates):

        # if args.use_linear_lr_decay:
        #     # decrease learning rate linearly
        #     utils.update_linear_schedule(
        #         agent.optimizer, j, num_updates,
        #         agent.optimizer.lr if args.algo == "acktr" else args.lr)

        # for step in range(args.num_steps):
        #     # Sample actions
        #     with torch.no_grad():
        #         value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
        #             rollouts.obs[step], rollouts.recurrent_hidden_states[step],
        #             rollouts.masks[step])

        #     # Obser reward and next obs
        #     if clip_action:
        #         obs, reward, done, infos = envs.step(torch.clamp(action, min=a_low, max=a_high))
        #     else:
        #         obs, reward, done, infos = envs.step(action)

        #     for info in infos:
        #         if 'episode' in info.keys():
        #             episode_rewards.append(info['episode']['r'])

        #     # If done then clean the history of observations.
        #     masks = torch.FloatTensor(
        #         [[0.0] if done_ else [1.0] for done_ in done])
        #     bad_masks = torch.FloatTensor(
        #         [[0.0] if 'bad_transition' in info.keys() else [1.0]
        #          for info in infos])
        #     rollouts.insert(obs, recurrent_hidden_states, action,
        #                     action_log_prob, value, reward, masks, bad_masks)

        # with torch.no_grad():
        #     next_value = actor_critic.get_value(
        #         rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
        #         rollouts.masks[-1]).detach()

        if args.gail:
            # if j >= 10:
            #     envs.venv.eval()

            gail_epoch = 1 #args.gail_epoch
            # if j < 10:
            #     gail_epoch = 100  # Warm up
            
            obfilt = None # utils.get_vec_normalize(envs)._obfilt    
            t0 = time.time()

            for _ in range(gail_epoch):
                discr.update(None, obfilt)

            tt_d += time.time() - t0

            # for step in range(args.num_steps):
            #     rollouts.rewards[step] = discr.predict_reward(
            #         rollouts.obs[step], rollouts.actions[step], args.gamma,
            #         rollouts.masks[step])

        # rollouts.compute_returns(next_value, args.use_gae, args.gamma,
        #                          args.gae_lambda, args.use_proper_time_limits)

        t0 = time.time()
        # value_loss, action_loss, dist_entropy = agent.update(rollouts)
        tt_g += time.time() - t0

        # rollouts.after_update()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #         or j == num_updates - 1) and args.save_dir != "":
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass

        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], os.path.join(save_path, model_name + ("T%d.pt" % total_num_steps)))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            # print(
            #     "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            #     .format(j, total_num_steps,
            #             int(total_num_steps / (end - start)),
            #             len(episode_rewards), np.mean(episode_rewards),
            #             np.median(episode_rewards), np.min(episode_rewards),
            #             np.max(episode_rewards), dist_entropy, value_loss,
            #             action_loss))

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        if args.eval_interval is not None and j % args.eval_interval == 0:
            ob_rms = None # utils.get_vec_normalize(envs).ob_rms  
            eval_episode_rewards = evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                                        args.num_processes, eval_log_dir, device, eval_envs, clip_action, a_low, a_high)

            """ Save results as text file. """            
            result_text = t_format("Step %8d " % (total_num_steps), 0) 
            result_text += t_format("(g%2.1f+d%2.1f)s" % (tt_g, tt_d), 1) 

            # c_reward_list = rollouts.rewards.to(device_cpu).detach().numpy()

            # result_text += " | [D] " + t_format("min: %.2f" % np.amin(c_reward_list), 0.5) + t_format(" max: %.2f" % np.amax(c_reward_list), 0.5)

            # print(c_reward_list) 

            result_text += " | [R_te] "
            result_text += t_format("min: %.2f" % np.min(eval_episode_rewards) , 1) + t_format("max: %.2f" % np.max(eval_episode_rewards), 1) \
                + t_format("Avg: %.2f (%.2f)" % (np.mean(eval_episode_rewards), np.std(eval_episode_rewards)/np.sqrt(len(eval_episode_rewards))), 2)
    
            # if args.il_algo.upper() == "VILD":
            #     ## check estimated worker noise
            #     estimated_worker_noise = discr.worker_net.get_worker_cov().to(device_cpu).detach().numpy().squeeze()
            #     if envs.action_space.shape[0] > 1:
            #         estimated_worker_noise = estimated_worker_noise.mean(axis=0)  #average across action dim
            #     result_text += " | w_noise: %s" % (np.array2string(estimated_worker_noise, formatter={'float_kind':lambda x: "%.5f" % x}).replace('\n', '') )
                    
            print(result_text)
            with open(result_path + ".txt", 'a') as f:
                print(result_text, file=f) 
            tt_g, tt_d = 0, 0


if __name__ == "__main__":
    main()
