import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data as data_utils
from torch import autograd
from os import path

from baselines.common.running_mean_std import RunningMeanStd

from ..model import Flatten
from ..algo.ail_utils import *  ## losses and sa_cat()
from colorama import init
from termcolor import cprint, colored
init(autoreset=True)
p_color = "yellow"

import matplotlib.pyplot as plt

device_cpu = torch.device("cpu")

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_dim=100):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        self.train()

    def forward(self, inputs):
        return self.main(inputs)

    def reward(self, inputs):
        return self.main(inputs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AIL():
    def __init__(self, observation_space, action_space, device, args, log_only=False):
        super(AIL, self).__init__()

        if log_only:
            self.m_return_list = self.load_expert_data(args)
            return 
        
        self.lr = args.il_lr    # larger learning rate for MLP
        self.action_dim = action_space.shape[0] 
        self.hidden_dim = 100

        self.state_dim = observation_space.shape[0]     # 

        self.device = device
        self.create_networks()

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.gail_batch_size = args.gail_batch_size
        self.label_expert = 1
        self.label_policy = -1
        self.reward_std = args.reward_std
        self.gp_lambda = args.gp_lambda
        self.m_return_list = self.make_dataset(args)

        if args.ail_saturate is None and args.ail_loss_type != "unhinged": 
            args.ail_saturate = 1

        if args.ail_loss_type == "logistic":
            self.adversarial_loss = Logistic_Loss()
        elif args.ail_loss_type == "unhinged":
            self.adversarial_loss = Unhinged_Loss() 
            if args.ail_saturate is None: args.ail_saturate = 0
        elif args.ail_loss_type == "sigmoid":
            self.adversarial_loss = Sigmoid_Loss() 
        elif args.ail_loss_type == "nlogistic":
            self.adversarial_loss = Normalized_Logistic_Loss()
        elif args.ail_loss_type == "apl":
            self.adversarial_loss = APL_Loss() 
        self.ail_saturate = args.ail_saturate

    def create_networks(self):
        self.trunk = Discriminator(self.state_dim + self.action_dim, self.hidden_dim).to(self.device) 
        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=self.lr)
        
    def make_dataset(self, args):
        self.load_expert_data(args)   # h5py demos are loaded into tensor. 
        expert_dataset = data_utils.TensorDataset(self.real_state_tensor, self.real_action_tensor)

        drop_last = len(expert_dataset) > self.gail_batch_size        
        self.expert_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=self.gail_batch_size,
            shuffle=True,   # important to shuffle the dataset. 
            drop_last=drop_last)

    def load_expert_data(self, args, verbose=1):    # also load non-expert data
        model_list = [1.0]
        if args.noise_prior != 0.0:
            model_list += [0.4, 0.3, 0.2, 0.1, 0.0]

        traj_deterministic = args.traj_deterministic
        demo_file_size = 10000

        self.index_worker_idx = [] 
        m_return_list = []
        index_start = 0
        expert_state_list, expert_action_list, expert_nstate_list, expert_reward_list, expert_mask_list, expert_id_list = [],[],[],[],[],[]

        traj_path = "./imitation_data/%s" % (args.env_name)

        for model_i in range(0, len(model_list)):
            m = model_list[model_i]

            if args.noise_type == "policy":     # policy noise (sub-optimal policies)
                traj_filename = traj_path + ("/%s_TRAJ-N%d_P%0.1f" % (args.env_name, demo_file_size, m))
            elif args.noise_type == "action":   # action noise
                traj_filename = traj_path + ("/%s_TRAJ-N%d_A%0.1f" % (args.env_name, demo_file_size, m))
            if traj_deterministic:
                traj_filename += "_det"
            else:
                traj_filename += "_sto" 

            hf = h5py.File(traj_filename + ".h5", 'r')            
            expert_mask = hf.get('mask_array')[:]
            expert_state = hf.get('obs_array')[:]
            # expert_nstate = hf.get('nobs_array')[:]
            expert_action = hf.get('act_array')[:]
            expert_reward = hf.get('reward_array')[:]

            step_num = expert_mask.shape[0]
            traj_num = step_num - np.sum(expert_mask)   
            m_return = np.sum(expert_reward) / traj_num

            m_return_list += [ m_return ]

            expert_id = np.ones((expert_mask.shape[0], 1)) * model_i
            
            if m != 1.0 and args.noise_prior != -1.0:
                if args.noise_prior == 0.5:
                    pair_num = 2000 # 10000 / 5 
                if args.noise_prior == 0.4:
                    pair_num = 1500
                if args.noise_prior == 0.3:
                    pair_num = 1000
                if args.noise_prior == 0.2:
                    pair_num = 500
                if args.noise_prior == 0.1:
                    pair_num = 200 

                if args.demo_sub_traj:
                    sub_num = pair_num // 50    # each sub traj has 50 sa-pairs.
                    index = []
                    ## data is split into sub_num chunks, and we randomly sample 50 pairs from each chunk. 
                    chuck_size = demo_file_size // sub_num
                    indexes_start = np.random.randint(0, chuck_size - 50, size=sub_num) 
                    for i in range(0, sub_num):
                        ii = indexes_start[i] + (i * chuck_size) 
                        index.append( np.arange(ii, ii + 50) )
                    index = np.hstack(index) 
                else:         
                    index = np.random.permutation(demo_file_size)[:pair_num]

                expert_mask = expert_mask[index]
                expert_state = expert_state[index]
                expert_action = expert_action[index]
                # expert_nstate = expert_nstate[index]  #next state is not used. 
                expert_reward = expert_reward[index]
                expert_id = expert_id[index]
                
            self.index_worker_idx += [ index_start + np.arange(0, expert_mask.shape[0] ) ] 
            index_start += expert_mask.shape[0] 

            expert_mask_list += [expert_mask]
            expert_state_list += [expert_state]
            expert_action_list += [expert_action]
            # expert_nstate_list += [expert_nstate]
            expert_reward_list += [expert_reward]
            expert_id_list += [expert_id]

            if verbose:
                print("%s TRAJ is loaded from %s with full_size %s: using data size %s steps and average return %s" % \
                    (colored(args.noise_type, p_color), colored(traj_filename, p_color), colored(step_num, p_color), colored(expert_state.shape[0] , p_color), \
                    colored( "%.2f" % (m_return), p_color )))

        expert_masks = np.concatenate(expert_mask_list, axis=0)
        expert_states = np.concatenate(expert_state_list, axis=0)
        expert_actions = np.concatenate(expert_action_list, axis=0)
        # expert_nstates = np.concatenate(expert_nstate_list, axis=0)
        expert_rewards = np.concatenate(expert_reward_list, axis=0)
        expert_ids = np.concatenate(expert_id_list, axis=0)

        self.real_mask_tensor = torch.FloatTensor(expert_masks).to(device_cpu)
        self.real_state_tensor = torch.FloatTensor(expert_states).to(device_cpu) 
        self.real_action_tensor = torch.FloatTensor(expert_actions).to(device_cpu) 
        # self.real_nstate_tensor = torch.FloatTensor(expert_nstates).to(device_cpu) 
        self.real_id_tensor = torch.LongTensor(expert_ids).to(device_cpu) 
        self.data_size = self.real_state_tensor.size(0) 

        self.worker_num = torch.unique(self.real_id_tensor).size(0) 

        print(self.real_state_tensor.size())
        print(self.real_action_tensor.size())

        if verbose:
            print("Total data pairs: %s, state dim %s, action dim %s" % \
                (colored(self.real_state_tensor.size(0), p_color), \
                colored(self.real_state_tensor.size(1), p_color), colored(self.real_action_tensor.size(1), p_color)
                ))
        return m_return_list

    def compute_grad_pen(self,
                         expert_data,
                         policy_data,
                         lambda_=10,
                         network=None):

        if expert_data.size(0) != policy_data.size(0):
            if expert_data.size(0) < policy_data.size(0):
                idx = np.random.permutation(policy_data.size(0))[: expert_data.size(0)]
                policy_data = policy_data[idx, :]
            else: 
                idx = np.random.permutation(expert_data.size(0))[: policy_data.size(0)]
                expert_data = expert_data[idx, :]
                
        # # DRAGAN 
        # alpha = torch.rand(expert_data.size()).to(expert_data.device)
        # mixup_data = alpha * expert_data + ((1 - alpha) * (expert_data + 0.5 * expert_data.std() * torch.rand(expert_data.size()).to(expert_data.device)))

        alpha = torch.rand(expert_data.size(0), 1)
        
        alpha = alpha.expand_as(expert_data).to(expert_data.device)
        mixup_data = alpha * expert_data + (1 - alpha) * policy_data


        mixup_data.requires_grad = True

        if network is None:
            network = self.trunk 

        disc = network(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.trunk.eval()

            d = self.trunk.reward(sa_cat(state, action))
        
            if self.ail_saturate == 1:
                reward = self.adversarial_loss.reward(d * self.label_policy, reduction=False)  # saturate  (positive)
            elif self.ail_saturate == -1:
                reward = -self.adversarial_loss.reward(d * self.label_expert, reduction=False)     # non-saturate (negative)
            elif self.ail_saturate == 0:
                reward = d 

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            if self.reward_std :
                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            else:
                return reward 

    def update(self, rollouts, obsfilt=None):
        self.trunk.train()

        rollouts_size = rollouts.get_batch_size()
        policy_mini_batch_size = self.gail_batch_size if rollouts_size > self.gail_batch_size \
                            else rollouts_size

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=policy_mini_batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(self.expert_loader,
                                              policy_data_generator):

            policy_state, policy_action = policy_batch[0], policy_batch[2]
            expert_state, expert_action = expert_batch[0], expert_batch[1]  
            
            # need to normalize the expert data using current policy statistics so that expert and policy data have the same normalization.
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)  
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)

            policy_d = self.trunk(sa_cat(policy_state, policy_action))
            expert_d = self.trunk(sa_cat(expert_state, expert_action))
            grad_pen = self.compute_grad_pen( sa_cat(expert_state, expert_action),
                                              sa_cat(policy_state, policy_action),
                                              self.gp_lambda)

            policy_loss = self.adversarial_loss(policy_d * self.label_policy)
            expert_loss = self.adversarial_loss(expert_d * self.label_expert)

            gail_loss = expert_loss + policy_loss

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

        return loss / n

## AIRL: Same training as GAIL, but the reward is the discriminator value. 
class AIRL(AIL):    
    def __init__(self, observation_space, action_space, device, args):
        args.ail_loss_type = "logistic"            
        args.ail_saturate = 0
        super(AIRL, self).__init__(observation_space, action_space, device, args)

class FAIRL(AIL):
    def __init__(self, observation_space, action_space, device, args):
        args.ail_loss_type = "logistic"            
        args.ail_saturate = 1
        super(FAIRL, self).__init__(observation_space, action_space, device, args)

    # override 
    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.trunk.eval()
            d = self.trunk.reward(sa_cat(state, action))
            
            reward = torch.exp(d) * -d

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            if self.reward_std :
                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            else:
                return reward 
