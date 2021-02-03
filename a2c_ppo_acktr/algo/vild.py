import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data as data_utils
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd

from ..model import Policy_Psi, Worker_Noise
from ..algo.ail import AIL 
from ..algo.ail_utils import *
from colorama import init
from termcolor import cprint, colored
init(autoreset=True)
p_color = "yellow"

import math 
import time 

device_cpu = torch.device("cpu")

class Discriminator_VILD(nn.Module):
    def __init__(self, num_inputs, hidden_dim=100, ail_loss_type="logistic"):
        super(Discriminator_VILD, self).__init__()

        if ail_loss_type == "logistic":    # default. 
            self.main = nn.Sequential(
                nn.Linear(num_inputs, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1))

        elif ail_loss_type == "unhinged":
            self.main = nn.Sequential(
                nn.Linear(num_inputs, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid())
        else:
            raise NotImplmentedError 

        self.main.train()

    def forward(self, inputs):
        return self.main(inputs)

    def reward(self, inputs):
        return self.main(inputs)

class VILD(AIL):
    def __init__(self, observation_space, action_space, device, args):
        self.ail_loss_type = args.ail_loss_type 
        if args.ail_loss_type == "unhinged":
            args.ail_saturate = 0
        super(VILD, self).__init__(observation_space, action_space, device, args)
        self.label_expert = 1     # used in vild repo
        self.label_policy = 0

        self.noise_t = 1e-8
        self.per_alpha = 2
        self.mini_batch_size = self.gail_batch_size
        self.entropy_coef = args.entropy_coef
        self.create_vild_network()
        self.behavior_cloning()

    # override to add sigmoid clipping for unhinged loss version. 
    def create_networks(self):
        self.trunk = Discriminator_VILD(self.state_dim + self.action_dim, self.hidden_dim).to(self.device)     
        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=self.lr)

    def create_vild_network(self):
        self.worker_net = Worker_Noise(self.state_dim, self.action_dim, worker_num=self.worker_num, device=self.device).to(self.device)
        self.worker_net.train() 
        self.optimizer_worker = torch.optim.Adam(self.worker_net.parameters(), lr=3e-4)     

        self.policy_psi_net = Policy_Psi(self.state_dim, self.action_dim, worker_num=self.worker_num, device=self.device).to(self.device)
        self.policy_psi_net.train() 
        self.optimizer_policy_psi = torch.optim.Adam(self.policy_psi_net.parameters(), lr=3e-4)

    # override. Split state and action, and add ids into dataset
    def make_dataset(self, args):
        self.load_expert_data(args)   # h5py demos are loaded into tensor. 
        expert_dataset = data_utils.TensorDataset(self.real_state_tensor, self.real_action_tensor, self.real_id_tensor)
        drop_last = len(expert_dataset) > self.gail_batch_size        
        self.expert_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=self.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    def compute_inference_reward(self, state, action):
        with torch.no_grad():
            self.trunk.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            if self.ail_loss_type == "logistic":    # binary_cross entropy. corresponding to GAIL-like variant.
                reward =  -F.logsigmoid(-d)  # minimize agent label score. 
            elif self.ail_loss_type == "unhinged":
                reward =  d
            if not hasattr(self.ret_rms.var, "__shape__"):
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8) 

    ## override to use BCE
    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.trunk.eval()
            d = self.trunk(torch.cat([state, action], dim=1))

            if self.ail_loss_type == "logistic":    # binary_cross entropy. corresponding to GAIL-like variant.
                reward =  -F.logsigmoid(-d)  # minimize agent label score. 
            elif self.ail_loss_type == "unhinged":
                reward =  d

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            if self.reward_std :
                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            else:
                return reward 

    ## sample data index based on covariance and compute importance weight. 
    def importance_sampling(self):
        worker_cov_k = self.worker_net.get_worker_cov(mean=True).cpu().detach().numpy()
        prob_k = 1 / (worker_cov_k)
        prob_k = prob_k / prob_k.sum()
        index, iw = [], [] 
        index_worker_idx_tmp = self.index_worker_idx.copy()

        for i in range(0, self.mini_batch_size):
            ## sample k from per_prob.
            choice_k = np.random.choice(self.worker_num, size=1, p=prob_k )[0]  
            index_i = np.random.randint(0, len(index_worker_idx_tmp[choice_k]))       
            index += [ index_i + index_worker_idx_tmp[choice_k][0] ]   

            ## remove the i-th sample to not re-sample it.
            index_worker_idx_tmp[choice_k] = np.delete(index_worker_idx_tmp[choice_k], index_i)

            iw_tmp = 1 / (self.worker_num * prob_k[choice_k])
            if self.per_alpha == 2: ## Truncate 
                iw_tmp = np.minimum(1, iw_tmp) 
            iw += [ iw_tmp ]   #iw of that sample is w(k) = p(k)/prob_k(k) = 1/ (K*prob_k(k))
        index = torch.LongTensor( np.array(index)) 
        iw = torch.FloatTensor(iw).to(self.device).unsqueeze(-1)
        return index, iw 

    # @override 
    def update(self, rollouts, obsfilt=None):
        self.trunk.train()

        index = torch.randperm(self.data_size)[0:self.mini_batch_size].to(self.device)
        self.update_inference(index=index)    # update policy psi  
        self.update_worker_noise(index=index)    # update worker noise 
        
        rollouts_size = rollouts.get_batch_size()
        policy_mini_batch_size = self.gail_batch_size if rollouts_size > self.gail_batch_size \
                            else rollouts_size

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=policy_mini_batch_size)

        loss = 0
        n = 0
        for policy_batch in policy_data_generator:

            ## mostly from vild code in git repo. 
            # fake
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))

            # real data 
            iw = 1
            if self.per_alpha > 0:  
                index, iw = self.importance_sampling()
            if self.per_alpha == 3: # no IW. 
                iw = 1

            # real 
            expert_state, expert_action, expert_id = self.real_state_tensor[index, :], self.real_action_tensor[index, :], self.real_id_tensor[index, :]
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)

            sample_actions, _, _, _ = self.policy_psi_net.sample_full( expert_state, expert_action, expert_id, symmetric=False)
            sample_actions = sample_actions.detach() 
            expert_d = self.trunk(torch.cat([expert_state, sample_actions], dim=1))

            if self.ail_loss_type == "unhinged":
                loss_fake = policy_d.mean()
                loss_real = (expert_d * iw).mean()
                gail_loss = -(loss_real - loss_fake)

            elif self.ail_loss_type == "logistic":
                if self.per_alpha > 0 and self.per_alpha != 3:
                    adversarial_loss_real = torch.nn.BCEWithLogitsLoss(weight=iw) 
                else:
                    adversarial_loss_real = torch.nn.BCEWithLogitsLoss() 
                adversarial_loss_fake = torch.nn.BCEWithLogitsLoss() 
                label_real = autograd.Variable(torch.FloatTensor(expert_d.size(0), 1).fill_(self.label_expert), requires_grad=False).to(self.device)
                label_fake = autograd.Variable(torch.FloatTensor(policy_d.size(0), 1).fill_(self.label_policy), requires_grad=False).to(self.device)
                loss_real = adversarial_loss_real(expert_d, label_real)
                loss_fake = adversarial_loss_fake(policy_d, label_fake)
                gail_loss = loss_real + loss_fake

            grad_pen = self.compute_grad_pen( torch.cat([expert_state, expert_action], dim=1),
                                              torch.cat([policy_state, policy_action], dim=1),
                                              self.gp_lambda)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

        return loss / n

    """ Function to update the parameters of q_psi """
    def update_inference(self, index):
        s_real = self.real_state_tensor[index, :].to(self.device)
        a_noise = self.real_action_tensor[index, :].to(self.device)
        worker_id = self.real_id_tensor[index].to(self.device)   # worker ID, LongTensor with value in [0, worker_num-1]
 
        worker_cov, worker_mu = self.worker_net(s_real, worker_id)        # each tensor size [batch_size, action_dim]
        worker_cov = worker_cov.data.detach()
        worker_mu = 0
            
        ent_coef = 0.001 # 0.01 seems to high and makes policy diverge

        sym = True 

        """ sample action from q_psi """
        sample_actions, log_probs, action_mean, action_log_std = self.policy_psi_net.sample_full( s_real, a_noise, worker_id, symmetric=sym)

        if sym:
            worker_cov = worker_cov.repeat(2, 1)
            s_real = s_real.repeat(2, 1)
            a_noise = a_noise.repeat(2, 1)
        
        error = (0.5 * (a_noise - sample_actions) ** 2 / worker_cov ).mean()    # [batch_size, action_dim] -> [1]. 
        rwd = self.compute_inference_reward( s_real, sample_actions).mean() - (ent_coef * log_probs).mean()     # [batch_size, 1] -> [1] 

        psi_loss = -(rwd / self.action_dim - error)    
        psi_loss = psi_loss * torch.min(worker_cov)    
        psi_loss += 0.001 * ((action_mean ** 2).mean() + (action_log_std ** 2).mean())  

        self.optimizer_policy_psi.zero_grad()
        psi_loss.backward()       
        self.optimizer_policy_psi.step()
        
    """ Function to update the parameters of worker net """
    def update_worker_noise(self, index):

        s_real = self.real_state_tensor[index, :].to(self.device)
        a_noise = self.real_action_tensor[index, :].to(self.device)
        worker_id = self.real_id_tensor[index].to(self.device) 

        worker_cov, worker_mu = self.worker_net(s_real, worker_id)

        sample_actions, _, _, _ = self.policy_psi_net.sample_full( s_real, a_noise, worker_id, symmetric=False)
    
        w_loss = 0.5 * ( (a_noise - sample_actions.data.detach()) ** 2 / worker_cov ).mean()
        w_loss += -0.5 * (self.noise_t**2) * (( 1 / worker_cov)).mean()    # the trace term    small and negigible. 
        w_loss += 0.5 * torch.log(worker_cov).mean()  # regularization term 
        
        self.optimizer_worker.zero_grad()
        w_loss.backward()
        self.optimizer_worker.step()

    def behavior_cloning(self, learning_rate=3e-4, bc_step=1000):

        bc_step_per_epoch = self.data_size / self.mini_batch_size 
        bc_epochs = math.ceil(bc_step/bc_step_per_epoch)

        train = data_utils.TensorDataset(self.real_state_tensor.to(self.device), self.real_action_tensor.to(self.device), self.real_id_tensor.unsqueeze(-1).to(self.device))
        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)

        optimizer_psi_bc = torch.optim.Adam(self.policy_psi_net.parameters(), lr=learning_rate) 
        
        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
                count = count + 1       

                action_mean_psi, _, _ = self.policy_psi_net( s_batch, a_batch, w_batch)
                loss_psi = 0.5 * ((action_mean_psi - a_batch) ** 2 ).mean()    ##
                optimizer_psi_bc.zero_grad()
                loss_psi.backward()
                optimizer_psi_bc.step()

        t1 = time.time()
        print("Psi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss_psi.item()))
