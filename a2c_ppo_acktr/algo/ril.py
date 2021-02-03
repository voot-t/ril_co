import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch import autograd
from os import path

from baselines.common.running_mean_std import RunningMeanStd

from ..model import Flatten
from ..algo.ail import AIL, Discriminator  
from ..algo.ail_utils import *
from colorama import init
from termcolor import cprint, colored
init(autoreset=True)
p_color = "yellow"

device_cpu = torch.device("cpu")

class RIL_CO(AIL):
    def __init__(self, observation_space, action_space, device, args):
        if not hasattr(self, 'b_size_multiplier'):        
            self.b_size_multiplier = 5  # 5 
        super(RIL_CO, self).__init__(observation_space, action_space, device, args)
        self.ril_prior = 0.5    # lambda in the paper. 

    ## @override 
    def create_networks(self):
        self.trunk_1 = Discriminator(self.state_dim + self.action_dim, self.hidden_dim).to(self.device) 
        self.trunk_2 = Discriminator(self.state_dim + self.action_dim, self.hidden_dim).to(self.device) 
        self.optimizer_1 = torch.optim.Adam(self.trunk_1.parameters(), lr=self.lr)
        self.optimizer_2 = torch.optim.Adam(self.trunk_2.parameters(), lr=self.lr)
        self.trunk = self.trunk_1   # alias for predict_reward

    ## @override
    def make_dataset(self, args):
        split = 0.5 
        print("b_size_multiplier %d" % self.b_size_multiplier) 

        self.load_expert_data(args)   # h5py demos are loaded into tensor. 
        data_size_1 = int( self.data_size * split )
        
        ## randomly split dataset into disjoint subsets. 
        indices = torch.randperm(self.data_size)
        self.real_state_tensor_1 = self.real_state_tensor[indices[:data_size_1], :]
        self.real_action_tensor_1 = self.real_action_tensor[indices[:data_size_1], :]
        self.real_state_tensor_2 = self.real_state_tensor[indices[data_size_1:], :]
        self.real_action_tensor_2 = self.real_action_tensor[indices[data_size_1:], :]
        self.data_size = None   #prevent possible errors.

        expert_dataset_1 = data_utils.TensorDataset(self.real_state_tensor_1, self.real_action_tensor_1)
        self.expert_loader_1 = torch.utils.data.DataLoader(
            dataset=expert_dataset_1,
            batch_size=self.gail_batch_size * (1+self.b_size_multiplier), # 1 gail-batch-size for true label training, and self.b_size_multiplier*gail-batch-size for pseudo-labeling. 
            shuffle=True)

        expert_dataset_2 = data_utils.TensorDataset(self.real_state_tensor_2, self.real_action_tensor_2)
        self.expert_loader_2 = torch.utils.data.DataLoader(
            dataset=expert_dataset_2,
            batch_size=self.gail_batch_size * (1+self.b_size_multiplier),
            shuffle=True)

    ## override
    def update(self, rollouts, obsfilt=None):
        self.trunk_1.train()
        self.trunk_2.train()

        rollouts_size = rollouts.get_batch_size()
        policy_mini_batch_size = self.gail_batch_size if rollouts_size > self.gail_batch_size else rollouts_size
        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=policy_mini_batch_size)

        loss_1, loss_2 = 0, 0
        n = 0
        for expert_batch_1, expert_batch_2, policy_batch in zip(self.expert_loader_1, 
                                              self.expert_loader_2, 
                                              policy_data_generator):

            policy_state, policy_action = policy_batch[0], policy_batch[2]

            # name data for convenience 
            expert_state_1, expert_action_1 = expert_batch_1[0], expert_batch_1[1]
            expert_state_2, expert_action_2 = expert_batch_2[0], expert_batch_2[1]

            if obsfilt is not None:
                expert_state_1 = obsfilt(expert_state_1.numpy(), update=False)
                expert_state_2 = obsfilt(expert_state_2.numpy(), update=False)

            expert_state_1 = torch.FloatTensor(expert_state_1).to(self.device)
            expert_state_2 = torch.FloatTensor(expert_state_2).to(self.device)

            expert_action_1 = expert_action_1.to(self.device)
            expert_action_2 = expert_action_2.to(self.device)
            expert_action_1_tr = expert_action_1[:self.gail_batch_size, :]
            expert_action_1_p  = expert_action_2[self.gail_batch_size:, :]
            expert_action_2_tr = expert_action_2[:self.gail_batch_size, :]
            expert_action_2_p  = expert_action_1[self.gail_batch_size:, :]

            expert_state_1_tr  = expert_state_1[:self.gail_batch_size, :] # from data_1 to train true-label term net 1
            expert_state_1_p   = expert_state_2[self.gail_batch_size:, :] # from data_2 to train pseudo-label term net 1

            expert_state_2_tr  = expert_state_2[:self.gail_batch_size, :] # from data_2 to train true-label term net 2
            expert_state_2_p   = expert_state_1[self.gail_batch_size:, :]  # from data_1 to train pseudo-label term net 2

            """ network 1 """
            # fake
            policy_d_1 = self.trunk_1(sa_cat(policy_state, policy_action))
            policy_loss_1 = self.adversarial_loss(policy_d_1 * self.label_policy)

            # real 
            expert_d_1_tr = self.trunk_1(sa_cat(expert_state_1_tr, expert_action_1_tr))
            expert_loss_1_tr = self.adversarial_loss(expert_d_1_tr * self.label_expert)

            # pseudo labeling. Get prediction from net 2 to select indices 
            with torch.no_grad():   
                expert_d_1_p = self.trunk_2(sa_cat(expert_state_1_p, expert_action_1_p)).detach().data.squeeze()
                index_p_1 = (expert_d_1_p < 0).nonzero()
            if index_p_1.size(0) > 1:
                index_p_1_sort = torch.argsort(expert_d_1_p[index_p_1], dim=0)[:self.gail_batch_size] # ascending 
                index_p_1 = index_p_1[index_p_1_sort].squeeze()
                
                expert_d_1_p = self.trunk_1(
                            sa_cat(expert_state_1_p[index_p_1,:], expert_action_1_p[index_p_1,:]))

                loss_p_1  = self.adversarial_loss(expert_d_1_p * self.label_policy)
                policy_loss_1 = (1-self.ril_prior) * policy_loss_1 + (self.ril_prior) * loss_p_1

            grad_pen_1 = self.compute_grad_pen(sa_cat(expert_state_1_tr, expert_action_1_tr),
                                               sa_cat(policy_state, policy_action),  
                                               self.gp_lambda, network=self.trunk_1)

            gail_loss_1 = expert_loss_1_tr + policy_loss_1
            loss_1 += (gail_loss_1 + grad_pen_1).item()
            
            """ network 2 """
            # fake
            policy_d_2 = self.trunk_2(sa_cat(policy_state, policy_action))
            policy_loss_2 = self.adversarial_loss(policy_d_2 * self.label_policy)

            # real 
            expert_d_2_tr = self.trunk_2(sa_cat(expert_state_2_tr, expert_action_2_tr))
            expert_loss_2_tr = self.adversarial_loss(expert_d_2_tr * self.label_expert)

            # pseudo labeling. Get prediction from net 1 to select indices 
            with torch.no_grad():   
                expert_d_2_p = self.trunk_1(sa_cat(expert_state_2_p, expert_action_2_p)).detach().data.squeeze()
                index_p_2 = (expert_d_2_p < 0).nonzero()
            if index_p_2.size(0) > 1:
                index_p_2_sort = torch.argsort(expert_d_2_p[index_p_2], dim=0)[:self.gail_batch_size]    # ascending 
                index_p_2 = index_p_2[index_p_2_sort].squeeze()
        
                expert_d_2_p = self.trunk_2(
                        sa_cat(expert_state_2_p[index_p_2,:], expert_action_2_p[index_p_2,:]))

                loss_p_2  = self.adversarial_loss(expert_d_2_p * self.label_policy)
                policy_loss_2 = (1-self.ril_prior) * policy_loss_2 + (self.ril_prior) * loss_p_2

            grad_pen_2 = self.compute_grad_pen(sa_cat(expert_state_2_tr, expert_action_2_tr),
                                               sa_cat(policy_state, policy_action), 
                                               self.gp_lambda, network=self.trunk_2)

            gail_loss_2 = expert_loss_2_tr + policy_loss_2
            loss_2 += (gail_loss_2 + grad_pen_2).item()

            n += 1

            self.optimizer_1.zero_grad()
            (gail_loss_1 + grad_pen_1).backward()
            self.optimizer_1.step()

            self.optimizer_2.zero_grad()
            (gail_loss_2 + grad_pen_2).backward()
            self.optimizer_2.step()

        return loss_1 / n

class RIL(AIL):
    def __init__(self, observation_space, action_space, device, args):
        if not hasattr(self, 'b_size_multiplier'):        
            self.b_size_multiplier = 5  # 5 
        super(RIL, self).__init__(observation_space, action_space, device, args)
        self.ril_prior = 0.5 

    ## @override 
    def create_networks(self):
        self.trunk = Discriminator(self.state_dim + self.action_dim, self.hidden_dim).to(self.device) 
        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=self.lr)

    ## @override
    def make_dataset(self, args):
        print("b_size_multiplier %d" % self.b_size_multiplier) 
        self.load_expert_data(args)   # h5py demos are loaded into tensor. 
        expert_dataset = data_utils.TensorDataset(self.real_state_tensor, self.real_action_tensor)

        drop_last = len(expert_dataset) > self.gail_batch_size        
        self.expert_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=self.gail_batch_size * self.b_size_multiplier,
            shuffle=True,
            drop_last=drop_last)

    ## override
    def update(self, rollouts, obsfilt=None):
        self.trunk.train()

        rollouts_size = rollouts.get_batch_size()
        policy_mini_batch_size = self.gail_batch_size if rollouts_size > self.gail_batch_size else rollouts_size
        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=policy_mini_batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(self.expert_loader, 
                                              policy_data_generator):

            policy_state, policy_action = policy_batch[0], policy_batch[2]

            # name data for convenience 
            expert_state, expert_action = expert_batch[0], expert_batch[1]

            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)

            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)

            expert_action_tr = expert_action[:self.gail_batch_size, :]
            expert_action_p  = expert_action[self.gail_batch_size:, :]
            expert_state_tr  = expert_state[:self.gail_batch_size, :] 
            expert_state_p   = expert_state[self.gail_batch_size:, :] 

            """ network 1 """
            # fake
            policy_d = self.trunk(sa_cat(policy_state, policy_action))
            policy_loss = self.adversarial_loss(policy_d * self.label_policy)

            # real 
            expert_d_tr = self.trunk(sa_cat(expert_state_tr, expert_action_tr))
            expert_loss_tr = self.adversarial_loss(expert_d_tr * self.label_expert)

            # pseudo labeling. Get prediction to select indices 
            with torch.no_grad():   
                expert_d_p = self.trunk(sa_cat(expert_state_p, expert_action_p)).detach().data.squeeze()
                index_p = (expert_d_p < 0).nonzero()
            if index_p.size(0) > 1:
                index_p_sort = torch.argsort(expert_d_p[index_p], dim=0)[:self.gail_batch_size] # ascending 
                index_p = index_p[index_p_sort].squeeze()
                expert_d_1_p = self.trunk(
                            sa_cat(expert_state_p[index_p,:], expert_action_p[index_p,:]))

                loss_p  = self.adversarial_loss(expert_d_p * self.label_policy)
                policy_loss = (1-self.ril_prior) * policy_loss + (self.ril_prior) * loss_p

            grad_pen = self.compute_grad_pen(sa_cat(expert_state_tr, expert_action_tr),
                                               sa_cat(policy_state, policy_action),  
                                               self.gp_lambda, network=self.trunk)

            gail_loss = expert_loss_tr + policy_loss
            loss += (gail_loss + grad_pen).item()
            
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

        return loss / n
