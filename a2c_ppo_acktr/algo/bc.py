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
from ..algo.ail import AIL
from ..algo.ail_utils import *
from colorama import init
from termcolor import cprint, colored
init(autoreset=True)
p_color = "yellow"

import matplotlib.pyplot as plt

device_cpu = torch.device("cpu")

class BC(AIL):
    def __init__(self, actor_critic, observation_space, action_space, device, args):
        super(BC, self).__init__(observation_space, action_space, device, args)
        self.actor_critic = actor_critic 
        self.lr = args.il_lr    # learning rate for MLP
        self.bc_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        self.dataloader_iterator = iter(self.expert_loader)


    ## override 
    def create_networks(self):
        pass 
        
    ## override 
    def update(self, rollouts=None, obsfilt=None):
        self.actor_critic.train()

        for i in range(5):  # to make bc and ail have the same number of total gradient steps. 
            try:
                expert_batch = next(self.dataloader_iterator)
            except StopIteration:
                self.dataloader_iterator = iter(self.expert_loader)
                expert_batch = next(self.dataloader_iterator)

            expert_state, expert_action = expert_batch[0], expert_batch[1]  

            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)

            _, action, _, _ = self.actor_critic.act(
                expert_state, None,
                None, deterministic=True)

            loss = nn.MSELoss()
            bc_loss = loss(action, expert_action) 

            # bc_loss = 0.5 * ((expert_action - action) ** 2 ).mean()    ##

            self.bc_optimizer.zero_grad()
            bc_loss.backward()
            self.bc_optimizer.step()

        return 
