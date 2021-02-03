import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

import math 

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


## Add From VILD.

class Policy_Psi(nn.Module):
    def __init__(self, state_dim, action_dim, device, worker_num=1, hidden_size=(100, 100), activation='tanh', param_std=0, log_std=0, a_bound=1, tanh_mean=1, squash_action=0):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        num_worker = worker_num
        self.tanh_mean = tanh_mean
        self.squash_action = squash_action
        self.num_worker = num_worker
        self.device = device 

        self.action_dim = action_dim

        self.param_std = param_std 
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.affine_layers = nn.ModuleList()

        last_dim = state_dim + action_dim         

        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        
        self.action_mean_k = []
        for i in range(0, num_worker):
            self.action_mean_k += [nn.Linear(last_dim, action_dim)]
            self.action_mean_k[i].weight.data.mul_(0.1)
            self.action_mean_k[i].bias.data.mul_(0.0)
        self.action_mean_k = nn.ModuleList(self.action_mean_k)

        if self.param_std == 1:
            self.log_std_out_k = []
            for i in range(0, num_worker):
                self.log_std_out_k += [nn.Linear(last_dim, action_dim)]
                self.log_std_out_k[i].weight.data.mul_(0.1)
                self.log_std_out_k[i].bias.data.mul_(0.0)        
            self.log_std_out_k = nn.ModuleList(self.log_std_out_k)
        elif self.param_std == 0:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim, num_worker) * log_std)
    
        self.a_bound = a_bound
        # assert self.a_bound == 1

        self.is_disc_action = False

        self.zero_mean = torch.FloatTensor(1, action_dim).fill_(0).to(self.device) 
        self.unit_var = torch.FloatTensor(1, action_dim).fill_(1).to(self.device)

        self.logprob_holder = torch.FloatTensor(1).to(self.device)

    def forward(self, state, noisy_action, worker_ids):
        x = torch.cat( (state, noisy_action), 1)

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = []
        for i in range(0, self.num_worker):
            action_mean += [self.action_mean_k[i](x)]
            if self.tanh_mean:
                action_mean[i] = torch.tanh(action_mean[i]) * self.a_bound

        ## gather action_mean according to worker_ids
        action_mean = torch.stack(action_mean, dim=2) # tensor [b_size, a_dim, num_k]
        worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.num_worker).to(self.device)
        action_mean = action_mean.gather(2, worker_ids_e)[:,:,0]

        if self.param_std == 1:
            action_log_std = []
            action_std = []

            for i in range(0, self.num_worker):
                action_log_std += [self.log_std_out_k[i](x) ]
                action_log_std[i] = torch.clamp(action_log_std[i], self.log_std_min, self.log_std_max)

            ## gather action_mean according to worker_ids
            action_log_std = torch.stack(action_log_std, dim=2) # tensor [b_size, a_dim, num_k]
            action_log_std = action_log_std.gather(2, worker_ids_e)[:,:,0]
            
        elif self.param_std == 0:
            worker_ids = worker_ids.type(torch.LongTensor).to(self.device)
            action_log_std_e = self.action_log_std.expand( state.size(0), -1, -1 ) # [b_size, a_dim, num_worker]
            worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.num_worker) ## expanded [batch_size, action_dim, num_worker]
            action_log_std = action_log_std_e.gather(2, worker_ids_e)[:,:,0]    ## [batch_size, noise_dim, num_worker] --> [batch_size, noise_dim]
        
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std
               
    def normal_log_density(self, x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)

    def sample_full(self, states, noisy_action, worker_id, symmetric=1):

        action_mean, action_log_std, action_std = self.forward(states, noisy_action, worker_id)

        # This line uses one epsilon to sample actions for all states samples.
        epsilon = torch.FloatTensor(action_mean.size()).data.normal_(0, 1).to(self.device)
        
        action_raw = action_mean + action_std * epsilon
        log_prob = self.normal_log_density(action_raw, action_mean, action_log_std, action_std)
        
        if self.squash_action:
            action = torch.tanh(action_raw) * self.a_bound     
            log_prob -= torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-8).sum(1, keepdim=True)    # ** this correction is only for a_bound = 1 !
        else: 
            action = action_raw

        if symmetric:
            action_sym_raw = action_mean - action_std * epsilon
            log_prob_sym = self.normal_log_density(action_sym_raw, action_mean, action_log_std, action_std) 

            if self.squash_action:
                action_sym = torch.tanh(action_sym_raw) * self.a_bound
                log_prob_sym -= torch.log(1 - torch.tanh(action_sym_raw).pow(2) + 1e-8).sum(1, keepdim=True)
            else:
                action_sym = action_sym_raw 

            ## concat them along batch dimension, return tensors with double batch size
            action = torch.cat( (action, action_sym), 0 )
            log_prob = torch.cat( (log_prob, log_prob_sym), 0 )

        return action, log_prob, action_mean, action_log_std

class Worker_Noise(nn.Module):
    def __init__(self, state_dim, action_dim, device, worker_num=1, worker_model=1, hidden_size=(256, 256), activation='relu', normalization=None):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.worker_num = worker_num
        self.device = device 

        """
        worker_model:
            1 = state independent covariance. Only model worker noise covariance. C(k) = diag[c_k]. (Used in the paper)
        """
        self.worker_model = worker_model
        self.action_dim = action_dim

        """ (log of) noise parameter for each worker. Taking log avoids using relu/clamp for non-negativity """
        self.worker_cov = nn.Parameter(torch.ones(self.action_dim, worker_num) * -1)

        if self.worker_model == 1:
            self.worker_mu_zero =  torch.zeros(1, self.action_dim).to(self.device) 
        else:
            raise NotImplementedError

    def forward(self, states, worker_ids):
        """ 
        This function return estimated noise covariance and mean-shift which are tensors of size [batch_size, action_dim].
        states size = [batch_size_1, state_dim]
        worker_ids size = [batch_size_2] Long tensor
        Depend on the worker_model, two batch sizes could be different.
        """
        worker_ids = worker_ids.type(torch.LongTensor).to(self.device)

        ## state independent. 
        if self.worker_model == 1:

            worker_ids_e = worker_ids.view(-1,1,1).expand(-1, self.action_dim, self.worker_num) ## expanded [batch_size, action_dim, worker_num]
            worker_cov_e = self.worker_cov.unsqueeze(0).expand(worker_ids.size(0), -1, -1)   ## [batch_size, action_dim, worker_num]
            worker_cov = worker_cov_e.gather(2, worker_ids_e)[:,:,0]    ## [batch_size, action_dim, worker_num] --> [batch_size, action_dim]

            return  torch.exp(worker_cov)  + 1e-8, self.worker_mu_zero.repeat(states.size(0), 1)
        else:
            raise NotImplementedError

    def get_worker_cov(self, mean=False):
        if mean:
            return torch.exp(self.worker_cov.mean(dim=0))    # mean across action dim.  return tensor size worker_num.
        else:
            return torch.exp(self.worker_cov)
