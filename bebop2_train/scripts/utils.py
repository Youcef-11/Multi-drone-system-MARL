#!/usr/bin/env python
import copy
import gym
import torch
import itertools
import random

import numpy as np
import torch.nn.functional as F

from collections import deque

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW
from torch.distributions.normal import Normal

from pytorch_lightning import LightningModule, Trainer


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()


# Create the replay buffer
class ReplayBuffer:
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
class RLDataset(IterableDataset):

    def __init__(self, buffer, sample_size=400):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience

# Update the target network
def polyak_average(net, target_net, tau=0.01):
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)

# Create the Deep Q-Network
class DQN(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + out_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),           
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(device)
        in_vector = torch.hstack((state, action))
        return self.net(in_vector.float())
    
#Create the gradient policy
class GradientPolicy(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims, max):
        super().__init__()

        self.max = torch.from_numpy(max).to(device)

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.linear_mu = nn.Linear(hidden_size, out_dims)
        self.linear_std = nn.Linear(hidden_size, out_dims)
        # self.linear_log_std = nn.Linear(hidden_size, out_dims)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device)
        x = self.net(obs.float())
        mu = self.linear_mu(x)
        std = self.linear_std(x)
        std = F.softplus(std) + 1e-3

        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)

        action = torch.tanh(action) * self.max
        return action, log_prob
    
# Soft actor-critic algorithm
class SAC(LightningModule):

    def __init__(self, env_name, capacity=100_000, batch_size=256, lr=1e-3, 
            hidden_size=256, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW, 
            samples_per_epoch=1_000, tau=0.05, alpha=0.02, epsilon=0.05):

        super().__init__()

        self.env = gym.make(env_name)

        obs_size = 17
        action_dims = 4
        max_action = np.array([1,1,1,1])
        self.episode = 0
        self.q_net1 = DQN(hidden_size, obs_size, action_dims)
        self.q_net2 = DQN(hidden_size, obs_size, action_dims)
        self.policy = GradientPolicy(hidden_size, obs_size, action_dims, max_action)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)

        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters()

        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f"{len(self.buffer)} samples in experience buffer. Filling...", end = '')
            score = self.play_episodes()
            print(f" : Score = {score}")


    @torch.no_grad()
    def play_episodes(self, policy=None):
        obs = self.env.reset()
        done = False

        score = 0

        while not done:
            if policy and random.random() > self.hparams.epsilon:
                action, _ = self.policy(obs)
                action = action.cpu().numpy()
            else:
                action = np.random.uniform(-1,1,4)
                
            next_obs, reward, done, info = self.env.step(action)
            exp = (obs, action, reward, done, next_obs)
            self.buffer.append(exp)
            obs = next_obs
            score += reward
        
        return score

    def forward(self, x):
        output = self.policy(x)
        return output

    def configure_optimizers(self):
        q_net_parameters = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters())
        q_net_optimizer = self.hparams.optim(q_net_parameters, lr=self.hparams.lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.lr)
        return [q_net_optimizer, policy_optimizer]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        states, actions, rewards, dones, next_states = batch
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        if optimizer_idx == 0:

            action_values1 = self.q_net1(states, actions)
            action_values2 = self.q_net2(states, actions)

            target_actions, target_log_probs = self.target_policy(next_states)

            next_action_values = torch.min(
                self.target_q_net1(next_states, target_actions),
                self.target_q_net2(next_states, target_actions)
            )
            next_action_values[dones] = 0.0

            expected_action_values = rewards + self.hparams.gamma * (next_action_values - self.hparams.alpha * target_log_probs)

            q_loss1 = self.hparams.loss_fn(action_values1, expected_action_values)
            q_loss2 = self.hparams.loss_fn(action_values2, expected_action_values)

            q_loss_total = q_loss1 + q_loss2
            self.log("episode/Q-Loss", q_loss_total)
            return q_loss_total

        elif optimizer_idx == 1:

            actions, log_probs = self.policy(states)

            action_values = torch.min(
                self.q_net1(states, actions),
                self.q_net2(states, actions)
            )

            policy_loss = (self.hparams.alpha * log_probs - action_values).mean()
            self.log("episode/Policy Loss", policy_loss)
            return policy_loss

    def training_epoch_end(self, training_step_outputs):
        score = self.play_episodes(policy=self.policy)

        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        print("\tScore =", score, end = '')

        # self.log("episode/episode_return", self.env.return_queue[-1])