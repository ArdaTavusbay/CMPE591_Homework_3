# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

gamma = 0.99

class VPG(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[256, 512, 256]):
        super(VPG, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_dim, hl[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hl)):
            layers.append(nn.Linear(hl[i-1], hl[i]))
            layers.append(nn.ReLU())
        # Output
        layers.append(nn.Linear(hl[-1], act_dim * 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Agent():
    def __init__(self):
        self.policy = VPG()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.rewards = []
        self.log_probs = []

    def decide_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        out = self.policy(state_tensor)
        action_mean, act_std = out.chunk(2, dim=-1)

        action_std = F.softplus(act_std) + 5e-2

        # NaN value fix
        if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
            print("[Warning]: NaN value, resetted parameters.")
            action_mean = torch.zeros_like(action_mean)
            action_std = torch.ones_like(action_std) * 0.1

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action).sum())
        return action

    def update_model(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns - returns.mean()
        loss = 0
        for log_prob, R in zip(self.log_probs, returns):
            loss += -log_prob * R

        self.optimizer.zero_grad()
        loss.backward()
        
        # prevent gradient explosion.
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []

    def add_reward(self, reward):
        self.rewards.append(reward)