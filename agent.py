import torch
from torch import optim
from model import VPG
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

gamma = 0.99

class Agent():
    def __init__(self):
        self.model = VPG()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.rewards = []
        self.log_probs = []
        self.entropies = []

        self.policy = self.model
        
    def decide_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        action_mean, act_std = self.model(state).chunk(2, dim=-1)

        action_std = F.softplus(act_std) + 5e-2

        if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
            print("Warning: NaN detected in policy output. Using random action.")
            action = torch.randn_like(action_mean)

            dummy_dist = Normal(torch.zeros_like(action_mean), torch.ones_like(action_std))
            log_prob = dummy_dist.log_prob(action).sum(dim=-1)
            entropy = dummy_dist.entropy().sum(dim=-1)
        else:
            normal_dist = Normal(action_mean, action_std)
            
            action = normal_dist.sample()
            
            log_prob = normal_dist.log_prob(action).sum(dim=-1)
            entropy = normal_dist.entropy().sum(dim=-1)
        
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        
        return action
    
    def add_reward(self, reward):
        self.rewards.append(reward)
    
    def update_model(self):
        if len(self.rewards) == 0:
            return
            
        R = 0
        returns = []
        
        # Calculate discounted returns
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        if len(returns) > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        if torch.isnan(returns).any():
            print("Warning: NaN detected in returns. Skipping update.")
            self.rewards = []
            self.log_probs = []
            self.entropies = []
            return
            
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        
        policy_loss = -(log_probs * returns.detach()).mean()
        
        if torch.isnan(policy_loss).any():
            print("Warning: NaN detected in policy loss. Skipping update.")
            self.rewards = []
            self.log_probs = []
            self.entropies = []
            return
        
        entropy_loss = -0.01 * entropies.mean()
        
        loss = policy_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        self.rewards = []
        self.log_probs = []
        self.entropies = []