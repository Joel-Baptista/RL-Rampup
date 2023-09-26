import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
from torch.distributions import Categorical

class Agent():
    def __init__(self, dim_states, n_actions, lr=0.0001, gamma = 0.99,epsilon_end = 0.1, e_decay = 0.0001) -> None:
        self.dim_states = dim_states
        self.n_actions = n_actions
        self.epsilon_end = epsilon_end
        self.e_decay = e_decay
        self.epsilon = 1
        self.gamma = gamma
        self.history = {}
        self.history["episodes"] = {}
        self.policy = Policy(dim_states=dim_states, dim_actions=n_actions, lr=lr)

        self.saved_log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))

        return action.item()

    def learn(self):
        Gt = []

        for i in range(0, len(self.rewards)):
            G = 0
            for j in range(i, len(self.rewards)):
                
                G += pow(self.gamma, j) * self.rewards[j]

            Gt.append(G)
        
        Gt = torch.tensor(Gt)
        Gt = (Gt - Gt.mean()) / (Gt.std() + 0.01)
        policy_gradient = []
        for k in range(0, len(self.rewards)):
            policy_gradient.append(-self.saved_log_probs[k] * Gt[k])
        
        policy_gradient = torch.cat(policy_gradient).sum()
        print(policy_gradient)
        policy_gradient.backward()
        self.policy.optimizer.step()
        self.policy.optimizer.zero_grad()
        
        self.saved_log_probs = []
        self.rewards = []
        self.policy.optimizer.step()

class Policy(nn.Module):
    def __init__(self, dim_states, dim_actions, lr) -> None:

        super(Policy, self).__init__()
        self.fc1 = nn.Linear(dim_states, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, dim_actions)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        
        x = torch.Tensor(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return self.softmax(x)