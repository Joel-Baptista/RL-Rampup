import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random

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
        self.Q = Qnetwork(dim_states=dim_states, dim_actions=n_actions, lr=lr)

    def choose_action(self, state):

        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:

            Q_values = torch.tensor([self.Q(torch.Tensor(np.append(state, i))) for i in range(0, self.n_actions)])
            action = torch.argmax(Q_values).item()

        return action

    def learn(self, state, action, reward, next_state):
        self.Q.optimizer.zero_grad()
        
        Q_values_target = torch.tensor([self.Q(torch.Tensor(np.append(next_state, i))) for i in range(0, self.n_actions)])
        Q_max_target = torch.max(Q_values_target)

        td_target = reward + self.gamma * Q_max_target

        Q_value = self.Q(torch.Tensor(np.append(state, action)))

        loss = self.Q.loss(Q_value, td_target)
        print(f"Loss: {loss.item()}")
        loss.backward()

        self.Q.optimizer.step()

        self.epsilon = self.epsilon - self.e_decay if self.epsilon > self.epsilon_end else self.epsilon_end


class Qnetwork(nn.Module):
    def __init__(self, dim_states, dim_actions, lr) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim_states + dim_actions, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

