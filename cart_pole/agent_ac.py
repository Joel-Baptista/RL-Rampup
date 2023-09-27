import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
from torch.distributions import Categorical

class AgentAC():
    def __init__(self, dim_states, n_actions, lr=0.0001, gamma = 0.99,epsilon_end = 0.1, e_decay = 0.0001,
    max_mem_size=100000, batch_size = 64) -> None:
        self.lr = lr
        self.dim_states = dim_states
        self.n_actions = n_actions
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.e_decay = e_decay
        self.epsilon = 1
        self.gamma = gamma
        self.history = {}
        self.history["episodes"] = {}
        self.policy = Policy(dim_states=dim_states, dim_actions=n_actions, lr=lr)
        self.Q_eval = DeepQNetwork(self.lr, self.dim_states, fc1_dims=256, fc2_dims=256, n_actions=n_actions)

        self.mem_counter = 0
        self.max_mem_size = max_mem_size

        self.state_memory = np.zeros((self.max_mem_size, dim_states), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem_size, dim_states), dtype=np.float32)
        self.action_memory = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.bool)


        self.saved_log_probs = []
        self.Gt = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        Q_value = self.Q_eval(state)
        print(Q_value[0])
        print(action.item())
        self.Gt.append(self.Q_eval(state)[::action.item()])

        return action.item()

    def learn_policy(self):
        
        print(self.Gt)
        Gt = torch.tensor(self.Gt)
        # Gt = (Gt - Gt.mean()) / (Gt.std() + 0.01)
        policy_gradient = []
        for k in range(0, len(self.saved_log_probs)):
            policy_gradient.append(-self.saved_log_probs[k] * Gt[k])
        
        policy_gradient = torch.cat(policy_gradient).sum()
        policy_gradient.backward()
        self.policy.optimizer.step()
        self.policy.optimizer.zero_grad()
        
        self.saved_log_probs = []
        self.Gt = []
        self.policy.optimizer.step()

    def learn_q(self, state, action, reward, state_):
        if self.mem_counter < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

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

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions) -> None:
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims 
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        
        state = torch.tensor(state, dtype=torch.float32)
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions