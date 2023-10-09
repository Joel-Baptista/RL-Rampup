import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions.normal import Normal 
import numpy as np
from agents.replay_buffer import ReplayBuffer

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], env=None, gamma=0.99, n_actions=[2], device= "cuda:3",
    max_size=1_000_000, fc1=256, fc2=256, batch_size=256, reward_scale=2, tau=0.005, chkpt_dir='') -> None:
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(
            alpha=alpha,
            state_dims=input_dims,
            n_actions=n_actions,
            name='actor',
            max_action=env.action_space.high,
            min_action=env.action_space.low,
            fc1_dims=fc1,
            fc2_dims=fc2,
            chkpt_dir=chkpt_dir,
            device=device
        )

        self.critic_1 = CriticNetwork(
            state_dims=input_dims,
            n_actions=n_actions,
            beta=beta,
            name='critic_1',
            fc1_dims=fc1,
            fc2_dims=fc2,
            chkpt_dir=chkpt_dir,
            device=device
        )

        self.critic_2 = CriticNetwork(
            state_dims=input_dims,
            n_actions=n_actions,
            beta=beta,
            name='critic_2',
            fc1_dims=fc1,
            fc2_dims=fc2,
            chkpt_dir=chkpt_dir,
            device=device
        )
        
        self.value = ValueNetwork(
            beta=beta,
            state_dims=input_dims,
            name='value',
            fc1_dims=fc1,
            fc2_dims=fc2,
            chkpt_dir=chkpt_dir,
            device=device
        )
        self.target_value = ValueNetwork(
            beta=beta,
            state_dims=input_dims,
            name='target_value',
            fc1_dims=fc1,
            fc2_dims=fc2,
            chkpt_dir=chkpt_dir,
            device=device
        )

        self.scale = reward_scale
        self.update_network_parameter(tau=1)

    def choose_action(self, observation, evaluate=False):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameter(self, tau=None):
        if tau is None:
            tau = self.tau 

        targets = self.target_value.state_dict()
        
        for key in self.value.state_dict():
            targets[key] = self.value.state_dict()[key].clone() * tau + self.target_value.state_dict()[key].clone()*(1-tau)

        self.target_value.load_state_dict(targets)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print("... load models ...")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None, None, None, None, None

        state, new_state, action, reward, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done,  dtype=T.bool).to(self.actor.device)

        value = self.value(state).view(-1)  
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(state, actions)
        q2_new_policy = self.critic_2(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(state, actions)
        q2_new_policy = self.critic_2(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        # actor_loss = -log_probs - critic_value
        # actor_loss = -critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1(state, action).view(-1)
        q2_old_policy = self.critic_2(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameter()

        return actor_loss.cpu().detach().numpy(), \
            critic_1_loss.cpu().detach().numpy(), \
            critic_2_loss.cpu().detach().numpy(), \
            value_loss.cpu().detach().numpy(), \
            log_probs.mean().cpu().detach().numpy()
            




class CriticNetwork(nn.Module):
    def __init__(self, state_dims, n_actions, beta, device, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir="") -> None:
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions[0]
        self.state_dims = state_dims[0] 

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + "_sac.pt")

        self.fc1 = nn.Linear(self.state_dims+ self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device(device if T.cuda.is_available() else 'cpu')

        self.to(self.device)
        print(self)

    def forward(self, state, action):
        action_value = F.relu(self.fc1(T.concat([state, action], axis=1)))
        action_value = F.relu(self.fc2(action_value))
        
        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class ValueNetwork(nn.Module):
    def __init__(self, state_dims, beta, device, fc1_dims=512, fc2_dims=512, name='value', chkpt_dir="") -> None:
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.state_dims = state_dims[0] 

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + "_sac.pt")
    
        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device(device if T.cuda.is_available() else 'cpu')

        self.to(self.device)
        print(self)

    def forward(self, state):
        action_value = F.relu(self.fc1(state))
        action_value = F.relu(self.fc2(action_value))
        
        v = self.v(action_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class ActorNetwork(nn.Module):
    def __init__(self, state_dims, n_actions, max_action, min_action,alpha, device, fc1_dims=512, fc2_dims=512, name='actor', chkpt_dir="") -> None:
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.state_dims = state_dims 
        self.n_actions = n_actions
        self.reparam_noise = 1e-6
        self.max_action = max_action
        self.min_action = min_action

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + "_actor.pt")

        self.fc1 = nn.Linear(*self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, *self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, *self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device(device if T.cuda.is_available() else 'cpu')

        self.to(self.device)
        
        print(self)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # TODO add tanh up in this business
        mu = self.mu(x)
        sigma = T.sigmoid(self.sigma(x))

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1) #clamp is faster than sigmoid function

        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        print(f"sigma: {sigma}")
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # actions_adjusted = self.adjust_action_values(T.tanh(actions)).to(self.device)
        actions_adjusted = T.tanh(actions) * T.tensor(self.max_action).to(self.device) 
        # actions_adjusted.to(self.device)
        log_probs = probabilities.log_prob(actions)
        print(f"log_probs1: {log_probs}")
        log_probs -= T.log(1-actions_adjusted.pow(2)+self.reparam_noise)
        print(f"log_probs2: {log_probs}")
        log_probs = T.tensor(log_probs).sum(-1, keepdim=True)
        print(f"log_probs3: {log_probs}")

        # print(f"mu: {mu}")
        # print(f"sigma: {sigma}")
        print("------------------------------")
        return actions_adjusted, log_probs


    def adjust_action_values(self, action):

        adjusted_action = action * ((self.max_action - self.min_action) / 2) + ((self.max_action + self.min_action) / 2)
        adjusted_action = T.clip(adjusted_action, min=self.min_action, max=self.max_action)

        return adjusted_action

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))