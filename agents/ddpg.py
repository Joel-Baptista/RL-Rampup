import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None, gamma=0.99, n_actions=2, 
    max_size=1_000_000, tau=0.005, fc1=400, fc2=300, batch_size=64, noise=0.1, device= "cuda:3", chkpt_dir="") -> None:
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size =batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(state_dims=input_dims, n_actions=n_actions, 
                                    fc1_dims=fc1, fc2_dims=fc2, name="actor", chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(state_dims=input_dims, n_actions=n_actions, 
                                        fc1_dims=fc1, fc2_dims=fc2, name="critic", chkpt_dir=chkpt_dir)

        self.target_actor = ActorNetwork(state_dims=input_dims, n_actions=n_actions, 
                                        fc1_dims=fc1, fc2_dims=fc2, name="target_actor", chkpt_dir=chkpt_dir)
        self.target_critic = CriticNetwork(state_dims=input_dims, n_actions=n_actions, 
                                        fc1_dims=fc1, fc2_dims=fc2, name="target_critic", chkpt_dir=chkpt_dir)

        self.device = T.device(device if T.cuda.is_available() else 'cpu')

        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=beta)

        self.update_network_parameter(tau=1)

    def update_network_parameter(self, tau=None):
        if tau is None:
            tau = self.tau 

        
        # actor_params = self.actor.named_parameters()
        # critic_params = self.critic.named_parameters()
        # target_critic_params = self.target_critic.named_parameters()
        # target_actor_params = self.target_actor.named_parameters()
        

        # critic_params_dict = dict(critic_params)
        # actor_params_dict = dict(actor_params)
        # target_critic_params_dict = dict(target_critic_params)
        # target_actor_params_dict = dict(target_actor_params)

        # for name in critic_params_dict:
        #     critic_params_dict[name] = tau * critic

        targets = self.target_actor.state_dict()
        
        for key in self.actor.state_dict():
            targets[key] = self.actor.state_dict()[key] * tau + self.target_actor.state_dict()[key]*(1-tau)

        self.target_actor.load_state_dict(targets)

        targets = self.target_critic.state_dict()
        
        for key in self.critic.state_dict():
            targets[key] = self.critic.state_dict()[key] * tau + self.target_critic.state_dict()[key]*(1-tau)

        self.target_critic.load_state_dict(targets)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print("..... saving models ....")
        T.save(self.actor.state_dict, self.actor.checkpoint_file)
        T.save(self.critic.state_dict, self.critic.checkpoint_file)
        T.save(self.target_actor.state_dict, self.target_actor.checkpoint_file)
        T.save(self.target_critic.state_dict, self.target_critic.checkpoint_file)

    def load_models(self):
        print("..... loading models ....")
        self.actor.load_state_dict(T.load(self.actor.checkpoint_file))
        self.critic.load_state_dict(T.load(self.critic.checkpoint_file))
        self.target_actor.load_state_dict(T.load(self.target_actor.checkpoint_file))
        self.target_critic.load_state_dict(T.load(self.target_critic.checkpoint_file))

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()
        
        state = T.tensor([observation], dtype=T.float32).to(self.device)
        actions = self.actor(state) * self.max_action
        if not evaluate:
            actions += T.randn(size=self.n_actions).to(self.device) * self.noise

        actions = T.clip(actions, min=self.min_action, max=self.max_action)
        self.actor.train()
        return actions[0].cpu().detach().numpy()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 0, 0
        
        state, new_state, action, reward, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state, dtype=T.float32).to(self.device)
        states_ = T.tensor(new_state, dtype=T.float32).to(self.device)
        actions = T.tensor(action, dtype=T.float32).to(self.device)
        rewards = T.tensor(reward, dtype=T.float32).to(self.device)
        dones = T.tensor(done, dtype=T.int).to(self.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor(states_)
        critic_value_ = self.target_critic(states_, target_actions)
        critic_value = self.critic(states, actions)

        # target = rewards + self.gamma * critic_value_*(1-dones)
        target = []

        for j in range(self.batch_size):
            target.append(rewards[j] + self.gamma * critic_value_[j]*(1-dones[j]))
        
        target = T.tensor(target).to(self.device)
        # print(target.size())
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic_optim.zero_grad()

        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()

        self.critic_optim.step()

        self.critic.eval()
        self.actor.train()
        self.actor_optim.zero_grad()
        new_policy_actions = self.actor(states)
        actor_loss = -self.critic(states, new_policy_actions)
        actor_loss = T.mean(actor_loss)

        actor_loss.backward()
        self.actor_optim.step()

        self.update_network_parameter()

        return actor_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy()


class CriticNetwork(nn.Module):
    def __init__(self, state_dims, n_actions, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir="") -> None:
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions[0]
        self.state_dims = state_dims[0] 

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + "_ddpg.pt")

        self.fc1 = nn.Linear(self.state_dims+ self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

    def forward(self, state, action):
        action_value = F.relu(self.fc1(T.concat([state, action], axis=1)))
        action_value = F.relu(self.fc2(action_value))
        
        q = self.q(action_value)

        return q

class ActorNetwork(nn.Module):
    def __init__(self, state_dims, n_actions, fc1_dims=512, fc2_dims=512, name='actor', chkpt_dir="") -> None:
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.state_dims = state_dims 

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + "_ddpg.pt")

        self.fc1 = nn.Linear(*self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, *self.n_actions)
        
    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu =  T.tanh(self.mu(x))

        return mu
        
