import os
import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class Agent:
    def __init__(self, input_dims, n_actions, env,  fc1, fc2, beta=0.001, action_std_init=0.6, action_std_decay=0.005, action_std_min=0.1,
    gamma=0.99, alpha=0.0003, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10, gae_lambda=0.95, chkpt_dir = "") -> None:
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(action_dim=n_actions, 
                                  input_dims=input_dims, 
                                  alpha=alpha, 
                                  fc1_dims= fc1,
                                  fc2_dims= fc2,
                                  chkpt_dir=chkpt_dir,
                                  action_std_init=action_std_init,
                                  action_std_decay=action_std_decay,
                                  action_std_min=action_std_min)
        self.critic = CriticNetwork(
            input_dims= input_dims, 
            alpha= beta,
            fc1_dims= fc1,
            fc2_dims= fc2, 
            chkpt_dir=chkpt_dir)

        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(
            state=state,
            action=action,
            probs=probs,
            vals=vals,
            reward=reward,
            done=done)
    
    def save_models(self):
        print("... saving models ...")

        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... load models ...")

        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def adjust_action_values(self, action):

        adjusted_action = action * ((self.max_action - self.min_action) / 2) + ((self.max_action + self.min_action) / 2)
        adjusted_action = T.clip(adjusted_action, min=self.min_action, max=self.max_action)

        return adjusted_action


    def choose_action(self, observation, evaluate=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        
        probs = T.squeeze(dist.log_prob(action)).item()
        print(f"probs: {probs}")
        action = T.squeeze(action).detach()
        value = T.squeeze(value).item()

        return self.adjust_action_values(action).cpu().numpy(), probs, value

    def learn(self):
        actor_loss_logging = []
        critic_loss_logging = []

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*vals_arr[k+1]*(1-int(done_arr[k])) - vals_arr[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(vals_arr).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_advantage = advantage[batch] * prob_ratio
                weighted_clipped_advatage = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = - T.min(weighted_advantage, weighted_clipped_advatage).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                actor_loss_logging.append(actor_loss.cpu().detach().numpy())
                critic_loss_logging.append(critic_loss.cpu().detach().numpy())
        
        self.memory.clear_memory()

        return np.mean(actor_loss_logging), np.mean(critic_loss_logging)




class PPOMemory:
    def __init__(self, batch_size) -> None:
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        # np.random.shuffle(indices) # if we shuffle the indices, don't we loose the concept of trajectory?
        np.random.shuffle(batch_start)
        batches =  [indices[i:i+self.batch_size] for i in batch_start]

        # print("self.states")
        # print(self.states)
        # print("batch_start")
        # print(batch_start)
        # print("indices")
        # print(indices)
        # print("batches")
        # print(batches)
        # print("------------------------------------------------------------")


        return  np.array(self.states), \
                np.array(self.actions), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.rewards), \
                np.array(self.dones), \
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, action_dim, input_dims, alpha, action_std_init, action_std_decay, action_std_min,
                 fc1_dims=256, fc2_dims=256, chkpt_dir="") -> None:
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_ppo.pt")
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, *action_dim),
            nn.Tanh()
        )
        self.action_dim = action_dim
        self.action_std = action_std_init
        self.action_std_decay = action_std_decay
        self.action_std_min = action_std_min

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:3" if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.action_var = T.full((*action_dim,), action_std_init * action_std_init).to(self.device)


    def set_action_std(self, action_std=None):
        
        if action_std is None:
            self.action_std = self.action_std - self.action_std_decay
        else:
            self.action_std = action_std
        if self.action_std < self.action_std_min: self.action_std = self.action_std_min
        
        self.action_var = T.full((*self.action_dim,), self.action_std * self.action_std).to(self.device)

    def forward(self, state):
        action_mean = self.actor(state)
        cov_mat = T.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir="") -> None:
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "critic_ppo.pt")
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:3" if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


