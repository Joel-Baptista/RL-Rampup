import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
import gym

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps: float = 1e-8, weight_decay: float = 0, amsgrad: bool = ...) -> None:
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)


        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)
                state['max_exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['max_exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, fc1, fc2, gamma=0.99) -> None:
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, fc1) 
        self.pi2 = nn.Linear(fc1, fc2) 
        self.pi = nn.Linear(fc2, n_actions)

        self.v1 = nn.Linear(*input_dims, fc1)
        self.v2 = nn.Linear(fc1, fc2)
        self.v = nn.Linear(fc2, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.states = []

    def forward(self, state):
        # for some reason, multiprocessing does not let multilayer??
        p2 = F.relu(self.pi1(state))
        # p2 = F.relu(self.pi2(p1))

        pi = self.pi(p2)

        v2 = F.relu(self.v1(state))
        # v2 = F.relu(self.v2(v1))
        v = self.v(v2)
        print("forward")

        return pi, v

    def calc_R(self, done):
        
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []

        for reward in self.rewards[::-1]:
            R = reward * self.gamma*R
            batch_return.append(R)

        batch_return.reverse()

        return T.tensor(batch_return, dtype=T.float)

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)
        pi, values = self.forward(states)
        values = values.squeeze()

        critic_loss = (returns-values)**2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)

        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, v =  self.forward(state)
        print("choose_action")
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action



class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                gamma, lr, name, global_ep_idx, env_id, n_games, t_max, fc1, fc2) -> None:
        super(Agent, self).__init__()

        self.n_games = n_games
        self.t_max = t_max
        self.local_actor_critic = ActorCritic(input_dims, n_actions, fc1=fc1, fc2=fc2)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer

    def run(self):
        print(f"run{self.name}")
        t_step = 1
        while self.episode_idx.value < self.n_games:
            done = False
            observation, _ = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            print(f"game{self.name}")
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                print(f"action{self.name}")
                
                
                observation_, reward, done, unkown, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % self.t_max == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                        
                        global_param._grad = local_param.grad
                    
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict()
                    )
                    self.local_actor_critic.clear_memory()
                
                t_step += 1
                observation = observation_
            
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1

            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)



