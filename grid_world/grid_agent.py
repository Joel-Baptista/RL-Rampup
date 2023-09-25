import numpy as np
import random 
import json

class GridAgent():
    def __init__(self, grid_size, n_actions, gamma = 0.99,epsilon_end = 0.1, e_decay = 0.0001) -> None:
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.epsilon_end = epsilon_end
        self.e_decay = e_decay
        self.epsilon = 1
        self.gamma = gamma
        self.history = {}
        self.history["episodes"] = {}
        self.state_count = np.zeros((grid_size, grid_size)) 
        
        self.Q = np.zeros((grid_size, grid_size, n_actions))

    def choose_action(self, state):

        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:

            max_array = np.max(self.Q[state]) == self.Q[state]
            
            if np.count_nonzero(max_array) > 1: # If more than one maximum, randomize the choice
                actions = []
                for i, is_max in enumerate(max_array):
                    if is_max==True: actions.append(i) 
                action = np.array(random.choice(actions))
            else:
                action = np.argmax(self.Q[state])


        return action

    def predict(self, state):
        max_array = np.max(self.Q[state]) == self.Q[state]
        
        if np.count_nonzero(max_array) > 1: # If more than one maximum, randomize the choice
            actions = []
            for i, is_max in enumerate(max_array):
                if is_max==True: actions.append(i) 
            action = np.array(random.choice(actions))
        else:
            action = np.argmax(self.Q[state])


        return action


    def q_learn(self, state, action, reward, next_state):

        self.state_count[state[0], state[1]] += 1

        self.Q[(state[0], state[1], action)] =self.Q[(state[0], state[1], action)] + (1 / self.state_count[state[0], state[1]])*(reward + self.gamma * np.max(self.Q[next_state]) - self.Q[(state[0], state[1], action)])
        # self.Q[(state[0], state[1], action)] =reward + self.gamma * np.max(self.Q[next_state])

        self.epsilon = self.epsilon - self.e_decay if self.epsilon > self.epsilon_end else self.epsilon_end

    def load_to_history(self, episode, state):
        
        if str(episode) not in self.history["episodes"].keys():
            # print(self.history.keys())
            self.history["episodes"][f"{episode}"] = []

        self.history["episodes"][f"{episode}"].append(state)
    
    def save_history(self, goal_pos, n, train_time):
        self.history["env_props"] = {}
        self.history["env_props"]["goal_pos"] = goal_pos
        self.history["env_props"]["grid_size"] = self.grid_size 

        self.history["train"] = {}
        self.history["train"]["time"] = train_time

        with open(f'history_{n}.json', 'w') as outfile:
            json.dump(self.history, outfile)

    def save_agent(self, n):
    
        with open(f'agent_{n}.json', 'w') as outfile:
            json.dump(self.Q.tolist(), outfile)
