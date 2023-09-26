import gym
from agent import Agent
from agent_buffer import AgentBuffer
import numpy as np
import torch


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # agent = Agent(dim_states=4, n_actions=2, lr=0.0001, gamma=0.9, epsilon_end=0.02, e_decay=0.0005)
    agent = AgentBuffer(gamma=0.99, epsilon=1.0, batch_size=256, n_actions=2,
        eps_end=0.01, input_dims=[4], lr=0.0001, max_mem_size=100_000)
    
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        j = 0
        while not done:
            frame = env.render(mode='rgb_array')

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            # reward = reward_shaping(observation, action)
            score += reward 

            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn(observation, action, reward, observation_)

            observation = observation_

        scores.append(score)
        # eps_history.append(agent.epsilion)

        avg_score = np.mean(scores[-100:])

        print('episode: ', i, 'score %.2f' % score, 
                'average_score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    
    torch.save(agent.Q.state_dict(), "/home/joel/PhD/RL-Skid2Mid/model.pt")
