import gym
from dq_learning import Agent
from utils import plotLearning
import numpy as np
import torch

if __name__ == "__main__":

    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilion=1.0, batch_size=256, n_actions=4,
    eps_end=0.01, input_dims=[8], lr=0.0001)

    print(agent.Q_eval.device)
    scores, eps_history = [], []
    n_games = 1

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward 

            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()

            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilion)

        avg_score = np.mean(scores[-100:])

        print('episode: ', i, 'score %.2f' % score, 
                'average_score %.2f' % avg_score,
                'epsilion %.2f' % agent.epsilion)

    x = [i+1 for i in range(n_games)]
    file_name = 'test.png'

    plotLearning(x, scores, eps_history, file_name)
    torch.save(agent.Q_eval.state_dict(), "/home/joel/PhD/RL-Skid2Mid/model.pt")





