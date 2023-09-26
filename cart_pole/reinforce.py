import gym
from agent_reinforce import Agent
import numpy as np
import torch


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    agent = Agent(4, 2)
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
            agent.rewards.append(reward)
            # reward = reward_shaping(observation, action)
            score += reward 

            observation = observation_

        agent.learn()
        scores.append(score)
        # eps_history.append(agent.epsilion)

        avg_score = np.mean(scores[-100:])

        print('episode: ', i, 'score %.2f' % score, 
                'average_score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    
    torch.save(agent.Q.state_dict(), "/home/joel/PhD/RL-Skid2Mid/model.pt")