import gym
from agents.ppo_discrete import Agent
import numpy as np
import os

config = {"n_games": 10,
          "env":'CartPole-v1',
          "chkpt_dir": "/home/joel/PhD/results/rl/models",
          "algorithm": "ppo",
          "alpha": 0.001,
          "beta": 0.001,
          "batch_size": 100,
          "experiment": "baseline2_no_reward",
          "n_epochs": 4,
          "fc2": 512,
          "debug": True}

if __name__== "__main__":
    print(config)
    
    chkpt_dir = os.path.join(config["chkpt_dir"], config["algorithm"] ,config["experiment"])
    env = gym.make(config["env"], render_mode="human")

    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, n_epochs= config["n_epochs"],
                  alpha=config["alpha"], batch_size=config["batch_size"], chkpt_dir=chkpt_dir)

    agent.load_models()

    score_history = []
    for i in range(config["n_games"]):
        
        
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:           
            
            action, _, _ = agent.choose_action(observation)

            observation_, reward, done, unkown, info = env.step(action)

            if done or unkown: done = True
    
            score += reward

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])


        print(f'episode {i}, score {score}, avg score {avg_score}')

    env.close()
