import gym
from agents.ppo import Agent
import numpy as np
import os

config = {"n_games": 20,
          "env":'BipedalWalker-v3',
          "chkpt_dir": "/home/joel/PhD/results/rl/models",
          "algorithm": "ppo",
          "alpha": 0.001,
          "beta": 0.001,
          "batch_size": 100,
          "experiment": "baseline6",
          "fc1": 1024,
          "fc2": 512,
          "debug": True}

if __name__== "__main__":
    print(config)
    
    chkpt_dir = os.path.join(config["chkpt_dir"], config["algorithm"] ,config["experiment"])
    env = gym.make(config["env"], render_mode="human")

    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape,
                fc1=config["fc1"], fc2=config["fc2"],
                chkpt_dir=chkpt_dir)

    if config["algorithm"] == "ppo":
        agent.actor.action_std = 0.1

    agent.load_models()

    score_history = []
    for i in range(config["n_games"]):
        
        
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:           
            
            if config["algorithm"] in ["ppo"]:
                action, _, _ = agent.choose_action(observation, evaluate=True)
            else:
                action = agent.choose_action(observation, evaluate=True)

            observation_, reward, done, unkown, info = env.step(action)

            if done or unkown: done = True
    
            score += reward
            # agent.remember(observation, action, reward, observation_, done)

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])


        print(f'episode {i}, score {score}, avg score {avg_score}')

    env.close()
