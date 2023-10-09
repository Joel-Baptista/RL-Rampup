import os
import gym
from agents.ppo_discrete import Agent
import wandb
import time
import numpy as np


def watch(logger: wandb, agent: Agent):
    logger.watch(agent.actor )
    logger.watch(agent.critic)

def unwatch(logger: wandb, agent: Agent):
    logger.unwatch(agent.actor )
    logger.unwatch(agent.critic)


# config = {"n_games": 2000,
#           "env":'CartPole-v1',
#           "chkpt_dir": "/root/models",
#           "experiment": "baseline1",
#           "alpha": 0.0003,
#           "algorithm": "ppo",
#           "batch_size": 64,
#           "debug": True}

config = {"n_games": 300,
          "N": 20,
          "env":'CartPole-v1',
          "chkpt_dir":  "/home/joel/PhD/results/rl/models",
          "experiment": "baseline2_no_reward",
          "batch_size": 5,
          "n_epochs": 4,
          "alpha": 0.0003,
          "algorithm": "ppo",
          "debug": False}

if __name__== "__main__":
    print(config)
    chkpt_dir = os.path.join(config["chkpt_dir"], config["algorithm"] ,config["experiment"])
    alg_dir = os.path.join(config["chkpt_dir"], config["algorithm"])
    
    if not os.path.isdir(alg_dir):
        os.mkdir(alg_dir)

    if not os.path.isdir(chkpt_dir):
        os.mkdir(chkpt_dir)

    env = gym.make(config["env"])
    

    if not config["debug"]: logger = wandb.init(project="Cart-Pole", config=config, name=f"{config['algorithm']}_{config['experiment']}")
    
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, n_epochs= config["n_epochs"],
                  alpha=config["alpha"], batch_size=config["batch_size"], chkpt_dir=chkpt_dir)

    if not config["debug"]: watch(logger, agent)

    st = time.time()
    cnt = 0
    score_history = []
    best_score = env.reward_range[0]
    n_steps=0
    learn_iters = 0

    for i in range(config["n_games"]):
        observation, _ = env.reset()
        done = False
        score = 0.0

        while not done:
            cnt += 1

            action, prob, val = agent.choose_action(observation)
            
            observation_, reward, done, unkwon, info = env.step(action)

            if done or unkwon: 
                done = True
                # if score <= 450: reward = -20

            n_steps += 1
            score += reward
            
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % config["N"] == 0:
                actor_loss, critic_loss = agent.learn()
                learn_iters += 1

                if not config["debug"]: 
                    logger.log({"iter": learn_iters, 
                                "actor_loss": actor_loss,
                                "critic_loss": critic_loss}, step=learn_iters)

            observation = observation_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-25:])

        if not config["debug"]: 
            logger.log({"episode": i, 
                        "time": time.time() - st, 
                        "score": score}, step=learn_iters)

        if avg_score > best_score:
            best_score = avg_score
            if not config["debug"]: agent.save_models()

        print(f'episode {i}, score {score}, avg score {avg_score}, learning_steps {learn_iters}')


