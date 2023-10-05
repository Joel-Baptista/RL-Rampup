import os
import gym
from agents.ppo import Agent
import wandb
import time
import numpy as np


def watch(logger: wandb, agent: Agent):
    logger.watch(agent.actor )
    logger.watch(agent.critic)

def unwatch(logger: wandb, agent: Agent):
    logger.unwatch(agent.actor )
    logger.unwatch(agent.critic)


config = {"n_games": 2000,
          "N": 50,
          "env":'BipedalWalker-v3',
          "chkpt_dir":  "/root/models",
          "experiment": "baseline2",
          "batch_size": 10,
          "n_epochs": 5,
          "alpha": 0.0003,
          "action_std_init": 0.6,
          "action_std_decay_rate": 0.005,
          "min_action_std": 0.1,
          "gae_lambda": 0.9,
          "algorithm": "ppo",
          "debug": False}


# config = {"n_games": 2000,
#           "N": 200,
#           "env":'BipedalWalker-v3',
#           "chkpt_dir":  "/home/joel/PhD/results/rl/models",
#           "experiment": "baseline1",
#           "batch_size": 20,
#           "n_epochs": 10,
#           "alpha": 0.0003,
#           "action_std_init": 0.6,
#           "action_std_decay_rate": 0.005,
#           "min_action_std": 0.1,
#           "gae_lambda": 0.9,
#           "algorithm": "ppo",
#           "debug": False}


if __name__== "__main__":
    print(config)
    chkpt_dir = os.path.join(config["chkpt_dir"], config["algorithm"] ,config["experiment"])
    alg_dir = os.path.join(config["chkpt_dir"], config["algorithm"])
    
    if not os.path.isdir(alg_dir):
        os.mkdir(alg_dir)

    if not os.path.isdir(chkpt_dir):
        os.mkdir(chkpt_dir)

    env = gym.make(config["env"])
    

    if not config["debug"]: logger = wandb.init(project="Bipedal", config=config, name=f"{config['algorithm']}_{config['experiment']}")
    
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.shape, env=env,n_epochs= config["n_epochs"],
                  alpha=config["alpha"], batch_size=config["batch_size"], action_std_decay=config["action_std_decay_rate"], 
                  action_std_min=config["min_action_std"], chkpt_dir=chkpt_dir, action_std_init=config["action_std_init"], 
                  gae_lambda=config["gae_lambda"])

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
                if score <= 450: reward = -20

            n_steps += 1
            score += reward
            
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % config["N"] == 0:
                actor_loss, critic_loss = agent.learn()
                learn_iters += 1
                agent.actor.set_action_std()
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

        print(f'episode {i}, score {score}, avg score {avg_score}, learning_steps {learn_iters}, agent_std {agent.actor.action_std}')


