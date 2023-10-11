import gym
from stable_baselines3 import PPO, SAC
import os
import torch as T

# config = {"chkpt_dir":  "/home/joel/PhD/results/rl/models",
#           "n_games": 2000,
#           "N": 500,
#           "env":'BipedalWalker-v3',
#           "experiment": "baseline1",
#           "batch_size": 100,
#           "fc1_dims": 1024,
#           "fc2_dims": 512,
#           "n_epochs": 2,
#           "policy_clip": 0.1,
#           "alpha": 0.0003,
#           "beta": 0.001,
#           "action_std_init": 0.6,
#           "action_std_decay_rate": 0.05,
#           "min_action_std": 0.05,
#           "gae_lambda": 0.9,
#           "algorithm": "ppo",
#           "debug": True}

config = {"n_games": 200000,
          "N": 500,
          "env":'BipedalWalker-v3',
          "chkpt_dir":  "/root/models",
          "experiment": "baseline1",
          "batch_size": 50,
          "fc1_dims": 1024,
          "fc2_dims": 512,
          "n_epochs": 2,
          "alpha": 0.0003,
          "beta": 0.001,
          "policy_clip": 0.1,
          "action_std_init": 0.6,
          "action_std_decay_rate": 0.005,
          "min_action_std": 0.1,
          "gae_lambda": 0.5,
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
    env.reset()
    device = T.device("cuda:3" if T.cuda.is_available() else 'cpu')

    model = PPO("MlpPolicy", env, verbose=1, device=device)
    
    TIMESTEPS = 100000
    for i in range(1, 30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=config["algorithm"])


    model.save(f"{chkpt_dir}/{TIMESTEPS*i}")



    '''episodes = 10

    for ep in range(episodes):
        obs =  env.reset()
        done = False
        while not done:
            env.render()
            obs, reward, done, info = env.step(env.action_space.sample())'''


    env.close()