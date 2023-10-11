import gym
from stable_baselines3 import PPO 
import os

config = {"chkpt_dir":  "/home/joel/PhD/results/rl/models",
          "n_games": 2000,
          "N": 500,
          "env":'BipedalWalker-v3',
          "experiment": "baseline1",
          "batch_size": 100,
          "fc1_dims": 1024,
          "fc2_dims": 512,
          "n_epochs": 2,
          "policy_clip": 0.1,
          "alpha": 0.0003,
          "beta": 0.001,
          "action_std_init": 0.6,
          "action_std_decay_rate": 0.05,
          "min_action_std": 0.05,
          "gae_lambda": 0.9,
          "algorithm": "ppo",
          "debug": True}



if __name__== "__main__":
    print(config)
    chkpt_dir = os.path.join(config["chkpt_dir"], config["algorithm"] ,config["experiment"])
    alg_dir = os.path.join(config["chkpt_dir"], config["algorithm"])
    
    if not os.path.isdir(alg_dir):
        os.mkdir(alg_dir)

    if not os.path.isdir(chkpt_dir):
        os.mkdir(chkpt_dir)

    env = gym.make(config["env"], render_mode ="human")
    env.reset()

    model = PPO.load(f"{chkpt_dir}/290000.zip", env=env)

    episodes = 10

    for ep in range(episodes):
        obs, _ =  env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, unkown, info = env.step(action)

    env.close()