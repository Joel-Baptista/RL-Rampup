from agents.a3c import Agent, ActorCritic, SharedAdam
import torch.multiprocessing as mp

config = {"chkpt_dir":  "/home/joel/PhD/results/rl/models",
          "env":'CartPole-v1',
          "experiment": "baseline1",
          "algorithm": "a3c",
          "n_games": 3000,
          "n_actions": 2,
          "input_dims": [4],
          "lr": 1e-4,
          "t_max": 5,
          "fc1": 256,
          "fc2": 256,
          "debug": True}

if __name__ == "__main__":
    print(config)

    global_actor_critic = ActorCritic(config["input_dims"], config["n_actions"], fc1=config["fc1"], fc2=config["fc2"])
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=config["lr"], betas=(0.92, 0.999))

    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                    optim,
                    config["input_dims"],
                    config["n_actions"],
                    gamma=0.99,
                    lr=config["lr"],
                    name=i,
                    global_ep_idx =global_ep,
                    env_id=config["env"],
                    t_max=config["t_max"],
                    n_games=config["n_games"],
                    fc1=config["fc1"],
                    fc2=config["fc2"])
                    for i in range(mp.cpu_count())]
                    # for i in range(1)]

    [w.start() for w in workers]
    [w.join() for w in workers]
