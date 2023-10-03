import wandb

class WandbLogger():
    def __init__(self, project, config) -> None:
        wandb.login()

        wandb.init(
            project=project,
            config = config
        )

    def get_config(self):
        return wandb.config

    def watch(self, model, criterion, log_type, log_freq):
        wandb.watch(models=model, criterion=criterion, log=log_type, log_freq=log_freq)

    def unwatch(self):
        wandb.unwatch()
    
    def log(self, step, **logs):
        wandb.log(logs, step=step)

    def save(self, file):
        wandb.save(file)