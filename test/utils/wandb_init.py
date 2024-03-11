import wandb
import random

def init_wandb(**kwargs):
    run = wandb.init(
        project = kwargs.get("proj_name", "diffusion_research"),
        config = kwargs["config"],
        name = kwargs.get("name", None)
    )
    return run


if __name__ == "__main__":
    import random
    run = init_wandb(proj_name = "diffusion_research")
    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        
        # log metrics to wandb
        run.log({"acc": acc, "loss": loss, "hi": loss + 0.5})
        
    # [optional] finish the wandb run, necessary in notebooks
    run.finish()
