import torch
import wandb

class HardStopException(Exception):
    """Custom exception to halt training on critical failures."""
    pass

class FallbackController:
    def __init__(self, config):
        self.config = config

    def hard_stop(self, reason: str, step: int, optimizer, model, config):
        """
        Serialize full training state before abort.
        This preserves the ability to diagnose and resume.
        """
        print(f"[HARD STOP] Step {step}: {reason}")
        
        # Serialize state
        checkpoint = {
            "step":           step,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "reason":         reason,
            "config":         config,
        }
        # Use a safe default for checkpoint_dir if not in config
        checkpoint_dir = getattr(config, "checkpoint_dir", "./checkpoints")
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        abort_path = f"{checkpoint_dir}/ABORT_step{step}.pt"
        torch.save(checkpoint, abort_path)
        
        if wandb.run is not None:
            wandb.alert(
                title=f"HARD STOP at step {step}",
                text=reason,
                level=wandb.AlertLevel.ERROR,
            )
        
        raise HardStopException(reason)
