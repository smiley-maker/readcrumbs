import wandb
import subprocess
import hashlib
from pathlib import Path


def get_git_commit_hash():
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_data_version(data_path=None):
    """
    Get data version by computing hash of data files.
    If data_path is provided, hash those files. Otherwise, return a placeholder.
    """
    if data_path is None:
        # Return a placeholder - user should specify their data path
        return "data-v1.0"
    
    data_path = Path(data_path)
    if not data_path.exists():
        return "unknown"
    
    # Compute hash of data files
    hasher = hashlib.sha256()
    if data_path.is_file():
        with open(data_path, "rb") as f:
            hasher.update(f.read())
    elif data_path.is_dir():
        for file_path in sorted(data_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
    
    return f"data-{hasher.hexdigest()[:8]}"


# Hyperparameters
hyperparameters = {
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    # Add more hyperparameters as needed
}

# Initialize wandb with config
wandb.init(
    project="readcrumbs",
    name="experiment-1",
    config={
        **hyperparameters,
        "code_version": get_git_commit_hash(),
        "data_version": get_data_version(),  # Update with your actual data path
    }
)

# Example training loop
for epoch in range(hyperparameters["epochs"]):
    # Simulate training metrics
    # Replace these with your actual training code
    
    # Log metrics for each epoch
    metrics = {
        "epoch": epoch + 1,
        "loss": 0.1 * (0.9 ** epoch),  # Example: decreasing loss
        "accuracy": 0.5 + 0.4 * (1 - 0.9 ** epoch),  # Example: increasing accuracy
        "f1_score": 0.5 + 0.4 * (1 - 0.9 ** epoch),  # Example: increasing f1
    }
    
    wandb.log(metrics)

# Log final metrics
wandb.log({
    "final_accuracy": metrics["accuracy"],
    "final_f1_score": metrics["f1_score"],
})

wandb.finish()
