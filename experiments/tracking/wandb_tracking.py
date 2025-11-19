import wandb
import subprocess
import hashlib
import tempfile
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


def save_model_artifact(model, model_name="model", model_type="pytorch", metadata=None, 
                        registered_model_name=None):
    """
    Save a trained model as a wandb artifact and optionally link it to a registered model.
    
    Args:
        model: The trained model object (PyTorch, TensorFlow, sklearn, etc.)
        model_name: Name for the model artifact
        model_type: Type of model ('pytorch', 'tensorflow', 'sklearn', 'pickle', etc.)
        metadata: Optional dictionary of additional metadata to attach to the artifact
        registered_model_name: Optional name of the registered model in Model Registry
    
    Returns:
        The wandb artifact object
    """
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=f"Trained {model_type} model",
        metadata=metadata or {}
    )
    
    # Create a temporary directory to save the model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / f"{model_name}.{_get_model_extension(model_type)}"
        
        # Save model based on type
        if model_type == "pytorch":
            import torch
            torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, model_path)
        elif model_type == "tensorflow":
            model.save(str(model_path))
        elif model_type == "sklearn":
            import joblib
            joblib.dump(model, model_path)
        elif model_type == "pickle":
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            # Default: try to save as pickle
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Add the model file to the artifact
        artifact.add_file(str(model_path))
        
        # Log the artifact
        wandb.log_artifact(artifact)
        
        # Link to registered model if specified
        if registered_model_name:
            artifact.wait()  # Wait for artifact to be logged
            run = wandb.run
            if run:
                # Link artifact to registered model
                run.link_artifact(
                    artifact,
                    f"{registered_model_name}:latest",
                    aliases=["latest"]
                )
    
    return artifact


def promote_model_to_stage(registered_model_name, alias="staging", metric_name="f1_score", 
                           metric_value=None, comparison="max", project_name=None):
    """
    Promote a model version to a specific stage (staging/production) in the Model Registry.
    
    Args:
        registered_model_name: Name of the registered model in Model Registry
        alias: Stage alias to assign ('staging' or 'production')
        metric_name: Name of the metric to use for comparison (e.g., 'f1_score', 'accuracy')
        metric_value: Optional metric value. If None, uses the latest model version
        comparison: How to compare models ('max' for higher is better, 'min' for lower is better)
        project_name: Optional project name. If None, uses current wandb project
    
    Returns:
        True if promotion was successful, False otherwise
    """
    try:
        api = wandb.Api()
        
        # Get project name from current run if not provided
        if project_name is None:
            project_name = wandb.run.project if wandb.run else "readcrumbs"
        
        # Access registered model
        registered_model_path = f"{project_name}/{registered_model_name}"
        registered_model = api.registered_model(registered_model_path)
        
        if metric_value is not None:
            # Find the best model based on metric
            best_version = None
            best_metric = float('-inf') if comparison == "max" else float('inf')
            
            for version in registered_model.versions:
                # Get metadata from the artifact
                try:
                    artifact = version.artifact
                    version_metadata = artifact.metadata or {}
                    version_metric = version_metadata.get(metric_name)
                    
                    if version_metric is not None:
                        if comparison == "max" and version_metric > best_metric:
                            best_metric = version_metric
                            best_version = version
                        elif comparison == "min" and version_metric < best_metric:
                            best_metric = version_metric
                            best_version = version
                except Exception:
                    continue
            
            if best_version:
                # Update aliases
                current_aliases = list(best_version.aliases) if best_version.aliases else []
                if alias not in current_aliases:
                    current_aliases.append(alias)
                    best_version.aliases = current_aliases
                    best_version.update()
                    print(f"Promoted model version {best_version.version} to '{alias}' stage "
                          f"(metric: {metric_name}={best_metric})")
                    return True
        else:
            # Promote the latest version
            if registered_model.versions:
                latest_version = registered_model.versions[0]
                current_aliases = list(latest_version.aliases) if latest_version.aliases else []
                if alias not in current_aliases:
                    current_aliases.append(alias)
                    latest_version.aliases = current_aliases
                    latest_version.update()
                    print(f"Promoted latest model version {latest_version.version} to '{alias}' stage")
                    return True
        
        return False
    except Exception as e:
        print(f"Error promoting model: {e}")
        print(f"Note: Make sure the registered model '{registered_model_name}' exists in the Model Registry.")
        return False


def save_and_register_model(model, model_name="model", model_type="pytorch", 
                            registered_model_name="readcrumbs-model", metadata=None,
                            auto_promote=False, promotion_stage="staging",
                            promotion_metric="f1_score", project_name=None):
    """
    Save a model as an artifact, register it, and optionally promote it based on performance.
    
    Args:
        model: The trained model object
        model_name: Name for the model artifact
        model_type: Type of model ('pytorch', 'tensorflow', 'sklearn', 'pickle', etc.)
        registered_model_name: Name of the registered model in Model Registry
        metadata: Optional dictionary of additional metadata
        auto_promote: If True, automatically promote to staging if it's the best model
        promotion_stage: Stage to promote to ('staging' or 'production')
        promotion_metric: Metric name to use for promotion comparison
        project_name: Optional project name. If None, uses current wandb project
    
    Returns:
        Tuple of (artifact, promoted) where promoted is True if model was promoted
    """
    # Save model artifact and link to registered model
    artifact = save_model_artifact(
        model=model,
        model_name=model_name,
        model_type=model_type,
        metadata=metadata,
        registered_model_name=registered_model_name
    )
    
    promoted = False
    if auto_promote and metadata and promotion_metric in metadata:
        # Wait a bit for artifact to be fully processed
        import time
        time.sleep(2)
        
        # Promote based on metric value
        promoted = promote_model_to_stage(
            registered_model_name=registered_model_name,
            alias=promotion_stage,
            metric_name=promotion_metric,
            metric_value=metadata[promotion_metric],
            comparison="max",  # Assuming higher is better for most metrics
            project_name=project_name
        )
    
    return artifact, promoted


def _get_model_extension(model_type):
    """Get the file extension for a given model type."""
    extensions = {
        "pytorch": "pth",
        "tensorflow": "h5",
        "sklearn": "joblib",
        "pickle": "pkl"
    }
    return extensions.get(model_type.lower(), "pkl")



if __name__ == "__main__":
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
    final_metrics = {
        "final_accuracy": metrics["accuracy"],
        "final_f1_score": metrics["f1_score"],
    }
    wandb.log(final_metrics)

    # Save the trained model as an artifact and register it in Model Registry
    # Example usage (uncomment and modify based on your model):
    # model = your_trained_model  # Replace with your actual model
    # 
    # # Prepare metadata with performance metrics
    # model_metadata = {
    #     "final_accuracy": final_metrics["final_accuracy"],
    #     "final_f1_score": final_metrics["final_f1_score"],
    #     "epochs": hyperparameters["epochs"],
    #     "learning_rate": hyperparameters["learning_rate"],
    #     "batch_size": hyperparameters["batch_size"],
    #     "code_version": wandb.config.get("code_version", "unknown"),
    #     "data_version": wandb.config.get("data_version", "unknown"),
    # }
    # 
    # # Save and register model with automatic promotion to staging if it's the best
    # artifact, promoted = save_and_register_model(
    #     model=model,
    #     model_name="readcrumbs-model",
    #     model_type="pytorch",  # or "tensorflow", "sklearn", "pickle"
    #     registered_model_name="readcrumbs-model",  # Name in Model Registry
    #     metadata=model_metadata,
    #     auto_promote=True,  # Automatically promote to staging if best model
    #     promotion_stage="staging",  # or "production"
    #     promotion_metric="f1_score"  # Metric to use for comparison
    # )
    # 
    # if promoted:
    #     print(f"Model automatically promoted to staging based on {promotion_metric}")
    # 
    # # Alternatively, manually promote to production after review:
    # # promote_model_to_stage(
    # #     registered_model_name="readcrumbs-model",
    # #     alias="production",
    # #     metric_name="f1_score",
    # #     comparison="max"
    # # )

    wandb.finish()