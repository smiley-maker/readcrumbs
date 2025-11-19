import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, train_test_split
import scipy.sparse as sparse
import boto3
import io
import joblib
import wandb
# When running this in EC2, wandb_tracking.py will be in the same folder...
from wandb_tracking import *


def load_data_from_s3(bucket : str, objectkey : str, client) -> pd.DataFrame:
    """Loads a parquet dataset from S3

    Args:
        bucket (str): The bucket name where the data is stored. 
        objectkey (str): The path to the file in the bucket. 
        client: boto3 client to handle data reading. 

    Returns:
        _type_: Dataframe loaded from parquet data in S3.
    """

    # Gets the object using the client
    obj = client.get_object(Bucket=bucket, Key=objectkey)
    
    # Creates a buffer so that pandas can read in the data 
    # (since it's not saved anywhere)
    buffer = io.BytesIO(obj['Body'].read())

    # Returns a pandas dataframe based on the parquet file read
    # from the buffer. 
    return pd.read_parquet(buffer)


def save_model_to_s3(model, bucket : str, objectkey : str, client):
    """Saves a trained model to S3 as a pickle file.

    Args:
        model (implicit model): Trained model
        bucket (str): The bucket name where the model should be stored.
        objectkey (str): The path where the model should be stored.
    """

    # Creates a buffer that the model can be placed in
    buffer = io.BytesIO()
    
    # Uses joblib to dump the model into the buffer as a pickle file.
    joblib.dump(model, buffer)

    # Moves the pointer in the buffer back to the beginning (where the model is)
    buffer.seek(0)

    # Puts the model file into S3 using the client.
    client.put_object(Bucket=bucket, Key=objectkey, Body=buffer)

def train_als_model(
        user_item_matrix, factors : int=50, regularization : float=0.01, iterations : int=15
    ):
    """Trains an ALS model using provided data and hyperparameters. 

    Args:
        user_item_matrix (sparse csr user-book matrix): User-item CSR matrix.
        factors (int, optional): The number of latent factors to compute. Defaults to 50.
        regularization (float, optional): The regularization factor to use. Defaults to 0.01.
        iterations (int, optional): The number of training iterations. Defaults to 15.

    Returns:
        implicit model: Returns the trained model
    """    

    # Initialize the ALS model
    model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
    
    # Train the model
    model.fit(user_item_matrix)
    
    return model

if __name__ == "__main__":
    # Create a client using boto3
    client = boto3.client("s3")

    # Load data
    bucket_name = 'readcrumbs'
    file_key = 'data/processed/ratings-small-v1.parquet'
    data = load_data_from_s3(
        bucket=bucket_name,
        objectkey=file_key,
        client=client
    )
    print(data.head())
    print(data.dtypes)
    

    # Create a user-item interaction matrix
    user_item_matrix = sparse.csr_matrix((data['review/score'], (data['User_id'].cat.codes, data['Title'].cat.codes)))

    # Split into training and testing CSR's
    train, test = train_test_split(user_item_matrix, train_percentage=0.9, random_state=42)
    
    # Train the ALS model using default params and training subset
    factors = 50
    regularization = 0.01
    iterations = 15
    model = train_als_model(
        user_item_matrix=train,
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )

    # Evaluate the model
    p_at_k = precision_at_k(model, train, test, K=10)
    print(f"Factors: {factors}, Reg: {regularization} -> Precision@10: {p_at_k}")
    
    # Save the trained model to a pickle file on s3
    model_file = "models/als_model-small-v1.pkl"

    save_model_to_s3(
        model=model, 
        bucket=bucket_name,
        objectkey=model_file,
        client=client
    )
    
    print("Model trained and saved successfully.")

    model_metadata = {
        "precision_at_k": precision_at_k,
        "iterations": iterations,
        "factors": factors,
        "regularization": regularization,
        "data_version": "data/processed/ratings-small-v1.parquet",
        "code_version": get_git_commit_hash(),
        "model_file": model_file,
        "promotion_metric": "precision_at_k"
    }

    print("Saving model to registry.")
    
    # Initialize wandb
    wandb.init(
        project="readcrumbs",
        name="small-v1",
        config={
            "precision_at_k": p_at_k,
            "regularization": regularization,
            "factors": factors,
            "iterations": iterations,
            "model_file": model_file,
            "code_version": get_git_commit_hash(),
            "data_version": "data/processed/ratings-small-v1.parquet",  # Update with your actual data path
        }
    )

    # Register model

    artifact = wandb.Artifact(
        name="readcrumbs-model-small-v1", 
        type="model"
    )

    # Save and register model with automatic promotion to staging if it's the best
    promotion_metric = "precision_at_k"
    artifact, promoted = save_and_register_model(
        model=model,
        model_name="readcrumbs-model-small-v1",
        model_type="pickle",
        registered_model_name="readcrumbs-model-small-v1",  # Name in Model Registry
        metadata=model_metadata,
        auto_promote=True,  # Automatically promote to staging if best model
        promotion_stage="staging",  # or "production"
        promotion_metric=promotion_metric  # Metric to use for comparison
    )
    
    if promoted:
        print(f"Model automatically promoted to staging based on {promotion_metric}")


    wandb.finish()