import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
import boto3
import io
import joblib

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
    client.put_object(Bucket=bucket_name, Key=file_key, Body=buffer)

def train_als_model(
        data : pd.DataFrame, factors : int=50, regularization : float=0.01, iterations : int=15
    ):
    """Trains an ALS model using provided data and hyperparameters. 

    Args:
        data (pandas DataFrame): The dataset to train the model with
        factors (int, optional): The number of latent factors to compute. Defaults to 50.
        regularization (float, optional): The regularization factor to use. Defaults to 0.01.
        iterations (int, optional): The number of training iterations. Defaults to 15.

    Returns:
        tuple[implicit model, sparse csr matrix]: Returns the trained model and the sparse user-book matrix.
    """

    # Create a user-item interaction matrix
    user_item_matrix = sparse.csr_matrix((data['review/score'], (data['User_id'].cat.codes, data['Title'].cat.codes)))    

    # Initialize the ALS model
    model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
    
    # Train the model
    model.fit(user_item_matrix)
    
    return model, user_item_matrix

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
    
    # Split data into training and test sets
#    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train the ALS model using default params
    model, train_user_item_csr = train_als_model(data)
    
    # Save the trained model to a pickle file on s3
    save_model_to_s3(
        model=model, 
        bucket=bucket_name,
        objectkey='models/als_model-small-v1.pkl',
        client=client
    )
    
    print("Model trained and saved successfully.")