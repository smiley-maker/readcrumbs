from fastapi import FastAPI, HTTPException, status
import boto3
import os
import datetime
import random
import joblib
import json
import numpy as np
from typing import List
from pydantic import BaseModel
import io

class MyFavorites(BaseModel):
    items: List[str] # List of favorite book titles
    userid: str # Unique user identifier
    
class PredictionResponse(BaseModel):
    recs: List[str] # titles of the recommended books

## Helper Functions
def load_supporting_tables_from_s3(table_name: str, client = None) -> dict:
    """Download and load supporting tables from S3 into memory without persisting it to disk.

    Args:
        table_name (str): Name of the table in S3.

    Returns:
        dict: Dictionary representing the table.
    """
    if client is None:
        client = boto3.client("s3")
    
    s3_bucket = 'readcrumbs'
    
    # Download the object as bytes into memory
    response = client.get_object(Bucket=s3_bucket, Key=table_name)
    table_bytes = response['Body'].read().decode('utf-8')
    table = json.loads(table_bytes)
    
    return table

def load_model_from_s3(model_name: str, client = None):
    """
    Download and load an ML model file from S3 into memory without persisting it to disk.
    
    Uses AWS credentials from environment variables if available:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_SESSION_TOKEN (optional, for temporary credentials)
    
    Falls back to IAM roles (if running on EC2, Lambda, ECS, etc.) or ~/.aws/credentials
    
    Required environment variables:
    - S3_MODEL_BUCKET: S3 bucket name
    - AWS_REGION: AWS region (optional, defaults to us-east-1)

    Args:
        model_name (str): The key/path of the model file in the S3 bucket.

    Returns:
        The loaded model object.
    """
    s3_bucket = 'readcrumbs'
    
    # Check if client is none, then create one
    if client is None:
        client = boto3.client('s3')
    
    # Get the model object from S3
    response = client.get_object(Bucket=s3_bucket, Key=model_name)
    
    # Load the model from the bytes in memory
    buffer = io.BytesIO(response['Body'].read())
    model = joblib.load(buffer)

    return model

def predict_using_model(data: MyFavorites, n_recs: int = 10) -> list[str]:
    """Gets n_recs recommendations based on the users favorite books.

    Args:
        data (MyFavorites): The users favorite book titles.
        n_recs (int, optional): Number of recommendations to return. Defaults to 10.

    Returns:
        list[str]: List of recommended book titles.
    """

    # Get the book ids for the user's favorite books
    my_favs_ids = [title_to_index[f] for f in data]
    # Get the item vectors for the user's favorite books using the model
    fav_vectors = [model.item_factors[int(i)] for i in my_favs_ids]
    #Average the vectors together to get a single user vector
    avg_vec = np.average(np.stack(fav_vectors), axis=0)
    # Get the top n_recs recommendations by finding the closest item vectors to the user vector
    recommendations = np.argsort(np.dot(avg_vec, model.item_factors.T))[:n_recs]
    # Convert the item ids back to titles
    return [index_to_title[str(i)] for i in recommendations]

def serialize_for_dynamodb(data):
    """
    Recursively serialize data for DynamoDB.
    Converts datetime objects to ISO format strings.
    """
    if isinstance(data, datetime.datetime):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: serialize_for_dynamodb(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_for_dynamodb(item) for item in data]
    else:
        return data



def get_dynamodb_table(table_name: str = None):
    """
    Get a DynamoDB table resource with proper credentials.
    
    Returns:
        boto3 DynamoDB Table resource
    """
#    table_name = os.environ.get("DDB_TABLE")
    if not table_name:
#        raise ValueError("DDB_TABLE environment variable not set.")
        table_name = "readcrumbs-logs"
    
    region = os.environ.get("AWS_REGION", "us-east-1")
    
    # Get AWS credentials from environment variables
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    
    # Create boto3 session with explicit credentials if available
    if aws_access_key_id and aws_secret_access_key:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region
        )
        dynamodb = session.resource("dynamodb")
    else:
        # Fall back to default credential chain (IAM roles, ~/.aws/credentials, etc.)
        dynamodb = boto3.resource("dynamodb", region_name=region)
    
    return dynamodb.Table(table_name)

def get_random_item_from_ddb():
    """
    Retrieve a random item from DynamoDB table.
    
    Returns:
        dict: A random item from the table, or None if table is empty
    """
    table = get_dynamodb_table("readcrumbs-logs")
    
    # Scan the table to get all items
    response = table.scan()
    items = response.get('Items', [])
    
    # Handle pagination if there are more items
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response.get('Items', []))
    
    if not items:
        return None
    
    # Return a random item
    return random.choice(items)


def save_to_ddb(data, table_name: str = None):
    """
    Save or update a dictionary of data to DynamoDB.
    Uses user_id (integer) as the primary key. If user_id already exists, the item will be updated.
    
    Args:
        data: Dictionary containing user_id (int) and other fields. user_id is used as primary key.
        table_name (str, optional): Name of the DynamoDB table. 
            If None, defaults to "readcrumbs-logs".
    
    Returns:
        dict: Response from DynamoDB put_item operation.
    """

    # Get the DynamoDB table with the table name
    table = get_dynamodb_table(table_name)
    
    # Serialize data for DynamoDB (convert datetime objects, etc.)
    serialized_data = serialize_for_dynamodb(data)
    
    # Ensure user_id exists (required as primary key)
    if 'userid' not in serialized_data:
        raise ValueError("userid is required in the request body")
    
    response = table.put_item(Item=serialized_data)
    print(response)
    return response

## API

app = FastAPI()

# Set up a client so we don't have to keep recreating it
s3_client = boto3.client("s3")

model = load_model_from_s3("models/als_model-small-v1.pkl", client=s3_client)
index_to_title = load_supporting_tables_from_s3("data/v1/index_to_title.json", client=s3_client)
title_to_index = load_supporting_tables_from_s3("data/v1/title_to_index.json", client=s3_client)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/random")
def get_random():
    """
    Get a random item from the DynamoDB table.
    
    Returns:
        dict: A random item from the table
    """
    random_item = get_random_item_from_ddb()
    if random_item is None:
        raise HTTPException(status_code=404, detail="No items found in table")
    return random_item


@app.post("/feedback")
def submit_feedback(feedback: dict):
    """Submit feedback to DynamoDB.

    Args:
        feedback (dict): Feedback data to store in DynamoDB.

    Raises: 
        HTTPException500: 500 error for any issues during saving feedback.

    Returns:
        dict: Response from DynamoDB put_item operation.
    """
    try:
        response = save_to_ddb(feedback, table_name="readcrumbs-feedback")
        return response
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error saving feedback: {str(e)}")


@app.post("/predict")
def predict(request: MyFavorites) -> PredictionResponse:
    """Get book recommendations from the users favorite titles. 

    Args:
        request (MyFavorites): Includes items, which is a list of favorite book titles,
                                and userid, a unique user identifier.

    Raises:
        HTTPException400: 400 error if no favorite books are provided.
        HTTPException500: 500 error for any other issues during prediction.

    Returns:
        PredictionResponse: The predicted book recommendations.
    """
    if len(request.items) < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Must enter at least one favorite book. You gave: {request.items}")

    try:
        # Get recommendations using the model
        recs = predict_using_model(request.items)

        # Log the request and prediction to DynamoDB
        logs = {
            "items": request.items,
            "userid": request.userid,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "prediction": recs
        }

        # Save the logs to DynamoDB
        save_to_ddb(logs, table_name="readcrumbs-logs")

        # Return the recommendations as a PredictionResponse
        return {
            "recs": recs
        }
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{str(e)}\Request: {request}")