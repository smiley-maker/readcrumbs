import fastapi
import boto3
import os
import datetime
import time
import random
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

'''
To-Do:
- [ ] Connect to S3 w/ model
- [ ] Create a function to load the model from S3
- [ ] Create a function to predict using the model
'''

## Helper Functions
def load_model_from_s3(model_name: str):
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
    s3_bucket = os.environ.get("S3_MODEL_BUCKET")
    if not s3_bucket:
        raise ValueError("S3_MODEL_BUCKET environment variable not set.")
    
    region = os.environ.get("AWS_REGION", "us-east-1")
    
    # Get AWS credentials from environment variables
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    
    # Create boto3 client with explicit credentials if available
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region
        )
    else:
        # Fall back to default credential chain (IAM roles, ~/.aws/credentials, etc.)
        s3 = boto3.client("s3", region_name=region)

    # Download model object as bytes into memory
    response = s3.get_object(Bucket=s3_bucket, Key=model_name)
    model_bytes = response['Body'].read()
    import pickle
    model = pickle.loads(model_bytes)
    return model

def predict_using_model(model, data):
    pass

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

def get_dynamodb_table():
    """
    Get a DynamoDB table resource with proper credentials.
    
    Returns:
        boto3 DynamoDB Table resource
    """
    table_name = os.environ.get("DDB_TABLE")
    if not table_name:
        raise ValueError("DDB_TABLE environment variable not set.")
    
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
    table = get_dynamodb_table()
    
    # Scan the table to get all items
    # Note: For very large tables, this could be expensive.
    # Consider optimizing with pagination or sampling if needed.
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

def save_to_ddb(data):
    """
    Save or update a dictionary of data to DynamoDB.
    Uses user_id (integer) as the primary key. If user_id already exists, the item will be updated.
    
    Uses AWS credentials from environment variables if available:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_SESSION_TOKEN (optional, for temporary credentials)
    
    Falls back to IAM roles (if running on EC2, Lambda, ECS, etc.) or ~/.aws/credentials
    
    Required environment variables:
    - DDB_TABLE: DynamoDB table name
    - AWS_REGION: AWS region (optional, defaults to us-east-1)
    
    Args:
        data: Dictionary containing user_id (int) and other fields. user_id is used as primary key.
    """
    table = get_dynamodb_table()
    
    # Serialize data for DynamoDB (convert datetime objects, etc.)
    serialized_data = serialize_for_dynamodb(data)
    
    # Ensure user_id exists (required as primary key)
    if 'user_id' not in serialized_data:
        raise ValueError("user_id is required in the request body")
    
    # Map user_id to pred-id (the table's primary key field name)
    # Keep user_id in the data as well for reference
    serialized_data['pred-id'] = serialized_data['user_id']

    response = table.put_item(Item=serialized_data)
    return response

## API
class MyFavorites(BaseModel):
    items: List[str]
    
class PredictionResponse(BaseModel):
    user_id: int
    req: MyFavorites

app = fastapi.FastAPI()

model = load_model_from_s3("<model_name>")

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
        raise fastapi.HTTPException(status_code=404, detail="No items found in table")
    return random_item

@app.post("/predict")
def predict(request: PredictionResponse):
    data = request.model_dump()
    data['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
    data['prediction'] = predict_using_model(model, data)
    save_to_ddb(data)
    return {"status": "ok"}