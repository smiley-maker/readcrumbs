import pandas as pd
import json
import boto3
from typing import Union
from io import BytesIO


def load_csv_data(bucket : str, objectkey: str, client) -> pd.DataFrame:
    """Loads CSV dataset from S3

    Args:
        bucket (str): The bucket name where the data is stored. 
        objectkey (str): The path to the file in the bucket. 
        client: boto3 client to handle data reading. 

    Returns:
        pd.DataFrame: The dataset read in from S3.
    """

    try: 
        # Get the object from S3
        print("Getting data from S3")
        response = client.get_object(Bucket=bucket, Key=objectkey)

        # The 'Body' is a streamable object (a botocore.response.StreamingBody).
        # pandas.read_csv can handle this stream directly, which avoids loading
        # the entire file into a Python string variable first.
        print("Reading CSV directly from S3 stream...")
        df = pd.read_csv(response['Body'])

        print(f"Dataframe with {len(df)} rows created.")
        print(df.head())

        return df
    except Exception as e:
        print(f"Error reading CSV from S3 storage: {e}")



def save_data(data : Union[pd.DataFrame, dict], bucket : str, objectkey : str, client) -> bool:
    """Saves either dataframe or dictionary data to S3.

    Args:
        data (DataFrame or dict): The data to save.
        bucket (str): The bucket name where the data is stored. 
        objectkey (str): The path to the file in the bucket. 
        client: boto3 client to handle data reading. 
    
    Returns:
        bool: True if successful upload, false otherwise. 
    """

    try:
        # Check if data is a DataFrame or dictionary
        if type(data) == pd.DataFrame:
            # We want to save as a parquet file. 
            # Need to create a buffer for pandas conversion process
            buffer = BytesIO()

            # Write the parquet file to the buffer
            data.to_parquet(buffer)

            # Set the dataset equal to the value of the buffer. 
            dataset = buffer.getvalue()
        
        elif type(data) == dict:
            # We want to save the data as JSON
            dataset = json.dumps(data)
        
        else:
            # Unsupported type
            print(f"Unsupported type: {type(data)}")
            return False

        client.put_object(
            Bucket=bucket,
            Key=objectkey,
            Body=dataset
        )
        
        print(f"Successfully saved data to s3://{bucket}/{objectkey}")
        return True
    except Exception as e:
        print(f"Error saving {type(data)} data to S3 url: s3://{bucket}/{objectkey}.")
        print(e)
        return False


def preprocess_data(df : pd.DataFrame, feats : list[str] = None) -> pd.DataFrame:
    """Preprocesses the reviews dataset by selecting relevant 
       columns and dropping null values. 

    Args:
        df (pd.DataFrame): original dataframe

    Returns:
        pd.DataFrame: processed dataset.
    """

    # Select features relevant for collaborative filtering, or
    # use provided ones.
    if not feats:
        feats = ["User_id", "Title", "review/score"]
    df = df[feats]

    # Drop any rows containing null values because all the features are necessary
    df = df.dropna(axis=0)

    # We only want to keep 4 and 5 star reviews because the implicit library works on 
    # the basis of interactions, and while the ratings will sway the model more towards
    # higher numbers, lower values aren't considered 'negative' and the model could
    # still recommend those books. 
    df = df[df["review/score"] >= 4]

    # Change the user id and title to categorical columns
    df["User_id"] = df["User_id"].astype("category")
    df["Title"] = df["Title"].astype("category")

    return df


def create_mappings(df : pd.DataFrame, title_col : str = "title_book") -> tuple[dict, dict]:
    """Creates mapping dictionaries for book titles to indices and vice versa.

    Args:
        df (pd.DataFrame): Dataframe containing book titles.
        title_col (str, optional): The column name containing book titles. Defaults to "title_book".
    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries:
            - title_to_index: Maps book titles to their corresponding indices.
            - index_to_title: Maps indices back to their corresponding book titles.
    """

    # Convert the book title into a categorical column (each name = new book)
    titles = df[title_col].astype("category")

    # Create a title to code mapping (for looking up factors)
    title_to_index = dict(zip(titles.cat.categories, titles.cat.codes))

    # Create the reverse mapping for code to title, used in displaying results
    index_to_title = dict(enumerate(titles.cat.categories))

    # Return dictionaries
    return title_to_index, index_to_title


if __name__=="__main__":
    print("Starting dataset processing...")
    # Create an S3 client
    print("Creating client")
    client = boto3.client("s3")

    # Load dataset from S3
    print("Loading dataset")
    df = load_csv_data(
        bucket="readcrumbs",
        objectkey="dataset/raw/book_ratings.csv",
        client=client
    )

    # Preprocess dataset
    print("Preprocessing data")
    df = preprocess_data(
        df=df
    )

    # Save processed dataset back to S3
    print("Saving Dataset")
    res = save_data(
        data=df,
        bucket="readcrumbs",
        objectkey="data/processed/ratings-small-v1.parquet",
        client=client
    )

    assert res

    if not res:
        print(f"Error saving processed dataset.")

    # Create and save mapping tables
    print("Creating Mapping Tables")
    tindex, indext = create_mappings(df, title_col="Title")
    print("Saving mapping tables")
    res = save_data(
        data=tindex,
        bucket="readcrumbs",
        objectkey="data/v1/title_to_index.json",
        client=client
    )

    assert res

    if not res:
        print(f"Error saving the title to index dictionary.")

    res = save_data(
        data=indext,
        bucket="readcrumbs",
        objectkey="data/v1/index_to_title.json",
        client=client
    )

    assert res

    if not res:
        print(f"Error saving the index to title dictionary.")