import pandas as pd
import json
import boto3
from typing import Union
from io import BytesIO, StringIO


def load_data(bucket : str, objectkey : str, client) -> pd.DataFrame:
    """Loads the original dataset from S3 storage.

    Args:
        bucket (str): The bucket name where the data is stored. 
        objectkey (str): The path to the file in the bucket. 
        client: boto3 client to handle data reading. 

    Returns:
        pd.DataFrame: Reads all lines from JSONL file and creates a dataframe. 
    """

    try:
        # Get the object from S3
        response = client.get_object(Bucket=bucket, Key=objectkey)
        
        # Read the content stream
        s3_body = response["Body"]

        # Initialize an empty list to store the parsed JSON objects.
        data = []

        # Iterate over each line in the stream
        # It must be decoded because the stream gives bytes. 
        for line in s3_body.iter_lines():
            data.append(json.loads(line.decode('utf-8')))
        
        # Convert the data list into a pandas dataframe
        df = pd.DataFrame(data)
        print("Dataframe created!")
        print(df.head())

        # Return the dataframe. 
        return df
    
    except Exception as e:
        print(f"Error reading JSONL object from S3: {e}")



def load_csv_data_optimized(bucket: str, objectkey: str, client) -> pd.DataFrame:
    """Loads CSV dataset from S3, streaming the body directly to pandas."""

    try: 
        # Get the object from S3
        print("Getting data from S3")
        response = client.get_object(Bucket=bucket, Key=objectkey)

        # The 'Body' is a streamable object (a botocore.response.StreamingBody).
        # pandas.read_csv can handle this stream directly, which avoids loading
        # the entire file into a Python string variable first.
        print("Reading CSV")
        df = pd.read_csv(response['Body'])

        print(f"Dataframe with {len(df):,} rows created.")
        # print(df.head()) # Keep this for verification if you like

        return df
    except Exception as e:
        print(f"Error reading CSV from S3 storage: {e}")
        return pd.DataFrame() # Return empty DF on failure

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

        # Read the response body and decode bytes to string
#        print("Encoding data to string")
#        csv_body = response['Body'].read().decode('utf-8')

        # Use StringIO to treat the string as a file-like object for pandas
#        print("STRINGIO")
#        data = StringIO(csv_body)

        # Read the data into a DataFrame
#        print("Reading as CSV")
#        df = pd.read_csv(data)

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
            print("Saving dataframe...")
            # We want to save as a parquet file. 
            # Need to create a buffer for pandas conversion process
            buffer = BytesIO()

            # Write the parquet file to the buffer
            print("Converting to parquet...")
            data.to_parquet(buffer)
#            data.to_csv(buffer)

            # Set the dataset equal to the value of the buffer. 
            print("Gettting value of buffer...")
            dataset = buffer.getvalue()
        
        elif type(data) == dict:
            # We want to save the data as JSON
            dataset = json.dumps(data)
        
        else:
            # Unsupported type
            print(f"Unsupported type: {type(data)}")
            return False

        print("Putting the object in aws s3")
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



def join_datasets(books : pd.DataFrame, meta : pd.DataFrame) -> pd.DataFrame:
    """The books dataset contains reviews for various books, 
       but doesn't contain the titles that we would want to return
       to the user when recommending books. So this function joins the
       two datasets, combining information from each.

    Args:
        books (pd.DataFrame): Amazon books dataset containing reviews
        meta (pd.DataFrame): Meta data about each book

    Returns:
        pd.DataFrame: A dataframe joined on parent_asin column.
    """

    # Perform an inner join with books and meta data on parent_asin.
    df = books.merge(meta, on="parent_asin", how="inner", suffixes=('_review', '_book'))

    # This dataframe contains columns we won't need. 
    # Let's select only the required ones, which would include
    # the rating, user_id, and title_book columns. 
    useful_cols = ["rating", "user_id", "title_book"]
    df = df[useful_cols]

    # Now that we have all necessary columns, let's remove all null values
    df = df.dropna(axis=0)

    # Return the preprocessed dataset.
    return df

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
    """_summary_

    Args:
        df (pd.DataFrame): _description_
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

    # Load books and meta datasets from S3 storage - too large (30gb)
#    books = load_data(
#        bucket="readcrumbs",
#        objectkey="dataset/raw/book_reviews.jsonl",
#        client=client
#    )

#    meta = load_data(
#        bucket="readcrumbs",
#        objectkey="dataset/raw/book_data.jsonl",
#        client=client
#    )

    # Obtain joined dataframe
#    df = join_datasets(books, meta)

    # Load smaller dataset
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

    print("Saving Dataset")
    res = save_data(
        data=df,
        bucket="readcrumbs",
        objectkey="data/processed/ratings-small-v1.parquet",
        client=client
    )

#    assert res

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