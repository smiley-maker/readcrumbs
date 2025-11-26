import pandas as pd
from experiments.training.preprocess import preprocess_data

# Unit test
## testing if we get the right feature columns after preprocessing
## Test if there are any null values in the data remaining
def test_preprocess_data():
    # Create a sample dataframe
    df = pd.read_csv("./data/raw/Books_rating.csv")

    # Get just a small sample for testing
    df = df.sample(n=100, random_state=42)

    # Preprocess the data
    processed_df = preprocess_data(df)

    # Check if the processed dataframe has the expected columns
    expected_columns = ["User_id", "Title", "review/score"]
    assert all(col in processed_df.columns for col in expected_columns), "Not all expected columns are present."

    # Check if there are any null values remaining
    assert not processed_df.isnull().values.any(), "There are still null values in the processed data."

    # Check if there are any reviews with a score less than 4. 
    assert (processed_df["review/score"] >= 4).all(), "There are reviews with a score less than 4."
    
    print("test_preprocess_data passed.")


if __name__ == "__main__":
    test_preprocess_data()