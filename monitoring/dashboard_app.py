import streamlit as st
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from decimal import Decimal

def fetch_dynamodb_table(table_name: str, region_name='us-east-1') -> pd.DataFrame:
    # Get the DynamoDB resource
    dynamodb = boto3.resource('dynamodb', region_name=region_name)
    # Retrieve the table
    table = dynamodb.Table(table_name)
    
    # Scan the table to get all items
    response = table.scan()
    # Collect all items
    data = response['Items']
    
    while 'LastEvaluatedKey' in response:
        # Continue scanning if there are more items
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        # Append new items to data list
        data.extend(response['Items'])
    
    # Convert list of items to DataFrame
    # Handle empty table case
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # This explicitly targets columns that are Decimals and turns them into floats/ints
    for col in df.columns:
        # Check if the first non-null element is a Decimal to determine column type
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(sample, Decimal):
            # Convert entire column from Decimal to float
            df[col] = df[col].apply(lambda x: float(x) if x else x)
            
    return df


# Get data from DynamoDB tables
logs = fetch_dynamodb_table('readcrumbs-logs')
feedback = fetch_dynamodb_table('readcrumbs-feedback')

# Start Streamlit app
st.title("Readcrumbs Monitoring Dashboard")

st.header("Item Coverage")
# Spread out the predictions because they are stored as a list in each dataframe entry.
df = logs.explode('prediction')
unique_books = df['prediction'].nunique()
st.metric("Unique Recommended Books", unique_books)

st.header("Input Diversity")
df = logs.explode('items')
unique_inputs = df['items'].nunique()
st.metric("Unique Input Combinations", unique_inputs)

st.metric("Total Predictions Made", len(logs))

st.header("User Engagement")
unique_users = df['userid'].nunique()
st.metric("Unique Users", unique_users)


st.header("Average Latency")
avg_latency = logs['latency'].mean()
st.metric("Average Prediction Latency (ms)", f"{avg_latency:.2f}")


st.header('Prediction Latency Over Time')

if 'latency' in logs.columns:
    # plot using plotly for interactivity
    fig = px.line(logs, x=logs.index, y='latency', title='Prediction Latency Over Time')
    st.plotly_chart(fig, use_container_width=True)
    fig.update_layout(xaxis_title='Timestamp', yaxis_title='Latency (ms)' )

else:
    st.info("No latency field present in data.")

st.header("Feedback Summary")
st.subheader("Recall At K=10")
if not feedback.empty:
    # Get user ids from userid+name format
    feedback['real_user_id'] = feedback['userid'].apply(lambda x: x.split("_")[0])
    # Calculate number of likes per user
    likes_per_user = feedback.groupby('real_user_id')['feedback'].sum()
    # Remove nonnumeric entries if any
    likes_per_user = pd.to_numeric(likes_per_user, errors='coerce').fillna(1)
    st.metric("Average Likes per User", f"{likes_per_user.mean():.2f}")
    st.metric("Total Likes", int(likes_per_user.sum()))
    # Each user got 10 recommendations, so recall@10 is likes/10
    recall_at_10 = likes_per_user.sum() / (len(likes_per_user) * 10)
    st.metric("Recall@10", f"{recall_at_10:.4f}")
else:
    st.info("No feedback data available.")



st.header("Top Recommended Books")
df = logs.explode('prediction')
top_books = df['prediction'].value_counts().head(10)

# Plot top recommended books
fig = px.bar(x=top_books.values, y=top_books.index, orientation='h', title='Top 10 Most Recommended Books')
st.plotly_chart(fig, use_container_width=True)
fig.update_layout(xaxis_title='Number of Recommendations', yaxis_title='Book Title')

st.header("Top Input Books")
df = logs.explode('items')
top_inputs = df['items'].value_counts().head(10)

# Plot top input books
fig = px.bar(x=top_inputs.values, y=top_inputs.index, orientation='h', title='Top 10 Input Books')
st.plotly_chart(fig, use_container_width=True)
fig.update_layout(xaxis_title='Number of Times Input', yaxis_title='Book Title')