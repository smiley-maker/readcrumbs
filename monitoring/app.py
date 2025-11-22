import streamlit as st
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

s3 = boto3.client('s3')
dynamodb = boto3.client('dynamodb')

# Helper functions
def get_data_from_dynamodb(table_name):
    response = dynamodb.scan(TableName=table_name)
    return response['Items']

def convert_to_df(items):
    return pd.DataFrame(items)


df = convert_to_df(get_data_from_dynamodb('prediction-logs'))

# ---------------------------- Streamlit app ----------------------------
st.title("Monitoring Dashboard")

df = convert_to_df(get_data_from_dynamodb('prediction-logs'))

# Convert columns to proper types
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
df['prediction'] = df['prediction'].astype(str)
if 'req' in df:
    df['req'] = df['req'].astype(str)

st.header('Prediction Latency Over Time')

if 'latency' in df.columns:
    # Plot latency over time if available
    fig1, ax1 = plt.subplots()
    sns.lineplot(x='datetime', y='latency', data=df, ax=ax1)
    ax1.set_title('Prediction Latency Over Time')
    st.pyplot(fig1)
else:
    st.info("No latency field present in data. Please ensure the backend logs the latency of predictions.")

st.header('Prediction Distribution (Target Drift)')

fig2, ax2 = plt.subplots()
sns.countplot(x='prediction', data=df, ax=ax2)
ax2.set_title('Distribution of Predicted Classes')
st.pyplot(fig2)

st.header('Collect User Feedback')

st.write("Click below to rate the most recent model prediction and help track accuracy.")

user_id_input = st.text_input("User ID", "")
recent = None
if user_id_input:
    try:
        uid = int(user_id_input)
        cur_user_rows = df[df['user_id'] == uid]
        if not cur_user_rows.empty:
            recent = cur_user_rows.sort_values('datetime', ascending=False).iloc[0]  # get latest
            st.write(f"Last prediction for User {user_id_input}:")
            st.code(dict(recent), language='json')
    except Exception:
        st.warning("Please enter a valid numeric user ID.")

if recent is not None:
    feedback = st.radio("Are these recommendations relevant to you?", ['Yes', 'No'])
    feedback_submitted = st.button("Submit Feedback")
    if feedback_submitted:

        feedback_table = 'prediction-feedback'
        record = {
            'user_id': {'N': str(recent['user_id'])},
            'datetime': {'S': str(recent['datetime'])},
            'prediction': {'S': str(recent['prediction'])},
            'feedback': {'S': feedback}
        }
        try:
            dynamodb.put_item(TableName=feedback_table, Item=record)
            st.success("Thank you for your feedback!")
        except Exception as e:
            st.error(f"Failed to submit feedback: {e}")

# Calculate live accuracy from feedback
    feedback_items = convert_to_df(get_data_from_dynamodb('prediction-feedback'))
    feedback_items['feedback'] = feedback_items['feedback'].astype(str)
    if not feedback_items.empty:
        acc = (feedback_items['feedback'] == 'Yes').mean()
        st.metric("Live Model Accuracy (from feedback)", f"{acc:.2%}")
    else:
        st.info("No feedback yet; accuracy cannot be computed.")





