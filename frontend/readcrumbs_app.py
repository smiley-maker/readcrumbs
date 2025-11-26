import streamlit as st
import requests
import uuid
import datetime

# API URL (probably need to change)
API_URL = "http://44.201.69.213:8000"

# Setup the streamlit page
title = "Readcrumbs"
description = "Provides book recommendations based on your favorite titles."
st.set_page_config(page_title=title, page_icon="📚")
st.title(title)
st.write(description)

if 'userid' not in st.session_state:
    st.session_state.userid = str(uuid.uuid4())

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

input_text = st.text_area("What are some of your favorite books?", height=150)

# Use the return value of st.button() to control when the prediction is made.
if st.button("Get Recommendations"):
    # If no favorite books are entered, show a warning.
    if input_text.strip() == "":
        st.warning("Please at least one favorite book.")
    else:
        # Connect with the API to make a prediction. Should pass a list of titles. 
        # The titles do have to match exactly what is in the dataset. 
        input_list = input_text.split(", ")

        # Create json data from input_list. This should match the expected input of the API.
        data_to_send = {
            "items": input_list,
            "userid": st.session_state.userid
        }

        response = requests.post(f"{API_URL}/predict", json=data_to_send)

        if response.status_code == 200:
            prediction = response.json()
            st.session_state.prediction = prediction["recs"]
        else:
            print("API problem...")
            st.session_state.prediction = []
        
# Display the results if they exist
if st.session_state.prediction:
    st.header("Recommended Books")
    for idx, book in enumerate(st.session_state.prediction):
        cols = st.columns([3, 1])
        # Shows the book title and a like button
        with cols[0]:
            st.markdown(f"**{idx+1}. {book}**")
        with cols[1]:
            if st.button("👍", key=f"like_{idx}"):
                feedback_data = {
                    "userid": st.session_state.userid+"_"+book.replace(" ", "_"), # Create unique user-book id
                    "recommendations": st.session_state.prediction, # Full list of recommendations
                    "feedback": 1, # 1 = like
                    "title": book, # The book being liked
                    "position": idx + 1, # Position in the recommendation list
                    "timestamp": str(datetime.datetime.now()), # Current timestamp
                }

                # Save feedback to DynamoDB using the API endpoint
                feedback_response = requests.post(f"{API_URL}/feedback", json=feedback_data)
                if feedback_response.status_code == 200:
                    st.markdown("Thanks for your feedback! :smiley:")
                else:
                    st.markdown("There was an issue submitting your feedback. Please try again.")

# Footer with name and link to GitHub repository
st.divider()
col1, col2 = st.columns(2)
with col1:
       st.markdown("Developed by Jordan Sinclair and Jordan Larson")
with col2:
    st.link_button("GitHub Repository", "https://github.com/smiley-maker/readcrumbs")