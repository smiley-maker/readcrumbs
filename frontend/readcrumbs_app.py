import streamlit as st
import requests
import uuid

# API URL (probably need to change)
API_URL = "http://3.237.48.140:8000"

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

# Button and Prediction Logic:
# Use the return value of st.button() to control when the prediction is made.
if st.button("Get Recommendations"):
    # If no review is entered, show a warning.
    if input_text.strip() == "":
        st.warning("Please at least one favorite book.")
    else:
        # Connect with the API to make a prediction. Should pass a list of titles. 
        input_list = input_text.split(", ")
        print(input_list)

        # Create json data from input_list. 
        data_to_send = {
            "items": input_list,
            "userid": st.session_state.userid
        }

        response = requests.post(f"{API_URL}/predict", json=data_to_send)
        print(response)

        if response.status_code == 200:
            prediction = response.json()
#            prediction = prediction["recs"]
            st.session_state.prediction = prediction["recs"]
        else:
            print("API problem...")
            st.session_state.prediction = []
        


# Display the results if they exist
if st.session_state.prediction:
    for book in st.session_state.prediction:
        st.markdown(f"- {book}")

    st.divider()

    feedback = st.selectbox("Please rate the recommendations:", ["Good", "Average", "Poor"])
    
    if st.button("Submit Feedback"):
        feedback_data = {
            "userid": st.session_state.userid,
            "recommendations": st.session_state.prediction,
            "feedback": feedback
        }

        feedback_response = requests.post(f"{API_URL}/feedback", json=feedback_data)
        if feedback_response.status_code == 200:
            st.success("Thank you for your feedback!")
        else:
            st.error("There was an error submitting your feedback. Please try again later.")


# Footer with name and link to GitHub repository
st.divider()
col1, col2 = st.columns(2)
with col1:
       st.markdown("Developed by Jordan Sinclair and Jordan Larson")
with col2:
    st.link_button("GitHub Repository", "https://github.com/smiley-maker/readcrumbs")