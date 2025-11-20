import streamlit as st
import requests

# API URL (probably need to change)
API_URL = "http://54.91.115.10:8000"

# Setup the streamlit page
title = "Readcrumbs"
description = "Provides book recommendations based on your favorite titles."
st.set_page_config(page_title=title, page_icon="📚")
st.title(title)
st.write(description)

input_text = st.text_area("What are some of your favorite books?", height=150)

# Button and Prediction Logic:
# Use the return value of st.button() to control when the prediction is made.
if st.button("Analyze Sentiment"):
    # If no review is entered, show a warning.
    if input_text.strip() == "":
        st.warning("Please enter a movie review to analyze.")
    else:
        # Connect with the API to make a prediction. Should pass a list of titles. 
        input_list = input_text.split(", ")

        # Create json data from input_list. 
        data_to_send = {
            "items": input_list
        }

        response = requests.post(f"{API_URL}/predict", json=data_to_send)

        if response.status_code == 200:
            prediction = response.json()
            prediction = prediction["recs"]
        
        # Display result
        st.subheader("Your Recommendations")
        for r in prediction:
            # display title
            st.text(r)

# Footer with name and link to GitHub repository
st.divider()
col1, col2 = st.columns(2)
with col1:
       st.markdown("Developed by Jordan Sinclair and Jordan Larson")
with col2:
    st.link_button("GitHub Repository", "https://github.com/smiley-maker/readcrumbs")