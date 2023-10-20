import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer



# Load the trained model
with open("SpamDetection.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the CountVectorizer for transforming text
with open("CountVectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Create a Streamlit web app
st.title("Spam Detection App")

# Input text box for the user to enter a message
user_input = st.text_input("Enter a message:")

# Function to make a prediction
def predict_spam(user_input):
    user_input_transformed = vectorizer.transform([user_input])
    prediction = model.predict(user_input_transformed)
    return prediction[0]

if st.button("Check for Spam"):
    if user_input:
        prediction = predict_spam(user_input)
        if prediction == 1:
            st.success("This message is not spam (ham).")
        else:
            st.error("This message is spam.")
    else:
        st.warning("Please enter a message for classification.")

# Optionally, you can provide some information about the model and data
#st.write("Model Accuracy:", ac_dt)
st.write("Model Description: This is a Naive Bayes model for spam detection.")
