import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('spam_detection_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer files not found. Please ensure 'spam_detection_model.pkl' and 'vectorizer.pkl' are in the same directory.")
        return None, None

def preprocess_text(text):
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(filtered_tokens)

model, vectorizer = load_assets()

if model and vectorizer:
    st.title("Spam Detection App")
    st.markdown("This app uses a Logistic Regression model to classify email text as 'Spam' or 'Not Spam'.")
    
    user_input = st.text_area("Enter your email text here:", height=200, placeholder="Enter text to classify...")
    
    if st.button("Predict"):
        if user_input:
            processed_input = preprocess_text(user_input)
            vectorized_input = vectorizer.transform([processed_input])
            prediction = model.predict(vectorized_input)
            
            st.markdown("---")
            if prediction[0] == 1:
                st.error("Prediction: SPAM")
            else:
                st.success("Prediction: NOT SPAM (HAM)")
        else:
            st.warning("Please enter some text to get a prediction.")
