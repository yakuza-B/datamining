import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ==========================================
# 1. Page Configuration & NLTK Setup
# ==========================================
st.set_page_config(page_title="Airline Review Analyzer", page_icon="✈️", layout="centered")

# Download NLTK data quietly (cached to run only once)
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# ==========================================
# 2. Load Trained Models & Vectorizers
# ==========================================
@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    chi2 = joblib.load('chi2_selector.pkl')
    lr_model = joblib.load('logistic_regression_model.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
    return tfidf, chi2, lr_model, nb_model

tfidf_vectorizer, chi2_selector, lr_model, nb_model = load_models()

# ==========================================
# 3. Text Preprocessing Function (Matches Training Exactly)
# ==========================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 1. Remove scraping artifacts
    text = re.sub(r'✅Trip Verified\|', '', text)
    text = re.sub(r'Not Verified\|', '', text)
    text = re.sub(r'✅Verified Review\|', '', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove punctuation, numeric digits, and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenize, remove stopwords, and lemmatize
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(processed_tokens)

# ==========================================
# 4. Streamlit User Interface
# ==========================================
st.title("✈️ Airline Review Sentiment Analyzer")
st.markdown("Enter an airline review below to predict whether a passenger would **Recommend** or **Not Recommend** the airline, based on our trained Machine Learning models.")

# Sidebar for model selection and info
with st.sidebar:
    st.header("Model Settings")
    selected_model = st.radio(
        "Choose Prediction Model:",
        ("Logistic Regression (Recommended)", "Multinomial Naive Bayes")
    )
    
    st.markdown("---")
    st.markdown("### 💡 Sample Reviews to Test:")
    if st.button("Load Positive Review"):
        st.session_state.sample_text = "The flight was great! The cabin crew were professional and attentive, and the meal served was good. Very smooth journey."
    if st.button("Load Negative Review"):
        st.session_state.sample_text = "Worst airline ever. The flight was delayed for hours, the seats were dirty and uncomfortable, and the staff was incredibly rude. Never flying again."

# Main Input Area
default_text = st.session_state.get('sample_text', "")
user_review = st.text_area("Paste or type the airline review here:", value=default_text, height=150)

if st.button("🔍 Analyze Sentiment", type="primary"):
    if not user_review.strip():
        st.warning("Please enter a review text to analyze.")
    else:
        with st.spinner("Processing text and generating prediction..."):
            # 1. Preprocess the user input
            cleaned_text = preprocess_text(user_review)
            
            # 2. Vectorize and apply feature selection
            text_tfidf = tfidf_vectorizer.transform([cleaned_text])
            text_selected = chi2_selector.transform(text_tfidf)
            
            # 3. Predict based on user selection
            if selected_model.startswith("Logistic"):
                model = lr_model
                model_name = "Logistic Regression"
            else:
                model = nb_model
                model_name = "Multinomial Naive Bayes"
                
            prediction = model.predict(text_selected)[0]
            probabilities = model.predict_proba(text_selected)[0]
            
            # Get probability for the 'yes' class (index 1 if classes are ['no', 'yes'])
            classes = model.classes_
            yes_index = np.where(classes == 'yes')[0][0]
            confidence = probabilities[yes_index] * 100
            
            # 4. Display Results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction", prediction.upper())
            
            with col2:
                st.metric("Confidence Score", f"{confidence:.1f}%")
            
            if prediction == 'yes':
                st.success(f"**Conclusion:** The {model_name} model predicts this review is **RECOMMENDED** with {confidence:.1f}% confidence. The passenger had a positive experience.")
            else:
                st.error(f"**Conclusion:** The {model_name} model predicts this review is **NOT RECOMMENDED** with {100 - confidence:.1f}% confidence. The passenger had a negative experience.")
                
            with st.expander("🔍 View Processed Text (Debug)"):
                st.code(cleaned_text)
