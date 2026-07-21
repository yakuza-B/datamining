import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ==========================================
# 1. Page Configuration & NLTK Setup
# ==========================================
st.set_page_config(
    page_title="Airline Review Analyzer", 
    page_icon="✈️", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Fix for SSL certificate errors when downloading NLTK data on Streamlit Cloud
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data (cached to run only once)
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
    # Note: Chi-Square selector is intentionally excluded here because the models 
    # were trained on the full TF-IDF matrix, not the reduced feature set.
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    lr_model = joblib.load('logistic_regression_model.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
    return tfidf, lr_model, nb_model

# Load models with error handling
try:
    tfidf_vectorizer, lr_model, nb_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False
    st.info("Please ensure your .pkl model files are in the same directory as this app.")
    st.stop()

# ==========================================
# 3. Text Preprocessing Function
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
    st.header("⚙️ Model Settings")
    selected_model = st.radio(
        "Choose Prediction Model:",
        ("Logistic Regression (Recommended)", "Multinomial Naive Bayes")
    )
    
    st.markdown("---")
    st.markdown("### 💡 Sample Reviews to Test:")
    if st.button("Load Positive Review", key="positive_btn"):
        st.session_state.sample_text = "The flight was great! The cabin crew were professional and attentive, and the meal served was good. Very smooth journey."
    if st.button("Load Negative Review", key="negative_btn"):
        st.session_state.sample_text = "Worst airline ever. The flight was delayed for hours, the seats were dirty and uncomfortable, and the staff was incredibly rude. Never flying again."
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance (Test Set)")
    st.markdown("**Logistic Regression**")
    st.metric("Accuracy", "90.22%")
    st.metric("F1-Score", "0.90")
    
    st.markdown("**Naive Bayes**")
    st.metric("Accuracy", "88.05%")
    st.metric("F1-Score", "0.88")

# Main Input Area
default_text = st.session_state.get('sample_text', "")
user_review = st.text_area("Paste or type the airline review here:", value=default_text, height=150)

if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not user_review.strip():
        st.warning("Please enter a review text to analyze.")
    else:
        with st.spinner("Processing text and generating prediction..."):
            # 1. Preprocess the user input
            cleaned_text = preprocess_text(user_review)
            
            # 2. Vectorize (Pass full TF-IDF matrix directly to the model)
            text_tfidf = tfidf_vectorizer.transform([cleaned_text])
            
            # 3. Predict based on user selection
            if selected_model.startswith("Logistic"):
                model = lr_model
                model_name = "Logistic Regression"
            else:
                model = nb_model
                model_name = "Multinomial Naive Bayes"
                
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]
            
            # Get probability for the 'yes' class
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
                
            with st.expander(" View Processed Text (Debug)"):
                st.code(cleaned_text)
            
            with st.expander("ℹ️ How It Works"):
                st.markdown("""
                1. **Text Cleaning**: Removes artifacts like "✅Trip Verified|", converts to lowercase, and removes special characters.
                2. **Tokenization & Lemmatization**: Breaks text into words and reduces them to their base form (e.g., "flying" → "fly").
                3. **TF-IDF Vectorization**: Converts text to numerical features based on word importance.
                4. **Prediction**: Applies the trained machine learning model to determine recommendation status.
                """)

# ==========================================
# 5. Footer Information
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>© 2026 Airline Review Sentiment Analyzer | Built with Streamlit</p>
    <p>Based on text mining and machine learning analysis of 7,730 airline reviews</p>
</div>
""", unsafe_allow_html=True)
