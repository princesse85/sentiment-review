import streamlit as st
#import pickle
import Joblib 

# ✅ Load the trained model
from Joblib import load
sentiment_model = load(model_file)
#with open("sentiment_model.pkl", "rb") as model_file:
    #sentiment_model = pickle.load(model_file)

# ✅ Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)

# ✅ Page styling
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.markdown(
    """
    <style>
    textarea {
        font-size: 16px !important;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #3e8e41;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ App title and description
st.title("🛍️ Ecommerce Customer Reviews Sentiment Analysis")
st.write("""
Enter a customer review below to determine whether it's **positive** or **negative** using a trained machine learning model.
""")

# ✅ Text input
user_input = st.text_area("✍️ Paste a customer review here:", height=150)

# ✅ Predict button
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        input_vector = tfidf_vectorizer.transform([user_input])
        prediction = sentiment_model.predict(input_vector)[0]
        sentiment = "✅ Positive 😊" if prediction == 1 else "⚠️ Negative 😠"
        st.success(f"Predicted Sentiment: **{sentiment}**")

