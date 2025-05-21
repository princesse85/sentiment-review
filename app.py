
import streamlit as st
import pickle


# 🎨 Custom CSS for UI
st.markdown(
    """
    <style>
    div.stTextArea > div > textarea {
        font-size: 16px;
        padding: 12px;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color:rgba(69, 80, 160, 0.34);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Load the sentiment model
with open('sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

# ✅ Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# 🧾 App Title and Description
st.title("🛍️ Ecommerce Customer Reviews Sentiment Analysis App")

st.write("""
Welcome to our Ecommerce Customer Reviews Analysis App.  
This simple tool analyzes customer reviews and predicts whether they are **positive** or **negative**.
""")

# 📝 User Input
user_input = st.text_area("📝 Enter a customer review below:", height=100)

# 🔍 Predict Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_features = tfidf_vectorizer.transform([user_input])
        prediction = sentiment_model.predict(input_features)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
        st.success(f"Predicted Sentiment: **{sentiment}**")
