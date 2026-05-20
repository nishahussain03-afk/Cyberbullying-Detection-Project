import streamlit as st
import pickle
import re

# LOAD MODEL
model = pickle.load(open("cyberbullying_model.pkl", "rb"))

vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# CLEAN FUNCTION
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# CYBERBULLYING KEYWORDS
bully_words = [
    "stupid",
    "idiot",
    "ugly",
    "hate",
    "useless",
    "loser",
    "dumb",
    "kill",
    "fool"
]

# TITLE
st.title("Cyberbullying Detection Web App")

st.write("Enter text to check whether it contains cyberbullying.")

# INPUT
user_input = st.text_area("Enter Text")

# BUTTON
if st.button("Predict"):

    cleaned = clean_text(user_input)

    # ML Prediction
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    # Keyword Detection
    keyword_detected = any(word in cleaned for word in bully_words)

    # FINAL DECISION
    if prediction == "bullying" or keyword_detected:

        st.error("Cyberbullying Detected")

    else:

        st.success("Non-Cyberbullying")

    # Show cleaned text
    st.write("Cleaned Text:", cleaned)