import re
import joblib
import streamlit as st


MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


st.set_page_config(
    page_title="CredibilityGuard",
    page_icon="🛡️",
    layout="centered"
)


st.title("🛡️ CredibilityGuard")

st.subheader("Fake News and Text Credibility Analyzer")

st.write(
    "Enter a news headline or article below to classify it "
    "using a trained machine learning model."
)


news_text = st.text_area(
    "Enter news text:",
    height=250,
    placeholder="Paste a news headline or article here..."
)


if st.button("Analyze News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")

    else:
        cleaned_text = clean_text(news_text)

        text_vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(text_vector)[0]

        probabilities = model.predict_proba(text_vector)[0]

        fake_probability = probabilities[0]
        real_probability = probabilities[1]

        st.divider()

        if prediction == 0:
            st.error("Prediction: FAKE NEWS")
            confidence = fake_probability * 100
        else:
            st.success("Prediction: REAL NEWS")
            confidence = real_probability * 100

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.info(
            "This is an automated machine-learning classification. "
            "It does not independently verify whether a claim is factually true."
        )

        st.divider()

        st.write("### Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fake",
                f"{fake_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Real",
                f"{real_probability * 100:.2f}%"
            )