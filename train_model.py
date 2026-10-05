import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


FAKE_PATH = "data/Fake.csv"
TRUE_PATH = "data/True.csv"
MODEL_DIR = "model"


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("Loading datasets...")

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

print("Fake dataset loaded:", len(fake_df))
print("Real dataset loaded:", len(true_df))


fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

print("\nTotal articles:", len(df))
print("Columns:", list(df.columns))


df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["content"] = df["title"] + " " + df["text"]

df["content"] = df["content"].apply(clean_text)

df = df[["content", "label"]]

df = df[df["content"].str.strip() != ""]

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSample data:")
print(df.head())


X = df["content"]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF transformation completed.")


print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed.")


print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake", "Real"]
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")


print("\n================================")
print("MODEL SAVED")
print("================================")

print("model/fake_news_model.pkl")
print("model/tfidf_vectorizer.pkl")

print("\nTraining completed successfully!")