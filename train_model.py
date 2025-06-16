import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load sentiment dataset from Hugging Face
print("🔄 Loading dataset...")
dataset = load_dataset("tweet_eval", "sentiment")

# Convert HuggingFace dataset to pandas DataFrame
df = dataset["train"].to_pandas()

# Map labels to readable strings
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
df["label"] = df["label"].map(label_map)

# Clean and filter
df = df.dropna(subset=["text", "label"])
df = df[df["text"].str.strip() != ""]

# Show label distribution
print("📊 Label distribution:\n", df["label"].value_counts())

# Split into train/test with stratification
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")

