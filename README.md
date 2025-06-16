# 🐦 Sentiment Analysis of Twitter Data

## 📖 Overview

This project performs **sentiment analysis** on live Twitter data to determine whether tweets about a particular keyword or hashtag are **positive** 😊, **negative** 😞, or **neutral** 😐. It leverages the Twitter API to fetch recent tweets in real-time and uses a **Logistic Regression** machine learning model trained on labeled text data to classify the sentiment of each tweet.

---

## ✨ Features

- 🐦 **Twitter Data Collection:** Connects to the Twitter API to fetch recent English-language tweets based on a user-specified keyword or hashtag.
- 🧹 **Text Preprocessing:** Cleans and vectorizes tweet texts using TF-IDF vectorization for effective feature extraction.
- 🤖 **Sentiment Classification:** Applies a Logistic Regression model trained on labeled sentiment data to classify tweets into Positive, Negative, or Neutral.
- 💻 **Interactive Interface:** Streamlit-based app for easy interaction, allowing users to input search terms and visualize sentiment distribution via charts.
- 💾 **Model Persistence:** Uses `joblib` to save and load trained models and vectorizers for quick inference.

---

## 🐦 Twitter API

The project uses Twitter's **v2 API** via the `tweepy` Python library to access recent tweets matching a query. It requires a **Twitter Developer Account** to obtain the necessary credentials (Bearer Token) for authentication.

- The API allows querying for recent tweets filtered by language (English) to ensure consistent sentiment analysis.
- Tweets are fetched in batches, limited by the maximum allowed per request.
- The retrieved tweet text is the input for sentiment classification.

For details on obtaining API credentials, visit [Twitter Developer Portal](https://developer.twitter.com/). 🔑

---

## 🤖 Machine Learning Model

### Logistic Regression

- Chosen for its simplicity and efficiency in classification tasks.
- Trained on a labeled dataset with examples of positive, negative, and neutral sentiments.
- Input features are generated using **TF-IDF vectorization**, which converts text data into numerical vectors that represent word importance in the corpus.
- After training, the model can predict the sentiment of unseen tweets with reasonable accuracy.

### TF-IDF Vectorizer

- Stands for **Term Frequency-Inverse Document Frequency**.
- Weighs words based on how important they are to a document relative to the entire corpus.
- Helps reduce the impact of common words that carry less meaning (e.g., "the", "and").

### Joblib

- Used for saving and loading the trained model and vectorizer efficiently.
- Allows quick deployment without retraining every time.

---

## 🚀 How to Use

1. 🔐 **Set up Twitter API credentials:** Obtain your Bearer Token from the Twitter Developer Portal and add it to the configuration file or environment variables.
2. 🏋️‍♂️ **Train the model:** Use `train_model.py` to train the Logistic Regression classifier on your dataset.
3. 🌐 **Run the Streamlit app:** Execute `app.py` to start the interactive web app.
4. 🔎 **Input a search query:** Enter a keyword or hashtag in the app to fetch live tweets and see their sentiment analysis results.
5. 📊 **View results:** Explore the tweet texts, their predicted sentiments, and a bar chart summarizing sentiment distribution.

---

## 📦 Requirements

- Python 3.7+
- Libraries: `tweepy`, `scikit-learn`, `pandas`, `streamlit`, `joblib`, `textblob`
- Twitter Developer Account for API access

---

## 🗂 Folder Structure
sentimental-analysis/
│
├── app.py                       # Streamlit app for user interface and live analysis

├── train_model.py               # Script to train and save the Logistic Regression model

├── sentiment_model.pkl          # Saved Logistic Regression model (after training)

├── vectorizer.pkl               # Saved TF-IDF vectorizer

├── README.md                    # This documentation

├── .gitignore                   # Git ignore rules

└── requirements.txt             # Python dependencies (optional)

