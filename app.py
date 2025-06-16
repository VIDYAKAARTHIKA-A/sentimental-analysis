# app.py

import streamlit as st
import pandas as pd
import tweepy
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def analyze_sentiment_ml(texts):
    features = vectorizer.transform(texts)
    return model.predict(features)

bearer_token = "YOUR_BEARER_TOKEN_HERE"  # Replace with your Twitter Bearer Token
client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAACEK2gEAAAAAtVCntejjN40Se1CmcRN9qiR0W3c%3DMXW52Uu6DjSuwmwV4ratKR37qk92g8tkbRu0eQewkoXe0NyIw3")

def fetch_tweets(query, max_tweets=50):
    tweets = client.search_recent_tweets(query=query, tweet_fields=["text", "lang"], max_results=100)
    tweet_texts = []

    if not tweets.data:
        return []

    for tweet in tweets.data:
        if tweet.lang == "en":
            tweet_texts.append(tweet.text)
            if len(tweet_texts) >= max_tweets:
                break

    return tweet_texts


st.title("🐦 Twitter Sentiment Analysis (ML-Powered)")

query = st.text_input("Enter a keyword or hashtag to search for tweets")
max_tweets = st.slider("Number of tweets to analyze", 10, 100, 50)

if st.button("Analyze"):
    if not query:
        st.warning("⚠️ Please enter a search keyword.")
    else:
        st.info(f"Fetching and analyzing up to {max_tweets} tweets about '{query}'...")

        tweets = fetch_tweets(query, max_tweets)

        if not tweets:
            st.error("No tweets found or API limit exceeded.")
        else:
            sentiments = analyze_sentiment_ml(tweets)

            df = pd.DataFrame({
                "Tweet": tweets,
                "Sentiment": sentiments
            })

            st.dataframe(df)
            sentiment_counts = df["Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)
