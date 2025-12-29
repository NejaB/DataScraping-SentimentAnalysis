import streamlit as st
import pandas as pd
import torch
from transformers import pipeline

# --- CONFIGURATION ---
st.set_page_config(page_title="Brand Monitor 2023", layout="wide")

# --- DATA LOADING ---
# We use @st.cache_data so the app doesn't reload the CSV every time you click a button
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None

# Load your three CSV files
df_reviews = load_data("review_data.csv")
df_products = load_data("product_data.csv")
df_testimonials = load_data("testimonial_data.csv")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Navigation")
selection = st.sidebar.radio(
    "Select a Section:",
    ["Reviews", "Products", "Testimonials"]
)

# --- MAIN CONTENT LOGIC ---

# 1. Initialize the Sentiment Analysis Pipeline
# The @st.cache_resource decorator ensures the model loads into memory only ONCE
@st.cache_resource
def load_sentiment_model():
    # This specific model is fine-tuned for SST-2 (Standard Sentiment Treebank)
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

if selection == "Reviews":
    st.title("Customer Reviews")
    #slider from january 2023 to december 2023
    month = st.select_slider("Select Month for yeat 2023:",
                     options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                     value="August")
    
    #print all reviews from selected month
    if df_reviews is not None:  
        st.write(f"Showing reviews from {month} 2023.")
        month_number = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"].index(month) + 1
        df_reviews['date'] = pd.to_datetime(df_reviews['date'], errors='coerce')
        filtered_reviews = df_reviews[df_reviews['date'].dt.month == month_number]
        st.dataframe(filtered_reviews)

        if not filtered_reviews.empty:
            # Sentiment analysis using transformers pipeline
            with st.spinner("Analyzing sentiments..."):
                sentiments = sentiment_analyzer(filtered_reviews['review_text'].tolist())
                filtered_reviews['Sentiment'] = [s['label'] for s in sentiments]
                filtered_reviews['Confidence'] = [s['score'] for s in sentiments]

            # 4. Display Summary Metrics
            pos_count = len(filtered_reviews[filtered_reviews['Sentiment'] == 'POSITIVE'])
            neg_count = len(filtered_reviews[filtered_reviews['Sentiment'] == 'NEGATIVE'])
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews", len(filtered_reviews))
            col2.metric("✅ Positive", pos_count)
            col3.metric("❌ Negative", neg_count)
            st.dataframe(filtered_reviews[[ 'Sentiment', 'Confidence']], use_container_width=True)

            avg_confidence = filtered_reviews.groupby('Sentiment')['Confidence'].mean()
            #add a barchsart to show negative and positive reviews 
            st.subheader("Average Confidence Scores")
            #chart includes average confidence scores
            st.bar_chart(avg_confidence)

            

    else:
        st.warning("No review data available.") 



elif selection == "Products":
    st.title("Product Catalog")
    if df_products is not None:
        df = pd.read_csv("product_data.csv")
        st.write(df.head(len(df_products)))
        st.write(f"Showing {len(df_products)} products.")
    else:
        st.warning("No product data available.")

elif selection == "Testimonials":
    st.title("Customer Testimonials")
    if df_testimonials is not None:
        st.write(f"Showing {len(df_testimonials)} testimonials.")
        st.table(df_testimonials.head(len(df_testimonials))) # .table looks great for product lists
    else:
        st.warning("No testimonial data available.")

