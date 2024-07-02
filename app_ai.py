import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import time
from google_play_scraper import Sort, reviews
from app_store_scraper import AppStore
from groq import Groq
import os
from typing import List, Dict

# Set page config
st.set_page_config(page_title="User Review sentiment analysis", layout="wide")

# Initialize session state for API key and data
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

if 'data' not in st.session_state:
    st.session_state.data = None

if 'summary' not in st.session_state:
    st.session_state.summary = None

# Function to update API key
def update_api_key():
    st.session_state.api_key = st.session_state.api_key_input

# Function to get the API key
def get_api_key():
    return st.session_state.api_key

# Function to scrape reviews from Google Play Store
@st.cache_data(ttl=3600)
def scrape_android_reviews(count: int) -> pd.DataFrame:
    try:
        result, _ = reviews(
            'com.SIBMobile',
            lang='en',
            country='in',
            sort=Sort.NEWEST,
            count=count
        )
        df = pd.DataFrame(result)
        df['date'] = pd.to_datetime(df['at']).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={'content': 'review', 'score': 'rating'})
        return df[['date', 'review', 'rating']]
    except Exception as e:
        st.error(f"Error scraping Android reviews: {str(e)}")
        return pd.DataFrame(columns=['date', 'review', 'rating'])

# Function to scrape reviews from Apple App Store
@st.cache_data(ttl=3600)
def scrape_ios_reviews(count: int) -> pd.DataFrame:
    try:
        app = AppStore(country='in', app_name='sib-mirror-mobile-banking', app_id=1184796899)
        app.review(how_many=count)
        df = pd.DataFrame(app.reviews)
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={'review': 'review', 'rating': 'rating'})
        return df[['date', 'review', 'rating']]
    except Exception as e:
        st.error(f"Error scraping iOS reviews: {str(e)}")
        return pd.DataFrame(columns=['date', 'review', 'rating'])

# Function to analyze sentiment
def analyze_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

# Function to query Groq API
def query_groq(prompt: str, max_tokens: int = 500) -> str:
    api_key = get_api_key()
    if not api_key:
        st.error("oops... looks like u haven't added the **magic spell**✨ in the side bar")
        return ""
    
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # Contact info to be displayed on hover
        contact_info = "Request at DTD"

        # HTML with hover effect and an "i" icon
        hover_html = f"""
        <style>
        .tooltip {{
        position: relative;
        display: inline-block;
        cursor: pointer;
        }}

        .tooltip .tooltiptext {{
        visibility: hidden;
        width: 220px;
        background-color: grey;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 100%; 
        left: 50%; 
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        }}

        .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
        }}
        </style>

        <p>Seems like the magic spell✨ is wrong !!! Try another one or contact the <span class="tooltip"><strong>Admin</strong>
        <span class="tooltiptext">{contact_info}</span>
        <span style="margin-left: 5px;">👁️</span>
        </span>.</p>
        """

    # Display the error message with hover effect
    st.markdown(hover_html, unsafe_allow_html=True)
    return ""

    # Display the error message with hover effect
    st.markdown(hover_html, unsafe_allow_html=True)
    return ""

# Function to prepare prompt
def prepare_prompt(question: str, reviews: List[str]) -> str:
    prompt = f"Based on the following app reviews, please answer this question in under 200 words: {question}\n\nReviews:\n"
    prompt += "\n".join(reviews)  
    return prompt

# Function to prepare brief summary prompt
def prepare_brief_summary_prompt(reviews: List[str]) -> str:
    prompt = f"Based on the following app reviews, provide a brief summary under 100 words of what customers have to say:\n\nReviews:\n"
    prompt += "\n".join(reviews[:10])  # Use only the first 10 reviews for brevity
    return prompt

# Function to display visualizations
def display_visualizations(df: pd.DataFrame):
    st.header("Sentiment Analysis")
    
    # Brief summary of customer sentiment
    if st.session_state.summary is None:
        with st.spinner("Generating summary..."):
            prompt = prepare_brief_summary_prompt(df['review'].tolist())
            st.session_state.summary = query_groq(prompt)
    st.write(st.session_state.summary)
    
    sentiment_counts = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral').value_counts()
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
    st.plotly_chart(fig)

# Main app function
def main():
    st.title("Mirror Plus Reviews sentiment analysis")

    # Sidebar
    st.sidebar.title("Settings")
    
    # API Key input (only created once)
    st.sidebar.text_input(
        "Enter a **magic spell✨** to summon AI",
        type="password",
        key="api_key_input",
        on_change=update_api_key
    )

    platform = st.sidebar.selectbox("Select Platform", ["iOS", "Android"])
    review_count = st.sidebar.slider("Number of reviews to fetch", 10, 200, 100)

    # Refresh data button
    if st.sidebar.button("Refresh Data"):
        with st.spinner("Scraping data..."):
            st.session_state.data = scrape_ios_reviews(review_count) if platform == "iOS" else scrape_android_reviews(review_count)
            st.session_state.summary = None  # Reset summary when data is refreshed
        st.success("Data scraped successfully!")

    # Main content area
    if st.session_state.data is not None:
        df = st.session_state.data

        # Apply sentiment analysis
        df['sentiment'] = df['review'].apply(analyze_sentiment)

        # Display reviews sorted by date
        st.header("Latest Reviews")
        st.dataframe(df.sort_values('date', ascending=False))

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rating", f"{df['rating'].mean():.2f}")
        with col2:
            st.metric("Total Reviews", len(df))
        with col3:
            st.metric("Average Sentiment", f"{df['sentiment'].mean():.2f}")

        # Visualizations
        display_visualizations(df)

        # FAQ and Custom Query sections
        st.header("Frequently Asked Questions")
        faq_questions = [
                    "what are the main pain points addressed by users, give it as a list.",
                    "What are the highlighted negatives?",
                    "What are the highlighted positives?",
                    "What improvements do users suggest for the app?",
                    "what do u suggest to this banking app better"
                ]

        selected_question = st.selectbox("Select a question", faq_questions)

        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                prompt = prepare_prompt(selected_question, df['review'].tolist())
                answer = query_groq(prompt)
                st.write(answer)

        st.header("Custom Query")
        user_query = st.text_input("Enter your question about the app")

        if st.button("Submit Query"):
            with st.spinner("Generating answer..."):
                prompt = prepare_prompt(user_query, df['review'].tolist())
                answer = query_groq(prompt)
                st.write(answer)
    else:
        st.info("Click 'Refresh Data' to load reviews.")

if __name__ == "__main__":
    main()
