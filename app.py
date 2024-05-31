import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from Sentiments import calculate_sentiments
import subprocess
import re
from datetime import date
import os

st.set_page_config(
    page_title="EY Dashboard", page_icon="📊", layout="wide"
)
adjust_top_pad = """
    <style>
        body {
            background-color: #ffffff;
        }
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

def load_data(search_term):
    csv_path = f'../Sentiment Analysis/data/employee_{search_term}.csv'
    if not os.path.exists(path=csv_path):
        subprocess.run(["python", "run_scrapy.py", search_term])
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    return df

def apply_filters(df, start_date=None, end_date=None):
    df['date'] = pd.to_datetime(df['date'], format='posted on %d %b %Y')
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return df

def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]

def make_dashboard(df, column='processed_review'):
    if df.empty:
        st.warning("No data available")
    else:
        df['month'] = df['date'].dt.month
        monthly_counts = df['month'].value_counts().sort_index()
        st.write("### Monthly Frequency of Posts")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=monthly_counts.index, y=monthly_counts.values, mode='lines+markers'))
        fig_line.update_layout(xaxis_title='Month', yaxis_title='Frequency', height=500, width=1200)
        st.plotly_chart(fig_line)

        col1, col2 = st.columns([40, 30])
        with col1:
            # Plot the top 3 uni-grams
            top_bi_grams = get_top_ngram(df[column], n=1)
            x, y = map(list, zip(*top_bi_grams))
            fig_bar_uni = px.bar(x=y, y=x, labels={'y': 'Frequency', 'x': 'Words'}, title='Top 10 Uni-grams')
            fig_bar_uni.update_layout(height=500)
            st.plotly_chart(fig_bar_uni)

        with col2:
            st.dataframe(df['review'])

        col1, col2 = st.columns([60, 40])
        with col1:
            # Create the pie chart for rating distribution
            label_counts = df['predicted_sentiment'].value_counts()
            labels = label_counts.index
            sizes = label_counts.values

            # Plot the pie chart
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=sizes)])
            fig_pie.update_layout(title='Sentiment Distribution', height=500)
            st.plotly_chart(fig_pie)

        with col2:
            wordcloud = WordCloud(
                background_color='white',
                stopwords=stopwords,
                max_words=100,
                max_font_size=30,
                scale=3,
                random_state=1,
            )
            wordcloud=wordcloud.generate(str(df[column]))
            fig=plt.figure(1, figsize=(12, 12))
            plt.axis('off')
            plt.imshow(wordcloud)
            st.write("WordCloud")
            st.pyplot(fig)

# Sidebar - Search Parameters
df = None
with st.sidebar:
    st.title("Sentiment Analysis")
    with st.form(key="search_form"):
        st.subheader("Search Parameters")
        search_term = st.text_input("Search term", key="search_term")
       
        start_date = st.date_input("Start Date", value=None)
        end_date = st.date_input("End Date", value=None)
        run = st.form_submit_button(label="Run")
    if run:
        df = load_data(search_term)
        df = apply_filters(df, start_date=start_date, end_date=end_date)
if df is not None:
    search_term = search_term.lower()
    company = search_term.title()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Calculate mean rating and sentiments
    mean_rating = df['Rating'].mean()
    sentiments = calculate_sentiments(df)
    
    # Header
    st.write(f"## *{company}*", f"<span style='font-size: 25px; margin-left: 10px;'>{mean_rating:.2f}⭐</span>", unsafe_allow_html=True)
    total_reviews_length = sentiments['processed_review'].apply(len).sum()
    total_reviews_count = len(sentiments)
    positive_reviews_count = len(sentiments[sentiments['predicted_sentiment'] == 'positive'])
    negative_reviews_count = len(sentiments[sentiments['predicted_sentiment'] == 'negative'])
    neutral_reviews_count = len(sentiments[sentiments['predicted_sentiment'] == 'neutral'])
    # Dashboard tabs
    
    st.markdown(f"<div style='display: flex; margin-top: 10px;'>\
                <div style='background-color: #1f77b4; color: white; padding: 10px; border-radius: 5px; margin-right: 10px;'>\
                    Total: {total_reviews_count}\
                </div>\
                <div style='background-color: #ff7f0e; color: white; padding: 10px; border-radius: 5px; margin-right: 10px;'>\
                    Positive: {positive_reviews_count}\
                </div>\
                <div style='background-color: #d62728; color: white; padding: 10px; border-radius: 5px; margin-right: 10px;'>\
                    Negative: {negative_reviews_count}\
                </div>\
                <div style='background-color: #2ca02c; color: white; padding: 10px; border-radius: 5px;'>\
                    Neutral: {neutral_reviews_count}\
                </div>\
            </div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["All", "Positive 😊", "Negative ☹️", "Neutral"])
    with tab1:
        make_dashboard(sentiments)
    with tab2:
        make_dashboard(sentiments[sentiments['predicted_sentiment']=='positive'])
    with tab3:
        make_dashboard(sentiments[sentiments['predicted_sentiment']=='negative'])
    with tab4:
        make_dashboard(sentiments[sentiments['predicted_sentiment']=='neutral'])
