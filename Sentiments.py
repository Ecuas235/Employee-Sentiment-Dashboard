import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from googletrans import Translator
import langid
import time

import string
import re

from autocorrect import Speller

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline



nltk.download('wordnet')
nltk.download("stopwords")

def change_lang(s):
  translator = Translator()
  if langid.classify(s)[0] != 'en':
    translation = translator.translate(s, dest = 'en')
    time.sleep(0.3)
    s = translation.text
  return s


def text_preprocessing(s):
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r'  ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s

def preprocess_dataframe(df):
  spell = Speller(lang='en')
  df.dropna(inplace=True)
  sentiments = pd.concat([df['likes'], df['dislikes']], ignore_index=True)
  #sentiments = sentiments.apply(lambda x: change_lang(x))
  dates = pd.concat([df['date'], df['date']], ignore_index=True)
  sentiments = [' '.join([spell(i) for i in x.split()]) for x in sentiments]
  processed_sents = np.array([text_preprocessing(text) for text in sentiments])
  cols = ["review", "processed_review", "date"]
  fin_df = pd.DataFrame(list(zip(sentiments, processed_sents, dates)), columns = cols)
  return fin_df
  
def calculate_sentiments(df):
  df = preprocess_dataframe(df)
  
  model_path = '../Sentiment Analysis/models/employee_sentiment/checkpoint-1000'
  model = AutoModelForSequenceClassification.from_pretrained(model_path)
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  
  classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
  text_data = df['processed_review'].tolist() 
  results = classifier(text_data)
  
  label_map = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive'
  }
  labels = [label_map[item['label']] for item in results]
  # predicted_sentiments = [model.config.id2label[label.item()] for label in results]
  df['predicted_sentiment'] = labels
  return df
