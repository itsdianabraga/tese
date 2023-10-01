import re
import pandas as pd
import unicodedata
from unidecode import unidecode
from pysentimiento.preprocessing import preprocess_tweet

# Load the CSV file into a DataFrame
df = pd.read_csv('tweets.csv')

# Display the first few rows of the DataFrame
print(df.head())

import string
import re
import nltk
from nltk.corpus import stopwords

# Download the NLTK stop words dataset
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    text = re.sub(r'\\u[0-9a-z]{4}', " ", text)  # Remove '\uXXXX' patterns

    # Remove URLs using regular expressions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Remove numeric numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize the text
    words = text.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Replace matches with spaces
    cleaned_text = ' '.join(words)

    # Normalize laughter (e.g., "hahaha" to "ha")
    text = preprocess_tweet(cleaned_text, lang="en", shorten=2)

    return text


import tensorflow as tf
from transformers import pipeline

# Specify the model and revision
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision="af0f99b"

sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, revision=revision)

results = sentiment_pipeline("nooo please dont")

print(results)

#df.to_csv('cleaned_dataset.csv', index=False)

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("i'm so sad", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

print(predicted_class_id)