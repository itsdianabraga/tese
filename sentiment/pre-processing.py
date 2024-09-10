import re
import pandas as pd
import unicodedata
from unidecode import unidecode
from pysentimiento.preprocessing import preprocess_tweet
import string
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Ensure required NLTK datasets are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset from CSV
df = pd.read_csv('tweets.csv')
print(df.head())  # Display the first few rows to check the data

# Function to preprocess text data
def preprocess_text(text):
    """
    Preprocess text by performing the following operations:
    - Convert to lowercase
    - Remove Unicode characters and URLs
    - Remove punctuation and numeric characters
    - Remove stopwords
    - Normalize repetitive patterns (e.g., laughter)
    
    Args:
        text (str): The input text to preprocess.
    
    Returns:
        str: Cleaned and preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove Unicode character patterns (e.g., '\uXXXX')
    text = re.sub(r'\\u[0-9a-z]{4}', " ", text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Remove numeric characters
    text = re.sub(r'\d+', '', text)

    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Join words back into a single string
    cleaned_text = ' '.join(words)

    # Normalize repetitive patterns like laughter (e.g., "hahaha" to "ha")
    cleaned_text = preprocess_tweet(cleaned_text, lang="en", shorten=2)

    return cleaned_text

# Test the preprocessing function with a sample text
sample_text = "This is a sample tweet! hahaha :) http://example.com"
cleaned_sample = preprocess_text(sample_text)
print(f"Original: {sample_text}\nCleaned: {cleaned_sample}")

# Set up sentiment analysis pipeline using a pre-trained model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision = "af0f99b"

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, revision=revision)

# Test the sentiment analysis pipeline with a sample text
sample_result = sentiment_pipeline("nooo please don't")
print(f"Sentiment Analysis Result: {sample_result}")

# Load tokenizer and model for sequence classification
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Tokenize a sample sentence and get the model's output logits
inputs = tokenizer("I'm so sad", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# Determine the predicted class based on logits
predicted_class_id = logits.argmax().item()
predicted_label = model.config.id2label[predicted_class_id]

print(f"Predicted class ID: {predicted_class_id}, Label: {predicted_label}")

# Optional: Save the cleaned DataFrame to a new CSV file
# df.to_csv('cleaned_dataset.csv', index=False)
