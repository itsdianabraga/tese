import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import torch


from textblob import TextBlob

# Preprocess the tweet (if necessary)
tweet = "There are 4 elephants in the gym"

def preprocess_tweet(tweet):
    # Remove URLs, special characters, and punctuation
    tweet = re.sub(r"http\S+|[^a-zA-Z\s]", "", tweet)

    # Tokenize the tweet into individual words
    tokens = word_tokenize(tweet)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Join the filtered tokens back into a single string
    preprocessed_tweet = " ".join(filtered_tokens)

    return preprocessed_tweet


# Preprocess the tweet
preprocessed_tweet = preprocess_tweet(tweet)

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Analyze the sentiment of the tweet
sentiment_scores = sia.polarity_scores(preprocessed_tweet)

# Interpret the sentiment scores
compound_score = sentiment_scores['compound']
if compound_score >= 0.45:
    sentiment = "positive"
elif compound_score <= -0.45:
    sentiment = "negative"
else:
    sentiment = "neutral"

# Print the sentiment
print("SentimentIntensityAnalyzer")
print(f"The score is {compound_score}. The tweet sentiment is {sentiment}.")
print(" ")


# SENTIMENT ANALYSIS
analysis = TextBlob(tweet)

if analysis.sentiment.polarity > 0.45:
    sentimento = "Positive"
elif analysis.sentiment.polarity < - 0.45:
    sentimento = "Negative"
else:
    sentimento = 'Neutral'

print("Text Blob")
print(f"The score is {analysis.sentiment.polarity}. The tweet sentiment is {sentimento}.")
print(" ")
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Preprocess and tokenize the text
text = tweet
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Perform sentiment analysis
outputs = model(**inputs)
logits = outputs.logits
predicted_labels = logits.argmax(dim=1)
sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
sentiment = sentiment_mapping[predicted_labels.item()]

print("BERT")
print(f"The sentiment of the text is {sentiment}.")
