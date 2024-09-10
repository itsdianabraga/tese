import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer as NLTKSentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pysentimiento import create_analyzer

# Create a Pysentimiento analyzer
analyzer = create_analyzer(task="sentiment", lang="en")

# Load the cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# Define class labels
class_labels = [-1, 0, 1]

# Sentiment analysis functions

def textblob_sentiment(text):
    """
    Sentiment analysis using TextBlob.
    Returns -1 for negative, 0 for neutral, and 1 for positive.
    """
    if not isinstance(text, str):
        text = str(text)
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.3:
        return 1
    elif analysis.sentiment.polarity < -0.3:
        return -1
    else:
        return 0

def vader_sentiment(text):
    """
    Sentiment analysis using VADER.
    Returns -1 for negative, 0 for neutral, and 1 for positive.
    """
    if not isinstance(text, str):
        text = str(text)
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] > 0.3:
        return 1
    elif sentiment['compound'] < -0.3:
        return -1
    else:
        return 0

def nltk_sentiment(text):
    """
    Sentiment analysis using NLTK's VADER.
    Returns -1 for negative, 0 for neutral, and 1 for positive.
    """
    if not isinstance(text, str):
        text = str(text)
    analyzer = NLTKSentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] > 0.3:
        return 1
    elif sentiment['compound'] < -0.3:
        return -1
    else:
        return 0

def bert_sentiment(text):
    """
    Sentiment analysis using BERTweet model.
    Returns -1 for negative, 0 for neutral, and 1 for positive.
    """
    tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
    model = AutoModelForSequenceClassification.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
    model.eval()
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    sentiment_label = torch.argmax(probs, dim=1).item()
    return -1 if sentiment_label == 0 else 0 if sentiment_label == 1 else 1

def pysentimiento(text):
    """
    Sentiment analysis using Pysentimiento.
    Returns -1 for negative, 0 for neutral, and 1 for positive.
    """
    a = analyzer.predict(text)
    if a.output == "NEG":
        return -1
    elif a.output == "NEU":
        return 0
    elif a.output == "POS":
        return 1

# Initialize lists for predictions and true labels
textblob_predictions = []
vader_predictions = []
nltk_predictions = []
bert_predictions = []
transformer_predictions = []
true_labels = []

# Iterate through the dataset and apply each model
for index, row in df.iterrows():
    text = row['cleaned_tweet']
    true_label = row['sentimento']  # True sentiment label
    true_labels.append(true_label)

    # Uncomment to use other models as needed
    # textblob_predictions.append(textblob_sentiment(text))
    # vader_predictions.append(vader_sentiment(text))
    nltk_predictions.append(pysentimiento(text))
    # bert_predictions.append(bert_sentiment(text))

transformer_predictions=[0, -1, 1, 1, 1, 1, -1, 0, 0, 1, 0, -1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 0, 1, 1, 0, 1, 0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 1, 0, -1, -1, -1, 1, 1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, 1, 0, 1, 1, 1, 0, 0, -1, 0, 0, 0, 1, 1, -1, -1, -1, 1, 0, 1, 1, -1, 0, -1, 0, -1, -1, 0, 0, 0, 0, -1, 1, 0, 1, -1, -1, -1, -1, -1, -1]
bert_predictions=[-1, -1, 0, 1, 0, 1, 0, -1, 0, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 1, -1, 0, 0, -1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, 1, 0, 0, 1, 1, -1, -1, -1, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0]
vader_predictions=[0, -1, -1, 1, 1, 1, 0, -1, 0, 1, -1, -1, 0, -1, -1, 1, 1, 1, 1, 1, 0, 1, 0, -1, -1, -1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, -1, -1, 1, 1, -1, 0, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 1, 1, 0, 1, 1, -1, 1, -1, 1, 0, 1, 1, -1, 1, -1, 1, 0, -1, 0, 0, 1, 0, -1, 1, -1, -1, -1, -1, -1, 0, -1, 0]
textblob_predictions=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 1, 0, 1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
nltk_predictions=[-1, -1, 0, 1, 0, 1, 0, -1, 0, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 1, -1, 0, 0, -1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, 1, 0, 0, 1, 1, -1, -1, -1, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0]

# Function to get label names
def label_names(label):
    if label == -1:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label == 1:
        return "Positive"

# Function to plot confusion matrices
def plot_confusion_matrix(confusion, accuracy, title):
    confusion_percentage = confusion / confusion.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_percentage, annot=False, cmap='Blues', 
                xticklabels=[label_names(label) for label in class_labels], 
                yticklabels=[label_names(label) for label in class_labels])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {title}')
    plt.text(2, -0.7, f'Accuracy: {accuracy:.2%}', fontsize=12, ha='center')
    plt.show()

# Evaluation of each model
models = {
    "TextBlob": textblob_predictions,
    "VADER": vader_predictions,
    "Pysentimiento": nltk_predictions,
    "BERTweet": bert_predictions,
    "DistilBERT": transformer_predictions
}

for name, predictions in models.items():
    confusion = confusion_matrix(true_labels, predictions, labels=class_labels)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    
    # Print evaluation metrics
    print(f"\n{name} Confusion Matrix:\n", confusion)
    print(f"{name} Accuracy:", accuracy)
    print(f"{name} Precision:", precision)
    print(f"{name} Recall:", recall)
    print(f"{name} F1-score:", f1)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion, accuracy, name)
