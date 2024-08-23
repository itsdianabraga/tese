import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer as NLTKSentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="en")

# Load your manually categorized tweets dataset as a DataFrame
# Replace 'your_data.csv' with the actual path to your dataset
df = pd.read_csv('cleaned_dataset.csv')

# Define the class labels as a list
class_labels = [-1,0,1]

import tensorflow as tf
from transformers import pipeline

# Specify the model and revision
#model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#revision="af0f99b"

#sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, revision=revision)

# Define functions for sentiment analysis with TextBlob, VADER, NLTK, and BERT

# TextBlob sentiment analysis
def textblob_sentiment(text):
    # Check if text is not a string, and if so, convert it to a string
    if not isinstance(text, str):
        text = str(text)

    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.3:
        sentimento = 1
    elif analysis.sentiment.polarity < - 0.3:
        sentimento = -1
    else:
        sentimento = 0
    return sentimento

# VADER sentiment analysis
def vader_sentiment(text):
    # Check if text is not a string, and if so, convert it to a string
    if not isinstance(text, str):
        text = str(text)

    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] > 0.3:
        sentimento = 1
    elif sentiment['compound'] < - 0.3:
        sentimento = -1
    else:
        sentimento = 0
    return sentimento


# NLTK sentiment analysis
def nltk_sentiment(text):
    if not isinstance(text, str):
        text = str(text)
    analyzer = NLTKSentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] > 0.3:
        sentimento = 1
    elif sentiment['compound'] < - 0.3:
        sentimento = -1
    else:
        sentimento = 0
    return sentimento


def bert_sentiment(text):
    tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis') #cite https://arxiv.org/abs/2106.09462

    model = AutoModelForSequenceClassification.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
    model.eval()

    # Tokenize and encode the text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=1)
    sentiment_label = torch.argmax(probs, dim=1).item()

    # Assuming labels 0, 1, 2 correspond to negative, neutral, and positive sentiment
    if sentiment_label == 0:
        sentimento = -1
    elif sentiment_label == 1:
        sentimento = 0
    elif sentiment_label == 2:
        sentimento = 1

    return sentimento

def pysentimiento(text):
    a = analyzer.predict(text)

    if a.output == "NEG":
        sentimento=-1
    elif a.output == "NEU":
        sentimento = 0
    elif a.output == "POS":
        sentimento =1

    return sentimento

# Initialize lists to store model predictions and true labels
textblob_predictions = []
vader_predictions = []
nltk_predictions = []
bert_predictions = []
transformer_predictions = []
true_labels = []

# Iterate through your dataset and apply each model
for index, row in df.iterrows():
    text = row['cleaned_tweet']
    true_label = row['sentimento']  # Replace 'label' with the actual column name containing labels
    true_labels.append(true_label)

    j=pysentimiento(text)

    # Apply each model
    #textblob_predictions.append(textblob_sentiment(text))
    #vader_predictions.append(vader_sentiment(text))
    nltk_predictions.append(pysentimiento(text))
    #bert_predictions.append(bert_sentiment(text))
    #transformer_predictions.append(transformer_sentiment(text))

transformer_predictions=[0, -1, 1, 1, 1, 1, -1, 0, 0, 1, 0, -1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 0, 1, 1, 0, 1, 0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 1, 0, -1, -1, -1, 1, 1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, 1, 0, 1, 1, 1, 0, 0, -1, 0, 0, 0, 1, 1, -1, -1, -1, 1, 0, 1, 1, -1, 0, -1, 0, -1, -1, 0, 0, 0, 0, -1, 1, 0, 1, -1, -1, -1, -1, -1, -1]
bert_predictions=[-1, -1, 0, 1, 0, 1, 0, -1, 0, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 1, -1, 0, 0, -1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, 1, 0, 0, 1, 1, -1, -1, -1, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0]
vader_predictions=[0, -1, -1, 1, 1, 1, 0, -1, 0, 1, -1, -1, 0, -1, -1, 1, 1, 1, 1, 1, 0, 1, 0, -1, -1, -1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, -1, -1, 1, 1, -1, 0, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 1, 1, 0, 1, 1, -1, 1, -1, 1, 0, 1, 1, -1, 1, -1, 1, 0, -1, 0, 0, 1, 0, -1, 1, -1, -1, -1, -1, -1, 0, -1, 0]
textblob_predictions=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 1, 0, 1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
nltk_predictions=[-1, -1, 0, 1, 0, 1, 0, -1, 0, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 1, -1, 0, 0, -1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, 1, 0, 0, 1, 1, -1, -1, -1, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0]

def label_names(label):
    if label == -1:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label == 1:
        return "Positive"


# Calculate the confusion matrix using VADER
textblob_confusion = confusion_matrix(true_labels, textblob_predictions, labels=class_labels)
textblob_accuracy = accuracy_score(true_labels, textblob_predictions)
textblob_confusion_percentage = textblob_confusion / textblob_confusion.sum(axis=1)[:, np.newaxis]

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j + 0.5, i + 0.5, f'{textblob_confusion[i][j]}\n({textblob_confusion_percentage[i][j]:.2%})',
                 ha='center', va='center', color='black', fontsize=12)

sns.heatmap(textblob_confusion_percentage, annot=False, cmap='Blues', xticklabels=[label_names(label) for label in class_labels], yticklabels=[label_names(label) for label in class_labels])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - TextBlob')

# Add accuracy to the figure
plt.text(2, -0.7, f'Accuracy: {textblob_accuracy:.2%}', fontsize=12, ha='center')

# Show the figure
plt.show()
print(textblob_accuracy)


# Calculate the confusion matrix using VADER
nltk_confusion = confusion_matrix(true_labels, nltk_predictions, labels=class_labels)
nltk_accuracy = accuracy_score(true_labels, nltk_predictions)
nltk_confusion_percentage = nltk_confusion / nltk_confusion.sum(axis=1)[:, np.newaxis]

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j + 0.5, i + 0.5, f'{nltk_confusion[i][j]}\n({nltk_confusion_percentage[i][j]:.2%})',
                 ha='center', va='center', color='black', fontsize=12)

sns.heatmap(nltk_confusion_percentage, annot=False, cmap='Blues', xticklabels=[label_names(label) for label in class_labels], yticklabels=[label_names(label) for label in class_labels])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Pysentimento')

# Add accuracy to the figure
plt.text(2, -0.7, f'Accuracy: {nltk_accuracy:.2%}', fontsize=12, ha='center')

# Show the figure
plt.show()
print(nltk_accuracy)

# Calculate the confusion matrix using VADER
vader_confusion = confusion_matrix(true_labels, vader_predictions, labels=class_labels)
vader_accuracy = accuracy_score(true_labels, vader_predictions)
vader_confusion_percentage = vader_confusion / vader_confusion.sum(axis=1)[:, np.newaxis]

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j + 0.5, i + 0.5, f'{vader_confusion[i][j]}\n({vader_confusion_percentage[i][j]:.2%})',
                 ha='center', va='center', color='black', fontsize=12)

sns.heatmap(vader_confusion_percentage, annot=False, cmap='Blues', xticklabels=[label_names(label) for label in class_labels], yticklabels=[label_names(label) for label in class_labels])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - VADER')

# Add accuracy to the figure
plt.text(2, -0.7, f'Accuracy: {vader_accuracy:.2%}', fontsize=12, ha='center')

# Show the figure
plt.show()
print(vader_accuracy)



#BERT

# Calculate the confusion matrix using VADER
bert_confusion = confusion_matrix(true_labels, bert_predictions, labels=class_labels)
bert_accuracy = accuracy_score(true_labels, bert_predictions)
bert_confusion_percentage = bert_confusion / bert_confusion.sum(axis=1)[:, np.newaxis]

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j + 0.5, i + 0.5, f'{bert_confusion[i][j]}\n({bert_confusion_percentage[i][j]:.2%})',
                 ha='center', va='center', color='black', fontsize=12)

sns.heatmap(bert_confusion_percentage, annot=False, cmap='Blues', xticklabels=[label_names(label) for label in class_labels], yticklabels=[label_names(label) for label in class_labels])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - BERTweet')

# Add accuracy to the figure
plt.text(2, -0.7, f'Accuracy: {bert_accuracy:.2%}', fontsize=12, ha='center')

# Show the figure
plt.show()
print(bert_accuracy)





#transformer

# Calculate the confusion matrix using VADER
transformer_confusion = confusion_matrix(true_labels, transformer_predictions, labels=class_labels)
transformer_accuracy = accuracy_score(true_labels, transformer_predictions)
transformer_confusion_percentage = transformer_confusion / transformer_confusion.sum(axis=1)[:, np.newaxis]

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j + 0.5, i + 0.5, f'{transformer_confusion[i][j]}\n({transformer_confusion_percentage[i][j]:.2%})',
                 ha='center', va='center', color='black', fontsize=12)

sns.heatmap(transformer_confusion_percentage, annot=False, cmap='Blues', xticklabels=[label_names(label) for label in class_labels], yticklabels=[label_names(label) for label in class_labels])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - DistilBERT')

# Add accuracy to the figure (adjust coordinates)
plt.text(2, -0.7, f'Accuracy: {transformer_accuracy:.2%}', fontsize=12, ha='center')

# Show the figure
plt.show()
print(transformer_accuracy)

# Calculate accuracy, precision, recall, and F1-score
textblob_accuracy = accuracy_score(true_labels, textblob_predictions)
vader_accuracy = accuracy_score(true_labels, vader_predictions)
nltk_accuracy = accuracy_score(true_labels, nltk_predictions)
bert_accuracy = accuracy_score(true_labels, bert_predictions)
transformer_accuracy = accuracy_score(true_labels, transformer_predictions)

textblob_precision = precision_score(true_labels, textblob_predictions, average='macro')
vader_precision = precision_score(true_labels, vader_predictions, average='macro')
nltk_precision = precision_score(true_labels, nltk_predictions, average='macro')
bert_precision = precision_score(true_labels, bert_predictions, average='macro')
transformer_precision = precision_score(true_labels, transformer_predictions, average='macro')


textblob_recall = recall_score(true_labels, textblob_predictions, average='macro')
vader_recall = recall_score(true_labels, vader_predictions, average='macro')
nltk_recall = recall_score(true_labels, nltk_predictions, average='macro')
bert_recall = recall_score(true_labels, bert_predictions, average='macro')
transformer_recall = recall_score(true_labels, transformer_predictions, average='macro')

textblob_f1 = f1_score(true_labels, textblob_predictions, average='macro')
vader_f1 = f1_score(true_labels, vader_predictions, average='macro')
nltk_f1 = f1_score(true_labels,nltk_predictions, average='macro')
bert_f1 = f1_score(true_labels, bert_predictions, average='macro')
transformer_f1 = f1_score(true_labels, transformer_predictions, average='macro')

print(true_labels)
# Print or use the confusion matrices and evaluation metrics as needed
print("TextBlob Confusion Matrix:")
print(textblob_confusion)
print("TextBlob Accuracy:", textblob_accuracy)
print("TextBlob Precision:", textblob_precision)
print("TextBlob Recall:", textblob_recall)
print("TextBlob F1-score:", textblob_f1)
print(textblob_predictions)

print("\nVADER Confusion Matrix:")
print(vader_confusion)
print("VADER Accuracy:", vader_accuracy)
print("VADER Precision:", vader_precision)
print("VADER Recall:", vader_recall)
print("VADER F1-score:", vader_f1)
print(vader_predictions)

print("\nNLTK Confusion Matrix:")
print(nltk_confusion)
print("NLTK Accuracy:", nltk_accuracy)
print("NLTK Precision:", nltk_precision)
print("NLTK Recall:", nltk_recall)
print("NLTK F1-score:", nltk_f1)
print(nltk_predictions)

print("\nBERT Confusion Matrix:")
print(bert_confusion)
print("BERT Accuracy:", bert_accuracy)
print("BERT Precision:", bert_precision)
print("BERT Recall:", bert_recall)
print("BERT F1-score:", bert_f1)
print(bert_predictions)

print("\nTransformer Confusion Matrix:")
print(transformer_confusion)
print("Transformer Accuracy:", transformer_accuracy)
print("Transformer Precision:", transformer_precision)
print("Transformer Recall:", transformer_recall)
print("Transformer F1-score:", transformer_f1)
print(transformer_predictions)

