#!/usr/bin/env python3

import json
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch

doc =open('C:/Users/diana/PycharmProjects/thesis/data_clean/limpeza_june.json', encoding='utf-8')
clean_data = json.load(doc)

tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
model = AutoModelForSequenceClassification.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
model.eval()

def bert_sentiment(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=1)
    sentiment_label = torch.argmax(probs, dim=1).item()

    return sentiment_label

i=0
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
for tweet in clean_data:
    i+=1
    clean_text= clean_data[tweet]['cleaned_tweet']
    sentiment_label=bert_sentiment(clean_text)
    clean_data[tweet]["sentiment"]=sentiment_map[sentiment_label]
    print(i)


with open('processados_june.json', 'w') as json_file:
    json.dump(clean_data, json_file, indent=4)