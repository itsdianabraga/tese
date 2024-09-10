#!/usr/bin/env python3

import json
import re
import string
import pandas as pd
from textblob import TextBlob
import spacy
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch

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

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Load SpaCy model and NLTK resources
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Initialize objects for various tasks
translator = Translator()
emotion_analyzer = create_analyzer(task="emotion", lang="en")
ps = PorterStemmer()
wn = WordNetLemmatizer()

# Define keywords for different categories
health_keywords = [
    "food", "sugar", "nutrition", "diet", "healthy eating", "balanced diet", "nutrient", "nutritional",
    "meal planning", "sleep", "sleeping patterns", "exercise", "physical activity", "workout", "fitness",
    "wellness", "health", "lifestyle", "life-style", "life style", "mental health", "immune system",
    "weight", "disease", "prevention", "pediatric", "active", "mental", "healthcare", "medical",
    "healthy habits", "unhealthy habits", "risk factors", "risk behaviors", "smoking", "tobacco", "nicotine",
    "substance addiction", "alcohol", "diet", "dieting", "overeating", "obesity", "overweight", "anorexia",
    "bulimia", "underweight", "food", "nutrition", "nutriment", "nutrient", "nourishment", "healthy food",
    "lack of exercise", "exercising", "physical exertion", "fitness", "sedentary", "inactivity", "procrastination",
    "laziness", "sleep quality", "sleep quantity", "sleep time", "sleep duration", "sleep cycle", "sleep disorder",
    "fatigue", "sleep apnea", "insomnia", "screen time", "misinformation", "disinformation", "diabetes",
    "insulin-dependent", "type 2 diabetes", "type II diabetes", "high blood pressure", "hypertension",
    "hypertensive", "cardiovascular disease", "heart failure", "heart muscle disease", "peripheral vascular disease",
    "stroke", "vascular disease", "heart attack", "unstable angina", "cancer", "tumor", "carcinoma",
    "malignancy", "lymphoma", "malignant neoplastic disease", "mental health", "anxiety", "depression",
    "noncommunicable disease", "NCD", "chronic disease", "chronic respiratory disease", "asthma",
    "chronic obstructive pulmonary disease", "COPD", "occupational lung disease", "obese", "body mass index"
]

food_keywords = [
    "obesity", "food", "nutrition", "diet", "healthy eating", "balanced diet", "nutrient", "nutritional",
    "meal planning", "overweight", "aliment", "obese", "nutrition", "nutriment", "nourishment",
    "alimentation", "healthy food", "body mass index"
]

keywords = [
    "healthy habits", "unhealthy habits", "risk factors", "risk behaviors", "drug", "smoking", "tobacco",
    "nicotine", "substance addiction", "alcohol", "drunkenness", "alcoholism", "chemical dependency",
    "substance abuse", "healthy lifestyle", "healthy life style", "healthy life-style", "unhealthy lifestyle",
    "unhealthy life style", "unhealthy life-style", "diet", "dieting", "overeating", "obesity", "overweight",
    "anorexia", "bulimia", "underweight", "food", "nutrition", "nutriment", "nutrient", "nourishment",
    "sustenance", "aliment", "alimentation", "healthy food", "physical exercise", "lack of exercise",
    "exercise", "exercising", "physical exertion", "workout", "fitness", "physical fitness", "sedentary",
    "inactivity", "procrastination", "laziness", "sleep quality", "sleep quantity", "sleep time", "sleep duration",
    "sleep cycle", "sleep disorder", "fatigue", "sleep apnea", "insomnia", "television", "tv", "screen time",
    "misinformation", "disinformation", "smartphone", "iphone", "android", "social media", "diabetes",
    "insulin-dependent", "type 2 diabetes", "type ii diabetes", "high blood pressure", "hypertension",
    "hypertensive", "cardiovascular disease", "heart failure", "heart muscle disease", "peripheral vascular disease",
    "stroke", "vascular disease", "heart attack", "unstable angina", "cancer", "tumor", "carcinoma",
    "malignancy", "lymphoma", "malignant neoplastic disease", "mental health", "anxiety", "depression",
    "noncommunicable disease", "NCD", "chronic disease", "chronic respiratory disease", "asthma",
    "chronic obstructive pulmonary disease", "copd", "occupational lung disease", "teacher", "school",
    "education", "school staff"
]

def translate_tweet(tweet):
    """
    Translates a tweet to English if it is not already in English.
    """
    try:
        translation = translator.translate(tweet, dest='en').text
    except Exception:
        translation = tweet  # If translation fails, return the original tweet
    # Clean up punctuation spacing in translation
    translation = re.sub(r"([.,!?;])(\w)", r'\1 \2', translation)
    translation = re.sub(r" ([.,!?;])", r'\1', translation)
    return translation

def normalize_slang(text):
    """
    Replaces common slang terms in the text with their expanded forms.
    """
    slang_replacements = {
        r'\bsmbdy\b': 'somebody', r'\bidgaf\b': "I don't give a fuck", r'\blol\b': 'laughing out loud',
        r'\bthx\b': 'thanks', r'\bnvm\b': 'never mind', r'\bidk\b': "I don't know", r'\bidc\b': "I don't care",
        r'\bbtw\b': 'by the way', r'\bomg\b': 'oh my god', r'\blmao\b': 'laughing my ass off', r'\bwtf\b': 'what the fuck',
        r'\bwth\b': 'what the heck', r'\bjk\b': 'just kidding', r'\bbc\b': 'because', r'\bty\b': 'thank you',
        r'\bly\b': 'love you', r'\bomw\b': 'on my way', r'\bimo\b': 'in my opinion', r'\biykyk\b': 'if you know, you know',
        r'\bsthu\b': 'shut the hell up', r'\byolo\b': 'you only live once', r'\b2day\b': 'today', r'\b2moro\b': 'tomorrow',
        r'\batm\b': 'at the moment', r'\bb4\b': 'before', r'\bl8r\b': 'later', r'\bgr8\b': 'great', r'\bily\b': 'I love you',
        r'\bpls\b': 'please', r'\bDM\b': 'direct message', r'\bnbd\b': 'no big deal', r'\bwyd\b': 'what are you doing',
        r'\bh8\b': 'hate'
    }
    for pattern, replacement in slang_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    """
    Processes the text by normalizing slang, removing URLs, mentions, punctuation,
    and stop words, and then returning the cleaned text.
    """
    text = text.lower()  # Convert text to lowercase
    text = normalize_slang(text)  # Replace slang terms
    text = re.sub(r'\\u[0-9a-z]{4}', " ", text)  # Remove unicode patterns
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@(\w+)', r'\1', text)  # Remove mentions
    text = preprocess_tweet(text, lang="en", shorten=2)  # Normalize laughter and other tweet-specific cleaning
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)  # Remove punctuation
    words = text.split()  # Tokenize the text into words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)  # Return cleaned text as a single string

def is_a_concern(result):
    """
    Determines if the result indicates a concern based on keywords or sentiment analysis.
    """
    if re.search(r'concern|doubt|worries|worry|issue|fear|preoccupation|worried|alarm|question', result):
        return True
    a = emotion_analyzer.predict(result)
    return a.output == "fear"

def process_tweets(data):
    """
    Processes tweets by filtering, translating, and cleaning the data, and then
    performing keyword matching and sentiment analysis.
    """
    clean_data = {}
    for tweet in data:
        found_kid_verb = False
        tweet_text = data[tweet]['result']
        result_en = translate_tweet(tweet_text) if data[tweet]["metadata"]["lang"] != "en" else tweet_text
        
        doc = nlp(result_en)
        found_kid_verb = any(
            (token.text == "kid" and token.pos_ == "VERB") or (token.text == "minor" and token.pos_ == "ADJ") for token in doc
        )
        
        if not found_kid_verb:
            if data[tweet]['query']["id"] in [59, 60, 64, 65, 66, 67, 100, 101, 102, 103]:
                result_en = result_en.lower()
                if any(keyword in result_en for keyword in health_keywords):
                    clean_data[tweet] = data[tweet]
                    clean_data[tweet]["result_en"] = result_en
                if any(keyword in result_en for keyword in food_keywords):
                    topic = clean_data[tweet]['query'].get('topic', [])
                    if isinstance(topic, list):
                        topic.append("school_nutrition")
                    else:
                        clean_data[tweet]['query']['topic'] = [topic, "school_nutrition"]
            else:
                clean_data[tweet] = data[tweet]
                clean_data[tweet]["result_en"] = result_en
    
    # Further processing of cleaned tweets
    for tweet in clean_data:
        result_en = clean_data[tweet]["result_en"]
        clean_data[tweet]['cleaned_tweet'] = preprocess_text(result_en)
        sentiment_label=bert_sentiment(clean_data[tweet]['cleaned_tweet'])
        clean_data[tweet]["sentiment"]=sentiment_map[sentiment_label]
        
        if is_a_concern(result_en):
            clean_data[tweet]['question'] = True
        titles = []
        descriptions = []
        matched_keywords_1 = []
        matched_keywords_2 = []
        topics = clean_data[tweet]['query']['topic']
        links = clean_data[tweet]['metadata'].get('links')
        
        if links is not None:
            titles = [link['title'] for link in links if link['title'] is not None]
            descriptions = [link['description'] for link in links if link['description'] is not None]
        
        if result_en:
            matched_keywords = [word for word in keywords if word.lower() in result_en.lower()]
            for title in titles:
                matched_keywords_1 += [word for word in keywords if word.lower() in title.lower()]
            for description in descriptions:
                matched_keywords_2 += [word for word in keywords if word.lower() in description.lower()]
            
            if isinstance(topics, list):
                if "school_nutrition" in topics:
                    merge_list = list(set(matched_keywords + matched_keywords_1 + matched_keywords_2 + ["school_nutrition"]))
                else:
                    merge_list = list(set(matched_keywords + matched_keywords_1 + matched_keywords_2))
            else:
                merge_list = list(set(matched_keywords + matched_keywords_1 + matched_keywords_2))
            
            if len(merge_list) == 1:
                clean_data[tweet]['query']['topic'] = merge_list[0]
            else:
                clean_data[tweet]['query']['topic'] = merge_list

    # Final filtering and cleanup
    clean_data_final = {}
    for tweet in clean_data:
        if clean_data[tweet]['query']['topic']:
            clean_data_final[tweet] = clean_data[tweet]
    
    REGX_USERNAME = r"@[A-Za-z0-9$-_@.&+]+"
    REGX_URL = r"https?://[A-Za-z0-9./]+"
    
    for tweet in clean_data_final.keys():
        clean_data_final[tweet]["nlp_process"].pop("lemmas", None)
        clean_data_final[tweet]["nlp_process"].pop("stems", None)
        tweet_text = clean_data[tweet]["result_en"]
        tweet_text = preprocess_text(tweet_text)
        emoji_pattern = r'emoji[\s\w]*?emoji'
        tweet_text = re.sub(emoji_pattern, '', tweet_text)
        tweet_text = re.sub(r'\bhashtag\b', '', tweet_text)
        tokens = [token.text for token in nlp(tweet_text)]
        tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation and len(t) > 2]
        clean_data_final[tweet]["nlp_process"]["tokens"] = tokens

    # Save final cleaned data to a JSON file
    with open('limpeza_june.json', 'w') as json_file:
        json.dump(clean_data_final, json_file, indent=4)

if __name__ == "__main__":
    with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_tweets_june.json', encoding='utf-8', 'r') as file:
        data = json.load(file)

    # Process the tweets
    process_tweets(data)
