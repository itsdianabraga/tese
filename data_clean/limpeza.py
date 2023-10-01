#!/usr/bin/env python3

#imports
import spacy
import langid
from googletrans import Translator
import string
import requests
import json
import re
from googletrans import Translator
import time
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# SENTIMENT ANALYSIS
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import torch
nlp = spacy.load("en_core_web_sm")
import string
import re
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import unicodedata
from unidecode import unidecode
from pysentimiento.preprocessing import preprocess_tweet

doc =open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_tweets_june.json', encoding='utf-8')
data = json.load(doc)
clean_data={}

API_KEY = 'AIzaSyBRh6hjazHm2lLbjwnq2imcQh61jtEb3J0'
ps = PorterStemmer()
wn = WordNetLemmatizer()
translator = Translator()
sia = SentimentIntensityAnalyzer()

from pysentimiento import create_analyzer
emotion_analyzer = create_analyzer(task="emotion", lang="en")

#variaveis para avaliar
health_keywords = [
    "food",
    "sugar",
    "nutrition",
    "diet",
    "healthy eating",
    "balanced diet",
    "nutrient",
    "nutritional",
    "meal planning",
    "sleep",
    "sleeping patterns",
    "exercise",
    "physical activity",
    "workout",
    "fitness",
    "wellness",
    "health",
    "lifestyle",
    "life-style",
    "life style,"
    "mental health",
    "immune system",
    "weight",
    "disease",
    "prevention",
    "pediatric",
    "active",
    "mental",
    "healthcare",
    "medical",
    "healthy habits",
    "unhealthy habits",
    "risk factors",
    "risk behaviors",
    "smoking", "tobacco", "nicotine", "substance addiction", "alcohol",
    "diet", "dieting", "overeating", "obesity", "overweight", "anorexia", "bulimia", "underweight",
    "food", "nutrition", "nutriment", "nutrient", "nourishment","healthy food",
    "lack of exercise", "exercising", "physical exertion", "fitness", "sedentary", "inactivity", "procrastination",
    "laziness",
    "sleep quality", "sleep quantity", "sleep time", "sleep duration", "sleep cycle",
    "sleep disorder", "fatigue", "sleep apnea", "insomnia",
    "screen time",
    "misinformation", "disinformation",
    "diabetes", "insulin-dependent", "type 2 diabetes", "type II diabetes",
    "high blood pressure", "hypertension", "hypertensive",
    "cardiovascular disease", "heart failure", "heart muscle disease", "peripheral vascular disease",
    "stroke", "vascular disease", "heart attack", "unstable angina",
    "cancer", "tumor", "carcinoma", "malignancy", "lymphoma", "malignant neoplastic disease",
    "mental health", "anxiety", "depression",
    "noncommunicable disease", "NCD", "chronic disease",
    "chronic respiratory disease", "asthma", "chronic obstructive pulmonary disease", "COPD",
    "occupational lung disease", "obese","body mass index"
]

food_keywords=["obesity","food","nutrition","diet","healthy eating","balanced diet","nutrient",
    "nutritional","meal planning", "overweight","aliment", "obese","nutrition", "nutriment", "nourishment",
    "alimentation", "healthy food","body mass index"]

keywords = [
    "healthy habits", "unhealthy habits", "risk factors", "risk behaviors",
    "drug", "smoking", "tobacco", "nicotine", "substance addiction", "alcohol",
    "drunkenness", "alcoholism", "chemical dependency", "substance abuse",
    "healthy lifestyle", "healthy life style", "healthy life-style", "unhealthy lifestyle",
    "unhealthy life style", "unhealthy life-style",
    "diet", "dieting", "overeating", "obesity", "overweight", "anorexia", "bulimia", "underweight",
    "food", "nutrition", "nutriment", "nutrient", "nourishment", "sustenance", "aliment",
    "alimentation", "healthy food",
    "physical exercise", "lack of exercise", "exercise", "exercising", "physical exertion",
    "workout", "fitness", "physical fitness", "sedentary", "inactivity", "procrastination",
    "laziness",
    "sleep quality", "sleep quantity", "sleep time", "sleep duration", "sleep cycle",
    "sleep disorder", "fatigue", "sleep apnea", "insomnia",
    "television", "tv", "screen time",
    "misinformation", "disinformation", "smartphone", "iphone", "android", "social media",
    "diabetes", "insulin-dependent", "type 2 diabetes", "type ii diabetes",
    "high blood pressure", "hypertension", "hypertensive",
    "cardiovascular disease", "heart failure", "heart muscle disease", "peripheral vascular disease",
    "stroke", "vascular disease", "heart attack", "unstable angina",
    "cancer", "tumor", "carcinoma", "malignancy", "lymphoma", "malignant neoplastic disease",
    "mental health", "anxiety", "depression",
    "noncommunicable disease", "NCD", "chronic disease",
    "chronic respiratory disease", "asthma", "chronic obstructive pulmonary disease", "copd",
    "occupational lung disease",
    "teacher", "school", "education", "school staff"
]

# Download the NLTK stop words dataset
stop_words = set(stopwords.words('english'))

#tradutor
translator = Translator()


def translate_tweet(tweet):
    try:
        translation = translator.translate(tweet, dest='en').text
    except Exception as e:
        translation = tweet
    translation = re.sub(r"([.,!?;])(\w)",r'\1 \2',translation)
    translation = re.sub(r" ([.,!?;])", r'\1', translation)
    return translation

def normalize_slang(text):
    # Define a dictionary of slang replacements
    slang_replacements = {
        r'\bsmbdy\b': 'somebody',
        r'\bidgaf\b': "I don't give a fuck",
        r'\blol\b': 'laughing out loud',
        r'\bthx\b': 'thanks',
        r'\bnvm\b': 'never mind',
        r'\bidk\b': "I don't know",
        r'\bidc\b': "I don't care",
        r'\bbtw\b': 'by the way',
        r'\bomg\b': 'oh my god',
        r'\blmao\b': 'laughing my ass off',
        r'\bwtf\b': 'what the fuck',
        r'\bwth\b': 'what the heck',
        r'\bjk\b': 'just kidding',
        r'\bbc\b': 'because',
        r'\bty\b': 'thank you',
        r'\bly\b': 'love you',
        r'\bomw\b': 'on my way',
        r'\bimo\b': 'in my opinion',
        r'\biykyk\b': 'if you know, you know',
        r'\bsthu\b': 'shut the hell up',
        r'\byolo\b': 'you only live once',
        r'\b2day\b': 'today',
        r'\b2moro\b': 'tomorrow',
        r'\batm\b': 'at the moment',
        r'\bb4\b': 'before',
        r'\bl8r\b': 'later',
        r'\bgr8\b': 'great',
        r'\bily\b': 'I love you',
        r'\bpls\b': 'please',
        r'\bDM\b': 'direct message',
        r'\bnbd\b': 'no big deal',
        r'\bwyd\b': 'what are you doing',
        r'\bh8\b': 'hate'
        # Add more replacements as needed
    }

    # Apply the replacements
    for pattern, replacement in slang_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    text = normalize_slang(text)

    text = re.sub(r'\\u[0-9a-z]{4}', " ", text)  # Remove '\uXXXX' patterns

    # Remove URLs using regular expressions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    #Tirar @
    text = re.sub(r'@(\w+)', r'\1', text)

    # Normalize laughter (e.g., "hahaha" to "ha")
    text = preprocess_tweet(text, lang="en", shorten=2)

    # Remove punctuation
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)

    # Tokenize the text
    words = text.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Replace matches with spaces
    cleaned_text = ' '.join(words)

    return cleaned_text

def is_a_concern(result):
    if re.search(r'concern|doubt|worries|worry|issue|fear|preoccupation|worried|alarm|question', result):
        return True
    a=emotion_analyzer.predict(result)
    if a.output=="fear":
        return True
    else:
        return False


for tweet in data:
    found_kid_verb = False
    tweet_text = data[tweet]['result']
    if data[tweet]["metadata"]["lang"] != "en":
        result_en = translate_tweet(tweet_text)
    else:
        result_en = tweet_text
    doc = nlp(result_en)
    for token in doc:
        # Check if the token is "kid" and "VERB"
        if (token.text == "kid" and token.pos_ == "VERB") or (token.text == "minor" and token.pos_=="ADJ"):
            found_kid_verb = True
            break
    if not found_kid_verb:
        if data[tweet]['query']["id"] in [59,60,64,65,66,67,100,101,102,103]:
            result_en = result_en.lower()  # Convert to lowercase for case-insensitive matching
            if any(keyword in result_en for keyword in health_keywords):
                clean_data[tweet]=data[tweet]
                clean_data[tweet]["result_en"]=result_en
            if any(keyword in result_en for keyword in food_keywords):
                if isinstance(clean_data[tweet]['query']['topic'],list):
                    clean_data[tweet]['query']['topic'].append("school_nutrition")
                else:
                    topics=[]
                    topics.append("school_nutrition")
                    topics.append(clean_data[tweet]['query']['topic'])
                    clean_data[tweet]['query']['topic']=topics
        else:
            clean_data[tweet]=data[tweet]
            clean_data[tweet]["result_en"] = result_en

for tweet in clean_data:
    result_en=clean_data[tweet]["result_en"]
    clean_data[tweet]['cleaned_tweet'] = preprocess_text(result_en)
    if is_a_concern(result_en):
        clean_data[tweet]['question']=True
    titulos = []
    descricoes = []
    matched_keywords_1 = []
    matched_keywords_2 = []
    topicos = clean_data[tweet]['query']['topic']
    links = clean_data[tweet]['metadata'].get('links')
    if links is not None:
        titulos = [link['title'] for link in links if link['title'] is not None]
        descricoes = [link['description'] for link in links if link['description'] is not None]
    if result_en is not None and result_en != "":
        matched_keywords = [word for word in keywords if word.lower() in result_en.lower()]
        for i in titulos:
            matched_keywords_1 = [word for word in keywords if word.lower() in i.lower()]
        for i in descricoes:
            matched_keywords_2 = [word for word in keywords if word.lower() in i.lower()]
        if isinstance(topicos, list):
            if "school_nutrition" in topicos:
                merge_list = list(
                    set(matched_keywords + matched_keywords_1 + matched_keywords_2 + ["school_nutrition"]))
            else:
                merge_list = list(set(matched_keywords + matched_keywords_1 + matched_keywords_2))
        else:
            merge_list = list(set(matched_keywords + matched_keywords_1 + matched_keywords_2))
        if len(merge_list) == 1:
            clean_data[tweet]['query']['topic'] = merge_list[0]
        else:
            clean_data[tweet]['query']['topic'] = merge_list


clean_data_final={}
for tweet in clean_data:
    print(clean_data[tweet]['query']['topic'])
    if clean_data[tweet]['query']['topic'] != []:
        clean_data_final[tweet]=clean_data[tweet]

REGX_USERNAME = r"@[A-Za-z0-9$-_@.&+]+"
REGX_URL = r"https?://[A-Za-z0-9./]+"

for tweet in clean_data_final.keys():
    clean_data_final[tweet]["nlp_process"].pop("lemmas", None)
    clean_data_final[tweet]["nlp_process"].pop("stems", None)
    tweeet=clean_data[tweet]["result_en"]
    # LIMPEZA DO TWEET
    tweeet=preprocess_text(tweeet)
    emoji_pattern = r'emoji[\s\w]*?emoji'
    tweeet=re.sub(emoji_pattern, '', tweeet)
    tweeet=re.sub(r'\bhashtag\b','',tweeet)
    tokens = [token.text for token in nlp(tweeet)]
    tokens = [t for t in tokens if
              t not in stop_words and
              t not in string.punctuation and
              len(t) > 2]
    clean_data_final[tweet]["nlp_process"]["tokens"]=tokens



print(len(clean_data))
print(len(clean_data_final))


with open('limpeza_june.json', 'w') as json_file:
    json.dump(clean_data_final, json_file, indent=4)