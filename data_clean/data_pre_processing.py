from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, pos_tag_sents

# import for bag of word
import numpy as np
# For the regular expression
import re

import string
# Textblob dependency
from textblob import TextBlob
from textblob import Word
# set to string
from ast import literal_eval

import os

#!/usr/bin/env python3

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import re
import time
import spacy
import unidecode
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS

dbFile = open("../files_txt/tweets_april.json", encoding='utf-8')
tweets_extracao = json.load(dbFile)
tweets_extracao1 = {}
nlp = spacy.blank("en")

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wn = WordNetLemmatizer()


#definição de variáveis de limpeza
REGX_USERNAME = r"@[A-Za-z0-9$-_@.&+]+"
REGX_URL = r"https?://[A-Za-z0-9./]+"

#variaveis de verificação
sentiments={"positive":0, "neutral":0, "negative":0}


i=0
for key, value in tweets_extracao.items():
    i=i+1
    tweet=value["result"]

    #LIMPEZA DO TWEET
    tweet = tweet.lower() #lower Casing
    #ponctuation
    tweet = re.sub(r'&amp;','&',tweet)
    tweet = re.sub(r'\n', ' ', tweet)
    tweet = re.sub(REGX_USERNAME, ' ', tweet)
    tweet = re.sub(REGX_URL, ' ', tweet)

    #SENTIMENT ANALYSIS
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        sentimento = "positive"
    elif analysis.sentiment.polarity == 0:
        sentimento = 'neutral'
    else:
        sentimento = 'negative'
    sentiments[sentimento] += 1

    tokens = [token.text for token in nlp(tweet)]

    tokens = [t for t in tokens if
              t not in STOP_WORDS and
              t not in string.punctuation and
              len(t) > 3]

    tokens = [t for t in tokens if not t.isdigit()]

    stemmed_tokens = [ps.stem(token) for token in tokens]

    lemmatized_tokens = [wn.lemmatize(token) for token in stemmed_tokens]

    print(tweet)
    print(tokens)
    print(stemmed_tokens)
    print(lemmatized_tokens)

    #test
    if i==100:
        print(sentiments)
        break


file=open("tweets_limpos.json","w", encoding='utf-8')
json.dump(tweets_extracao1,file,indent=4,ensure_ascii=False)