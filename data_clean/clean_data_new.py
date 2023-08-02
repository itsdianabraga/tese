#!/usr/bin/env python3

#imports
import spacy
import langid
from googletrans import Translator
import string
import requests
import json
import re
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
nlp = spacy.blank("en")

#definição de variáveis de limpeza
REGX_USERNAME = r"@[A-Za-z0-9$-_@.&+]+"
REGX_URL = r"https?://[A-Za-z0-9./]+"

doc =open('C:/Users/diana/PycharmProjects/thesis/tweets_june.json', encoding='utf-8')
db = json.load(doc)
db1={}

API_KEY = 'AIzaSyBRh6hjazHm2lLbjwnq2imcQh61jtEb3J0'
ps = PorterStemmer()
wn = WordNetLemmatizer()
translator = Translator()
sia = SentimentIntensityAnalyzer()

count=0

class TranslationError(Exception):
    pass

def translate_tweet(tweet,linguagem):
    # Detect the language of the tweet
    if linguagem is not None:
        language = linguagem
    else:
        language = langid.classify(tweet)[0]

    # Translate the tweet to Portuguese if it's not already in Portuguese
    if language != 'pt':
        try:
            translation = translator.translate(tweet, dest='pt').text
        except Exception as e:
            translation = tweet
        translation = re.sub(r"([.,!?;])(\w)",r'\1 \2',translation)
        translation = re.sub(r" ([.,!?;])", r'\1', translation)
        return translation
    else:
        return tweet

def get_country_from_coords(lat, lng):
    url = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={API_KEY}'
    response = requests.get(url)
    data = response.json()
    country = None
    for component in data['results'][0]['address_components']:
        if 'country' in component['types']:
            country = component['long_name']
            break
    return country
v=0

start = time.time() #controlo dos limites da API twitter

for key, value in db.items():
    print(v)
    v+=1
    try:
        result=value["result"]
        link_tweet = re.findall(r'(?:https?:\/\/|www\.)\S+', result)
        if link_tweet != []:
            link_tweet = link_tweet[0]
        else:
            link_tweet = None
        q = re.search(r'\?', value["result"])
        hash=re.findall(r'\#\w+',value["result"])
        mencionados=re.findall((r'@\w+'),value["result"])
        metadata=value.get("metadata")
        user=value.get("user")
        if q != None:
            question = True
        else:
            question = False
        tweet_id=key
        contexto = []
        mencoes = []
        if hash ==[]:
            hash=None
        context=metadata.get("context")
        if context is not None:
            for i in context:
                contexto.append({"type": i.get("type"), "text": i.get("normalized_text"), "prob": i.get("probability")})
        else:
            contexto=None
        l = value.get("metadata")
        urls=[]
        media = []
        mks=[]
        if l.get("links") is not None:
            for i in l.get("links"):
                if i.get("media_key") is None:
                    title = i.get("title")
                    description = i.get("description")
                    if i.get("expanded_url") is not None:
                        url = i.get("expanded_url")
                    elif i.get("unwound_url") is not None:
                        url = i.get("unwound_url")
                    elif i.get("url") is not None:
                        url = i.get("url")
                    match_tweet=re.match(r'https\:\/\/twitter\.com\/',url)
                    images = i.get("images")
                    if images is not None:
                        if len(images) == 2 and images[0]['height'] * images[0]['width'] <= images[1]['height'] * images[1]['width']:
                            images.pop(0)
                            media.append({"type": "news_image", "url": images[0].get("url")})
                        elif len(images) == 2 and images[0]['height'] * images[0]['width'] >= images[1]['height'] * images[1]['width']:
                            images.pop(1)
                            media.append({"type": "news_image", "url": images[0].get("url")})
                        else:
                            for b in images:
                                media.append({"type": "news_image", "url": images[b].get("url")})
                    if title is not None:
                        titulo_pt=translate_tweet(title,value["metadata"]["lang"])
                    else:
                        titulo_pt=None
                    if description is not None:
                        description_pt=translate_tweet(description,value["metadata"]["lang"])
                    else:
                        description_pt=None
                    if match_tweet is not None:
                        link_tweet = url
                    else:
                        urls.append({"title":title, "titule_pt":titulo_pt, "description":description, "description_pt":description_pt, "url":url})
                else:
                    for f in l.get("media"):
                        if i.get("media_key") == f.get("media_key"):
                            mks.append(f.get("media_key"))
                            tipo=f.get("type")
                            if f.get("url") is not None:
                                url_media=f.get("url")
                            elif i.get("expanded_url") is not None:
                                url_media = i.get("expanded_url")
                            elif i.get("unwound_url") is not None:
                                url_media = i.get("unwound_url")
                            elif i.get("url") is not None:
                                url_media = i.get("url")
                            media.append({"type": tipo, "url": url_media})
        else:
            urls=None
        if l.get("media") is not None:
            for f in l.get("media"):
                if f.get("media_key") not in mks:
                    mks.append(f.get("media_key"))
                    tipo = f.get("type")
                    if f.get("url") is not None:
                        url_media = f.get("url")
                    media.append({"type": tipo, "url": url_media})
        #traducoes e resultados
        result=re.sub(r'&amp;','&',result)
        result = ' '.join(re.findall(r'(?:https?:\/\/|www\.)\S+|(\S+)', result))
        if result is not None and result != "":
            result_portugues=translate_tweet(result,value["metadata"]["lang"])
        else:
            result_portugues=value["result"]
            result=value["result"]
        #pontuacao do tweet
        pontuation_mentions=int(value["metadata"]["pontuation_mentions"])+int(value["metadata"]["public_metrics"]["impression_count"])+int(value["metadata"]["public_metrics"]["like_count"])+int(value["metadata"]["public_metrics"]["quote_count"])+int(value["metadata"]["public_metrics"]["reply_count"])+int(value["metadata"]["public_metrics"]["retweet_count"])
        #localização
        location=value["metadata"]["location"]
        url = f'https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={API_KEY}'
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'OK':
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
            if lat==0 and lng==0:
                country="United Kingdom"
            else:
                country = get_country_from_coords(lat, lng)
        else:
            country = None

        # LIMPEZA DO TWEET
        tweet = result.lower()  # lower Casing

        # ponctuation
        tweet = re.sub(r'&amp;', '&', tweet)
        tweet = re.sub(r'\n', ' ', tweet)
        tweet = re.sub(REGX_USERNAME, ' ', tweet)
        tweet = re.sub(REGX_URL, ' ', tweet)

        # Sentiment analysis
        sentiment_scores = sia.polarity_scores(tweet)

        # Interpret the sentiment scores
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.45:
            sentimento = "Positive"
        elif compound_score <= -0.45:
            sentimento = "Negative"
        else:
            sentimento = "Neutral"

        tokens = [token.text for token in nlp(tweet)]
        tokens = [t for t in tokens if
                  t not in STOP_WORDS and
                  t not in string.punctuation and
                  len(t) > 3]
        tokens = [t for t in tokens if not t.isdigit()]
        lemmatized_tokens = [wn.lemmatize(token) for token in tokens]
        stemmed_tokens = [ps.stem(token) for token in lemmatized_tokens]

        if urls==[]:
            urls=None
        if mencoes==[]:
            mencoes=None
        if contexto==[]:
            contexto=None
        if media==[]:
            media=None
        if link_tweet==None:
            link_tweet = ''.join(["https://twitter.com/t/status/", str(value["id_tweet"])])

        db1[key] = {
            "id_tweet": value["id_tweet"],
            "query": {
                "id": value["query"]["id"],
                "query": value["query"]["query"],
                "main_topic": value["query"]["main_topic"],
                "topic": value["query"]["topic"]
            },
            "result": result,
            "result_pt": result_portugues,
            "link_tweet":link_tweet,
            "question": question,
            "sentiment": sentimento,
            "user": {
                "username": user["username"],
                "name": user["name"],
                "verified": user["verified"],
                "profile_photo": user["profile_photo"]
            },
            "nlp_process":{
                "tokens": tokens,
                "lemmas": lemmatized_tokens,
                "stems": stemmed_tokens
            },
            "metadata": {
                "pontuation": pontuation_mentions,
                "pontuation_mentions":value["metadata"]["pontuation_mentions"],
                "lang":value["metadata"]["lang"],
                "location": country,
                "links":urls,
                "media":media,
                "hashtags":hash,
                "mentions":mencoes,
                "entities":contexto,
                "public_metrics":value["metadata"]["public_metrics"]
            }
        }
    except Exception as e:
        g=2
        print("deu um erro aqui!")

end= time.time()
diferenca= end-start
print(f"Demorei {diferenca} segundos")

file = open("clean_tweets_june.json", "w", encoding='utf-8')
json.dump(db1, file, indent=4, ensure_ascii=False)