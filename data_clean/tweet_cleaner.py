#!/usr/bin/env python3

# Imports
import spacy
import langid
from googletrans import Translator
import requests
import json
import re
import time
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS

# Constants
API_KEY = 'AIzaSyBRh6hjazHm2lLbjwnq2imcQh61jtEb3J0'
REGX_USERNAME = r"@[A-Za-z0-9$-_@.&+]+"
REGX_URL = r"https?://[A-Za-z0-9./]+"

# Initialize NLP tools
nlp = spacy.blank("en")
ps = PorterStemmer()
wn = WordNetLemmatizer()
translator = Translator()

# Custom exception for translation errors
class TranslationError(Exception):
    pass

def translate_tweet(tweet, language=None):
    """Translate a tweet to Portuguese if it's not already in Portuguese."""
    if language is None:
        language = langid.classify(tweet)[0]
    
    if language != 'pt':
        try:
            translation = translator.translate(tweet, dest='pt').text
        except Exception:
            translation = tweet
        # Adjust spacing around punctuation
        translation = re.sub(r"([.,!?;])(\w)", r'\1 \2', translation)
        translation = re.sub(r" ([.,!?;])", r'\1', translation)
        return translation
    return tweet

def get_country_from_coords(lat, lng):
    """Retrieve the country name from latitude and longitude coordinates."""
    url = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={API_KEY}'
    response = requests.get(url)
    data = response.json()
    for component in data['results'][0]['address_components']:
        if 'country' in component['types']:
            return component['long_name']
    return None

def process_tweet(key, value):
    """Process individual tweet and extract required information."""
    try:
        result = value["result"]
        
        # Extract links, hashtags, and mentions
        link_tweet = re.findall(r'(?:https?:\/\/|www\.)\S+', result)
        link_tweet = link_tweet[0] if link_tweet else None
        question = bool(re.search(r'\?', result))
        hash_tags = re.findall(r'\#\w+', result) or None
        mentions = re.findall(r'@\w+', result) or None

        # Context and media processing
        metadata = value.get("metadata", {})
        user = value.get("user", {})
        context_info = metadata.get("context", [])
        context_entities = [{"type": i.get("type"), "text": i.get("normalized_text"), "prob": i.get("probability")} for i in context_info] or None

        # Process links and media
        urls, media, mks = [], [], []
        links = metadata.get("links", [])
        for link in links:
            title, description = link.get("title"), link.get("description")
            url = link.get("expanded_url") or link.get("unwound_url") or link.get("url")
            images = link.get("images")
            if images:
                images.sort(key=lambda img: img['height'] * img['width'], reverse=True)
                media.append({"type": "news_image", "url": images[0].get("url")})
            if url and not re.match(r'https\:\/\/twitter\.com\/', url):
                urls.append({
                    "title": title,
                    "titule_pt": translate_tweet(title, metadata.get("lang")) if title else None,
                    "description": description,
                    "description_pt": translate_tweet(description, metadata.get("lang")) if description else None,
                    "url": url
                })
            for f in metadata.get("media", []):
                if link.get("media_key") == f.get("media_key"):
                    mks.append(f.get("media_key"))
                    media.append({"type": f.get("type"), "url": f.get("url")})

        # Translation and cleaning
        result = re.sub(r'&amp;', '&', result)
        result = ' '.join(re.findall(r'(?:https?:\/\/|www\.)\S+|(\S+)', result))
        result_portugues = translate_tweet(result, metadata.get("lang"))

        # NLP Processing
        tokens = [token.text for token in nlp(result.lower())]
        tokens = [t for t in tokens if t not in STOP_WORDS and t not in string.punctuation and len(t) > 3 and not t.isdigit()]
        lemmatized_tokens = [wn.lemmatize(token) for token in tokens]
        stemmed_tokens = [ps.stem(token) for token in lemmatized_tokens]

        # Location processing
        location = metadata.get("location")
        if location:
            url = f'https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={API_KEY}'
            response = requests.get(url)
            data = response.json()
            if data['status'] == 'OK':
                lat = data['results'][0]['geometry']['location']['lat']
                lng = data['results'][0]['geometry']['location']['lng']
                country = get_country_from_coords(lat, lng) if (lat != 0 and lng != 0) else "United Kingdom"
            else:
                country = None

        # Compile processed data
        return {
            "id_tweet": value["id_tweet"],
            "query": value["query"],
            "result": result,
            "result_pt": result_portugues,
            "link_tweet": link_tweet or ''.join(["https://twitter.com/t/status/", str(value["id_tweet"])]),
            "question": question,
            "sentiment": None,
            "user": {
                "username": user.get("username"),
                "name": user.get("name"),
                "verified": user.get("verified"),
                "profile_photo": user.get("profile_photo")
            },
            "nlp_process": {
                "tokens": tokens,
                "lemmas": lemmatized_tokens,
                "stems": stemmed_tokens
            },
            "metadata": {
                "pontuation": sum(int(value["metadata"].get(key, 0)) for key in ["pontuation_mentions", "public_metrics"]["impression_count", "public_metrics"]["like_count", "public_metrics"]["quote_count", "public_metrics"]["reply_count", "public_metrics"]["retweet_count"]),
                "pontuation_mentions": value["metadata"]["pontuation_mentions"],
                "lang": value["metadata"]["lang"],
                "location": country,
                "links": urls or None,
                "media": media or None,
                "hashtags": hash_tags,
                "mentions": mentions,
                "entities": context_entities,
                "public_metrics": value["metadata"]["public_metrics"]
            }
        }
    except Exception as e:
        print(f"Error processing tweet {key}: {e}")
        return None

def main():
    # Load tweets data
    with open('C:/Users/diana/PycharmProjects/thesis/tweets_june.json', encoding='utf-8') as doc:
        db = json.load(doc)

    # Initialize the processed tweets dictionary
    db1 = {}
    
    for v, (key, value) in enumerate(db.items()):
        print(f"Processing tweet {v}...")
        processed_data = process_tweet(key, value)
        if processed_data:
            db1[key] = processed_data

    # Save processed tweets to a file
    with open("clean_tweets_june.json", "w", encoding='utf-8') as file:
        json.dump(db1, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
