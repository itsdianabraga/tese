import requests
from googletrans import Translator

#CODIGO GEOCODE

# Replace YOUR_API_KEY with your actual Google Maps API key
API_KEY = 'AIzaSyBRh6hjazHm2lLbjwnq2imcQh61jtEb3J0'

locations = [
    "Portage, Wis.",
    "Baraboo, WI",
    "Beaver Dam, Wisconsin",
    "Cheshire East"
]

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

for location in locations:
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={API_KEY}'
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        lat = data['results'][0]['geometry']['location']['lat']
        lng = data['results'][0]['geometry']['location']['lng']
        country = get_country_from_coords(lat, lng)
        if country:
            print(location, '=>', country)
    else:
        print(location, '=>', None)


# CODIGO DA TRADUÇÃO
import langid
from googletrans import Translator
import re

def translate_tweet(tweet):
    # Detect the language of the tweet
    language = langid.classify(tweet)[0]
    print(language)

    # Translate the tweet to Portuguese if it's not already in Portuguese
    if language != 'pt':
        translator = Translator()
        translation = translator.translate(tweet, dest='pt').text
        translation = re.sub(r"([.,!?;])(\w)",r'\1 \2',translation)
        translation = re.sub(r" ([.,!?;])", r'\1', translation)
        return translation
    else:
        return tweet

# Example usage
text= "#GodMorningMonday\nWhen a person, who consumes tobacco, emits smoke on smoking hookah, beedi or cigarette, then that smoke entering into the bodies of his small children causes damage. \nFor more information watch 👀 youtube ▶️ channel \nSant Rampal ji maharaj https://t.co/llaMo0bIZj"
translated_text = translate_tweet(text)
print(f"Original text: {text}")
print(f"Translated text: {translated_text}")
