#!/usr/bin/env python3

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import re
from datetime import datetime, timedelta

#----------------------------------CONNECTION TO TWITTER API--------------------------------------------------------------------------

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAO4YcgEAAAAAcOOrgk%2BiM1ZW%2BnbcK7AjReTucFk%3DaJpjnqXa50nbsA3O4YDDmT10Zav1JGTJfNJyMBmxGVevze3gML'
search_url = "https://api.twitter.com/2/tweets/search/all?"

def bearer_oauth(r):
    """Attaches the Bearer token to the request header."""
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    """Handles connection to the Twitter API with retry logic for robustness."""
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    
    max_retries = 90
    retry_count = 0
    while True:
        response = session.get(url, auth=bearer_oauth, params=params)
        if response.status_code == 200:
            break
        elif retry_count >= max_retries:
            time.sleep(90)
        else:
            retry_count += 1
            time.sleep(5)
    return response.json()

def process_tweet_data(data, tweets_extracao, ids):
    """Processes tweet data, extracting relevant information and updating the tweets_extracao dictionary."""
    info_media = []

    if data["meta"]["result_count"] > 0:
        for i in data["data"]:
            main_key = i["id"]
            if main_key not in ids:
                ids.append(main_key)
                process_media(i, data, info_media)
                k, user_info = process_user(i, data)
                process_metadata(i, k, data, info_media, tweets_extracao, user_info)
                info_media = []

def process_media(i, data, info_media):
    """Processes media attachments in the tweet."""
    if i.get("attachments") is not None:
        m = i.get("attachments").get("media_keys")
        if m:
            for a in m:
                for media in data["includes"]["media"]:
                    if media["media_key"] == a:
                        info_media.append(media)
                        break
        else:
            info_media = None
    else:
        info_media = None

def process_user(i, data):
    """Extracts and returns user information from the tweet."""
    k = -1
    user_info = {}
    for p in range(len(data["includes"]["users"])):
        if data["includes"]["users"][p]["id"] == i["author_id"]:
            k = p
            user_info["username"] = data["includes"]["users"][p]["username"]
            user_info["name"] = data["includes"]["users"][p]["name"]
            user_info["profile_photo"] = data["includes"]["users"][p]["profile_image_url"]
            user_info["verified"] = data["includes"]["users"][p]["verified"]
            break
    return k, user_info

def process_metadata(i, k, data, info_media, tweets_extracao, user_info):
    """Processes and updates metadata for the tweet."""
    pontuacao = 20 if data["includes"]["users"][k]["verified"] else 1
    local = data["includes"]["users"][k].get("location")
    context = i.get("entities", {}).get("annotations")
    h = i.get("entities", {}).get("urls")

    id = extract_tweet_id(i["text"])
    update_extraction_data(i, id, pontuacao, local, context, h, info_media, tweets_extracao, user_info)

def extract_tweet_id(text):
    """Extracts and returns the tweet ID from the text."""
    n_original = re.sub(r'\n', r' ', text)
    n = re.search(r'^(.+?)(?=\s+https?://)', n_original)
    return n.group(1) if n else n_original

def update_extraction_data(i, id, pontuacao, local, context, h, info_media, tweets_extracao, user_info):
    """Updates the tweet extraction data."""
    if id in tweets_extracao:
        users = tweets_extracao[id]["user"]["username"]
        update_existing_tweet(i, id, pontuacao, tweets_extracao, user_info, users)
    else:
        create_new_tweet(i, id, pontuacao, local, context, h, info_media, tweets_extracao, user_info)

def update_existing_tweet(i, id, pontuacao, tweets_extracao, user_info, users):
    """Updates an existing tweet's data."""
    if isinstance(users, list):
        if user_info["username"] not in users:
            users.append(user_info["username"])
            update_pontuation(i, id, pontuacao, tweets_extracao, user_info)
    elif users != user_info["username"]:
        users = [tweets_extracao[id]["user"]["username"], user_info["username"]]
        update_pontuation(i, id, pontuacao, tweets_extracao, user_info)

    tweets_extracao[id]["user"]["username"] = users
    update_public_metrics(i, id, tweets_extracao)

def update_pontuation(i, id, pontuacao, tweets_extracao, user_info):
    """Updates the pontuation for mentions."""
    if user_info["verified"]:
        tweets_extracao[id]["metadata"]["pontuation_mentions"] += 20
    else:
        tweets_extracao[id]["metadata"]["pontuation_mentions"] += 1

def update_public_metrics(i, id, tweets_extracao):
    """Updates the public metrics for the tweet."""
    metrics = tweets_extracao[id]["metadata"]["public_metrics"]
    metrics["impression_count"] += i["public_metrics"]["impression_count"]
    metrics["like_count"] += i["public_metrics"]["like_count"]
    metrics["quote_count"] += i["public_metrics"]["quote_count"]
    metrics["reply_count"] += i["public_metrics"]["reply_count"]
    metrics["retweet_count"] += i["public_metrics"]["retweet_count"]

def create_new_tweet(i, id, pontuacao, local, context, h, info_media, tweets_extracao, user_info):
    """Creates a new tweet entry in the extraction data."""
    tweets_extracao[id] = {
        "id_tweet": i["id"],
        "query": {
            "id": key["queryID"],
            "query": key["query"],
            "main_topic": key["main_topic"],
            "topic": theme
        },
        "result": i["text"],
        "user": user_info,
        "metadata": {
            "pontuation_mentions": pontuacao,
            "lang": i["lang"],
            "location": local,
            "links": h,
            "media": info_media,
            "context": context,
            "public_metrics": i["public_metrics"]
        }
    }

def handle_rate_limiting(start, count):
    """Handles rate limiting based on API usage."""
    end = time.time()
    diferenca = end - start
    if count % 300 == 0 and diferenca < 900:
        time.sleep(915 - diferenca)
        start = time.time()
    elif diferenca > 900:
        f = count % 300
        time.sleep(f * 3)
        start = time.time()
    return start

def main():
    tweets_extracao = {}  # Dictionary to store extracted tweets
    ids = []  # To control the IDs
    start = time.time()  # To control Twitter API limits

    # Get the current date and time
    now = datetime.utcnow()
    start_time = (now - timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0).isoformat() + "Z"
    end_time = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"

    with open('C:/Users/diana/PycharmProjects/thesis/queries/queries.json', 'r') as doc:
        db = json.load(doc)

    for key in db:
        theme = re.findall(r"\"(.*?)\"", key["query"])[0]
        query_params = {
            'query': key["query"],
            'start_time': start_time,
            "end_time": end_time,
            "max_results": 500,
            "tweet.fields": "author_id,created_at,geo,public_metrics,attachments,entities,lang",
            "expansions": "author_id,geo.place_id,attachments.media_keys,referenced_tweets.id.author_id",
            "media.fields": "url",
            "user.fields": "username,name,location,verified,profile_image_url"
        }
        json_response = connect_to_endpoint(search_url, query_params)
        process_tweet_data(json_response, tweets_extracao, ids)

        if json_response["meta"].get("next_token"):
            while json_response["meta"].get("next_token"):
                query_params["next_token"] = json_response["meta"]["next_token"]
                json_response = connect_to_endpoint(search_url, query_params)
                process_tweet_data(json_response, tweets_extracao, ids)
                start = handle_rate_limiting(start, count)

    with open("../tweets_june.json", "w", encoding='utf-8') as
