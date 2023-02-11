#!/usr/bin/env python3
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
from builtins import format


#----------------------------------CONNECTION TO TWITTER API--------------------------------------------------------------------------

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFdCiQEAAAAASlcblnjSnYEbgMIzB0Kd4SxDm6c%3D1eLJSoSTYavew0Ye0ZGMmvvrfC8SOvycnxRZvjImZtQjbgXaeX'

search_url = "https://api.twitter.com/2/tweets/{tweet_id}"

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    max_retries = 5
    retry_count = 0
    while True:
        response = session.get(url, auth=bearer_oauth, params=params)
        if response.status_code == 200:
            break
        elif retry_count == max_retries:
            raise Exception(response.status_code, response.text)
        else:
            retry_count += 1
            time.sleep(5)
    return response.json()

#####################################################################################################


doc =open('C:/Users/diana/PycharmProjects/thesis/tweets_db.json', encoding='utf-8')
db = json.load(doc)
db1= {}
db2={}

count=0
start = time.time()

for key, value in db.items():
    print(key)
    tweet_id=key
    surl = search_url.format(tweet_id=tweet_id)
    query_params = {"tweet.fields": "entities"}
    count+=1
    end = time.time()
    diferenca = end - start
    if count % 300 == 0 and diferenca < 900:
        print("To much requests, time to sleep")
        time.sleep(900 - diferenca)
        start = time.time()
    elif count % 300 == 0 and diferenca >=900:
        time.sleep(5)
        start = time.time()
    json_response = connect_to_endpoint(surl, query_params)
    data = json.dumps(json_response, indent=4, sort_keys=True)
    data = json.loads(data)
    if data.get("data") is not None:
        if data["data"].get("entities") is not None:
            tags=[]
            contexto=[]
            mencoes=[]
            x=data["data"]["entities"]
            hashtags=x.get("hashtags")
            if hashtags is not None:
                for i in hashtags:
                    tags.append(i.get("tag"))
            else:
                tags=None
            context=x.get("annotations")
            if context is not None:
                for i in context:
                    contexto.append({"type": i.get("type"), "text": i.get("normalized_text"), "prob": i.get("probability")})
            else:
                contexto=None
            mentions=x.get("mentions")
            if mentions is not None:
                for i in mentions:
                    mencoes.append({"id": i.get("id"), "username": i.get("username")})
            else:
                mencoes=None
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
                    images = i.get("images")
                    if images is not None:
                        media.append({"type": "news_image", "url": images[0].get("url")})
                    urls.append({"title":title, "description":description, "url":url})
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
        else:
            media=None
        db1[key] = {
            "id_tweet": value.get("id_tweet"),
            "query": {
                "id": value["query"]["id"],
                "query": value["query"]["query"]
            },
            "result": value["result"],
            "verified": value["verified"],
            "metadata": {
                "iduser": value["metadata"]["iduser"],
                "lang": value["metadata"]["lang"],
                "location": value["metadata"]["location"],
                "links":urls,
                "media":media,
                "hashtags": tags,
                "mentions": mencoes,
                "entities": contexto,
                "public_metrics": value["metadata"]["public_metrics"]
            }
        }
    else:
        db2[key] = value
    file = open("tweets_db_clean.json", "w", encoding='utf-8')
    file2 = open("tweets_perdidos.json", "w", encoding='utf-8')
    json.dump(db1, file, indent=4, ensure_ascii=False)
    json.dump(db2, file2, indent=4, ensure_ascii=False)




print("ja ta")



