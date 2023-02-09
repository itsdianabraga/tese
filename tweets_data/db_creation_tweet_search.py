#!/usr/bin/env python3
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time


#----------------------------------CONNECTION TO TWITTER API--------------------------------------------------------------------------

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFdCiQEAAAAASlcblnjSnYEbgMIzB0Kd4SxDm6c%3D1eLJSoSTYavew0Ye0ZGMmvvrfC8SOvycnxRZvjImZtQjbgXaeX'

search_url = "https://api.twitter.com/2/tweets/search/all?"

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params, tweets_extracao):
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


#tweets= open('tweetsv1.json','w',encoding='utf-8')
doc=open('/queries/queries_kids.json', 'r')
db = json.load(doc)

#dbFile = open("tweets_dec.json", encoding='utf-8')
#tweets_extracao = json.load(dbFile)

tweets_extracao= {}
ids=[]
info_media= []
next_token_true=0
count=0
start = time.time()

for key in db:
    query_params = {'query': key["query"],
                    'start_time':"2023-01-01T00:00:00Z",
                    "end_time":"2023-01-31T00:00:00Z",
                    "max_results":500,
                    "tweet.fields":"author_id,created_at,geo,public_metrics,attachments,entities,lang",
                    "expansions":"author_id,geo.place_id,attachments.media_keys",
                    "media.fields":"url",
                    "user.fields":"name,location,verified"}
    print(key["queryID"])
    json_response = connect_to_endpoint(search_url, query_params, tweets_extracao)
    data = json.dumps(json_response, indent=4, sort_keys=True)
    data = json.loads(data)
    count+=1
    k=-1
    time.sleep(2)
    if data["meta"]["result_count"]>0:
        for i in data["data"]:
            main_key = i["id"]
            if tweets_extracao.get(main_key) == None:
                ids.append(i["id"])
                if i.get("attachments") is not None:
                    m = i.get("attachments")
                    m = m.get("media_keys")
                    if m is not None:
                        if len(m) == 1:
                            media_key = m[0]
                            for l in range(len(data["includes"]["media"])):
                                if data["includes"]["media"][l]["media_key"] == media_key:
                                    info_media.append(data["includes"]["media"][l])
                                    break
                        else:
                            for a in m:
                                media_key = a
                                for l in range(len(data["includes"]["media"])):
                                    if data["includes"]["media"][l]["media_key"] == media_key:
                                        info_media.append(data["includes"]["media"][l])
                                        break
                    else:
                        info_media = None
                else:
                    info_media = None
                for p in range(len(data["includes"]["users"])):
                    if data["includes"]["users"][p]["id"]==i["author_id"]:
                        k=p
                        break
                if data["includes"]["users"][k].get("location") is not None:
                    local=data["includes"]["users"][k].get("location")
                else:
                    local=None
                if i.get("entities") is not None:
                    h= i.get("entities")
                    context=h.get("annotations")
                    h= h.get("urls")
                else:
                    h=None
                    context=None
                tweets_extracao[main_key]={
                        "id_tweet": i["id"],
                        "query": {
                            "id":key["queryID"],
                            "query": key["query"]
                        },
                        "result":i["text"],
                        "verified":data["includes"]["users"][k]["verified"],
                        "metadata": {
                            "iduser":i["author_id"],
                            "lang":i["lang"],
                            "location":local,
                            "links": h,
                            "media":info_media,
                            "context":context,
                            "public_metrics":i["public_metrics"]
                            }
                    }
                procura_media=[]
                info_media=[]
        if data["meta"].get("next_token") is not None:
            next_t=data["meta"].get("next_token")
            next_token_true=1
        else:
            next_token_true=0
    end = time.time()
    diferenca = end - start
    if count % 300 == 0 and diferenca < 900:
        time.sleep(915 - diferenca)
        start = time.time()
    elif diferenca > 900:
        f = count % 300
        time.sleep(f * 3)
        start = time.time()
        print("wi")
    while next_token_true==1:
        query_params = {'query': key["query"],
                        'start_time':"2023-01-01T00:00:00Z",
                        "end_time":"2023-01-31T00:00:00Z",
                        "max_results": 500,
                        "next_token": next_t,
                        "tweet.fields": "author_id,created_at,geo,public_metrics,attachments,entities,lang",
                        "expansions": "author_id,geo.place_id,attachments.media_keys",
                        "media.fields": "url",
                        "user.fields": "name,location,verified"}
        json_response = connect_to_endpoint(search_url, query_params, tweets_extracao)
        data = json.dumps(json_response, indent=4, sort_keys=True)
        data = json.loads(data)
        count+=1
        k = 0
        time.sleep(2)
        for i in data["data"]:
            if tweets_extracao.get(main_key) == None:
                ids.append(i["id"])
                if i.get("attachments") is not None:
                    m = i.get("attachments")
                    m = m.get("media_keys")
                    if m is not None:
                        if len(m) == 1:
                            media_key = m[0]
                            for l in range(len(data["includes"]["media"])):
                                if data["includes"]["media"][l]["media_key"] == media_key:
                                    info_media.append(data["includes"]["media"][l])
                                    break
                        else:
                            for a in m:
                                media_key = a
                                for l in range(len(data["includes"]["media"])):
                                    if data["includes"]["media"][l]["media_key"] == media_key:
                                        info_media.append(data["includes"]["media"][l])
                                        break
                    else:
                        info_media = None
                else:
                    info_media = None
                for p in range(len(data["includes"]["users"])):
                    if data["includes"]["users"][p]["id"] == i["author_id"]:
                        k = p
                        break
                if data["includes"]["users"][k].get("location") is not None:
                    local = data["includes"]["users"][k].get("location")
                else:
                    local = None
                main_key = i["id"]
                if i.get("entities") is not None:
                    h = i.get("entities")
                    context = h.get("annotations")
                    h = h.get("urls")
                else:
                    h = None
                    context = None
                tweets_extracao[main_key] = {
                    "id_tweet": i["id"],
                    "query": {
                        "id": key["queryID"],
                        "query": key["query"]
                    },
                    "result": i["text"],
                    "verified": data["includes"]["users"][k]["verified"],
                    "metadata": {
                        "iduser": i["author_id"],
                        "lang": i["lang"],
                        "location": local,
                        "links": h,
                        "media": info_media,
                        "public_metrics": i["public_metrics"]
                    }
                }
                info_media = []
        if data["meta"].get("next_token") is not None:
            next_t=data["meta"].get("next_token")
            next_token_true=1
        else:
            next_token_true=0
        end = time.time()
        diferenca = end - start
        if count % 300 == 0 and diferenca<900:
            time.sleep(915 - diferenca)
            start=time.time()
        elif diferenca > 900:
            f = count % 300
            time.sleep(f*3)
            start=time.time()
        '''
        if count % 298 == 0:
            end = time.time()
            diferenca = end - start
            print("vou descansar")
            if diferenca > 900:
                time.sleep(15)
            else:
                time.sleep(915 - diferenca)
            start = time.time()
        '''

ficheiro = open("../tweets_db.json", "w", encoding='utf-8')
json.dump(tweets_extracao, ficheiro, ensure_ascii=False, indent=4)