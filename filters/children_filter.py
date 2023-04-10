#!/usr/bin/env python3

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import re
import time
import spacy

#dbFile = open("tweets_new.json", encoding='utf-8')
#tweets_extracao = json.load(dbFile)
tweets_extracao1 = {}
nlp = spacy.load('en_core_web_sm')

a = nlp("CRIME | A woman in Maryland was arrested Monday after police said she screamed obscenities and exposed herself to children trick-or-treating on Halloween night.\nhttps://t.co/edfATSocos")
b = nlp("A woman in Maryland was arrested Monday after police said she screamed obscenities and exposed herself to children trick-or-treating on Halloween night. https://t.co/2Rjui0SICz")
c = a.similarity(b)

print(c)

time.sleep(50)

'''
for key, value in tweets_extracao.items():
    i=re.search(r'(\b(kid|child|girl|boy|baby|teenager|youngster|children|infant|adolescent|juvenile|minor|kiddie|youth)\b)',value["result"])
    q=re.search(r'\?',value["result"])
    if i != None:
        kidsmention=True
    else:
        kidsmention=False
    if q != None:
        question=True
    else:
        question=False
    tweets_extracao[key] = {
        "id_tweet": value.get("id_tweet"),
        "query": {
            "id": value["query"]["id"],
            "query": value["query"]["query"],
            "similarity": value["query"]["similarity"]
        },
        "result": value["result"],
        "verified": value["verified"],
        "question": question,
        "kidsmention": kidsmention,
        "metadata": {
            "iduser": value["metadata"]["iduser"],
            "location": value["metadata"]["location"],
            "media": value["metadata"]["media"],
            "public_metrics": value["metadata"]["public_metrics"]
        }
    }




file=open("tweets_new1.json","w", encoding='utf-8')
json.dump(tweets_extracao1,file,indent=4,ensure_ascii=False)
'''