#!/usr/bin/env python3
import spacy
import json

dbFile = open("C:/Users/diana/PycharmProjects/tese/tweets_new.json", encoding='utf-8')
tweets_extracao = json.load(dbFile)
tweets_extracao1 = {}
nlp = spacy.load('en_core_web_lg')

tweets_pertencentes=[]
matriz_superior=[]
u=0
ids={}

for key, value in tweets_extracao.items():
    ids = {}
    matriz_superior.append(key)
    if key not in tweets_pertencentes:
        numero=1
        ids[value["metadata"]["iduser"]] = {"verified": value["verified"]}
        main_id=key
        count_max=value["metadata"]["public_metrics"]["like_count"]
        if u==0:
            tweets_extracao1[key]={
                    "id_tweet": value.get("id_tweet"),
                    "query": {
                        "id": value["query"]["id"],
                        "query": value["query"]["query"],
                        "similarity": value["query"]["similarity"]
                    },
                    "result": value["result"],
                    "verified": value["verified"],
                    "question": value["question"],
                    "kidsmention": value["kidsmention"],
                    "mentions": {
                        "number": 0,
                        "ids":{}
                    },
                    "metadata": {
                        "iduser": value["metadata"]["iduser"],
                        "location": value["metadata"]["location"],
                        "media": value["metadata"]["media"],
                        "public_metrics": value["metadata"]["public_metrics"]
                    }
            }
            u=1
        else:
            for keys,values in tweets_extracao.items():
                if values["query"]["id"]==value["query"]["id"]:
                    if keys not in matriz_superior:
                        a=nlp(value["result"])
                        b=nlp(values["result"])
                        c = a.similarity(b)
                        if c>0.88:
                            if keys not in tweets_pertencentes:
                                tweets_pertencentes.append(keys)
                            if keys not in matriz_superior:
                                matriz_superior.append(keys)
                            numero+=1
                            ids[values["metadata"]["iduser"]] = {"verified": values["verified"]}
                            if count_max<values["metadata"]["public_metrics"]["like_count"]:
                                main_id=keys
                                count_max=values["metadata"]["public_metrics"]["like_count"]
                                ids[values["metadata"]["iduser"]] = {"verified": values["verified"]}
                            else:
                                ids[values["metadata"]["iduser"]] = {"verified": values["verified"]}
            if numero==1:
                numero=0
                ids={}
            else:
                del ids[tweets_extracao[main_id]["metadata"]["iduser"]]
                numero-=1
            tweets_extracao1[main_id] = {
                "id_tweet": tweets_extracao[main_id]["id_tweet"],
                "query": {
                    "id": tweets_extracao[main_id]["query"]["id"],
                    "query": tweets_extracao[main_id]["query"]["query"],
                    "similarity": tweets_extracao[main_id]["query"]["similarity"]
                },
                "result": tweets_extracao[main_id]["result"],
                "verified": tweets_extracao[main_id]["verified"],
                "question": tweets_extracao[main_id]["question"],
                "kidsmention": tweets_extracao[main_id]["kidsmention"],
                "mentions": {
                    "number": numero,
                    "ids": ids
                },
                "metadata": {
                    "iduser": tweets_extracao[main_id]["metadata"]["iduser"],
                    "location": tweets_extracao[main_id]["metadata"]["location"],
                    "media": tweets_extracao[main_id]["metadata"]["media"],
                    "public_metrics": tweets_extracao[main_id]["metadata"]["public_metrics"]
                }
            }
            print(tweets_extracao1)


file=open("tweets_new1.json","w", encoding='utf-8')
json.dump(tweets_extracao1,file,indent=4,ensure_ascii=False)
