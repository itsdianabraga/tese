#!/usr/bin/env python3

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import spacy
import re
import csv


#----------------------------------CONNECTION TO TWITTER API--------------------------------------------------------------------------

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFdCiQEAAAAASlcblnjSnYEbgMIzB0Kd4SxDm6c%3D1eLJSoSTYavew0Ye0ZGMmvvrfC8SOvycnxRZvjImZtQjbgXaeX'

search_url = "https://api.twitter.com/2/tweets/counts/all?"

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    response = session.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

#####################################################################################################


header = ['queryID', 'query', 'similarity'] #Heahter de Database Count

#Opening json file to save all the new keys related to the different topics
similaridade= open('../keywords_definitions/similarity.txt', 'w')
doc=open('../keywords_definitions/synonyms.json', 'r')
db = json.load(doc)

#------------------------------------------------------------------------------
# SIMILARITY

nlp = spacy.load('en_core_web_lg')

totalkeys=[]

for key in db.keys():
	for i in db[key]:
		totalkeys.append(i)

#print(" ", *totalkeys, sep = ", ", file=similaridade)
print(len(totalkeys))
similarity=[]
list=[]
matriz_superior=[]
dbcount=[]
u=0
id=1
#------------------------------------------------------------------------------
# QUERIES CRIATION
'''
for j in totalkeys:
	j = re.sub(r'_', r' ', j)
	query_params = {'query': "\"" + j + "\" -is:retweet -is:quote -is:reply",
					'start_time': "2022-10-08T00:00:00Z", "end_time": "2022-11-08T00:00:00Z"}
	dbcount.append([id, query_params["query"], None, data["meta"]["total_tweet_count"]])
	id += 1
'''
for j in totalkeys:
	for i in totalkeys:
		u=0
		if i not in matriz_superior: #Each combination is calculated only once
			for key in db.keys():
				if j in db[key] and i in db[key]:
					p=re.sub(r'_', r' ', j)
					k=re.sub(r'_', r' ', i)
					a = nlp(p)
					b = nlp(k)
					c = a.similarity(b)
					if c < 0:
						c = 0
					list.append(c) #Similarity matrix
					u=1
			if u!=1:
				p = re.sub(r'_', r' ', j)
				k = re.sub(r'_', r' ', i)
				a=nlp(p)
				b=nlp(k)
				c=a.similarity(b)
				if c<0:
					c=0
				list.append(c)
				if id%298==0 and id!=0:
					print(dbcount)
					#time.sleep(900)
				query_params = {'query': "\""+p+"\""+" \""+k+"\" -is:retweet -is:quote -is:reply", 'start_time': "2022-10-08T00:00:00Z", "end_time": "2022-11-08T00:00:00Z"}
				'''
				json_response = connect_to_endpoint(search_url, query_params)
				data=json.dumps(json_response, indent=4, sort_keys=True)
				data=json.loads(data)
				'''
				dbcount.append([id,query_params["query"],c])
				id += 1
		else:
			list.append(0)
	similarity.append(list)
	print(j, *list, sep=", ", file=similaridade)
	list=[]
	matriz_superior.append(j)

print(similarity)


with open('collectioncountunic.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write multiple rows
    writer.writerows(dbcount)