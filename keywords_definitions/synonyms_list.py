import nltk
from nltk.corpus import wordnet
import json
import spacy
import en_core_web_sm
import re

#Opening json file to save all the new keys related to the different topics
'''doc=open('synonyms.json', 'w')'''
similaridade= open('venv/similarity.txt', 'w')
abovefiftyrelated= open('abovefiftyrelated.txt','w')
belowfiftyrelated= open('belowfiftyrelated.txt','w')
abovefifty= open('abovefifty.txt','w')
belowfifty= open('belowfifty.txt','w')

#Declaration of variables
totalkeys=[]
synonyms=[]
docs={}
similarity=[]

keys=["vulnerability", "habits", "lifestyle", "diets", "food", "healthy_food", "nutrition", "economical_resources", "physical_exercise", "sleep_quality",
	  "sexual_education", "NCDs", "diseases", "activities", "computer", "smartphone", "TV", "health", "noncommunicable diseases",
	  "children", "fitness", "chronic diseases", "cardiovascular diseases", "cancer", "diabetes", "chronic respiratory diseases"]

# Search for synonyms for each topic
for k in keys:
	synonyms.append(k)
	totalkeys.append(k)
	for syn in wordnet.synsets(k):
		for l in syn.lemmas():
			if l.name() not in synonyms and l.name not in totalkeys:
				synonyms.append(l.name())
				totalkeys.append(l.name())
	docs[k]=synonyms
	synonyms=[]

#json.dump(docs, doc, ensure_ascii=False, indent=4)