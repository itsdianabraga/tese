import nltk
from nltk.corpus import wordnet
import json
import spacy
import en_core_web_sm
import re
'''
#Opening json file to save all the new keys related to the different topics
doc=open('synonyms.json', 'w')
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

keys=["vulnerability", "habits", "lifestyle", "diet", "food", "healthy_food", "nutrition", "economical_resources", "physical_exercise", "sleep_quality","NCDs", "activities", "computer", "smartphone", "TV", "noncommunicable_diseases", "fitness", "chronic_diseases", "cardiovascular_diseases", "cancer", "diabetes", "chronic_respiratory_diseases"]

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
'''
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# List of words
words = ["vulnerability", "habits", "lifestyle", "diets", "food", "healthy_food", "nutrition",
         "economical_resources", "physical_exercise", "sleep_quality", "sexual_education", "NCDs",
         "diseases", "activities", "computer", "smartphone", "TV", "health", "noncommunicable diseases",
         "children", "fitness", "chronic diseases", "cardiovascular diseases", "cancer", "diabetes",
         "chronic respiratory diseases"]

# Combine the words into a single string
text = " ".join(words)

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

# Display the generated word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Turn off the axis
plt.show()
