import nltk
from nltk.corpus import wordnet
import json
import spacy
import en_core_web_sm
import re
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
