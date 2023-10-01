import gensim
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from pysentimiento.preprocessing import preprocess_tweet
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download the NLTK stop words dataset
stop_words = set(stopwords.words('english'))

import gensim
from gensim import corpora, models

# Sample list of preprocessed tweets
preprocessed_tweets = [
    "in britain, children sent schools 75% calories lunch ultra-processed foods",
    "physical activity crucial children health well-being regular exercise helps build strong bones reduces risk obesity diabetes cardiovascular disease fitkids healthybodies",
    "summer fun begun gl homes arden residents loving fun events planned incredible clubhouse like dancing acting classes youth summer soccer season sports amp fitness classes daily food trucks summer camp call 800-910-3402 glhomes",
    "saturday's summer garden party entertainment food amp drinks children amp adults 12 rcm jazz trio 12.45 action amanda 1.30 jazz trio return 2.15 silva ace fitness trigga street dance 3pm ends bbq cakes pimms wine amp beer free entrance",
    "2 air india fliers stranded russia foodselderly superfood seniorfitness video stranded passengers air india flight including children elderly found battling language barriers unfamiliar food substandard accommodation",
    "usa store gold standard 100 whey protein powder",
    "debbie smith 63 years young 4 sets dips 10 reps looking fountain youth gym working active find gym workout exercise health strong longislandgym portwashington newyork longisland fitness",
    "college boys back summer empire blessed godisgood sports speed speedandagility strength strengthandconditioning run florida training exercise fitness workout brevardcounty spacecoast youth kids gym summer college"
]

# Tokenize and create a dictionary
tokenized_tweets = [tweet.split() for tweet in preprocessed_tweets]
print(tokenized_tweets)
dictionary = corpora.Dictionary(tokenized_tweets)

# Create a Document-Term Matrix (DTM)
dtm = [dictionary.doc2bow(tokens) for tokens in tokenized_tweets]

# Define the number of topics
num_topics = 1  # You can adjust this based on your dataset

# Apply LDA
lda_model = gensim.models.LdaModel(dtm, num_topics=num_topics, id2word=dictionary, passes=15)

# Print the top words for each topic
for topic_id, topic_words in lda_model.print_topics(num_words=25):
    print(f"Topic {topic_id + 1}: {topic_words}")

# Initialize a list to store top keywords for each topic
top_keywords_per_topic = []

# Extract the top keywords for each topic
for topic_id, topic_words in lda_model.print_topics(num_words=25):
    keywords = [word.split("*")[1].strip('"') for word in topic_words.split(" + ")]
    top_keywords_per_topic.append(keywords)

print(top_keywords_per_topic)

# Create a word cloud for each topic's keywords
for topic_id, keywords in enumerate(top_keywords_per_topic):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))

    # Plot the word cloud
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Topic {topic_id + 1} Keywords Word Cloud")
    plt.axis("off")
    plt.show()

from collections import Counter

# Combine all tokenized tweets into a single list of tokens
all_tokens = [token for tweet_tokens in tokenized_tweets for token in tweet_tokens]

# Count the frequency of each token
token_counts = Counter(all_tokens)

# Get the most common tokens
most_common_tokens = token_counts.most_common(10)  # You can adjust the number as needed

# Print the most common tokens and their frequencies
for token, count in most_common_tokens:
    print(f"{token}: {count}")