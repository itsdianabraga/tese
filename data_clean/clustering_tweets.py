# Import necessary libraries
import numpy as np
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

def cluster_text(dist_text, processedTweets, dist_threshold_txt):
    cluster_id = 1
    for i in range(dist_text.shape[0]):
        try:
            doc_dist_array_txt = dist_text[i,]
            # Identify document ids where the tweets are similar
            similarDocs = np.where(doc_dist_array_txt <= dist_threshold_txt)[0]
            processedTweets.loc[processedTweets.index.isin(similarDocs) & processedTweets['cluster_ref'].isna(), 'cluster_ref'] = cluster_id
            cluster_id = cluster_id + 1
        except ValueError:
            continue

    return processedTweets




# %%%%%%%%% Calculate Cosine distance of documents %%%%%%%%%%%%%%

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def get_clusters(processedTweets, similarity_threshold):
       tweets = processedTweets['text'].tolist()

       #Define count vectorizer parameters
       vectorizer =  CountVectorizer(max_df=1.0, min_df=1, stop_words=stopwords, tokenizer=tokenize_only)

       # Get document term frequency matix
       dtf_matrix = vectorizer.fit_transform(tweets) #fit the vectorizer to tweets which does not contain'htttp' and 'RT'
       dist_text = np.around(abs(1 - cosine_similarity(dtf_matrix)),2)

# --------1D clustering based on distance measure
       # Pre clustering setup
       processedTweets['cluster_ref'] = None # Tweets that are 1-similarithy% similar, as defined by dist_threshold
       # Clustering of tweets based on text
       processedTweets = cluster_text(dist_text,processedTweets,dist_threshold_txt = (1-similarity_threshold))

       return processedTweets

# Define your list of stopwords
stopwords = stopwords.words('english')

# Define your list of tweets
tweets = ["my cat is so fat", "my cat is really fat, uau", "my mother is so sad"]

# Define a similarity threshold (e.g., 0.85)
similarity_threshold = 0.70

# Call the get_clusters function
processed_tweets = get_clusters(pd.DataFrame({'text': tweets}), similarity_threshold)

# Access the clustered tweets
clustered_tweets = processed_tweets[processed_tweets['cluster_ref'].notnull()]

print(clustered_tweets)

# Count the number of unique cluster IDs
num_clusters = processed_tweets['cluster_ref'].nunique()

# Print the number of unique cluster IDs
print(f"Number of unique cluster IDs: {num_clusters}")

