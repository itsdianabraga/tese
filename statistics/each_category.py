import json
import random
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from geopy.geocoders import Nominatim
import spacy

# Load the JSON file containing tweets data
with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_june.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Function to cluster tweets based on cosine similarity
def cluster_text(dist_text, processedTweets, dist_threshold_txt):
    cluster_id = 1
    for i in range(dist_text.shape[0]):
        try:
            doc_dist_array_txt = dist_text[i,]
            # Identify similar tweets
            similarDocs = np.where(doc_dist_array_txt <= dist_threshold_txt)[0]
            processedTweets.loc[processedTweets.index.isin(similarDocs) & processedTweets['cluster_ref'].isna(), 'cluster_ref'] = cluster_id
            cluster_id += 1
        except ValueError:
            continue
    return processedTweets

# Tokenization function to preprocess text
def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    return filtered_tokens

# Function to calculate cosine similarity and generate clusters
def get_clusters(processedTweets, similarity_threshold):
    tweets = processedTweets['text'].tolist()
    
    # Vectorize tweets using CountVectorizer
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words=stopwords, tokenizer=tokenize_only)
    dtf_matrix = vectorizer.fit_transform(tweets)
    
    # Calculate cosine similarity matrix
    dist_text = np.around(abs(1 - cosine_similarity(dtf_matrix)), 2)
    
    # Initialize cluster references
    processedTweets['cluster_ref'] = None
    processedTweets = cluster_text(dist_text, processedTweets, dist_threshold_txt=(1-similarity_threshold))
    
    return processedTweets

# Define stopwords list
stopwords = stopwords.words('english')

# Load the spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Initialize dictionaries to store aggregated results
tweet_count = {}
tweets = {}
pontuation_sum = {}
sentiment_count = {}
top_results_global = {}
top_results_media_links = {}
top_results_verified = {}
top_results_question = {}
hashtags = {}
lemmas = {}
sentiment_percentages = {}
geographical_distribution = {}
geographical_percentages = {}
engagement_metrics = {}
coordenadas = {}
loc = Nominatim(user_agent="GetLoc")
localizacoes = []

# Iterate through the tweets and aggregate data
for tweet in data:
    topic = data[tweet]['query']['topic']
    pontuation_mentions = data[tweet]['metadata']['pontuation_mentions']
    pontuation = data[tweet]['metadata']['pontuation']
    sentiment = data[tweet]['sentiment']
    original = data[tweet]["result"]
    result = data[tweet]['result_en']
    result_pt = data[tweet]['result_pt']
    topic_hashtags = data[tweet]['metadata']['hashtags']
    topic_lemmas = data[tweet]['nlp_process']['lemmas']
    location = data[tweet]['metadata']['location']
    media = data[tweet]['metadata']['media']
    links = data[tweet]['metadata']['links']
    question = data[tweet]['question']
    verified = data[tweet]['user']['verified']
    username = data[tweet]['user']['username']
    name = data[tweet]['user']['name']
    profile_photo = data[tweet]['user']['profile_photo']
    likes = data[tweet]['metadata']['public_metrics']['like_count']
    retweets = data[tweet]['metadata']['public_metrics']['retweet_count']
    impressions = data[tweet]['metadata']['public_metrics']['impression_count']
    quotes = data[tweet]['metadata']['public_metrics']['quote_count']
    reply = data[tweet]['metadata']['public_metrics']['reply_count']
    link_tweet = data[tweet]['link_tweet']

    if media:
        media = media[0]["url"]

    if isinstance(username, list):
        username = username[0]

    # Update or initialize metrics for topics
    topics = topic if isinstance(topic, list) else [topic]
    for t in topics:
        tweet_count.setdefault(t, 0)
        tweet_count[t] += 1

        tweets.setdefault(t, [])
        tweets[t].append(result)

        engagement_metrics.setdefault(t, {
            'total_tweets': 0, 'total_impressions': 0, 'total_likes': 0,
            'total_retweets': 0, 'total_quotes': 0, 'total_replys': 0
        })
        engagement_metrics[t]['total_tweets'] += 1
        engagement_metrics[t]['total_likes'] += likes
        engagement_metrics[t]['total_retweets'] += retweets
        engagement_metrics[t]['total_impressions'] += impressions
        engagement_metrics[t]['total_quotes'] += quotes
        engagement_metrics[t]['total_replys'] += reply

        sentiment_count.setdefault(t, {"Positive": 0, "Neutral": 0, "Negative": 0})
        sentiment_count[t][sentiment] += 1

        pontuation_sum.setdefault(t, 0)
        pontuation_sum[t] += pontuation_mentions

        geographical_distribution.setdefault(t, {})

        if location not in localizacoes:
            localizacoes.append(location)
            getLoc = loc.geocode(location)
            coordenadas[location] = [getLoc.latitude, getLoc.longitude]

        if location:
            geographical_distribution[t].setdefault(location, [0, coordenadas[location][0], coordenadas[location][1]])
            geographical_distribution[t][location][0] += 1

        hashtags.setdefault(t, [])
        if topic_hashtags:
            hashtags[t].extend(topic_hashtags)

        lemmas.setdefault(t, [])
        if topic_lemmas:
            lemmas[t].extend(topic_lemmas)

        top_results_global.setdefault(t, [])
        top_results_media_links.setdefault(t, [])
        top_results_verified.setdefault(t, [])
        top_results_question.setdefault(t, [])

        # Store tweet data based on conditions
        tweet_data = {
            'username': username, 'name': name, "verified": verified,
            'result': original, 'result_en': result, 'result_pt': result_pt,
            'link': link_tweet, 'pontuation': pontuation, 'mentions': pontuation_mentions,
            'profile_photo': profile_photo, 'sentiment': sentiment,
            'media': media, 'total_impressions': impressions,
            'total_likes': likes, 'total_retweets': retweets,
            'total_quotes': quotes, 'total_replys': reply
        }

        top_results_global[t].append(tweet_data)
        if media or links:
            top_results_media_links[t].append(tweet_data)
        if verified:
            top_results_verified[t].append(tweet_data)
        if question:
            top_results_question[t].append(tweet_data)

# Clustering process and sentiment analysis
similarity_threshold = 0.70
for topic in tweets:
    processed_tweets = get_clusters(pd.DataFrame({'text': tweets[topic]}), similarity_threshold)
    num_clusters = processed_tweets['cluster_ref'].nunique()
    engagement_metrics[topic]['total_clusters'] = num_clusters

# Calculate sentiment percentages
for topic in sentiment_count:
    total_tweets = sum(sentiment_count[topic].values())
    sentiment_percentages[topic] = {sentiment: (count / total_tweets * 100) for sentiment, count in sentiment_count[topic].items()}

# Calculate geographical percentages
for topic in geographical_distribution:
    total_tweets = tweet_count[topic]
    geographical_percentages[topic] = {location: (count[0] / total_tweets * 100) for location, count in geographical_distribution[topic].items()}

# Calculate the most common hashtags and keywords
hashtag_counts = {topic: dict(Counter(hashtags[topic]).most_common(50)) for topic in hashtags}
keyword_counts = {topic: dict(Counter(lemmas[topic]).most_common(50)) for topic in lemmas}

# Sort the top results for each topic by pontuation
for topic in top_results_global:
    top_results_global[topic] = sorted(top_results_global[topic], key=lambda x: x['pontuation'], reverse=True)[:10]
    top_results_media_links[topic] = sorted(top_results_media_links[topic], key=lambda x: x['pontuation'], reverse=True)[:10]
    top_results_verified[topic] = sorted(top_results_verified[topic], key=lambda x: x['pontuation'], reverse=True)[:10]
    top_results_question[topic] = sorted(top_results_question[topic], key=lambda x: x['pontuation'], reverse=True)[:10]

# Collecting final data
data = []

for topic in tweet_count:
    data.append({
        'topic': topic, 'created_at': today, 'total_tweets': tweet_count[topic],
        'total_pontuation': pontuation_sum[topic], 'daily_statistics': daily_statistics,
        'top_tweets': top_results_global[topic], 'top_media_links': top_results_media_links[topic],
        'top_verified': top_results_verified[topic], 'top_question': top_results_question[topic],
        'geographical_distribution': geographical_distribution[topic], 'hashtags': hashtag_counts[topic],
        'keywords': keyword_counts[topic], 'sentiment_percentage': sentiment_percentages[topic],
        'geographical_percentages': geographical_percentages[topic], 'engagement_metrics': engagement_metrics[topic]
    })

# Write statistics to a JSON file
with open('statistics_each_new.json', 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, indent=4)
