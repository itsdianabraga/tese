import json
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# Load the JSON file containing tweets data
with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_tweets_june.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize variables for overall statistics
total_tweets = len(data)
total_pontuation_mentions = 0
total_sentiment_count = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
total_geographical_distribution = {}
top_results_global = []
top_results_media_links = []
top_results_verified = []

# Initialize counters for top topics, hashtags, and lemmas
top_topics = Counter()
top_hashtags = Counter()
lemmas = Counter()

# Initialize engagement metrics
engagement_metrics = {
    'total_tweets': 0, 'total_impressions': 0, 'total_likes': 0,
    'total_retweets': 0, 'total_quotes': 0, 'total_replys': 0
}

# Initialize geolocator for location processing
loc = Nominatim(user_agent="GetLoc")

# Iterate through the tweets and gather data
for tweet in data:
    topic = data[tweet]['query']['topic']
    pontuation = data[tweet]['metadata']['pontuation']
    result = data[tweet]['result']
    result_pt = data[tweet]['result_pt']
    media = data[tweet]['metadata']['media']
    links = data[tweet]['metadata']['links']
    hashtags = data[tweet]['metadata']['hashtags']
    lemmas_list = data[tweet]['nlp_process']['lemmas']
    pontuation_mentions = data[tweet]['metadata']['pontuation_mentions']
    likes = data[tweet]['metadata']['public_metrics']['like_count']
    retweets = data[tweet]['metadata']['public_metrics']['retweet_count']
    impressions = data[tweet]['metadata']['public_metrics']['impression_count']
    quotes = data[tweet]['metadata']['public_metrics']['quote_count']
    reply = data[tweet]['metadata']['public_metrics']['reply_count']
    verified = data[tweet]['user']['verified']
    username = data[tweet]['user']['username']
    name = data[tweet]['user']['name']
    profile_photo = data[tweet]['user']['profile_photo']
    link_tweet = data[tweet]['link_tweet']

    # Update the top topics, hashtags, and lemmas
    top_topics[topic] += 1
    top_hashtags.update(hashtags)
    lemmas.update(lemmas_list)

    # Ensure media is correctly processed
    if media:
        media = media[0]["url"]

    if isinstance(username, list):
        username = username[0]

    # Update engagement metrics
    engagement_metrics['total_tweets'] += 1
    engagement_metrics['total_likes'] += likes
    engagement_metrics['total_retweets'] += retweets
    engagement_metrics['total_impressions'] += impressions
    engagement_metrics['total_quotes'] += quotes
    engagement_metrics['total_replys'] += reply

    # Compile top results
    tweet_data = {
        'username': username, 'name': name, 'verified': verified, 'result': result,
        'result_pt': result_pt, 'link': link_tweet, 'pontuation': pontuation,
        'mentions': pontuation_mentions, 'profile_photo': profile_photo, 'media': media,
        'total_impressions': impressions, 'total_likes': likes, 'total_retweets': retweets,
        'total_quotes': quotes, 'total_replys': reply
    }
    top_results_global.append(tweet_data)

    if media or links:
        top_results_media_links.append(tweet_data)

    if verified:
        top_results_verified.append(tweet_data)

# Sort the top results by 'pontuation' and keep only the top 10
top_results_global_sorted = sorted(top_results_global, key=lambda x: x['pontuation'], reverse=True)[:10]
top_results_media_links_sorted = sorted(top_results_media_links, key=lambda x: x['pontuation'], reverse=True)[:10]
top_results_verified_sorted = sorted(top_results_verified, key=lambda x: x['pontuation'], reverse=True)[:10]

# Determine the top 20 topics of the day and the top 50 hashtags
top_topics_day = top_topics.most_common(20)
top_hashtags_day = top_hashtags.most_common(50)

# Words to exclude from keyword analysis
excluded_words = [
    "children", "child", "kid", "youngster", "minor", "baby", "teenager", "infant", 
    "adolescent", "juvenile", "kiddie", "youth", "pediatrics", "neonatal"
]

# Filter out excluded words from lemmas and get the top 50 trending keywords
filtered_lemmas = [(word, count) for word, count in lemmas.items() if word not in excluded_words]
trending_keywords = sorted(filtered_lemmas, key=lambda x: x[1], reverse=True)[:50]

# Update overall statistics from tweets
for tweet in data:
    pontuation_mentions = data[tweet]['metadata']['pontuation_mentions']
    sentiment = data[tweet]['sentiment']
    location = data[tweet]['metadata']['location']

    total_pontuation_mentions += pontuation_mentions
    total_sentiment_count[sentiment] += 1

    if location:
        total_geographical_distribution.setdefault(location, 0)
        total_geographical_distribution[location] += 1

# Calculate sentiment percentages
total_sentiment_percentages = {
    sentiment: count / total_tweets * 100 for sentiment, count in total_sentiment_count.items()
}

# Calculate geographical percentages and coordinates
total_geographical_percentages = {}
geographical_distribution = {}

for location, count in total_geographical_distribution.items():
    percentage = count / total_tweets * 100
    total_geographical_percentages[location] = percentage

    getLoc = loc.geocode(location)
    if getLoc:
        geographical_distribution[location] = [count, getLoc.latitude, getLoc.longitude]

# Convert lists of top topics, hashtags, and keywords to dictionaries
dict_topics = dict(top_topics_day)
dict_hashtags = dict(top_hashtags_day)
dict_keyword = dict(trending_keywords)

# Prepare the final statistics dictionary
statistics = {
    "Overall Statistics": {
        "engagement_metrics": engagement_metrics,
        "sentiment_count": total_sentiment_count,
        "sentiment_percentages": total_sentiment_percentages,
        "geographical": geographical_distribution,
        "geographical_percentages": total_geographical_percentages,
        "topics": dict_topics,
        "hashtags": dict_hashtags,
        "keywords": dict_keyword,
        "top_10_global": top_results_global_sorted,
        "top_10_media": top_results_media_links_sorted,
        "top_10_verified": top_results_verified_sorted
    }
}

# Write the statistics to a JSON file
with open('statistics.json', 'w', encoding='utf-8') as file:
    json.dump(statistics, file, indent=4)
