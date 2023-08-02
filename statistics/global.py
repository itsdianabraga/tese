import json
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
# importing geopy library
from geopy.geocoders import Nominatim

# Load the JSON file
with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_tweets_june.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Variables to store the overall statistics
total_tweets = len(data)
total_pontuation_mentions = 0
total_sentiment_count = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
total_geographical_distribution = {}
top_results_global = {}
top_results_media_links = {}
top_results_verified = {}
engagement_metrics = {'total_tweets': 0, 'total_impressions': 0, 'total_likes': 0, 'total_retweets': 0, 'total_quotes': 0, 'total_replys': 0}

# Variables to store the top topics, hashtags, lemmas, and engagement metrics
top_topics = Counter()
top_hashtags = Counter()
lemmas = Counter()
top_results_global=[]
top_results_media_links=[]
top_results_verified=[]
loc = Nominatim(user_agent="GetLoc")

# Iterate through the tweets
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

    # Update the top topics and hashtags
    top_topics[topic] += 1
    top_hashtags.update(hashtags)

    # Update the lemmas count
    lemmas.update(lemmas_list)

    if media is not None:
        media=media[0]["url"]

    if isinstance(username, list):
        username=username[0]

    # Update the engagement metrics
    engagement_metrics['total_tweets'] += 1
    engagement_metrics['total_likes'] += likes
    engagement_metrics['total_retweets'] += retweets
    engagement_metrics['total_impressions'] += impressions
    engagement_metrics['total_quotes'] += quotes
    engagement_metrics['total_replys'] += reply

    top_results_global.append({'username':username,
                               'name': name,
                               "verified": verified,
                               'result': result,
                               'result_pt': result_pt,
                               'link': link_tweet,
                               'pontuation': pontuation,
                                'mentions': pontuation_mentions,
                               'profile_photo': profile_photo,
                               'media':media, 'total_impressions': impressions,
                                      'total_likes': likes,
                                      'total_retweets': retweets,
                                      'total_quotes': quotes,
                                      'total_replys': reply
                                      })

    if media or links:
        top_results_media_links.append({'username':username,
                               'name': name,
                               "verified": verified,
                               'result': result,
                                'result_pt': result_pt,
                                'link': link_tweet,
                               'pontuation': pontuation,
                                'mentions': pontuation_mentions,
                               'profile_photo': profile_photo,
                               'media':media, 'total_impressions': impressions,
                                      'total_likes': likes,
                                      'total_retweets': retweets,
                                      'total_quotes': quotes,
                                      'total_replys': reply
                                      })

    if verified:
        top_results_verified.append({'username':username,
                               'name': name,
                               "verified": verified,
                               'result': result,
                                'result_pt': result_pt,
                                'link': link_tweet,
                               'pontuation': pontuation,
                                'mentions': pontuation_mentions,
                               'profile_photo': profile_photo,
                               'media':media, 'total_impressions': impressions,
                                      'total_likes': likes,
                                      'total_retweets': retweets,
                                      'total_quotes': quotes,
                                      'total_replys': reply
                                      })

## Sort the lists based on 'pontuation'
top_results_global_sorted = sorted(top_results_global, key=lambda x: x['pontuation'], reverse=True)
top_results_media_links_sorted = sorted(top_results_media_links, key=lambda x: x['pontuation'], reverse=True)
top_results_verified_sorted = sorted(top_results_verified, key=lambda x: x['pontuation'], reverse=True)

# Keep the top 10 results
top_results_global_sorted = top_results_global_sorted[:10]
top_results_media_links_sorted = top_results_media_links_sorted[:10]
top_results_verified_sorted = top_results_verified_sorted[:10]

# Calculate the top 10 topics of the day
top_topics_day = top_topics.most_common(20)

# Calculate the top 10 hashtags
top_hashtags_day = top_hashtags.most_common(50)

# Words to exclude
excluded_words = ["children", "child", "kid", "youngster", "minor", "baby", "teenager", "infant", "adolescent", "juvenile", "kiddie", "youth", "pediatrics", "neonatal"]

# Create a Counter object from the lemmas list
lemmas_counter = Counter(dict(lemmas))

# Filter out the excluded words
filtered_lemmas = [(word, count) for word, count in lemmas_counter.items() if word not in excluded_words]

# Sort the filtered lemmas by count in descending order
sorted_lemmas = sorted(filtered_lemmas, key=lambda x: x[1], reverse=True)

# Get the top 10 most common items
trending_keywords = sorted_lemmas[:50]

# Iterate through the tweets
for tweet in data:
    pontuation_mentions = data[tweet]['metadata']['pontuation_mentions']
    sentiment = data[tweet]['sentiment']
    location = data[tweet]['metadata']['location']

    # Update overall statistics
    total_pontuation_mentions += pontuation_mentions
    total_sentiment_count[sentiment] += 1

    total_geographical_distribution.setdefault(location, 0)
    total_geographical_distribution[location] += 1

# Calculate sentiment percentages for the overall data
total_sentiment_percentages = {}
for sentiment, count in total_sentiment_count.items():
    percentage = count / total_tweets * 100
    total_sentiment_percentages[sentiment] = percentage

geographical_distribution={}
# Calculate geografic percentages for the overall data
total_geographical_percentages = {}
for geographical, count in total_geographical_distribution.items():
    percentage = count / total_tweets * 100
    total_geographical_percentages[geographical] = percentage
    getLoc = loc.geocode(geographical)
    geographical_distribution[geographical]=[count,getLoc.latitude,getLoc.longitude]


#turning lists to simple dicts
dict_topics={}
dict_hashtags={}
dict_keyword={}

for topic, count in top_topics_day:
    dict_topics[topic]=count

for hashtag, count in top_hashtags_day:
    dict_hashtags[hashtag]=count

for keyword, count in trending_keywords:
    dict_keyword[keyword]=count


# Prepare the data to be written in a JSON file
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

# Write the data to a JSON file
with open('statistics.json', 'w') as file:
    json.dump(statistics, file, indent=4)

# Rest of your code...
