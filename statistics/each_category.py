import json
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from geopy.geocoders import Nominatim
# Load the JSON file
with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_tweets_june.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Dictionary to store the tweet count, pontuation_mentions sum, and sentiment count for each topic and sentiment category
tweet_count = {}
pontuation_sum = {}
sentiment_count = {}
top_results_global = {}
top_results_media_links = {}
top_results_verified = {}
hashtags = {}
lemmas = {}
sentiment_percentages = {}
geographical_distribution = {}
geographical_percentages ={}
engagement_metrics={}
coordenadas={}
loc = Nominatim(user_agent="GetLoc")
localizacoes=[]

# Iterate through the tweets
for tweet in data:
    topic = data[tweet]['query']['topic']
    pontuation_mentions = data[tweet]['metadata']['pontuation_mentions']
    pontuation = data[tweet]['metadata']['pontuation']
    sentiment = data[tweet]['sentiment']
    result = data[tweet]['result']
    result_pt = data[tweet]['result_pt']
    topic_hashtags = data[tweet]['metadata']['hashtags']
    topic_lemmas = data[tweet]['nlp_process']['lemmas']
    location = data[tweet]['metadata']['location']
    media = data[tweet]['metadata']['media']
    links = data[tweet]['metadata']['links']
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


    # Update tweet count
    tweet_count.setdefault(topic, 0)
    tweet_count[topic] += 1

    if media is not None:
        media=media[0]["url"]

    if isinstance(username, list):
        username=username[0]


    #Update tweet metrics
    engagement_metrics.setdefault(topic, {'total_tweets': 0, 'total_impressions': 0, 'total_likes': 0, 'total_retweets': 0, 'total_quotes': 0, 'total_replys': 0})
    engagement_metrics[topic]['total_tweets'] +=1
    engagement_metrics[topic]['total_likes'] += likes
    engagement_metrics[topic]['total_retweets'] += retweets
    engagement_metrics[topic]['total_impressions'] += impressions
    engagement_metrics[topic]['total_quotes'] += quotes
    engagement_metrics[topic]['total_replys'] += reply

    # Update sentiment count for the topic
    sentiment_count.setdefault(topic, {"Positive":0, "Neutral":0, "Negative":0})
    sentiment_count[topic][sentiment] += 1

    # Update pontuation_mentions sum
    pontuation_sum.setdefault(topic, 0)
    pontuation_sum[topic] += pontuation_mentions

    geographical_distribution.setdefault(topic, {})

    if location not in localizacoes:
        localizacoes.append(location)
        getLoc = loc.geocode(location)
        coordenadas[location] = [getLoc.latitude,getLoc.longitude]

    # Update geographical distribution
    if location is not None:
        geographical_distribution[topic].setdefault(location, [0,coordenadas[location][0],coordenadas[location][1]])
        geographical_distribution[topic][location][0] += 1

    # Update hashtags for the topic
    hashtags.setdefault(topic, [])
    if topic_hashtags is not None:  # Check if hashtags exist for the topic
        hashtags[topic].extend(topic_hashtags)

    # Update keywords for the topic
    lemmas.setdefault(topic, [])
    if topic_lemmas is not None:  # Check if lemmas exist for the topic
        lemmas[topic].extend(topic_lemmas)

    # Update top results
    top_results_global.setdefault(topic, [])
    top_results_media_links.setdefault(topic, [])
    top_results_verified.setdefault(topic, [])

    top_results_global[topic].append({'username':username,
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
        top_results_media_links[topic].append({'username':username,
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
        top_results_verified[topic].append({'username':username,
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

# Sort the top results for each topic
for topic in top_results_global:
    top_results_global[topic]= sorted(top_results_global[topic], key=lambda x: x['pontuation'], reverse=True)
    top_results_global[topic] = top_results_global[topic][:10]  # Select the top 10 results

for topic in top_results_media_links:
    top_results_media_links[topic]=sorted(top_results_media_links[topic], key=lambda x: x['pontuation'], reverse=True)
    top_results_media_links[topic] = top_results_media_links[topic][:10]  # Select the top 10 results

for topic in top_results_verified:
    top_results_verified[topic]=sorted(top_results_verified[topic], key=lambda x: x['pontuation'], reverse=True)
    top_results_verified[topic] = top_results_verified[topic][:10]  # Select the top 10 results

for topic in sentiment_count:
    total_tweets = sum(sentiment_count[topic].values())
    percentages = {}
    for sentiment, count in sentiment_count[topic].items():
        percentage = count / total_tweets * 100
        percentages[sentiment] = percentage
    sentiment_percentages[topic] = percentages

geo_distribution={}
for topic in geographical_distribution:
    total_tweets = tweet_count[topic]
    percentages = {}
    for sentiment, count in geographical_distribution[topic].items():
        percentage = count[0] / total_tweets * 100
        percentages[sentiment] = percentage
    geographical_percentages[topic] = percentages

hashtag_counts={}
keyword_counts={}

# Count hashtags and keywords for each topic
for topic in hashtags:
    hashtag = Counter(hashtags[topic])
    hashtag_counts[topic] = dict(hashtag.most_common(50))

for topic in lemmas:
    keyword = Counter(lemmas[topic])
    keyword_counts[topic] = dict(keyword.most_common(50))

# Create a dictionary to store the data
data = []

# Iterate over the tweet_count dictionary
for topic, count in tweet_count.items():
    topic_data = {
        "topic": topic,
        "engagement_metrics": engagement_metrics.get(topic, {}),
        "sentiment_count": sentiment_count.get(topic, {}),
        "sentiment_percentages": sentiment_percentages.get(topic, {}),
        "geographical": geographical_distribution.get(topic, {}),
        "geographical_percentages": geographical_percentages.get(topic, {}),
        "hashtags": hashtag_counts.get(topic, {}),
        "keywords":keyword_counts.get(topic, {}),
        "top_10_global": top_results_global.get(topic, []),
        "top_10_media": top_results_media_links.get(topic, []),
        "top_10_verified": top_results_verified.get(topic, [])
    }

    # Add the topic data to the dictionary
    data.append(topic_data)

# Write the data to a JSON file
with open('statistics_each.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)


'''
# Generate word clouds for hashtags for each topic
for topic, topic_hashtags in hashtags.items():
    if len(topic_hashtags) > 0:  # Check if there are any hashtags for the topic
        # Concatenate all hashtags into a single string
        hashtags_text = ' '.join(topic_hashtags)

        # Create the word cloud
        font_path = "path_to_your_font_file.ttf"
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(hashtags_text)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Word Cloud - Hashtags for {topic}")
        plt.axis('off')
        plt.show()
    else:
        print(f"No hashtags available for the topic: {topic}")

# Generate word clouds for lemmas for each topic
for topic, topic_lemmas in lemmas.items():
    if len(topic_lemmas) > 0:  # Check if there are any hashtags for the topic
        # Concatenate all hashtags into a single string
        hashtags_text = ' '.join(topic_lemmas)

        # Create the word cloud
        font_path = "path_to_your_font_file.ttf"
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(hashtags_text)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Word Cloud - Topics for {topic}")
        plt.axis('off')
        plt.show()
    else:
        print(f"No hashtags available for the topic: {topic}")
        
'''
