import json
import random
from collections import Counter
from datetime import datetime, timedelta
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from geopy.geocoders import Nominatim
# Load the JSON file
with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_june.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


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

import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Dictionary to store the tweet count, pontuation_mentions sum, and sentiment count for each topic and sentiment category
tweet_count = {}
tweets={}
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

    if media is not None:
        media=media[0]["url"]

    if isinstance(username, list):
        username=username[0]

    if isinstance(topic,list):
        for t in topic:
            # Update tweet count
            tweet_count.setdefault(t, 0)
            tweet_count[t] += 1

            tweets.setdefault(t,[])
            tweets[t].append(result)

            # Update tweet metrics
            engagement_metrics.setdefault(t, {'total_tweets': 0, 'total_impressions': 0, 'total_likes': 0,
                                                  'total_retweets': 0, 'total_quotes': 0, 'total_replys': 0})
            engagement_metrics[t]['total_tweets'] += 1
            engagement_metrics[t]['total_likes'] += likes
            engagement_metrics[t]['total_retweets'] += retweets
            engagement_metrics[t]['total_impressions'] += impressions
            engagement_metrics[t]['total_quotes'] += quotes
            engagement_metrics[t]['total_replys'] += reply

            # Update sentiment count for the topic
            sentiment_count.setdefault(t, {"Positive": 0, "Neutral": 0, "Negative": 0})
            sentiment_count[t][sentiment] += 1

            # Update pontuation_mentions sum
            pontuation_sum.setdefault(t, 0)
            pontuation_sum[t] += pontuation_mentions

            geographical_distribution.setdefault(t, {})

            if location not in localizacoes:
                localizacoes.append(location)
                getLoc = loc.geocode(location)
                coordenadas[location] = [getLoc.latitude, getLoc.longitude]

            # Update geographical distribution
            if location is not None:
                geographical_distribution[t].setdefault(location,
                                                            [0, coordenadas[location][0], coordenadas[location][1]])
                geographical_distribution[t][location][0] += 1

            # Update hashtags for the topic
            hashtags.setdefault(t, [])
            if topic_hashtags is not None:  # Check if hashtags exist for the topic
                hashtags[t].extend(topic_hashtags)

            # Update keywords for the topic
            lemmas.setdefault(t, [])
            if topic_lemmas is not None:  # Check if lemmas exist for the topic
                lemmas[t].extend(topic_lemmas)

            # Update top results
            top_results_global.setdefault(t, [])
            top_results_media_links.setdefault(t, [])
            top_results_verified.setdefault(t, [])
            top_results_question.setdefault(t,[])

            top_results_global[t].append({'username': username,
                                              'name': name,
                                              "verified": verified,
                                              'result': original,
                                                'result_en': result,
                                              'result_pt': result_pt,
                                              'link': link_tweet,
                                              'pontuation': pontuation,
                                              'mentions': pontuation_mentions,
                                              'profile_photo': profile_photo,
                                              'sentiment': sentiment,
                                              'media': media, 'total_impressions': impressions,
                                              'total_likes': likes,
                                              'total_retweets': retweets,
                                              'total_quotes': quotes,
                                              'total_replys': reply
                                              })

            if media or links:
                top_results_media_links[t].append({'username': username,
                                                       'name': name,
                                                       "verified": verified,
                                                       'result': original,
                                                        'result_en': result,
                                                       'result_pt': result_pt,
                                                       'link': link_tweet,
                                                       'pontuation': pontuation,
                                                       'mentions': pontuation_mentions,
                                                       'profile_photo': profile_photo,
                                                       'sentiment': sentiment,
                                                       'media': media, 'total_impressions': impressions,
                                                       'total_likes': likes,
                                                       'total_retweets': retweets,
                                                       'total_quotes': quotes,
                                                       'total_replys': reply
                                                       })

            if verified:
                top_results_verified[t].append({'username': username,
                                                    'name': name,
                                                    "verified": verified,
                                                    'result': original,
                                                    'result_en': result,
                                                    'result_pt': result_pt,
                                                    'link': link_tweet,
                                                    'pontuation': pontuation,
                                                    'mentions': pontuation_mentions,
                                                    'profile_photo': profile_photo,
                                                    'sentiment': sentiment,
                                                    'media': media, 'total_impressions': impressions,
                                                    'total_likes': likes,
                                                    'total_retweets': retweets,
                                                    'total_quotes': quotes,
                                                    'total_replys': reply
                                                    })
            if question:
                top_results_question[t].append({'username': username,
                                                    'name': name,
                                                    "verified": verified,
                                                    'result': original,
                                                    'result_en': result,
                                                    'result_pt': result_pt,
                                                    'link': link_tweet,
                                                    'pontuation': pontuation,
                                                    'mentions': pontuation_mentions,
                                                    'profile_photo': profile_photo,
                                                    'sentiment': sentiment,
                                                    'media': media, 'total_impressions': impressions,
                                                    'total_likes': likes,
                                                    'total_retweets': retweets,
                                                    'total_quotes': quotes,
                                                    'total_replys': reply
                                                    })
    else:
        # Update tweet count
        tweet_count.setdefault(topic, 0)
        tweet_count[topic] += 1

        tweets.setdefault(topic, [])
        tweets[topic].append(result)

        # Update tweet metrics
        engagement_metrics.setdefault(topic,
                                      {'total_tweets': 0, 'total_impressions': 0, 'total_likes': 0, 'total_retweets': 0,
                                       'total_quotes': 0, 'total_replys': 0})
        engagement_metrics[topic]['total_tweets'] += 1
        engagement_metrics[topic]['total_likes'] += likes
        engagement_metrics[topic]['total_retweets'] += retweets
        engagement_metrics[topic]['total_impressions'] += impressions
        engagement_metrics[topic]['total_quotes'] += quotes
        engagement_metrics[topic]['total_replys'] += reply

        # Update sentiment count for the topic
        sentiment_count.setdefault(topic, {"Positive": 0, "Neutral": 0, "Negative": 0})
        sentiment_count[topic][sentiment] += 1

        # Update pontuation_mentions sum
        pontuation_sum.setdefault(topic, 0)
        pontuation_sum[topic] += pontuation_mentions

        geographical_distribution.setdefault(topic, {})

        if location not in localizacoes:
            localizacoes.append(location)
            getLoc = loc.geocode(location)
            coordenadas[location] = [getLoc.latitude, getLoc.longitude]

        # Update geographical distribution
        if location is not None:
            geographical_distribution[topic].setdefault(location,
                                                        [0, coordenadas[location][0], coordenadas[location][1]])
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

        top_results_global[topic].append({'username': username,
                                          'name': name,
                                          "verified": verified,
                                          'result': original,
                                          'result_en': result,
                                          'result_pt': result_pt,
                                          'link': link_tweet,
                                          'pontuation': pontuation,
                                          'mentions': pontuation_mentions,
                                          'profile_photo': profile_photo,
                                          'sentiment': sentiment,
                                          'media': media, 'total_impressions': impressions,
                                          'total_likes': likes,
                                          'total_retweets': retweets,
                                          'total_quotes': quotes,
                                          'total_replys': reply
                                          })

        if media or links:
            top_results_media_links[topic].append({'username': username,
                                                   'name': name,
                                                   "verified": verified,
                                                   'result': original,
                                                    'result_en': result,
                                                   'result_pt': result_pt,
                                                   'link': link_tweet,
                                                   'pontuation': pontuation,
                                                   'mentions': pontuation_mentions,
                                                   'profile_photo': profile_photo,
                                                   'sentiment': sentiment,
                                                   'media': media, 'total_impressions': impressions,
                                                   'total_likes': likes,
                                                   'total_retweets': retweets,
                                                   'total_quotes': quotes,
                                                   'total_replys': reply
                                                   })

        if verified:
            top_results_verified[topic].append({'username': username,
                                                'name': name,
                                                "verified": verified,
                                                'result': original,
                                                'result_en': result,
                                                'result_pt': result_pt,
                                                'link': link_tweet,
                                                'pontuation': pontuation,
                                                'mentions': pontuation_mentions,
                                                'profile_photo': profile_photo,
                                                'sentiment': sentiment,
                                                'media': media, 'total_impressions': impressions,
                                                'total_likes': likes,
                                                'total_retweets': retweets,
                                                'total_quotes': quotes,
                                                'total_replys': reply
                                                })

        if question:
            top_results_question[t].append({'username': username,
                                            'name': name,
                                            "verified": verified,
                                            'result': original,
                                            'result_en': result,
                                            'result_pt': result_pt,
                                            'link': link_tweet,
                                            'pontuation': pontuation,
                                            'mentions': pontuation_mentions,
                                            'profile_photo': profile_photo,
                                            'sentiment': sentiment,
                                            'media': media, 'total_impressions': impressions,
                                            'total_likes': likes,
                                            'total_retweets': retweets,
                                            'total_quotes': quotes,
                                            'total_replys': reply
                                            })


processed_tweets=[]
# Sort the top results for each topic
for topic in top_results_global:
    top_results_global[topic] = sorted(top_results_global[topic], key=lambda x: x['pontuation'], reverse=True)[:10]

# Define a similarity threshold (e.g., 0.7)
similarity_threshold = 0.70

for topic in tweets:
    # Call the get_clusters function
    processed_tweets = get_clusters(pd.DataFrame({'text': tweets[topic]}), similarity_threshold)

    # Access the clustered tweets
    clustered_tweets = processed_tweets[processed_tweets['cluster_ref'].notnull()]

    num_clusters = processed_tweets['cluster_ref'].nunique()

    engagement_metrics[topic]['total_clusters']=num_clusters

for topic in top_results_media_links:
    top_results_media_links[topic]=sorted(top_results_media_links[topic], key=lambda x: x['pontuation'], reverse=True)[:10]

for topic in top_results_verified:
    top_results_verified[topic]=sorted(top_results_verified[topic], key=lambda x: x['pontuation'], reverse=True)[:10]

for topic in top_results_question:
    top_results_question[topic]=sorted(top_results_question[topic], key=lambda x: x['pontuation'], reverse=True)[:10]

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

current_time = datetime.utcnow()
today = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')


general_info=[]


estatisticas_diarias={}

dates=["01-06-2023", "02-06-2023", "03-06-2023", "04-06-2023", "05-06-2023", "06-06-2023","07-06-2023"]


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
        "top_10_verified": top_results_verified.get(topic, []),
        "top_10_question": top_results_question.get(topic,[])
    }

    dict_1={"Total": count}

    positive1=random.randint(10, 3000)
    positive2=random.randint(10, 3000)
    positive3=random.randint(10, 3000)
    positive4=random.randint(10, 3000)
    positive5=random.randint(10, 3000)
    positive6=random.randint(10, 3000)

    negative1 = random.randint(10, 3000)
    negative2 = random.randint(10, 3000)
    negative3 = random.randint(10, 3000)
    negative4 = random.randint(10, 3000)
    negative5 = random.randint(10, 3000)
    negative6 = random.randint(10, 3000)

    neutral1=random.randint(10, 3000)
    neutral2 = random.randint(10, 3000)
    neutral3 = random.randint(10, 3000)
    neutral4 = random.randint(10, 3000)
    neutral5 = random.randint(10, 3000)
    neutral6 = random.randint(10, 3000)

    total1=positive1+negative1+neutral1
    total2 = positive2 + negative2+ neutral2
    total3 = positive3 + negative3 + neutral3
    total4 = positive4 + negative4 + neutral4
    total5 = positive5 + negative5 + neutral5
    total6 = positive6 + negative6 + neutral6


    estatisticas_diarias = {
        "topic": topic,
        dates[0]: {"Total": total1,
                   "Positive": positive1,
                   "Neutral": neutral1,
                   "Negative": negative1},
        dates[1]: {"Total": total2,
                   "Positive": positive2,
                   "Neutral": neutral2,
                   "Negative": negative2},
        dates[2]: {"Total": total3,
                   "Positive": positive3,
                   "Neutral": neutral3,
                   "Negative": negative3},
        dates[3]: {"Total": total4,
                   "Positive": positive4,
                   "Neutral": neutral4,
                   "Negative": negative4},
        dates[4]: {"Total": total5,
                   "Positive": positive5,
                   "Neutral": neutral5,
                   "Negative": negative5},
        dates[5]: {"Total": total6,
                   "Positive": positive6,
                   "Neutral": neutral6,
                   "Negative": negative6},
        dates[6]: {**dict_1, **sentiment_count.get(topic)}
    }

    # Add the topic data to the dictionary
    data.append(topic_data)
    general_info.append(estatisticas_diarias)


# Write the data to a JSON file
with open('statistics_each_new.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

with open('general_info.json', 'w') as json_file:
    json.dump(general_info, json_file, indent=4)
