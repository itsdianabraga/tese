# Twitter Sentiment Analysis
import sys
import csv
import tweepy
import matplotlib.pyplot as plt
from aylienapiclient import textapi

from collections import Counter

from tweepy import client

## Twitter credentials
consumer_key = 'jnIYl9SSJPW47RU2xbmPLl3cL'
consumer_secret = '6eHyHigGFZAJsPl1wic3hFPsqzmnyzZEnvbMgukkWEv3DwcCLO'
access_token = '1496955716137013248-3vBcSuT5pECWT4tPORfoAk51RYLGFg'
access_token_secret = 'DKE8i9hagLlFGYZMTDzuxN7wYzUqMRvSPI0yGak3K83As'

## set up an instance of Tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

## AYLIEN credentials
application_id = "18f40e1f"
application_key = "0aeafb76a5eeb8c6d897283791e69b67"

## set up an instance of the AYLIEN Text API
client = textapi.Client(application_id, application_key)

## search Twitter for something that interests you
query = input("What subject do you want to analyze for this example? \n")
number = input("How many Tweets do you want to analyze? \n")

results = api.search_tweets(
   lang="en",
   q=query + " -rt",
   count=number,
   result_type="recent"
)

print("--- Gathered Tweets \n")

## open a csv file to store the Tweets and their sentiment
file_name = 'Sentiment_Analysis_of_{}_Tweets_About_{}.csv'.format(number, query)

with open(file_name, 'w', newline='') as csvfile:
   csv_writer = csv.DictWriter(
       f=csvfile,
       fieldnames=["Tweet", "Sentiment"]
   )
   csv_writer.writeheader()

   print("--- Opened a CSV file to store the results of your sentiment analysis... \n")

## tidy up the Tweets and send each to the AYLIEN Text API
   for c, result in enumerate(results, start=1):
       tweet = result.text
       tidy_tweet = tweet.strip().encode('ascii', 'ignore')

       if len(tweet) == 0:
           print('Empty Tweet')
           continue

       response = client.Sentiment({'text': tidy_tweet})
       csv_writer.writerow({
           'Tweet': response['text'],
           'Sentiment': response['polarity']
       })

       print("Analyzed Tweet {}".format(c))