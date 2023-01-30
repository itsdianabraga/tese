import tweepy
import configparser
import pandas as pd

#read configs

api_key = 'jnIYl9SSJPW47RU2xbmPLl3cL'
api_key_secret = '6eHyHigGFZAJsPl1wic3hFPsqzmnyzZEnvbMgukkWEv3DwcCLO'

access_token = '1496955716137013248-3vBcSuT5pECWT4tPORfoAk51RYLGFg'
access_token_secret = 'DKE8i9hagLlFGYZMTDzuxN7wYzUqMRvSPI0yGak3K83As'

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFdCiQEAAAAASlcblnjSnYEbgMIzB0Kd4SxDm6c%3D1eLJSoSTYavew0Ye0ZGMmvvrfC8SOvycnxRZvjImZtQjbgXaeX'

#authentication

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api=tweepy.API(auth)


tweet=[]
likes=[]
user=[]
timeline=[]
hashtags=[]
tags=[]
t=[]
medias=[]
m={}


q="(diet) AND (-filter:retweets OR quotes OR replies) AND (filter:news)"

for status in tweepy.Cursor(api.search_tweets,
                            q,
                            lang="en",
                            tweet_mode="extended").items(500):
    tweet.append(status.full_text)
    likes.append(status.favorite_count)
    timeline.append(status.created_at)
    user.append(status.user.screen_name)
    hashtags=status.entities['hashtags']
    for i in range(len(hashtags)):
        tags.append(hashtags[i]['text'])
        medias.append(hashtags[i]['text'])
    t.append(tags)
    tags=[]
    hashtags=[]

for i in medias:
    if i.lower() in m:
        m[i.lower()]+=1
    else:
        m[i.lower()]=1

print(len(tweet))
print(dict(sorted(m.items(), key=lambda item: -item[1])))

df = pd.DataFrame({'user':user,'tweets':tweet, 'likes':likes, 'time':timeline, 'tags':t})

df.to_csv('nutrition.csv')

import time

start = time.time()
print("hello")
time.sleep(2)
end = time.time()
print(end - start)

