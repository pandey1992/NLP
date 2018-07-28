#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 12:55:25 2018
"""
# Importing the libraries
import tweepy
import re
import pickle
import matplotlib.pyplot as plt
from tweepy import OAuthHandler
# Initializing the keys
consumer_key = ''
consumer_secret = '' 
access_token = ''
access_secret =''

# Initializing the tokens
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
#args = ['Bank of America','Citibank','JP Morgan','Wells Fargo']
args = ['Bank of America','Citibank','JP Morgan','Wells Fargo']
api = tweepy.API(auth,timeout=10)

for query in args:
    # Fetching the tweets
    list_tweets = []
    if len(args) == 4:
        for status in tweepy.Cursor(api.search,q=query+" -filter:retweet",lang='en',result_type='recent').items(1000):
            list_tweets.append(status.text)
        
    # Loading the vectorizer and classfier
    with open('clog_model.pickle','rb') as f:
        clf = pickle.load(f)
        
    with open('tfidf_model.pickle','rb') as f:
        tfidf = pickle.load(f)    
        
    total_pos = 0
    total_neg = 0
    
    # Preprocessing the tweets and predicting sentiment
    for tweet in list_tweets:
        tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
        tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
        tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
        tweet = tweet.lower()
        tweet = re.sub(r"that's","that is",tweet)
        tweet = re.sub(r"there's","there is",tweet)
        tweet = re.sub(r"what's","what is",tweet)
        tweet = re.sub(r"where's","where is",tweet)
        tweet = re.sub(r"it's","it is",tweet)
        tweet = re.sub(r"who's","who is",tweet)
        tweet = re.sub(r"i'm","i am",tweet)
        tweet = re.sub(r"she's","she is",tweet)
        tweet = re.sub(r"he's","he is",tweet)
        tweet = re.sub(r"they're","they are",tweet)
        tweet = re.sub(r"who're","who are",tweet)
        tweet = re.sub(r"ain't","am not",tweet)
        tweet = re.sub(r"wouldn't","would not",tweet)
        tweet = re.sub(r"shouldn't","should not",tweet)
        tweet = re.sub(r"can't","can not",tweet)
        tweet = re.sub(r"couldn't","could not",tweet)
        tweet = re.sub(r"won't","will not",tweet)
        tweet = re.sub(r"\W"," ",tweet)
        tweet = re.sub(r"\d"," ",tweet)
        tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
        tweet = re.sub(r"\s+[a-z]$"," ",tweet)
        tweet = re.sub(r"^[a-z]\s+"," ",tweet)
        tweet = re.sub(r"\s+"," ",tweet)
        sent = clf.predict(tfidf.transform([tweet]).toarray())
        if sent[0] == 1:
            total_pos += 1
        else:
            total_neg += 1
        
    # Visualizing the results
    import matplotlib.pyplot as plt
    import numpy as np
    objects = ['Positive','Negative']
    x_pos = np.arange(len(objects))
    plt.figure(figsize=(12,10))
    plt.subplot(4,1,i+1)
    plt.bar(x_pos,[total_pos/10,total_neg/10],alpha=0.5)
    plt.xticks(x_pos,objects)
    plt.ylabel('Percentage')
    plt.title('Percentage of Postive and Negative Tweets for {}'.format(query))
    
    plt.show()
