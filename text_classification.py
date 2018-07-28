#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 10:45:47 2018

Text Classification
"""

#Importing the libraries
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
#Importing the dataset

reviews = load_files('txt_sentoken/')
X,y = reviews.data,reviews.target

#Storing as Pickle file(a byte file)
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
#Unpickling the dataset
with open('X.pickle','rb') as f:
    X=pickle.load(f)
with open('y.pickle','rb') as f:
    y=pickle.load(f)
    
#Creating the corpus
corpus =[]
for i in range(len(X)):
    review = re.sub(r'\W', ' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-zA-Z]\s+',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)
#Creating BOW model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000,min_df=4,max_df=0.6,stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()
#Transforming BOW into Tf-Idf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X=transformer.fit_transform(X).toarray()
#Creating Tf-Idf model directly
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()
#train-test split
text_train,text_test,sent_train,sent_test = train_test_split(X,y,test_size=0.2,random_state=0)
#Training LogisticRegression Model
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(text_train,sent_train)
sent_pred = log_model.predict(text_test)
#Getting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)

#pickling the classifier
with open('log_model.pickle','wb') as f:
    pickle.dump(log_model,f)
#picking Tfidf vectorizer
with open('tfidf_model.pickle','wb') as f:
    pickle.dump(vectorizer,f)
#Unpickling the classifier and Tfif
with open('log_model.pickle','rb') as f:
    clf = pickle.load(f)

with open('tfidf_model.pickle','rb') as f:
    tfidf = pickle.load(f)

sample = ['hey man you are a great person, have a great life']
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))



