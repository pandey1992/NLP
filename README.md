# NLP
This will include projects related to Natural Language processing in python.
I have used used Cornell Sentiment analysis dataset(polarity dataset). You could download the dataset in the below link:
http://www.cs.cornell.edu/people/pabo/movie-review-data/
After cleaning and using lemmatization and transforming to Tfidf Vector, I trained the model using SVM,RFC and logistic regression.
Logistic regression gave a accuracy of 84.75%
Used pickle to persist these classifier and Tfidf Vectorizer and used these classifier to analyze 500 recent tweets regarding all top US banks and converted them to percentage to see percentage of negative and positive tweets.
