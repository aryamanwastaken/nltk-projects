# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import xgboost
from sklearn.model_selection import RandomizedSearchCV
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os   # for file handling
from sklearn.naive_bayes import MultinomialNB    # for naive bayes classifier

# importing files
for dirname, _, filenames in os.walk('./text-classification-naive-bayes'):   # for walking through the directory
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("./text-classification-naive-bayes/Corona_NLP_train.csv", encoding='latin1')  # reading the csv file
train.head()  # printing the first five rows of the dataset


def drop(p):  # for dropping the unnecessary columns
    p.drop(["UserName", "ScreenName", "Location", "TweetAt"], axis=1, inplace=True)  # dropping the columns
drop(train)  # calling the function
train.head()  # printing the first five rows of the dataset


train["Sentiment"].value_counts() # printing the value counts of the target variable

def rep(t): # for replacing the values in the target variable
    # replacing the values
    d = {"Sentiment": {'Positive': 0, 'Negative': 1, "Neutral": 2, "Extremely Positive": 3, "Extremely Negative": 4}}
    t.replace(d, inplace=True)
rep(train)  # calling the function
train.head()  # printing the first five rows of the dataset

tweettoken = TweetTokenizer(strip_handles=True, reduce_len=True)  # for tokenizing the tweets
lemmatizer = WordNetLemmatizer()  # for lemmatizing the words
stemmer = PorterStemmer()  # for stemming the words

collect = []  # for collecting the preprocessed tweets

def preprocess(t):  # for preprocessing the tweets
    tee = re.sub('[^a-zA-Z]', " ", t)   # for removing the special characters
    tee = tee.lower()  # for converting the tweets into lower case
    res = tweettoken.tokenize(tee)  # for tokenizing the tweets
    # for removing the stopwords
    for i in res:   
        if i in stopwords.words('english'):  # checking if the word is a stopword
            res.remove(i)  # removing the stopword
    rest = []  # for collecting the lemmatized words 
    for k in res:   # for lemmatizing the words 
        rest.append(lemmatizer.lemmatize(k))    # lemmatizing the words
    ret = " ".join(rest)  # for joining the words
    collect.append(ret)  # appending the preprocessed tweets
for j in range(41157):   # for preprocessing the tweets
    preprocess(train["OriginalTweet"].iloc[j])      # calling the functioN
collect[:5]  # printing the first five preprocessed tweets

# for creating the bag of words model
def bow(ll):  
    cv = CountVectorizer(max_features=200)  # for creating the bag of words model
    x = cv.fit_transform(ll).toarray()  # for creating the sparse matrix
    return x  # returning the sparse matrix
y = bow(collect)  # calling the function
collect[:5]  # printing the first five preprocessed tweets

len(y[0][:])  # printing the length of the sparse matrix

# for creating the tf-idf model
def tfidf(xx):
    cv = TfidfVectorizer(max_features=4000)  # for creating the tf-idf model
    x = cv.fit_transform(xx).toarray()  # for creating the sparse matrix
    return x  # returning the sparse matrix
values = train["Sentiment"].values  # for collecting the target variable

# for splitting the dataset into training and testing set
(x_train, x_test, y_train, y_test) = train_test_split(y, values, train_size=0.75, random_state=42)

# for creating the naive bayes classifier
clf = MultinomialNB()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)  # for predicting the values

print("Confusion Matrix for BoW:")  # for printing the confusion matrix
print(confusion_matrix(y_test, y_pred))  # printing the confusion matrix

print("Score for BoW:", clf.score(x_test, y_test))  # for printing the score

y = tfidf(collect)  # calling the function
# for splitting the dataset into training and testing set
(x_train, x_test, y_train, y_test) = train_test_split(y, values, train_size=0.75, random_state=42)

clf = MultinomialNB()  # for creating the naive bayes classifier
clf.fit(x_train, y_train)  # for fitting the model

y_pred = clf.predict(x_test)  # for predicting the values

print("Confusion Matrix for TF-IDF:")  # for printing the confusion matrix
print(confusion_matrix(y_test, y_pred))  # printing the confusion matrix

print("Score for TF-IDF:", clf.score(x_test, y_test))  # for printing the score


"""EXPLANATIONS"""
# EXPLANATION ON SCORE METHOD
# .score() method provides the mean accuracy on the given test data and labels. It's the fraction
#  of correctly predicted samples out of the total samples.

# EXPLANAION ON DIFFERENCES IN SCORES
# The difference in scores between BoW and TF-IDF can be explained by the fact that TF-IDF takes into account the frequency of words in the document.
# The confusion matrices provide a detailed breakdown of true positives, false positives, true negatives, and false negatives for each class. 
# By analyzing these matrices, we can understand where each model is making mistakes and potentially why one model might be outperforming the other.


"""
# EXTRAS
"""
# This is inefficient and can be improved by using the set() data structure to store the stopwords and then check membership against this set.
# The line if i in stopwords.words('english'): checks if a word is a stopword by repeatedly calling stopwords.words('english') for every word in every tweet. 
# It's better to call stopwords.words('english') once, store the result in a set, and then check membership against this set.
"""
stop_words_set = set(stopwords.words('english'))
for i in res:
    if i in stop_words_set:
        res.remove(i)
"""