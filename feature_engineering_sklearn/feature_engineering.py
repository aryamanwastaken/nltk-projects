import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re #regex
from nltk.corpus import stopwords #stopwords
from nltk.tokenize import word_tokenize #tokenizer
from nltk import pos_tag #part of speech 
from sklearn.model_selection import train_test_split #splitting data
from sklearn.base import BaseEstimator, TransformerMixin #custom sklearn transformer
from sklearn.pipeline import Pipeline #sklearn pipeline
from sklearn.feature_extraction.text import TfidfVectorizer #tfidf vectorizer
from sklearn.preprocessing import StandardScaler #standard scaler
from sklearn.pipeline import FeatureUnion #combining features
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.model_selection import GridSearchCV #grid search
from sklearn.metrics import classification_report #evaluation metric


df = pd.read_csv('./feature_engineering_sklearn/input/train.csv')

# dropping missing values
df.dropna(axis=0, inplace=True)
df.set_index('id', inplace = True)

df.head()

# stopwords
stopWords = set(stopwords.words('english'))

# adjectives 
def adjectives(text):
    tagged = pos_tag(word_tokenize(text))
    return len([word for word, pos in tagged if pos in ['JJ', 'JJR', 'JJS']])

# nouns
def nouns(text):
    tagged = pos_tag(word_tokenize(text))
    return len([word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']])

# verbs
def verbs(text):
    tagged = pos_tag(word_tokenize(text))
    return len([word for word, pos in tagged if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])

#creating a function to encapsulate preprocessing, to mkae it easy to replicate on  submission data
def processing(df):
    #lowering and removing punctuation
    df['processed'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
    
    #numerical feature engineering
    #total length of sentence
    df['length'] = df['processed'].apply(lambda x: len(x))
    #get number of words
    df['words'] = df['processed'].apply(lambda x: len(x.split(' ')))
    df['words_not_stopword'] = df['processed'].apply(lambda x: len([t for t in x.split(' ') if t not in stopWords]))
    #get the average word length
    df['avg_word_length'] = df['processed'].apply(lambda x: np.mean([len(t) for t in x.split(' ') if t not in stopWords]) if len([len(t) for t in x.split(' ') if t not in stopWords]) > 0 else 0)
    #get the average word length
    df['commas'] = df['text'].apply(lambda x: x.count(','))
    
    # get number of adjectives
    df['adjectives'] = df['text'].apply(adjectives)
    
    # get number of nouns
    df['nouns'] = df['text'].apply(nouns)
    
    # get number of verbs
    df['verbs'] = df['text'].apply(verbs)

    return(df)

df = processing(df)

df.head()

# splitting data
features= [c for c in df.columns.values if c  not in ['id','text','author']]
numeric_features= [c for c in df.columns.values if c  not in ['id','text','author','processed']]
target = 'author'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)
X_train.head()


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
   
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

# PIPELINES
text = Pipeline([
                ('selector', TextSelector(key='processed')),
                ('tfidf', TfidfVectorizer( stop_words='english'))
            ])

text.fit_transform(X_train)

length =  Pipeline([
                ('selector', NumberSelector(key='length')),
                ('standard', StandardScaler())
            ])

length.fit_transform(X_train)

words =  Pipeline([
                ('selector', NumberSelector(key='words')),
                ('standard', StandardScaler())
            ])
words_not_stopword =  Pipeline([
                ('selector', NumberSelector(key='words_not_stopword')),
                ('standard', StandardScaler())
            ])
avg_word_length =  Pipeline([
                ('selector', NumberSelector(key='avg_word_length')),
                ('standard', StandardScaler())
            ])
commas =  Pipeline([
                ('selector', NumberSelector(key='commas')),
                ('standard', StandardScaler()),
            ])

adjectives = Pipeline([
                ('selector', NumberSelector(key='adjectives')),
                ('standard', StandardScaler())
            ])

nouns = Pipeline([
                ('selector', NumberSelector(key='nouns')),
                ('standard', StandardScaler())
            ])

verbs = Pipeline([
                ('selector', NumberSelector(key='verbs')),
                ('standard', StandardScaler())
            ])

feats = FeatureUnion([('text', text), 
                      ('length', length),
                      ('words', words),
                      ('words_not_stopword', words_not_stopword),
                      ('avg_word_length', avg_word_length),
                      ('commas', commas),
                      ('adjectives', adjectives),
                      ('nouns', nouns),
                      ('verbs', verbs)])

feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(X_train)

pipeline = Pipeline([
    ('features', feats),
    ('classifier', LogisticRegression(random_state=42)),
])

# fitting our model
pipeline.fit(X_train, y_train)

# evaluating performance
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

# grid search
pipeline.get_params().keys()

# hyperparameters
hyperparameters = { 'features__text__tfidf__max_df': [0.9, 0.95],
                    'features__text__tfidf__ngram_range': [(1,1), (1,2)]
                  }

# cross validation
clf = GridSearchCV(pipeline, hyperparameters, cv=5)
 
# Fit and tune model
clf.fit(X_train, y_train)

# best parameters
clf.best_params_

# evaluating performance
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

np.mean(preds == y_test)

# load submission data
submission = pd.read_csv('./feature_engineering_sklearn/input/test.csv')

#preprocessing
submission = processing(submission)
predictions = clf.predict_proba(submission)

# generating a submission file
preds = pd.DataFrame(data=predictions, columns = clf.best_estimator_.named_steps['classifier'].classes_)

#generating a submission file
result = pd.concat([submission[['id']], preds], axis=1)
result.set_index('id', inplace = True)
result.head()