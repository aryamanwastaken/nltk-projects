import os # file paths
import gensim # word2vec
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import matplotlib.gridspec as gridspec # subplots
from sklearn.preprocessing import LabelEncoder  # encode target labels
from sklearn.linear_model import LogisticRegression # classifier
import warnings # ignore warnings

warnings.filterwarnings('ignore') # ignore warnings

print(os.listdir("../input")) # check input files

data = pd.read_json('../input/train.json') # read train data
test = pd.read_json('../input/test.json') # read test data

print('Training data shape: {}'.format(data.shape)) # print shape of train data
print('Test data shape: {}'.format(test.shape)) # print shape of test data

# Target variable 
target = data.cuisine

data['ingredient_count'] = data.ingredients.apply(lambda x: len(x)) # number of ingredients per recipe

# Function to flatten lists
def flatten_lists(lst): 
    """Remove nested lists."""
    return [item for sublist in lst for item in sublist] # flatten lists
f = plt.figure(figsize=(14,8)) # plot size
gs = gridspec.GridSpec(2, 2) # grid size

ax1 = plt.subplot(gs[0, :]) # first row, span all columns
data.ingredient_count.value_counts().hist(ax=ax1) # plot histogram
ax1.set_title('Recipe richness', fontsize=12)

ax2 = plt.subplot(gs[1, 0]) # second row, first column
pd.Series(flatten_lists(list(data['ingredients']))).value_counts()[:20].plot(kind='barh', ax=ax2) # plot bar chart
ax2.set_title('Most popular ingredients', fontsize=12)

ax3 = plt.subplot(gs[1, 1]) # second row, second column
data.groupby('cuisine').mean()['ingredient_count'].sort_values(ascending=False).plot(kind='barh', ax=ax3) # plot bar chart
ax3.set_title('Average number of ingredients in cuisines', fontsize=12) 

plt.show() # show plot

# Feed a word2vec with the ingredients
w2v = gensim.models.Word2Vec(list(data.ingredients), size=350, window=10, min_count=2, iter=20) 
# it is a good idea to use a size of 350-500 dimensions, window of 10 words, a min_count threshold of 2-5, and start with 20 iterations

w2v.most_similar(['meat']) # checking the most similar words to 'meat'

w2v.most_similar(['chicken']) # checking the most similar words to 'chicken'

# function to create document vectors
def document_vector(doc): 
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
    doc = [word for word in doc if word in w2v.wv.vocab] # remove out-of-vocabulary words
    return np.mean(w2v[doc], axis=0) # average words vectors

data['doc_vector'] = data.ingredients.apply(document_vector) # apply function to train data
test['doc_vector'] = test.ingredients.apply(document_vector) # apply function to test data

lb = LabelEncoder() # encode target labels
y = lb.fit_transform(target) # fit and transform target variable

X = list(data['doc_vector']) # create X variable
X_test = list(test['doc_vector']) # create X_test variable

clf = LogisticRegression(C=100) # create classifier

clf.fit(X, y) # fit classifier

y_test = clf.predict(X_test) # predict on test data
y_pred = lb.inverse_transform(y_test) # inverse transform predictions

test_id = [id_ for id_ in test.id] # create test id
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine']) # create submission file
sub.to_csv('clf_output.csv', index=False) # write submission file


