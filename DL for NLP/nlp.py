# LESSON 1 - Deep Learning and Natural Language

"""
For this lesson you must research and list 10 impressive applications of deep learning methods in the field of natural language processing. Bonus points if you can link to a research paper that demonstrates the example.

1. Machine Translation - https://arxiv.org/abs/1706.03762
2. Speech Recognition - https://arxiv.org/abs/1512.02595
3. Sentiment Analysis - https://arxiv.org/abs/1810.04805
4. Text Generation - https://arxiv.org/abs/2005.14165
5. Conversational Agents - https://arxiv.org/abs/1911.00536
6. Named Entity Recognition - https://arxiv.org/abs/1603.01360
7. Summarization - https://arxiv.org/abs/1705.04304
8. Question Answering - https://arxiv.org/abs/1810.04805
9. Language Modelling - https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
10. Emotion Detection - https://dl.acm.org/doi/10.1145/2818346.2830593
"""


########################################################################################################################################

# LESSON 2 - CLEANING TEXT DATA

import nltk
from nltk.tokenize import word_tokenize

# Download punkt dataset
nltk.download('punkt')

# Function to manually tokenize text
def manual_tokenize(text):
    # Split by whitespace and convert to lowercase
    tokens = text.split() 
    tokens = [token.lower() for token in tokens]
    return tokens

# Function to tokenize text using NLTK
def nltk_tokenize(text):
    tokens = word_tokenize(text)
    return tokens

# Choose pride and prejudice as the text from Project Gutenberg website
filename = 'pride_and_prejudice.txt'

# Read the book
with open(filename, 'rt', encoding='utf-8') as file:
    text = file.read()

# Manual Tokenization
manual_tokens = manual_tokenize(text)

# NLTK Tokenization
nltk_tokens = nltk_tokenize(text)

# Save the tokens to new files
with open('manual_tokens.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(manual_tokens))

with open('nltk_tokens.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(nltk_tokens))

print("Tokenization complete. Files saved as 'manual_tokens.txt' and 'nltk_tokens.txt'")


########################################################################################################################################

# LESSON 3 - BAG OF WORDS MODEL


""""
Bag-of-words implemented with scikit-learn
"""

from sklearn.feature_extraction.text import TfidfVectorizer # import TF-IDF vectorizer
import re

# Sample text documents
texts = [
    "The quick brown fox jumped over the lazy dog.",
    "The dog.",
    "The fox"
]

# Basic text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Clean the texts
cleaned_texts = [clean_text(text) for text in texts]

# Create the transform
vectorizer = TfidfVectorizer()

# Tokenize and build vocab
vectorizer.fit(cleaned_texts)

# Summarize
print("Vocabulary: ", vectorizer.vocabulary_)
print("IDF: ", vectorizer.idf_)

# Encode document
vector = vectorizer.transform([cleaned_texts[0]])

# Summarize encoded vector
print("Shape: ", vector.shape)
print("Encoded Vector: ", vector.toarray())

"""
Bag-of-words implemented with keras
"""

from keras.preprocessing.text import Tokenizer # import tokenizer

# Define documents
docs = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!'
]

# Basic text cleaning
cleaned_docs = [clean_text(doc) for doc in docs]

# Create the tokenizer
tokenizer = Tokenizer()

# Fit the tokenizer on the documents
tokenizer.fit_on_texts(cleaned_docs)

# Summarize what was learned
print("Word Counts: ", tokenizer.word_counts)  # dictionary mapping words to their number of occurrences
print("Document Count: ", tokenizer.document_count) # number of documents that were used to fit the Tokenizer
print("Word Index: ", tokenizer.word_index) # dictionary mapping words and their uniquely assigned integers
print("Word Docs: ", tokenizer.word_docs) # dictionary mapping words and how many documents each appeared in
 
# Integer encode documents
encoded_docs = tokenizer.texts_to_matrix(cleaned_docs, mode='count') # mode can be 'binary', 'count', 'tfidf', 'freq'
print("Encoded Documents:\n", encoded_docs) # one-hot encoded vectors for each document

########################################################################################################################################

# LESSON 4 - WORD EMBEDDING REPRESENTATION

import gensim  # gensim for word embedding
import string # string for text cleaning
from gensim.models import Word2Vec # import Word2Vec model
from nltk.tokenize import word_tokenize, sent_tokenize # import NLTK tokenizer
from sklearn.decomposition import PCA # import PCA for dimensionality reduction
from matplotlib import pyplot # import pyplot for plotting
import nltk # import NLTK

# Download punkt dataset
nltk.download('punkt')

# Function to clean and tokenize text
def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    sentences = sent_tokenize(text) # split into sentences
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences] # tokenize each sentence
    return tokenized_sentences

# Choose pride and prejudice as the text from Project Gutenberg website
filename = 'pride_and_prejudice.txt'
with open(filename, 'r', encoding='utf-8') as file:
    text = file.read()  # read the book

# Clean and tokenize text
tokenized_text = clean_and_tokenize(text)

# Train Word2Vec model
model = Word2Vec(tokenized_text, min_count=5, vector_size=100, window=5)

# Summarize the loaded model
print(model)

# Summarize vocabulary
words = list(model.wv.index_to_key)  # get all words in the vocabulary
print(words[:10])  # print first 10 words

# Fit a 2D PCA model to the vectors
X = model.wv[words]
pca = PCA(n_components=2) # 2 principal components
result = pca.fit_transform(X)  # fit PCA model to the vectors

# Create a scatter plot of the projection
pyplot.figure(figsize=(10, 10))
for i, word in enumerate(words[:50]):  # plot first 50 words
    pyplot.scatter(result[i, 0], result[i, 1]) # plot word
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1])) # annotate word
pyplot.show() # show plot

########################################################################################################################################

# LESSON 5 - LEARNED EMBEDDINGS

from keras.preprocessing.text import Tokenizer  # for tokenization
from keras.preprocessing.sequence import pad_sequences # for padding
from keras.models import Sequential # for sequential model
from keras.layers import Dense, Embedding, Flatten # for dense, embedding and flatten layers
import numpy as np # for numpy arrays

# Sample sentences and labels
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!', # positive labels
        'Poor effort',
        'Not good', # negative labels
        'Could have done better',
        'Poor work',
        'Needs improvement']

# Binary classification labels
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs) # fit tokenizer on documents

# Integer encode the documents
encoded_docs = tokenizer.texts_to_sequences(docs) 
print(f"Encoded Docs: {encoded_docs}") # one-hot encoded vectors for each document

# Pad documents to a max length
max_length = 5 # max length of a document
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post') # pad documents to max length
print(f"Padded Docs: {padded_docs}") # padded one-hot encoded vectors for each document

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length)) # embedding layer
model.add(Flatten()) # flatten layer
model.add(Dense(1, activation='sigmoid')) # dense layer

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

# Summarize the model
print(model.summary())

# Train the model
model.fit(padded_docs, labels, epochs=50, verbose=0)


########################################################################################################################################

# LESSON 6 - CLASSIFYING TEXT

"""
Setting Up the Word Embedding Layer:
    I decide on the vocabulary size based on my dataset. It's usually between 10,000 and 100,000 words. For larger datasets, I might go higher.
    The embedding dimensions are a bit of a balancing act. I've experimented with anything from 50 to 300. While higher dimensions capture more nuanced word relationships, they do increase the computational load.
    Sometimes, I jumpstart the process with pre-trained embeddings like GloVe or Word2Vec. This is particularly handy when my training data is limited.


Tweaking the Convolutional Layer:
    The number of filters is something I play around with. Starting from 32, I might go up to 256 or more, depending on how complex the patterns I'm trying to learn are.
    For the kernel size, I usually oscillate between 3, 5, and 7. This choice dictates how many words the layer looks at in one go. Smaller sizes are great for picking up on things like bigrams, while larger sizes help in understanding longer phrases.
    I typically stick to a stride of 1, so the filter moves across the sentence one word at a time.
    
    
Pooling Layer Considerations:
    I often use MaxPooling to reduce the output size from the convolutional layer. A pool size of 2 works well for me.


Fully-Connected and Output Layers:
    I add one or more dense layers with a non-linear activation function like ReLU. The number of units in these layers depends on how complex my task is.
    For the output layer, a single neuron with a sigmoid activation function does the trick for binary classification. For more categories, I switch to softmax.


Regularization Techniques:
    To avoid overfitting, I use dropout after pooling or dense layers.
    Sometimes, I also include batch normalization after convolutional or dense layers to stabilize the learning process.
"""
