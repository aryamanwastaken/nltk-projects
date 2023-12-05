# LESSON 7 : Movie Review Sentiment Analysis Project

import os  # for path operations
import string   # for string operations
import numpy as np  # for array operations
from keras.preprocessing.text import Tokenizer  # for tokenization
from keras.preprocessing.sequence import pad_sequences  # for padding
from keras.models import Sequential  # for neural networks
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D  # for neural networks
from sklearn.model_selection import train_test_split  # for splitting the data

# Function to load and clean the dataset
def load_doc(filename):
    with open(filename, 'r', encoding='utf-8') as file:  # open the file as read only
        text = file.read()
    tokens = text.split()  # split into tokens by white space
    table = str.maketrans('', '', string.punctuation)  # remove punctuation from each token
    tokens = [w.translate(table).lower() for w in tokens]  # remove punctuation from each word
    return ' '.join(tokens)  # join tokens into a string

def process_docs(directory):
    documents = list() # list for storing all reviews
    for filename in os.listdir(directory): # iterate through files in the directory
        if not filename.endswith(".txt"):  # skip files that do not have a .txt extension
            continue 
        path = directory + '/' + filename  # create the full path of the file to open
        doc = load_doc(path)  # load the doc
        documents.append(doc)  # add to the list
    return documents  # return the list

positive_docs = process_docs('txt_sentoken/pos')  # load positive reviews
negative_docs = process_docs('txt_sentoken/neg')  # load negative reviews
docs = positive_docs + negative_docs  # combine positive and negative reviews

tokenizer = Tokenizer() # create the tokenizer
tokenizer.fit_on_texts(docs) # fit the tokenizer on the documents

encoded_docs = tokenizer.texts_to_sequences(docs) # integer encode the documents
max_length = max([len(s.split()) for s in docs]) # find the maximum length of a list
X = pad_sequences(encoded_docs, maxlen=max_length, padding='post') # pad sequences

y = np.array([1 for _ in range(len(positive_docs))] + [0 for _ in range(len(negative_docs))]) # create labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split the data

model = Sequential() # create the model
model.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length=max_length)) # embedding layer
model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) # convolutional layer
model.add(GlobalMaxPooling1D()) # pooling layer
model.add(Dense(1, activation='sigmoid')) # output layer

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   # compile the model

print(model.summary())  # summarize the model
 
model.fit(X_train, y_train, epochs=10, verbose=2) # fit the model

loss, accuracy = model.evaluate(X_test, y_test, verbose=0) # evaluate the model
print(f'Test Accuracy: {accuracy}') # evaluate the model

# Function to predict sentiment of new reviews
def predict_sentiment(review, vocab, tokenizer, max_length, model):
    tokens = review.split() # split into tokens by white space
    tokens = [w.lower() for w in tokens] # convert to lower case
    tokens = [w for w in tokens if w in vocab] # remove out of vocabulary words
    line = ' '.join(tokens) # join tokens into a string
    encoded = tokenizer.texts_to_sequences([line]) # integer encode the document
    padded = pad_sequences(encoded, maxlen=max_length, padding='post') # pad documents to a max length of 4 words
    return model.predict(padded, verbose=0)  # predict sentiment for the review

new_review = 'This movie is fantastic! I really loved it.' # test with a new review
print(predict_sentiment(new_review, tokenizer.word_index, tokenizer, max_length, model)) # predict sentiment
