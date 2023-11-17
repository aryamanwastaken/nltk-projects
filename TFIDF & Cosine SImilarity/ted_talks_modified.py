# PROGRAM_2

import math
import pandas as pd
import numpy as np
from collections.abc import Iterable

# Load the ted_main.csv dataset
ted_talks = pd.read_csv('ted_main.csv')
descriptions = ted_talks['description'].tolist()
query = "personal development innovation"

# Function to compute term frequency (TF) for each document
def compute_tf(docs_list):
    
    tf_list = []
    for doc in docs_list:  # Iterate through the list of documents
        doc_words = doc.split()  # Split the document into words
        word_dict = dict.fromkeys(set(doc_words), 0)  # Create a dictionary with each word and its frequency count initialized to 0
        for word in doc_words:  # Count the occurrences of each word in the document
            word_dict[word] += 1  # Compute the term frequency
        tf_list.append(word_dict)  # Append the TF to the list
    return tf_list

# Compute TF for each document in the dataset
tf_docs = compute_tf(descriptions)

# Function to compute normalized TF for a given text (document or query)
def compute_normalized_tf(text):
    
    words = text.lower().split()  # Convert the text to lowercase
    tf_dict = dict.fromkeys(set(words), 0)  # Create a dictionary with each word and its frequency count initialized to 0
    for word in words:  # Count the occurrences of each word in the text
        tf_dict[word] = words.count(word) / float(len(words))  # Compute the term frequency
    return tf_dict

# Compute normalized TF for the query
query_tf = compute_normalized_tf(query)

# Function to calculate cosine similarity using only TF
def cosine_similarity_tf(doc_tf, query_tf):
    
    dot_product = 0
    doc_magnitude = 0
    query_magnitude = 0
    
    for word in query_tf:  # Iterate through each word in the query
        dot_product += query_tf.get(word, 0) * doc_tf.get(word, 0)  # Compute the dot product of the TF values
        doc_magnitude += doc_tf.get(word, 0) ** 2  # Compute the magnitude of the TF values
        query_magnitude += query_tf.get(word, 0) ** 2  # Compute the magnitude of the TF values
    doc_magnitude = np.sqrt(doc_magnitude)  # Compute the magnitude of the TF values
    query_magnitude = np.sqrt(query_magnitude)  # Compute the magnitude of the TF values
    
    if doc_magnitude * query_magnitude == 0:  # If the denominator is 0 return 0
        return 0
    else:  # If the denominator is not 0 return the cosine similarity
        return dot_product / (doc_magnitude * query_magnitude)

# Function to rank documents based on cosine similarity using TF
def rank_similarity_docs_tf(tf_docs, query_tf):
    
    cos_sim = []
    for doc_tf in tf_docs:  # Iterate through the list of TF values for each document
        cos_sim.append(cosine_similarity_tf(doc_tf, query_tf))  # Compute cosine similarity
    return cos_sim

# Rank documents based on similarity to the query using TF
similarity_scores_tf = rank_similarity_docs_tf(tf_docs, query_tf)

# Flatten the list of cosine similarity scores
def flatten(lis):
    for item in lis:  # Iterate through the list 
        if isinstance(item, Iterable) and not isinstance(item, str):  # Check if the item is iterable
            for x in flatten(item):  # Iterate through the nested list
                yield x
        else:  # If the item is not iterable yield the item
            yield item 

# Print the flattened list of cosine similarity scores
flattened_scores = list(flatten(similarity_scores_tf))  
print(flattened_scores)  
