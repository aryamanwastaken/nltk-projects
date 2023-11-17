# PROGRAM_0


import math
import pandas as pd  # for dataframes 
import numpy as np # for arrays
from IPython.display import Image # for displaying images

# Define the documents and the query string
doc1 = "I want to start learning to charge something in life"
doc2 = "reading something about life no one else knows"
doc3 = "Never stop learning"
query = "life learning"

# Function to compute term frequency (TF) for each document
def compute_tf(docs_list):
    for doc in docs_list:
        # Split the document into words
        doc_lst = doc.split(" ")
        # Create a dictionary with each word and its frequency count initialized to 0
        wordDict = dict.fromkeys(set(doc_lst), 0)

        # Count the occurrences of each word in the document
        for token in doc_lst:
            wordDict[token] +=  1
        # Create a DataFrame to display the term frequencies
        df = pd.DataFrame([wordDict])
        idx = 0
        new_col = ["Term Frequency"]    
        df.insert(loc=idx, column='Document', value=new_col)  # add new column to the dataframe
        print(df)

# Compute term frequency for each document
compute_tf([doc1, doc2, doc3])

# Function to compute normalized term frequency
def termFrequency(term, document):
    normalizeDocument = document.lower().split()  # Convert the document to lowercase
    return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))  # Compute the term frequency

# Function to compute normalized TF for each document
def compute_normalizedtf(documents):
    tf_doc = []
    for txt in documents:  # Iterate through the list of documents
        sentence = txt.split()  # Split the document into words
        norm_tf = dict.fromkeys(set(sentence), 0)  # Create a dictionary with each word and its frequency count initialized to 0
        for word in sentence:  # Count the occurrences of each word in the document
            norm_tf[word] = termFrequency(word, txt)  # Compute the term frequency
        tf_doc.append(norm_tf)  # Append the normalized TF to the list
        df = pd.DataFrame([norm_tf])  # Create a DataFrame to display the normalized TF 
        idx = 0  
        new_col = ["Normalized TF"]   # Add new column to the dataframe
        df.insert(loc=idx, column='Document', value=new_col)  # Add new column to the dataframe
        print(df)
    return tf_doc

# Compute normalized term frequency for each document
tf_doc = compute_normalizedtf([doc1, doc2, doc3])

# Function to compute inverse document frequency (IDF)
def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0  
    for doc in allDocuments:  # Iterate through the list of documents
        if term.lower() in doc.lower().split():  # Check if the term is present in the document
            numDocumentsWithThisTerm += 1  # Increment the count of documents containing the term

    if numDocumentsWithThisTerm > 0:  # If the term is present in at least one document
        return 1.0 + math.log(float(len(allDocuments)) / numDocumentsWithThisTerm)  # Compute IDF
    else:  # If the term is not present in any document
        return 1.0

# Function to compute IDF for each term in the documents
def compute_idf(documents):
    idf_dict = {}  # Create a dictionary to store IDF for each term
    for doc in documents:  # Iterate through the list of documents
        sentence = doc.split()  # Split the document into words
        for word in sentence:  # Iterate through the words in the document
            idf_dict[word] = inverseDocumentFrequency(word, documents)  # Compute IDF for the word
    return idf_dict  # Return the dictionary

# Compute IDF for each term in the documents
idf_dict = compute_idf([doc1, doc2, doc3])

# Function to compute TF-IDF for all documents with respect to the query
def compute_tfidf_with_alldocs(documents, query):
    tf_idf = []
    index = 0  
    query_tokens = query.split()  # Split the query into words
    df = pd.DataFrame(columns=['doc'] + query_tokens)  # Create a DataFrame to display the TF-IDF
    for doc in documents:  # Iterate through the list of documents
        df['doc'] = np.arange(0, len(documents))  # Add a column to the DataFrame to display the document number
        doc_num = tf_doc[index]  # Get the normalized TF for the document
        sentence = doc.split()  # Split the document into words
        for word in sentence:  # Iterate through the words in the document
            for text in query_tokens:  # Iterate through the words in the query
                if(text == word):  # Check if the word in the document matches the word in the query
                    idx = sentence.index(word)  # Get the index of the word in the document
                    tf_idf_score = doc_num[word] * idf_dict[word]  # Compute TF-IDF
                    tf_idf.append(tf_idf_score)  # Append the TF-IDF to the list
                    df.iloc[index, df.columns.get_loc(word)] = tf_idf_score  # Add the TF-IDF to the DataFrame
        index += 1  
    df.fillna(0, axis=1, inplace=True)  # Replace NaN values with 0
    return tf_idf, df  

# Compute TF-IDF for all documents
documents = [doc1, doc2, doc3]
tf_idf, df = compute_tfidf_with_alldocs(documents, query)  # Call the function to compute TF-IDF
print(df)

# Function to compute normalized TF for the query string
def compute_query_tf(query):
    query_norm_tf = {}  # Create a dictionary to store normalized TF for the query
    tokens = query.split()  # Split the query into words
    for word in tokens:  # Iterate through the words in the query
        query_norm_tf[word] = termFrequency(word, query)  # Compute the term frequency
    return query_norm_tf  # Return the dictionary

# Compute normalized TF for the query
query_norm_tf = compute_query_tf(query)
print(query_norm_tf)

# Function to compute IDF for the query string
def compute_query_idf(query):
    idf_dict_qry = {}  # Create a dictionary to store IDF for the query
    sentence = query.split()  # Split the query into words
    documents = [doc1, doc2, doc3]  # Create a list of documents
    for word in sentence:  # Iterate through the words in the query
        idf_dict_qry[word] = inverseDocumentFrequency(word, documents)  # Compute IDF for the word
    return idf_dict_qry

# Compute IDF for the query
idf_dict_qry = compute_query_idf(query)
print(idf_dict_qry)

# Function to compute TF-IDF for the query string
def compute_query_tfidf(query):
    tfidf_dict_qry = {}  # Create a dictionary to store TF-IDF for the query
    sentence = query.split()  # Split the query into words
    for word in sentence:  # Iterate through the words in the query
        tfidf_dict_qry[word] = query_norm_tf[word] * idf_dict_qry[word]  # Compute TF-IDF for the word
    return tfidf_dict_qry  

# Compute TF-IDF for the query
tfidf_dict_qry = compute_query_tfidf(query)
print(tfidf_dict_qry)

# Function to calculate cosine similarity between query and a document
def cosine_similarity(tfidf_dict_qry, df, query, doc_num):
    dot_product = 0
    qry_mod = 0
    doc_mod = 0
    tokens = query.split()  # Split the query into words
   
    for keyword in tokens:  # Iterate through the words in the query
        dot_product += tfidf_dict_qry[keyword] * df[keyword][df['doc'] == doc_num]  # Compute the dot product
        qry_mod += tfidf_dict_qry[keyword] * tfidf_dict_qry[keyword]  # Compute the query vector magnitude
        doc_mod += df[keyword][df['doc'] == doc_num] * df[keyword][df['doc'] == doc_num]  # Compute the document vector magnitude
    qry_mod = np.sqrt(qry_mod)  # Compute the query vector magnitude
    doc_mod = np.sqrt(doc_mod)  # Compute the document vector magnitude
    denominator = qry_mod * doc_mod  # Compute the denominator of the cosine similarity formula
    cos_sim = dot_product / denominator  # Compute the cosine similarity
     
    return cos_sim

# Function to flatten nested lists
def flatten(lis):
    for item in lis:  # Iterate through the list
        if isinstance(item, Iterable) and not isinstance(item, str):  # Check if the item is iterable
            for x in flatten(item): # Iterate through the nested list
                yield x 
        else: # If the item is not iterable
            yield item

# Function to rank documents based on cosine similarity
def rank_similarity_docs(data):
    cos_sim = []
    for doc_num in range(0, len(data)):  # Iterate through the documents
        cos_sim.append(cosine_similarity(tfidf_dict_qry, df, query, doc_num).tolist())  # Compute cosine similarity
    return cos_sim

# Rank documents based on similarity to the query
similarity_docs = rank_similarity_docs(documents)
doc_names = ["Document1", "Document2", "Document3"]  # Create a list of document names
print(doc_names)
print(list(flatten(similarity_docs)))  # Print the list of cosine similarity scores
