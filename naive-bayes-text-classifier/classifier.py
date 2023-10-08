import csv
from collections import defaultdict

# reading training data from train.csv
with open("./naive-bayes-text-classifier/train.csv", "r") as file:
    reader = csv.reader(file)
    training_data = list(reader)

# calculating the total number of documents in the training data
total_docs = len(training_data)

# initializing dictionaries to store counts
doc_counts = defaultdict(int)  # count of documents per class (e.g., positive, negative)
word_counts = defaultdict(lambda: defaultdict(int))  # count of each word per class
total_words = defaultdict(int)  # total words per class

# processing each document and its label in the training data
for doc, label in training_data:
    # incrementing the document count for the given label
    doc_counts[label] += 1
    
    # spliting the document into words and count them for the given label
    for word in doc.split():
        word_counts[label][word] += 1
        total_words[label] += 1

# calculating prior probabilities for each class
# prior probability = (num of documents of a class) / (total documents)
prior_probs = {label: count / total_docs for label, count in doc_counts.items()}

# calculating likelihood probabilities for each word given a class
# likelihood probability = (count of a word in a class) / (total words in that class)
likelihood_probs = {}
for label, words in word_counts.items():
    for word, count in words.items():
        likelihood_probs[(word, label)] = count / total_words[label]

# writing the calculated probabilities to model.csv
with open("./naive-bayes-text-classifier/model.csv", "w") as file:
    writer = csv.writer(file)
    
    # writing prior probabilities
    writer.writerow(["PP"])
    for label, prob in prior_probs.items():
        writer.writerow([label, prob])
    
    # writing likelihood probabilities
    writer.writerow(["LP"])
    for (word, label), prob in likelihood_probs.items():
        writer.writerow([word, label, prob])
