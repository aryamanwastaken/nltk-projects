import csv
from collections import defaultdict
import math

# reading the trained model from model.csv
with open("./naive-bayes-text-classifier/model.csv", "r") as file:
    reader = csv.reader(file)
    model_data = list(reader)

# reading the test data from test.csv
with open("./naive-bayes-text-classifier/test.csv", "r") as file:
    reader = csv.reader(file)
    test_data = list(reader)

# initializing dictionaries to store prior and likelihood probabilities
prior_probs = {}
likelihood_probs = defaultdict(lambda: defaultdict(float))
mode = None  # To track the current section in model.csv

# extracting probabilities from the model data
for row in model_data:
    
    if not row:  # checking for empty rows
        continue
    
    if row[0] == "PP":
        mode = "PP"
        continue
    
    elif row[0] == "LP":
        mode = "LP"
        continue

    # Extracting prior probabilities
    if mode == "PP":
        prior_probs[row[0]] = float(row[1])
    
    # extracting likelihood probabilities
    elif mode == "LP":
        word, label, prob = row
        likelihood_probs[word][label] = float(prob)

# listing to store predictions
predictions = []

# predicting class label for each document in the test set
for doc, actual_label in test_data:
    max_prob = float('-inf')  # initialize with negative infinity
    predicted_label = None
    
    # calculating the probability for each class
    for label, prior_prob in prior_probs.items():
        prob = math.log(prior_prob)
        
        for word in doc.split():
            # adding log likelihood of each word. Use a small value for unseen words.
            prob += math.log(likelihood_probs[word].get(label, 1e-5))
        
        # updating predicted label if current class has higher probability
        if prob > max_prob:
            max_prob = prob
            predicted_label = label
    predictions.append((doc, predicted_label, actual_label))

# writing predictions to test_predictions.csv
with open("./naive-bayes-text-classifier/test_predictions.csv", "w") as file:
    
    writer = csv.writer(file)
    
    for doc, predicted_label, actual_label in predictions:
        output_statement = f"{doc}, predicted: {predicted_label}, actual: {actual_label}"
        writer.writerow([output_statement])
