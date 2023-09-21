import nltk
from nltk.corpus import gutenberg   # using the gutenberg corpus as an example https://www.nltk.org/book/ch02.html
from nltk import bigrams            # for generating bigrams
from collections import Counter     # for counting the number of occurrences of each type

nltk.download('gutenberg')          # download the corpus

# Get the first 500 sentences from the corpus
sentences = gutenberg.sents()[:500]
# Flatten the list of sentences into a list of words
words = [word for sent in sentences for word in sent]

tokens = [word.lower() for word in words]  # convert all words to lowercase

bi_grams = list(bigrams(tokens))    # generate bigrams from the tokens

word_freq = Counter(tokens)         # count the number of occurrences of each type
bigram_freq = Counter(bi_grams)     # count the number of occurrences of each bigram

# (count(w1 w2)/count(w1))
bigram_probabilities = {bigram: freq/word_freq[bigram[0]] for bigram, freq in bigram_freq.items()}  # calculate the probability of each bigram

# Function to predict the next word
def predict_next_word(inpW, bigram_probabilities):
    predictions = {}
    for (w1, w2), prob in bigram_probabilities.items():
        if w1 == inpW:
            predictions[w2] = prob

    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)  # sort the predictions by probability

    output = "Possible next words: "
    for word, prob in sorted_predictions:
        output += f"{word}: {prob:.6f}, "
    return output[:-2]  # remove the trailing comma and space

user_input = input("Please enter a word: ").lower()  # user input
print(predict_next_word(user_input, bigram_probabilities))  # print the output
