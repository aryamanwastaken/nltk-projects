import nltk
from nltk.corpus import gutenberg   # using the gutenberg corpus as an example https://www.nltk.org/book/ch02.html
import Levenshtein                  # for calculating the Levenshtein distance

from collections import Counter     # for counting the number of occurrences of each type


nltk.download('gutenberg')          # download the corpus
words = gutenberg.words()           # get the words from the corpus
N = len(words)                      # get the total number of words in the corpus
vocabulary = set(words)             # get the vocabulary (unique words) in the corpus


word_freq = Counter(words)          # count the number of occurrences of each type


relative_freq = {word: freq/N for word, freq in word_freq.items()}  # calculate the relative frequency of each type


def get_closest_words(user_string, n=5):  # function to get the closest words to the input string

    # calculate the Levenshtein distance between the input string and each word in the vocabulary
    distances = [(word, Levenshtein.distance(user_string, word)) for word in vocabulary] 
    sorted_dist = sorted(distances, key=lambda x: x[1])   # sort the distances by the second element (the distance)

    closest_words = sorted_dist[:n]  # get the n closest words
    return [(word, relative_freq[word]) for word, _ in closest_words]  # return the closest words and their probabilities

def process_input(XYZ):   # function to process the input string

    if XYZ in vocabulary:   # if the input string is in the vocabulary, return the probability
        return f'"{XYZ}" is a complete and correct word as per corpus Gutenberg, and its probability is {relative_freq[XYZ]:.6f}'
    
    else:   # else return the closest words
        closest_words = get_closest_words(XYZ)
        output = [f'"{word}" with probability {prob:.6f}' for word, prob in closest_words] # format the output
        return f'"{XYZ}" is not in the corpus. Closest words are: ' + ', '.join(output)  # output for 5 closest words


user_string = input("Please enter a word: ")  # user input
print(process_input(user_string))             # print the output
