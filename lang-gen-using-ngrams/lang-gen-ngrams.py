# Importing necessary libraries and modules
from nltk.util import bigrams, ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize
from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd

# Example sentences for demonstration
text = [['I','need','to','book', 'ticket', 'to', 'Australia' ], ['I', 'want', 'to' ,'read', 'a' ,'book', 'of' ,'Shakespeare']]

# Displaying bigrams for the first sentence
print("bigrams: ", list(bigrams(text[0])))

# Displaying trigrams for the second sentence
print("n-grams: ", list(ngrams(text[1], n=3)))

# Loading the CSV file containing Donald Trump's tweets or statements
df = pd.read_csv('lang-gen-using-ngrams/realdonaldtrump.csv')

# Displaying the first few rows of the dataframe for a quick look
df.head()

# Tokenizing the 'content' column of the dataframe to prepare for model training
trump_corpus = list(df['content'].apply(word_tokenize))

# Setting the n-gram size to 3 (trigram)
n = 3

# Using the padded_everygram_pipeline to prepare training data. 
# This function returns two iterators: one for the training data and another for the padded sentences.
train_data, padded_sents = padded_everygram_pipeline(n, trump_corpus)

# Initializing the MLE (Maximum Likelihood Estimation) model for trigrams
trump_model = MLE(n)

# Training the model using the prepared data
trump_model.fit(train_data, padded_sents)

# Function to generate sentences using the trained model
def generate_sent(model, num_words, random_seed=42):
    
    # Utility to convert a list of tokens back into a coherent sentence
    detokenize = TreebankWordDetokenizer().detokenize
    
    content = []
    
    # Generating tokens using the model
    for token in model.generate(num_words, random_seed=random_seed):
        # Skipping the start token
        if token == '<s>':
            continue
        # Breaking the loop if the end token is encountered
        if token == '</s>':
            break
        # Appending the generated token to the content list
        content.append(token)
    
    # Detokenizing the content list to form a coherent sentence and returning it
    return detokenize(content)

# Generating and displaying sentences using the trained model
# Example generated sentences:
# 1. "for 200 years . Thank you so!"
# 2. "she treated me fairly, they will be incredible—best in"
# 3. "America is great and we will make it even better."
# 4. "Fake news media has been very unfair to me."
# 5. "We have the best economy in the history of our country."
print(generate_sent(trump_model, num_words=15, random_seed=42))
print(generate_sent(trump_model, num_words=20, random_seed=0))
