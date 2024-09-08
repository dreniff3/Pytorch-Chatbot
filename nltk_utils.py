'''
    This module...
'''

import nltk
from nltk.stem.porter import PorterStemmer

# download pre-trained tokenizer so .word_tokenize works
# nltk.data.path = [r'C:\Users\Don Reniff\AppData\Roaming\nltk_data']
# nltk.download('punkt')

# create stemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    '''
        Splits a sentence into meaningful units.
    '''
    return nltk.word_tokenize(sentence)


def stem(word):
    '''
        Generates the root form of words (using a crude heuristic that chops
        the end of words).
    '''
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass


s = "How long does shipping take?"
print(s)
s = tokenize(s)
print(s)

words = ["organize", "organizes", "organizing"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)
